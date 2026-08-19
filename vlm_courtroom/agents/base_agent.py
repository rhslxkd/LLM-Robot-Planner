from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import base64
import os
import requests

from vlm_courtroom.config import get_model, OLLAMA_BASE_URL, OPENAI_API_KEY_ENV


class Message:
    def __init__(self, sender: str, content: str, role: str = "user"):
        self.sender = sender
        self.content = content
        self.role = role  # 'user', 'model', or specific agent role

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "content": self.content,
            "role": self.role
        }


class VLMAgent(ABC):
    def __init__(
        self,
        name: str,
        role: str,
        model_role: str = "DEFAULT",
        backend: str = "gemini",
        ollama_model: Optional[str] = None,
        gemini_model: Optional[str] = None,
        openai_model: Optional[str] = None,
    ):
        """
        backend: "gemini" (기존 Vertex AI 경로, 기본값) / "ollama" (로컬 서버) / "openai" (GPT)

        gemini_model: backend="gemini"일 때, 지정하면 역할별 매핑(AGENT_MODEL_MAP,
                      Judge=pro/나머지=flash 혼합)을 무시하고 4개 에이전트 전부 이 모델
                      하나로 통일한다. 백본 비교 실험의 공정성을 위해 필수적으로 씀
                      (예: "gemini-2.5-flash"만 단독으로, 혹은 "gemini-2.5-pro"만 단독으로).
                      None이면 기존처럼 역할별 혼합 매핑을 씀(공정 비교용은 아님, 참고용).
        ollama_model: backend="ollama"일 때 사용할 모델 태그 (예: "qwen2.5vl:7b").
        openai_model: backend="openai"일 때 사용할 모델명 (예: "gpt-4o", "gpt-4o-mini").
        """
        self.name = name
        self.role = role
        self.model_role = model_role
        self.backend = backend
        self.ollama_model = ollama_model
        self.gemini_model = gemini_model
        self.openai_model = openai_model
        self.memory: List[Message] = []

        if backend == "gemini":
            # gemini_model이 지정되면 4개 에이전트 전부 동일 모델로 통일(공정 비교용).
            # 지정 안 하면 기존 역할별(Judge=pro 등) 혼합 매핑 사용.
            self.model = get_model(model_role, model_name=gemini_model)
        elif backend == "ollama":
            if not ollama_model:
                raise ValueError(
                    f"[{name}] backend='ollama'인데 ollama_model이 지정되지 않았습니다."
                )
            self.model = None
        elif backend == "openai":
            if not openai_model:
                raise ValueError(
                    f"[{name}] backend='openai'인데 openai_model이 지정되지 않았습니다."
                )
            self.model = None
        else:
            raise ValueError(
                f"[{name}] 알 수 없는 backend: {backend!r} (gemini/ollama/openai만 지원)"
            )

    def add_to_memory(self, message: Message):
        self.memory.append(message)

    @abstractmethod
    def process(self, context: Dict[str, Any]) -> Message:
        """
        Process the current context and return a response Message.
        Context can include the image, previous messages, etc.
        """
        pass

    def generate_response(self, prompt: str, image_path: Optional[str] = None) -> str:
        """VLM 호출 진입점. backend에 따라 Gemini/Ollama/OpenAI로 분기."""
        if self.backend == "gemini":
            return self._generate_gemini(prompt, image_path)
        elif self.backend == "ollama":
            return self._generate_ollama(prompt, image_path)
        elif self.backend == "openai":
            return self._generate_openai(prompt, image_path)
        else:
            return f"Error: unknown backend {self.backend!r}"

    def _generate_gemini(self, prompt: str, image_path: Optional[str] = None) -> str:
        """기존 Vertex AI(Gemini) 호출 로직 — 변경 없음."""
        try:
            from vertexai.generative_models import Part

            contents = [prompt]
            if image_path:
                with open(image_path, "rb") as f:
                    image_data = f.read()
                mime_type = "image/jpeg"  # Default
                if image_path.lower().endswith(".png"):
                    mime_type = "image/png"

                image_part = Part.from_data(data=image_data, mime_type=mime_type)
                contents.append(image_part)

            response = self.model.generate_content(contents)
            return response.text
        except Exception as e:
            return f"Error generating response: {e}"

    def _generate_ollama(self, prompt: str, image_path: Optional[str] = None,
                          max_retries: int = 3, base_delay: float = 5.0) -> str:
        import time
        payload: Dict[str, Any] = {"model": self.ollama_model, "prompt": prompt, "stream": False}
        if image_path:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            payload["images"] = [base64.b64encode(image_bytes).decode("utf-8")]

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=300)
                resp.raise_for_status()
                return resp.json().get("response", "")
            except Exception as e:
                last_error = e
            if attempt == max_retries:
                break
            delay = base_delay * (2 ** attempt)
            print(f"[{self.name}] ⚠️ Ollama 에러 (시도 {attempt+1}/{max_retries+1}): {last_error} -> {delay:.0f}초 후 재시도")
            time.sleep(delay)
        return f"Error generating response (Ollama): {last_error}"

    def _generate_openai(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        max_retries: int = 4,
        base_delay: float = 5.0,
    ) -> str:
        """OpenAI Chat Completions API(vision) 호출. Gemini 쪽과 동일하게
        429/5xx 등 일시적 에러는 지수 백오프로 재시도한다.

        TODO: API 키 로드 방식(환경변수 vs 키 파일)이 실제 서버 환경과 맞는지 확인 필요 --
        지금은 config.OPENAI_API_KEY_ENV(기본 "OPENAI_API_KEY") 환경변수를 읽는다.
        """
        import time

        api_key = os.environ.get(OPENAI_API_KEY_ENV)
        if not api_key:
            return f"Error generating response: {OPENAI_API_KEY_ENV} 환경변수가 설정되지 않음"

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_path:
            with open(image_path, "rb") as f:
                image_data = f.read()
            mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
            b64 = base64.b64encode(image_data).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64}"},
            })

        payload = {
            "model": self.openai_model,
            "messages": [{"role": "user", "content": content}],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                is_transient = resp.status_code in (429, 500, 502, 503, 504)
            except requests.exceptions.RequestException as e:
                last_error = e
                is_transient = True

            if not is_transient or attempt == max_retries:
                break
            delay = base_delay * (2 ** attempt)
            print(
                f"[{self.name}] ⚠️ OpenAI 일시적 에러 (시도 {attempt + 1}/{max_retries + 1}): "
                f"{last_error} -> {delay:.0f}초 후 재시도"
            )
            time.sleep(delay)

        return f"Error generating response (OpenAI): {last_error}"