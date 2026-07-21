from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import base64
import requests

from vlm_courtroom.config import get_model, OLLAMA_BASE_URL


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
    ):
        """
        backend: "gemini" (기존 Vertex AI 경로, 기본값 — 기존 호출부와 100% 호환)
                 "ollama"  (로컬 Ollama 서버로 전환. ollama_model 필수)
        ollama_model: backend="ollama"일 때 사용할 모델 태그
                      (예: "qwen2.5vl:7b", "llava:13b", "llama3.2-vision:11b")
        """
        self.name = name
        self.role = role
        self.model_role = model_role
        self.backend = backend
        self.ollama_model = ollama_model
        self.memory: List[Message] = []

        if backend == "gemini":
            # 기존 동작 그대로: Vertex AI GenerativeModel 인스턴스 생성
            self.model = get_model(model_role)
        elif backend == "ollama":
            if not ollama_model:
                raise ValueError(
                    f"[{name}] backend='ollama'인데 ollama_model이 지정되지 않았습니다."
                )
            # Ollama는 Vertex AI 모델 객체가 필요 없음 (HTTP API로 직접 호출)
            self.model = None
        else:
            raise ValueError(f"[{name}] 알 수 없는 backend: {backend!r} (gemini 또는 ollama만 지원)")

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
        """VLM 호출 진입점. backend에 따라 Gemini 또는 Ollama로 분기."""
        if self.backend == "gemini":
            return self._generate_gemini(prompt, image_path)
        elif self.backend == "ollama":
            return self._generate_ollama(prompt, image_path)
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

    def _generate_ollama(self, prompt: str, image_path: Optional[str] = None) -> str:
        """Ollama 로컬 서버(/api/generate) 호출. Vision 입력은 base64 인코딩된
        images 리스트로 전달 (Ollama API 규격)."""
        try:
            payload: Dict[str, Any] = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
            }

            if image_path:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                payload["images"] = [base64.b64encode(image_bytes).decode("utf-8")]

            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=300,  # 로컬 모델은 응답이 느릴 수 있어 넉넉히
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except requests.exceptions.RequestException as e:
            return f"Error generating response (Ollama connection): {e}"
        except Exception as e:
            return f"Error generating response (Ollama): {e}"