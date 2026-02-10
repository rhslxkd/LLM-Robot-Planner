# 🏛️ VLM Courtroom: 상세 코드 리뷰 (Detailed Code Review)

이 문서는 **VLM Courtroom** 프로젝트의 모든 파일에 대해, **실제 코드**와 함께 **상세한 설명**을 제공합니다. 

---

## 📂 프로젝트 구조 (Project Structure)

```
HELM_Brain/
├── vlm_courtroom/
│   ├── __init__.py            # 패키지 인식용 파일
│   ├── config.py              # Vertex AI 설정 및 초기화
│   ├── main_court.py          # 메인 실행 파일
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py      # 에이전트 부모 클래스
│   │   └── specific_agents.py # 4명의 특수 에이전트 정의
│   └── court/
│       ├── __init__.py
│       └── courtroom.py       # 재판 진행 및 시각화
```

---

## 0️⃣ `__init__.py` 의 역할

```python
# (비어 있음)
```

### 💡 "얘는 도대체 역할이 뭐야??"
- **정체**: 이 파일은 파이썬에게 **"이 폴더(`vlm_courtroom`)는 단순한 폴더가 아니라, 불러와서 쓸 수 있는 '패키지(Package)'야!"** 라고 알려주는 명찰입니다.
- **기능**:
    - 이 파일이 있어야 다른 파이썬 파일에서 `from vlm_courtroom import ...` 처럼 폴더 내부의 코드를 불러올 수 있습니다.
    - 보통은 비워두지만, 패키지를 불러올 때 초기화해야 할 코드가 있다면 여기에 적기도 합니다.
- **결론**: 내용은 없어도 **존재 자체가 중요**한 파일입니다. 없으면 `import` 에러가 날 수 있습니다.

---

## 1️⃣ `vlm_courtroom/config.py`

이 파일은 구글 Vertex AI (Gemini)를 사용하기 위한 **설정 파일**입니다.

### 📜 전체 코드
```python
import os
import json
import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account

# Determine the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
KEY_PATH = "/mnt/d/_SECRETS/keys/vertex/google_vertex_key.json"
PROJECT_ID = "kaggle-genai-477714"
LOCATION = "us-central1"

# Map roles to specific models
AGENT_MODEL_MAP = {
    "JUDGE": "gemini-1.5-pro",
    "COORDINATE": "gemini-1.5-flash",
    "PROSECUTOR": "gemini-1.5-flash",
    "DEFENSE": "gemini-1.5-flash"
}

def init_vertex_ai():
    """Initializes Vertex AI with the service account key."""
    if not os.path.exists(KEY_PATH):
        raise FileNotFoundError(f"Key file not found at {KEY_PATH}")
        
    try:
        with open(KEY_PATH, 'r') as f:
            key_data = json.load(f)
            project_id = key_data.get('project_id')
            
        credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
        vertexai.init(project=project_id, location=LOCATION, credentials=credentials)
        print(f"✅ Vertex AI Initialized for project: {project_id}")
        return project_id
    except Exception as e:
        print(f"❌ Failed to initialize Vertex AI: {e}")
        raise e

def get_model(role: str = "DEFAULT"):
    """
    Returns a configured GenerativeModel instance based on the agent's role.
    Defaults to gemini-1.5-flash if role is not found.
    """
    model_name = AGENT_MODEL_MAP.get(role.upper(), "gemini-1.5-flash")
    return GenerativeModel(model_name)
```

### 🔍 상세 설명
1. **`KEY_PATH`**: 인증 키(`google_vertex_key.json`)가 있는 절대 경로입니다. 이 파일이 없으면 AI에 접속할 수 없습니다.
2. **`AGENT_MODEL_MAP`**: 각 역할별로 어떤 AI 모델을 쓸지 정해둔 지도입니다.
    - **JUDGE (판사)**: 가장 똑똑해야 하므로 `gemini-1.5-pro`를 씁니다.
    - **나머지**: 속도가 중요하므로 빠르고 저렴한 `gemini-1.5-flash`를 씁니다.
3. **`init_vertex_ai()`**: 프로그램이 시작될 때 딱 한 번 호출됩니다. 키 파일을 읽어서 구글 서버에 "나 접속할게!"라고 신고(인증)하는 함수입니다.
4. **`get_model(role)`**: 각 에이전트가 생성될 때 "저 무슨 모델 써야 해요?"라고 물어보는 함수입니다. 역할(`role`)을 주면 그에 맞는 모델(`GenerativeModel`)을 꺼내줍니다.

---

## 2️⃣ `vlm_courtroom/agents/base_agent.py`

모든 에이전트들의 **어머니(부모 클래스)**입니다. 공통적인 기능(AI와 대화하기)을 여기서 정의합니다.

### 📜 핵심 코드 (Generate Response)
```python
def generate_response(self, prompt: str, image_path: Optional[str] = None) -> str:
    """Helper to generate content from Gemini."""
    try:
        from vertexai.generative_models import Part
        
        contents = [prompt]
        if image_path:
            # 이미지가 있으면 읽어서 contents에 추가
            with open(image_path, "rb") as f:
                image_data = f.read()
            # 확장자에 따라 MIME 타입 결정
            mime_type = "image/jpeg" 
            if image_path.lower().endswith(".png"):
                mime_type = "image/png"
            
            # Vertex AI 전용 이미지 객체 생성
            image_part = Part.from_data(data=image_data, mime_type=mime_type)
            contents.append(image_part)

        # 텍스트 + 이미지 묶어서 전송!
        response = self.model.generate_content(contents)
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"
```
### 🔍 상세 설명
- **역할**: 에이전트가 말을 할 수 있게 해주는 입과 귀입니다.
- **이미지 처리**: `image_path`가 들어오면, 컴퓨터가 이해하는 0과 1의 데이터(`image_data`)로 변환하고, 이를 AI에게 보낼 수 있는 편지 봉투(`Part`)에 담습니다.
- **전송**: 텍스트 프롬프트와 이미지 봉투를 `contents` 리스트에 담아 `model.generate_content`로 보냅니다.

---

## 3️⃣ `vlm_courtroom/agents/specific_agents.py`

4명의 특수 요원들이 정의된 곳입니다.

### 📜 CoordinateAgent (좌표 생성 요원)
```python
class CoordinateAgent(VLMAgent):
    def __init__(self, name="CoordinateAgent", reset_db=False):
        super().__init__(name, "Coordinate Generator", model_role="COORDINATE")
        # ChromaDB (기억 저장소) 초기화
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        if reset_db:
            # reset_db가 참이면 기억 삭제 (테스트용)
            try:
                self.chroma_client.delete_collection("scene_coordinates")
                print(f"[{self.name}] 🗑️ Existing VectorDB collection deleted (Reset).")
            except Exception:
                pass 
        self.collection = self.chroma_client.get_or_create_collection(name="scene_coordinates")

    def process(self, context):
        # ... 프롬프트 작성 ...
        prompt = """
        Task:
        1. Analyze the scene and Explain your path planning logic.
        2. Generate 10 sequential (x, y) coordinates.
        Output Format: ## Scene Analysis ... ## Coordinates
        Example: [{"x": 1, "y": 2}]
        """
        # ... 응답 파싱 및 DB 저장 ...
```
### 🔍 상세 설명
- **ChromaDB**: AI가 생성한 경로를 `./chroma_db` 폴더에 영구 저장합니다. 나중에 "아까 비슷한 상황에서 어떻게 했지?"라고 검색할 때 쓰입니다.
- **`reset_db`**: 테스트할 때마다 예전 데이터가 쌓이면 헷갈리므로, 싹 지우고 시작하는 옵션입니다.
- **Prompt**: 단순히 좌표만 뱉는 게 아니라, **"왜 그렇게 했는지(Logic)"** 설명하도록 하여 사용자가 이해하기 쉽게 만듭니다.

---

## 4️⃣ `vlm_courtroom/court/courtroom.py`

재판을 지휘하고 결과를 그려주는 **감독관**입니다.

### 📜 시각화 및 파일 저장 코드 (Visualize Path)
```python
def visualize_path(self, image_path: str, verdict_text: str, robot_pos: tuple = None, scale: float = None):
    # ... 좌표 파싱 및 그래프 그리기 (matplotlib) ...
    
    # Imports for saving
    import os
    import shutil
    from datetime import datetime

    input_filename = os.path.basename(image_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 원본 관리: 프로젝트 내부 inputs 폴더로 복사 (기록용)
    project_input_dir = os.path.join(os.getcwd(), "vlm_courtroom", "inputs")
    os.makedirs(project_input_dir, exist_ok=True)
    target_input_path = os.path.join(project_input_dir, input_filename)
    
    if os.path.abspath(image_path) != os.path.abspath(target_input_path):
        shutil.copy2(image_path, target_input_path)
    
    # 2. 결과 저장: 프로젝트 내부 outputs 폴더에만 저장!
    project_output_dir = os.path.join(os.getcwd(), "vlm_courtroom", "outputs")
    os.makedirs(project_output_dir, exist_ok=True)
    
    output_filename = f"{filename_no_ext}_verdict_{timestamp}{ext}"
    output_path_project = os.path.join(project_output_dir, output_filename)
    
    plt.savefig(output_path_project)
    print(f"🖼️ Saved verdict to Project Outputs: {output_path_project}")
```
### 🔍 상세 설명
- **역할**: 재판이 끝나고 판결이 나오면, 이미지를 불러와서 경로를 그립니다.
- **파일 관리 (File Management)**:
    1. **Input 복사**: D드라이브에 있는 원본을 프로젝트 안(`vlm_courtroom/inputs`)으로 복사해옵니다. 나중에 원본이 없어져도 프로젝트는 돌아가게 하기 위함입니다.
    2. **Output 저장**: 결과 파일은 **오직** 프로젝트 안(`vlm_courtroom/outputs`)에만 저장합니다. D드라이브 원본 폴더는 깨끗하게 유지됩니다.

---

## 5️⃣ `vlm_courtroom/main_court.py`

사용자가 실행하는 **조종석(Cockpit)** 파일입니다.

### 📜 주요 실행 코드
```python
def main():
    # 1. AI 접속
    init_vertex_ai()
    
    # 2. 법정 개정 (DB 리셋 포함)
    court = VLMCourt(reset_db=True) 

    # 3. 이미지 설정 (Windows 경로)
    IMAGE_DIR = "/mnt/d/Datasets/HELM/Input_images/go2/" 
    image_filename = "brax (1).png"
    image_path = os.path.join(IMAGE_DIR, image_filename)
    
    # 4. 보정 값 (Calibration)
    # 로봇 위치 (950, 550) / 스케일 (300px = 1m)
    robot_pos = (950, 550) 
    scale = 300.0
    
    # 5. 재판 시작!
    court.run_case(scenario, image_path=image_path, robot_pos=robot_pos, scale=scale)
```
### 🔍 상세 설명
- **`court = VLMCourt(reset_db=True)`**: 여기서 법정을 세우면서 `reset_db=True`를 넘겨서, 시작할 때마다 벡터 DB를 깨끗이 비웁니다.
- **경로 설정**: 윈도우의 데이터셋 경로를 `IMAGE_DIR`로 잡아두어, 파일명만 바꾸면 쉽게 다른 이미지도 테스트할 수 있습니다.
- **Calibration**: `brax (1).png` 사진에 맞춰서 로봇 위치와 크기를 수동으로 잡아준 값입니다. 다른 사진을 쓸 때는 이 값을 바꿔줘야 정확하게 그려집니다.
