import os
import json
import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account

# Determine the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
KEY_PATH = "/home/user/hyeonsoo/Keys/Key/google_vertex_key.json"
PROJECT_ID = "kaggle-genai-477714"
LOCATION = "us-central1"

# --- Ollama (로컬 VLM 백엔드 비교 실험용) ---
OLLAMA_BASE_URL = "http://localhost:11434"

# --- OpenAI (GPT 백엔드 비교 실험용) ---
# TODO: 실제 키 경로/환경변수명 확인 후 맞출 것. 우선 환경변수 OPENAI_API_KEY 사용을 기본으로 함.
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# Map roles to specific models
AGENT_MODEL_MAP = {
    "JUDGE": "gemini-2.5-pro",
    "COORDINATE": "gemini-2.5-flash",
    "PROSECUTOR": "gemini-2.5-flash",
    "DEFENSE": "gemini-2.5-flash"
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


def get_model(role: str = "DEFAULT", model_name: str = None):
    """
    Returns a configured GenerativeModel instance based on the agent's role.

    model_name이 명시되면 AGENT_MODEL_MAP을 무시하고 그 모델을 그대로 쓴다 --
    이게 있어야 "Gemini 백본"을 Pro/Flash 혼합이 아니라 4개 에이전트 전부 동일 모델로
    통일해서 공정하게 비교할 수 있다 (Ollama 백본이 이미 그렇게 동작하는 것과 동일하게).
    model_name이 없으면 기존처럼 역할별 매핑(AGENT_MODEL_MAP)을 사용한다.
    """
    if model_name is None:
        model_name = AGENT_MODEL_MAP.get(role.upper(), "gemini-1.5-flash")
    return GenerativeModel(model_name)