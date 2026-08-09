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


def get_model(role: str = "DEFAULT"):
    """
    Returns a configured GenerativeModel instance based on the agent's role.
    Defaults to gemini-1.5-flash if role is not found.
    (Ollama 백엔드에서는 이 함수를 호출하지 않음 — VLMAgent가 backend="ollama"일 때
    Vertex AI 모델 생성을 건너뛰도록 되어 있음.)
    """
    model_name = AGENT_MODEL_MAP.get(role.upper(), "gemini-1.5-flash")
    return GenerativeModel(model_name)