import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
import json
import os

KEY_PATH = "/mnt/d/_SECRETS/keys/vertex/google_vertex_key.json"

def main():
    print(f"--- [Step 1] 경로 확인: {KEY_PATH} ---")
    if not os.path.exists(KEY_PATH):
        print("❌ 에러: 키 파일을 찾을 수 없어! 경로를 다시 확인해.")
        return
    print("✅ 키 파일 발견!")

    try:
        # 2. JSON에서 프로젝트 ID 읽기
        with open(KEY_PATH, 'r') as f:
            key_data = json.load(f)
            project_id = key_data['project_id']
        
        # 3. 자격 증명 및 초기화
        credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
        vertexai.init(project=project_id, location="us-central1", credentials=credentials)
        
        # 4. 모델 호출 (버전 명시 필수)
        model = GenerativeModel("gemini-2.5-flash")
        
        print(f"--- [Step 2] Gemini에게 말 거는 중... (Project: {project_id}) ---")
        response = model.generate_content("안녕? 연결 성공했으면 'HELM 시스템 가동'이라고 짧게 대답해줘.")
        
        print("\n[Gemini의 대답]:")
        print(response.text)
        print("\n✅ 모든 연결이 완벽해! 이제 진짜 시작할 수 있어.")

    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")

if __name__ == "__main__":
    main()