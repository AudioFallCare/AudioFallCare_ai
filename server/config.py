"""
서버 환경 설정

Pydantic BaseSettings로 환경변수 관리
"""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SPRING_SERVER_URL: str = "https://audiofallcare-was-test.onrender.com"
    MODEL_PATH: str = "models/best_model.pt"
    WS_HOST: str = "0.0.0.0"
    WS_PORT: int = 8000
    THRESHOLD: float = 0.5
    SAMPLE_RATE: int = 16000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
