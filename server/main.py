"""
FastAPI 앱 진입점
"""
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 프로젝트 루트를 sys.path에 추가 (src 모듈 import 가능하도록)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from server.config import settings
from server.routes import router as http_router
from server.websocket import router as ws_router
from src.inference import FallDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 모델 로드/해제"""
    model_path = PROJECT_ROOT / settings.MODEL_PATH

    try:
        detector = FallDetector(
            model_path=str(model_path),
            threshold=settings.THRESHOLD,
            sample_rate=settings.SAMPLE_RATE,
        )
        app.state.detector = detector

        # checkpoint 메타 정보 저장 (model info 엔드포인트용)
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
        app.state.checkpoint_meta = {
            "accuracy": checkpoint.get("accuracy"),
            "f1": checkpoint.get("f1"),
        }
        logger.info("AI model loaded successfully from %s", model_path)
    except FileNotFoundError:
        logger.warning("Model file not found at %s - server running without model", model_path)
        app.state.detector = None
        app.state.checkpoint_meta = {}

    app.state.active_connections = 0

    yield

    logger.info("Shutting down AI server")


app = FastAPI(
    title="AudioFallCare AI Server",
    description="실시간 오디오 낙상 감지 AI 서버",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(http_router)
app.include_router(ws_router)


if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",
        host=settings.WS_HOST,
        port=settings.WS_PORT,
        reload=True,
    )
