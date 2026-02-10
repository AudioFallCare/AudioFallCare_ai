"""
HTTP 엔드포인트 (health, model info)
"""
from fastapi import APIRouter, Request

router = APIRouter(prefix="/ai")


@router.get("/health")
async def health(request: Request):
    """서버 상태 확인"""
    app = request.app
    detector = app.state.detector
    return {
        "status": "ok",
        "modelLoaded": detector is not None,
        "activeConnections": app.state.active_connections,
        "device": str(detector.device) if detector else None,
    }


@router.get("/model/info")
async def model_info(request: Request):
    """모델 정보 반환"""
    app = request.app
    detector = app.state.detector
    if detector is None:
        return {"error": "Model not loaded"}

    checkpoint = app.state.checkpoint_meta
    return {
        "accuracy": checkpoint.get("accuracy"),
        "f1": checkpoint.get("f1"),
        "threshold": detector.threshold,
        "sampleRate": detector.sample_rate,
        "maxLength": detector.max_length,
        "device": str(detector.device),
    }
