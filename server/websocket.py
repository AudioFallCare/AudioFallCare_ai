"""
WebSocket 오디오 스트리밍 핸들러
"""
import logging

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.config import settings
from server.notifier import notify_fall

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/audio/stream")
async def audio_stream(websocket: WebSocket):
    """
    실시간 오디오 스트리밍 WebSocket 엔드포인트

    Query params:
        code: 연결 코드
        guardianId: 보호자 ID
        recorderId: 리코더 ID
    """
    code = websocket.query_params.get("code", "")
    guardian_id = websocket.query_params.get("guardianId", "")
    recorder_id = websocket.query_params.get("recorderId", "")

    await websocket.accept()
    app = websocket.app
    app.state.active_connections += 1

    logger.info(
        "WebSocket connected: code=%s guardianId=%s recorderId=%s",
        code, guardian_id, recorder_id,
    )

    detector = app.state.detector

    try:
        while True:
            # 3초 단위 오디오 청크 (binary) 수신
            data = await websocket.receive_bytes()

            # int16 PCM → float32 numpy array
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # [실험 코드] 현재 들어오는 소리의 크기를 측정
            rms = np.sqrt(np.mean(audio_chunk**2))
            peak = np.max(np.abs(audio_chunk))
            std = np.std(audio_chunk)
            logger.info(f"[Audio Check] RMS: {rms:.6f} | Peak: {peak:.6f} | STD: {std:.6f}")
            
            # 소리 크기가 충분히 작으면 무시(잡음 방지)
            if rms < 0.1:
                continue

            # AI 추론
            result = detector.predict_stream(audio_chunk, settings.SAMPLE_RATE)
            logger.info(f"[AI Result] Confidence: {result['confidence']:.4f}")

            # 낙상 감지 시 Spring 서버에 알림
            if result["fall"]:
                logger.warning(
                    "Fall detected! code=%s confidence=%.4f soundType=%s",
                    code, result["confidence"], result["soundType"],
                )
                await notify_fall(
                    guardian_id=guardian_id,
                    recorder_id=recorder_id,
                    confidence=result["confidence"],
                    sound_type=result["soundType"],
                )

            # 결과를 리코더에도 응답
            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: code=%s recorderId=%s", code, recorder_id)
    except Exception:
        logger.exception("WebSocket error: code=%s", code)
    finally:
        app.state.active_connections -= 1
