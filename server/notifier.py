"""
Spring 백엔드로 낙상 알림 전송
"""
import logging
from datetime import datetime, timezone

import httpx

from server.config import settings

logger = logging.getLogger(__name__)


async def notify_fall(
    guardian_id: str,
    recorder_id: str,
    confidence: float,
    sound_type: str,
) -> bool:
    """
    Spring 백엔드에 낙상 감지 알림을 전송한다.

    Returns:
        True if notification was sent successfully, False otherwise.
    """
    payload = {
        "guardianId": guardian_id,
        "recorderId": recorder_id,
        "confidence": confidence,
        "soundType": sound_type,
        "detectedAt": datetime.now(timezone.utc).isoformat(),
    }

    url = f"{settings.SPRING_SERVER_URL}/api/internal/fall"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            logger.info("Fall notification sent: guardianId=%s confidence=%.4f", guardian_id, confidence)
            return True
    except httpx.HTTPStatusError as e:
        logger.error("Spring server returned %s: %s", e.response.status_code, e.response.text)
    except httpx.ConnectError:
        logger.error("Cannot connect to Spring server at %s", url)
    except Exception:
        logger.exception("Unexpected error sending fall notification")

    return False
