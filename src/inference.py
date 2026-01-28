"""
모델 추론 스크립트
"""


def predict(audio_data):
    """
    오디오 데이터로 낙상 여부 판별

    Args:
        audio_data: 오디오 데이터

    Returns:
        dict: {fall: bool, confidence: float, soundType: str}
    """
    return {
        "fall": False,
        "confidence": 0.0,
        "soundType": "unknown"
    }


if __name__ == "__main__":
    result = predict(None)
    print(result)
