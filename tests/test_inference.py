"""
추론 테스트
"""
from src.inference import predict


def test_predict_returns_dict():
    """predict 함수가 올바른 형식을 반환하는지 테스트"""
    result = predict(None)

    assert isinstance(result, dict)
    assert "fall" in result
    assert "confidence" in result
    assert "soundType" in result
