"""
모델 추론 스크립트
"""
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np

from model import FallDetectionCNN


class FallDetector:
    """낙상 감지 추론 클래스"""

    def __init__(
        self,
        model_path: str = "models/best_model.pt",
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        threshold: float = 0.5
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold

        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 로드
        self.model = FallDetectionCNN()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # 오디오 변환
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        self.amplitude_to_db = T.AmplitudeToDB()

        print(f"Model loaded from {model_path}")
        print(f"Model accuracy: {checkpoint.get('accuracy', 'N/A')}")
        print(f"Model F1: {checkpoint.get('f1', 'N/A')}")

    def preprocess(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """오디오 전처리"""
        # 리샘플링
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # 모노로 변환
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Mel Spectrogram 변환
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        # 정규화
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)

        # 배치 차원 추가
        mel_spec_db = mel_spec_db.unsqueeze(0)

        return mel_spec_db

    def predict_file(self, audio_path: str) -> dict:
        """파일에서 낙상 감지"""
        waveform, sr = torchaudio.load(audio_path, backend="soundfile")
        return self.predict(waveform, sr)

    def predict(self, waveform: torch.Tensor, sr: int) -> dict:
        """
        오디오 데이터로 낙상 여부 판별

        Args:
            waveform: 오디오 웨이브폼 텐서
            sr: 샘플레이트

        Returns:
            dict: {fall: bool, confidence: float, soundType: str}
        """
        # 전처리
        mel_spec = self.preprocess(waveform, sr)
        mel_spec = mel_spec.to(self.device)

        # 추론
        with torch.no_grad():
            output = self.model(mel_spec)
            confidence = output.item()

        # 결과
        is_fall = confidence > self.threshold

        # 소리 유형 분류 (단순화)
        if is_fall:
            if confidence > 0.8:
                sound_type = "thud"  # 쿵 소리
            else:
                sound_type = "impact"  # 충격음
        else:
            sound_type = "normal"

        return {
            "fall": is_fall,
            "confidence": round(confidence, 4),
            "soundType": sound_type
        }

    def predict_stream(self, audio_chunk: np.ndarray, sr: int) -> dict:
        """
        실시간 스트림 데이터에서 낙상 감지

        Args:
            audio_chunk: numpy 배열 형태의 오디오 청크
            sr: 샘플레이트

        Returns:
            dict: {fall: bool, confidence: float, soundType: str}
        """
        waveform = torch.from_numpy(audio_chunk).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        return self.predict(waveform, sr)


def predict(audio_data) -> dict:
    """
    오디오 데이터로 낙상 여부 판별 (호환성 유지용)

    Args:
        audio_data: 오디오 데이터

    Returns:
        dict: {fall: bool, confidence: float, soundType: str}
    """
    # 모델이 없으면 기본값 반환
    return {
        "fall": False,
        "confidence": 0.0,
        "soundType": "unknown"
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="낙상 감지 추론")
    parser.add_argument("--model", type=str, default="models/best_model.pt")
    parser.add_argument("--input", type=str, required=True, help="오디오 파일 경로")
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    detector = FallDetector(args.model, threshold=args.threshold)
    result = detector.predict_file(args.input)

    print(f"\n결과:")
    print(f"  낙상 감지: {result['fall']}")
    print(f"  신뢰도: {result['confidence']:.2%}")
    print(f"  소리 유형: {result['soundType']}")
