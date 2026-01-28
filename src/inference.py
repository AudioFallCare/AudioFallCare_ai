"""
모델 추론 스크립트

librosa: 오디오 로드 (호환성)
torchaudio: Mel Spectrogram 변환 (GPU 가속)
"""
import torch
import torchaudio.transforms as T
import librosa
import numpy as np

try:
    from model import FallDetectionCNN
except ImportError:
    from src.model import FallDetectionCNN


class FallDetector:
    """낙상 감지 추론 클래스"""

    def __init__(
        self,
        model_path: str = "models/best_model.pt",
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        max_length: int = 3,
        threshold: float = 0.5
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length
        self.max_samples = sample_rate * max_length
        self.threshold = threshold

        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 로드
        self.model = FallDetectionCNN()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # torchaudio 변환기 (GPU 가속)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        ).to(self.device)
        self.amplitude_to_db = T.AmplitudeToDB().to(self.device)

        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
        print(f"Model accuracy: {checkpoint.get('accuracy', 'N/A')}")
        print(f"Model F1: {checkpoint.get('f1', 'N/A')}")

    def preprocess(self, waveform: np.ndarray, sr: int) -> torch.Tensor:
        """오디오 전처리"""
        # 리샘플링 (librosa)
        if sr != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)

        # 고정 길이로 패딩/자르기
        if len(waveform) > self.max_samples:
            waveform = waveform[:self.max_samples]
        elif len(waveform) < self.max_samples:
            padding = self.max_samples - len(waveform)
            waveform = np.pad(waveform, (0, padding), mode='constant')

        # numpy to tensor
        waveform = torch.from_numpy(waveform).float()

        # 채널 차원 추가 (1, samples)
        waveform = waveform.unsqueeze(0)

        # GPU로 이동
        waveform = waveform.to(self.device)

        # torchaudio로 Mel Spectrogram 변환 (GPU 가속)
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        # 정규화
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)

        # 배치 차원 추가
        mel_spec_db = mel_spec_db.unsqueeze(0)

        return mel_spec_db

    def predict_file(self, audio_path: str) -> dict:
        """파일에서 낙상 감지"""
        # librosa로 오디오 로드 (호환성 좋음)
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return self.predict(waveform, sr)

    def predict(self, waveform: np.ndarray, sr: int) -> dict:
        """
        오디오 데이터로 낙상 여부 판별

        Args:
            waveform: 오디오 웨이브폼 (numpy array)
            sr: 샘플레이트

        Returns:
            dict: {fall: bool, confidence: float, soundType: str}
        """
        # 전처리
        mel_spec = self.preprocess(waveform, sr)

        # 추론
        with torch.no_grad():
            output = self.model(mel_spec)
            confidence = output.item()

        # 결과
        is_fall = confidence > self.threshold

        # 소리 유형 분류
        if is_fall:
            if confidence > 0.8:
                sound_type = "thud"
            else:
                sound_type = "impact"
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
        return self.predict(audio_chunk, sr)


def predict(audio_data) -> dict:
    """
    오디오 데이터로 낙상 여부 판별 (호환성 유지용)

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
