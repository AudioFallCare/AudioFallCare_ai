"""
SAFE 데이터셋 로드
Sound Analysis for Fall Event Detection

librosa: 오디오 로드 (호환성)
torchaudio: Mel Spectrogram 변환 (GPU 가속)
"""
import os
import glob
import numpy as np
import torch
import torchaudio.transforms as T
import librosa


class SAFEDataset(torch.utils.data.Dataset):
    """SAFE 데이터셋 (낙상/비낙상 이진 분류용)

    파일명 형식: AA-BBB-CC-DDD-FF.wav
    - AA: Fold 번호 (01-10)
    - BBB: 랜덤 코드
    - CC: 환경 ID
    - DDD: 시퀀스 번호
    - FF: 클래스 (01=낙상, 02=비낙상)
    """

    def __init__(
        self,
        data_dir: str = "data",
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        max_length: int = 3,
        train: bool = True,
        test_fold: int = 10,
        device: str = None
    ):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length
        self.max_samples = sample_rate * max_length
        self.train = train

        # 디바이스 설정
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # torchaudio 변환기 (GPU 가속 가능)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        self.amplitude_to_db = T.AmplitudeToDB()

        # WAV 파일 목록 로드
        all_files = glob.glob(os.path.join(data_dir, "*.wav"))

        if len(all_files) == 0:
            raise ValueError(f"No WAV files found in {data_dir}")

        # 파일 정보 파싱
        self.files = []
        self.labels = []

        for filepath in all_files:
            filename = os.path.basename(filepath)
            parts = filename.replace('.wav', '').split('-')

            if len(parts) >= 5:
                fold = int(parts[0])
                label = 1 if parts[4] == '01' else 0

                if train and fold != test_fold:
                    self.files.append(filepath)
                    self.labels.append(label)
                elif not train and fold == test_fold:
                    self.files.append(filepath)
                    self.labels.append(label)

        fall_count = sum(self.labels)
        non_fall_count = len(self.labels) - fall_count

        print(f"{'Train' if train else 'Test'} 데이터: {len(self.files)}개")
        print(f"  - 낙상: {fall_count}개")
        print(f"  - 비낙상: {non_fall_count}개")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        label = self.labels[idx]

        # librosa로 오디오 로드 (호환성 좋음)
        waveform, sr = librosa.load(filepath, sr=self.sample_rate, mono=True)

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

        # torchaudio로 Mel Spectrogram 변환 (GPU 가속 가능)
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        # 정규화
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)

        label = torch.tensor(label, dtype=torch.float32)

        return mel_spec_db, label


if __name__ == "__main__":
    # 데이터셋 로드 테스트
    train_dataset = SAFEDataset("data", train=True)
    test_dataset = SAFEDataset("data", train=False)

    # 샘플 확인
    mel_spec, label = train_dataset[0]
    print(f"Mel Spectrogram shape: {mel_spec.shape}")
    print(f"Label: {label}")
