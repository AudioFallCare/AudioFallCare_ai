"""
데이터셋 다운로드 및 로드
ESC-50 데이터셋 사용
"""
import os
import urllib.request
import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T


ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"

# 낙상 관련 클래스 (ESC-50에서 충격음, 유리 깨지는 소리 등)
FALL_RELATED_CLASSES = [
    "glass_breaking",      # 유리 깨지는 소리
    "door_wood_knock",     # 문 두드리는 소리 (충격음)
    "footsteps",           # 발걸음
    "door_wood_creaks",    # 문 삐걱거리는 소리
]

# 비낙상 클래스 (일상 소리)
NON_FALL_CLASSES = [
    "clock_tick",
    "snoring",
    "breathing",
    "coughing",
    "sneezing",
    "crying_baby",
    "laughing",
    "keyboard_typing",
    "mouse_click",
]


def download_esc50(data_dir: str = "data"):
    """ESC-50 데이터셋 다운로드"""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ESC-50-master.zip")
    extract_path = os.path.join(data_dir, "ESC-50-master")

    if os.path.exists(extract_path):
        print("ESC-50 데이터셋이 이미 존재합니다.")
        return extract_path

    print("ESC-50 데이터셋 다운로드 중...")
    urllib.request.urlretrieve(ESC50_URL, zip_path)

    print("압축 해제 중...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    os.remove(zip_path)
    print("다운로드 완료!")

    return extract_path


class ESC50Dataset(Dataset):
    """ESC-50 데이터셋 (낙상/비낙상 이진 분류용)"""

    def __init__(
        self,
        data_dir: str = "data/ESC-50-master",
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        train: bool = True,
        fold: int = 5
    ):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.train = train

        # Mel Spectrogram 변환
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        self.amplitude_to_db = T.AmplitudeToDB()

        # 메타데이터 로드
        meta_path = os.path.join(data_dir, "meta", "esc50.csv")
        self.meta = pd.read_csv(meta_path)

        # 낙상/비낙상 필터링
        fall_mask = self.meta['category'].isin(FALL_RELATED_CLASSES)
        non_fall_mask = self.meta['category'].isin(NON_FALL_CLASSES)
        self.meta = self.meta[fall_mask | non_fall_mask].reset_index(drop=True)

        # 라벨 생성 (1: 낙상 관련, 0: 비낙상)
        self.meta['label'] = self.meta['category'].isin(FALL_RELATED_CLASSES).astype(int)

        # Train/Test 분리 (fold 기반)
        if train:
            self.meta = self.meta[self.meta['fold'] != fold].reset_index(drop=True)
        else:
            self.meta = self.meta[self.meta['fold'] == fold].reset_index(drop=True)

        print(f"{'Train' if train else 'Test'} 데이터: {len(self.meta)}개")
        print(f"  - 낙상 관련: {self.meta['label'].sum()}개")
        print(f"  - 비낙상: {(1 - self.meta['label']).sum()}개")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]

        # 오디오 로드 (soundfile 백엔드 사용)
        audio_path = os.path.join(self.data_dir, "audio", row['filename'])
        waveform, sr = torchaudio.load(audio_path, backend="soundfile")

        # 리샘플링
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # 모노로 변환
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Mel Spectrogram 변환
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        # 정규화
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)

        label = torch.tensor(row['label'], dtype=torch.float32)

        return mel_spec_db, label


if __name__ == "__main__":
    # 데이터셋 다운로드 테스트
    data_path = download_esc50("data")

    # 데이터셋 로드 테스트
    train_dataset = ESC50Dataset(data_path, train=True)
    test_dataset = ESC50Dataset(data_path, train=False)

    # 샘플 확인
    mel_spec, label = train_dataset[0]
    print(f"Mel Spectrogram shape: {mel_spec.shape}")
    print(f"Label: {label}")
