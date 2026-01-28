"""
데이터 전처리 스크립트
"""
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt


def load_audio(file_path: str, target_sr: int = 16000):
    """
    오디오 파일 로드

    Args:
        file_path: 오디오 파일 경로
        target_sr: 목표 샘플레이트

    Returns:
        waveform: 오디오 웨이브폼 (torch.Tensor)
        sample_rate: 샘플레이트
    """
    waveform, sr = torchaudio.load(file_path, backend="soundfile")

    # 리샘플링
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    # 모노로 변환
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform, sr


def extract_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 512
) -> torch.Tensor:
    """
    Mel Spectrogram 추출

    Args:
        waveform: 오디오 웨이브폼
        sample_rate: 샘플레이트
        n_mels: Mel 필터 수
        n_fft: FFT 윈도우 크기
        hop_length: 홉 길이

    Returns:
        mel_spec_db: Mel Spectrogram (dB)
    """
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    amplitude_to_db = T.AmplitudeToDB()

    mel_spec = mel_transform(waveform)
    mel_spec_db = amplitude_to_db(mel_spec)

    return mel_spec_db


def extract_mfcc(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mfcc: int = 40
) -> torch.Tensor:
    """
    MFCC 추출

    Args:
        waveform: 오디오 웨이브폼
        sample_rate: 샘플레이트
        n_mfcc: MFCC 계수 수

    Returns:
        mfcc: MFCC 특징
    """
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc
    )

    mfcc = mfcc_transform(waveform)
    return mfcc


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """텐서 정규화 (평균 0, 표준편차 1)"""
    return (tensor - tensor.mean()) / (tensor.std() + 1e-8)


def extract_features(audio_data, feature_type: str = "mel"):
    """
    오디오에서 특징 추출

    Args:
        audio_data: 오디오 데이터 (파일 경로 또는 텐서)
        feature_type: 특징 유형 ("mel" 또는 "mfcc")

    Returns:
        features: 추출된 특징
    """
    # 파일 경로인 경우 로드
    if isinstance(audio_data, str):
        waveform, sr = load_audio(audio_data)
    else:
        waveform = audio_data
        sr = 16000

    if feature_type == "mel":
        features = extract_mel_spectrogram(waveform, sr)
    elif feature_type == "mfcc":
        features = extract_mfcc(waveform, sr)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    return normalize(features)


def visualize_audio(file_path: str, save_path: str = None):
    """
    오디오 시각화 (Waveform + Mel Spectrogram)

    Args:
        file_path: 오디오 파일 경로
        save_path: 저장 경로 (None이면 화면에 표시)
    """
    waveform, sr = load_audio(file_path)
    mel_spec = extract_mel_spectrogram(waveform, sr)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Waveform
    axes[0].plot(waveform.squeeze().numpy())
    axes[0].set_title("Waveform")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Amplitude")

    # Mel Spectrogram
    img = axes[1].imshow(
        mel_spec.squeeze().numpy(),
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )
    axes[1].set_title("Mel Spectrogram")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Mel Frequency")
    plt.colorbar(img, ax=axes[1], format='%+2.0f dB')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="오디오 전처리")
    parser.add_argument("--input", type=str, required=True, help="오디오 파일 경로")
    parser.add_argument("--visualize", action="store_true", help="시각화")
    parser.add_argument("--output", type=str, default=None, help="시각화 저장 경로")

    args = parser.parse_args()

    # 특징 추출 테스트
    features = extract_features(args.input)
    print(f"Features shape: {features.shape}")

    # 시각화
    if args.visualize:
        visualize_audio(args.input, args.output)
