# AudioFallCare AI

소리 기반 낙상 감지 AI 모델

## 개요

오디오 데이터를 분석하여 낙상 여부를 판별하는 AI 모델입니다.

## 기능

- 실시간 오디오 스트림 분석
- 낙상 소리 감지 및 분류
- 신뢰도(confidence) 점수 반환

## 프로젝트 구조

```
AudioFallCare_ai/
├── data/               # 학습 데이터
├── models/             # 학습된 모델
├── src/                # 소스 코드
│   ├── train.py        # 모델 학습
│   ├── inference.py    # 추론
│   └── preprocess.py   # 데이터 전처리
├── notebooks/          # Jupyter 노트북
├── tests/              # 테스트 코드
├── requirements.txt    # 의존성
└── README.md
```

## 설치

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 사용법

### 학습

```bash
python src/train.py --data ./data --epochs 100
```

### 추론

```bash
python src/inference.py --model ./models/model.pt --input audio.wav
```

## 출력 형식

```json
{
  "fall": true,
  "confidence": 0.95,
  "soundType": "thud"
}
```

## 관련 레포지토리

- [AudioFallCare_web](https://github.com/AudioFallCare/AudioFallCare_web) - 프론트엔드
- [AudioFallCare_was](https://github.com/AudioFallCare/AudioFallCare_was) - 백엔드
- [AudioFallcare_docs](https://github.com/AudioFallCare/AudioFallcare_docs) - 문서
