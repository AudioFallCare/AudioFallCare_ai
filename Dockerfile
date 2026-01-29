FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 (libsndfile - soundfile/librosa용)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# CPU 전용 PyTorch 설치 (이미지 크기 절감)
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu \
    torchaudio==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 나머지 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --no-deps \
    -r requirements.txt \
    && pip install --no-cache-dir \
    fastapi uvicorn[standard] httpx pydantic-settings \
    librosa soundfile numpy

# 소스 복사
COPY src/ src/
COPY server/ server/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
