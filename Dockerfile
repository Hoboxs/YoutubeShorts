FROM python:3.10-slim

# =========================
# System deps
# =========================
RUN apt-get update && apt-get install -y \
    ffmpeg \
    fontconfig \
    git \
    build-essential \
    libsndfile1 \
    espeak-ng \
    espeak-ng-data \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# =========================
# Pin setuptools<70 so pkg_resources remains available
# to pip's isolated build environments
# =========================
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "setuptools<70" wheel setuptools-scm

# =========================
# Base ML deps
# =========================
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    cython \
    packaging

# =========================
# Torch (required by whisper)
# =========================
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchaudio==2.2.2

# =========================
# Whisper — try wheel first, fall back to git
# =========================
RUN pip install --no-cache-dir openai-whisper==20231117 || \
    pip install --no-cache-dir git+https://github.com/openai/whisper.git

# =========================
# App requirements (make sure openai-whisper is NOT in here)
# =========================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# =========================
# Install fonts to system directory for ffmpeg/subtitle rendering
# =========================
RUN mkdir -p /usr/local/share/fonts && \
    cp Fonts/*.ttf /usr/local/share/fonts/ && \
    cp Fonts/*.otf /usr/local/share/fonts/ 2>/dev/null || true && \
    fc-cache -fv /usr/local/share/fonts/

CMD ["python", "app.py"]