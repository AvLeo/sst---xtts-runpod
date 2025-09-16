FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python -m pip install --upgrade pip && \
    pip install "faster-whisper==1.0.*" fastapi uvicorn[standard] pydantic soundfile numpy \
               langdetect

COPY stt_app.py /app/app.py
EXPOSE 8001
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8001","--workers","1"]
