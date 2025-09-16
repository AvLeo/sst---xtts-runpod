FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python -m pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 && \
    pip install "TTS==0.22.*" fastapi uvicorn[standard] numpy soundfile pydantic

ENV HF_HOME=/cache/hf
ENV COQUI_TOS_AGREED=1
#ENV SPEAKER_WAV=/voice/locutor.wav
COPY tts_app.py /app/app.py
EXPOSE 8002
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8002","--workers","1"]
