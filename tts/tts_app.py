from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse, JSONResponse
from TTS.api import TTS
import os, io, numpy as np, soundfile as sf
import torch

try:
    from torch.serialization import add_safe_globals
    from TTS.tts.configs.xtts_config import XttsConfig
    add_safe_globals([XttsConfig])
except Exception as e:
    print("safe_globals skip:", e)

app = FastAPI(title="TTS · XTTS-v2")
_tts = None

def get_tts():
    global _tts
    if _tts is None:
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    return _tts

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/speak")
async def speak(
    text: str = Form(...),
    lang: str = Form("es"),         # 'es','en','pt','fr','it','zh'
    speed: float = Form(1.0),       # 0.8–1.2 razonable
    use_speaker: bool = Form(True), # usar clon si existe
):
    tts = get_tts()
    sr = 24000
    spk = os.getenv("SPEAKER_WAV") if use_speaker else None

    if spk and os.path.exists(spk):
        audio = tts.tts(text=text, language=lang, speaker_wav=spk, speed=speed)
    else:
        audio = tts.tts(text=text, language=lang, speed=speed)

    buf = io.BytesIO()
    sf.write(buf, np.array(audio), sr, format="WAV")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav",
                             headers={"X-Lang": lang})
