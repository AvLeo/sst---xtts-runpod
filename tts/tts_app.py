from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse, JSONResponse
from TTS.api import TTS
import os, io, numpy as np, soundfile as sf

# --- Torch 2.6 safe loader: allowlist para XTTS ---
import torch
try:
    from torch.serialization import add_safe_globals
    from TTS.tts.configs.xtts_config import XttsConfig
    add_safe_globals([XttsConfig])
except Exception as e:
    print("safe_globals skip:", e)

app = FastAPI(title="TTS · XTTS-v2")

_tts = None
_default_speaker = None  # nombre de speaker integrado del modelo (si existe)

def get_tts():
    global _tts, _default_speaker
    if _tts is None:
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        # intentar descubrir speakers integrados (p.ej. "female-en-5", etc.)
        try:
            sm = _tts.synthesizer.tts_model.speaker_manager
            keys = list(getattr(sm, "speakers", {}).keys())
            _default_speaker = keys[0] if keys else None
            print("XTTS default_speaker:", _default_speaker)
        except Exception as e:
            print("no se pudo inspeccionar speakers:", e)
    return _tts

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/speak")
async def speak(
    text: str = Form(...),
    lang: str = Form("es"),        # es|en|pt|fr|it|zh
    speed: float = Form(1.0),
    speaker: str | None = Form(None),    # nombre del speaker integrado (opcional)
    use_speaker_wav: bool = Form(True),  # usar clon si existe SPEAKER_WAV
):
    tts = get_tts()
    sr = 24000
    spk_wav = os.getenv("SPEAKER_WAV") if use_speaker_wav else None

    try:
        # 1) Si hay clon (SPEAKER_WAV), úsalo
        if spk_wav and os.path.exists(spk_wav):
            audio = tts.tts(text=text, language=lang, speaker_wav=spk_wav, speed=speed)
            used = f"wav:{os.path.basename(spk_wav)}"
        else:
            # 2) Si vino "speaker" por form, úsalo
            name = speaker or _default_speaker
            if not name:
                return JSONResponse({"error": "No hay SPEAKER_WAV y el modelo no trae speakers integrados."}, status_code=400)
            audio = tts.tts(text=text, language=lang, speaker=name, speed=speed)
            used = f"name:{name}"
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    buf = io.BytesIO()
    sf.write(buf, np.array(audio), sr, format="WAV")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav",
                             headers={"X-Lang": lang, "X-Speaker": used})
