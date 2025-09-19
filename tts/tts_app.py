from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse, JSONResponse
from TTS.api import TTS
import os, io, numpy as np, soundfile as sf
from num2words import num2words
import re
# --- Torch 2.6: allowlist para XTTS (evita UnpicklingError) ---
import torch
try:
    from torch.serialization import add_safe_globals
    from TTS.tts.configs.xtts_config import XttsConfig
    add_safe_globals([XttsConfig])
except Exception as e:
    print("safe_globals skip:", e)

app = FastAPI(title="TTS · XTTS-v2 robust")

_tts = None
_default_speaker = None

def get_tts():
    global _tts, _default_speaker
    if _tts is None:
        print(">> Cargando XTTS-v2…")
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        try:
            sm = _tts.synthesizer.tts_model.speaker_manager
            keys = list(getattr(sm, "speakers", {}).keys())
            _default_speaker = keys[0] if keys else None
            print(">> XTTS default_speaker:", _default_speaker)
        except Exception as e:
            print(">> No pude inspeccionar speakers integrados:", e)
    return _tts

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/models")
def models():
    tts = get_tts()
    try:
        sm = tts.synthesizer.tts_model.speaker_manager
        keys = list(getattr(sm, "speakers", {}).keys())
        return {"model_name": tts.synthesizer.tts_model.__class__.__name__,
                "num_speakers": len(keys),
                "speakers": keys}
    except Exception as e:
        return JSONResponse({"error": f"no speakers: {e}"}, status_code=500)

@app.get("/config")
def config():
    p = os.getenv("SPEAKER_WAV")
    return {"SPEAKER_WAV": p, "exists": bool(p and os.path.exists(p))}

@app.get("/speakers")
def speakers():
    tts = get_tts()
    try:
        sm = tts.synthesizer.tts_model.speaker_manager
        keys = list(getattr(sm, "speakers", {}).keys())
        return {"speakers": keys}
    except Exception as e:
        return JSONResponse({"error": f"no speakers: {e}"}, status_code=500)

def _norm_lang(lang: str) -> str:
    lang = (lang or "en").lower()
    # normalizamos zh -> “zh” (no zh-cn/zh-CN)
    if lang.startswith("zh"):
        return "zh"
    if lang not in {"es","en","pt","fr","it","zh"}:
        print(">> lang desconocido, usando en:", lang)
        return "en"
    return lang

def preprocess_es(text: str, pause="…"):
    # 1) números a palabras
    def _num(m):
        n = int(m.group(0))
        return num2words(n, lang="es")
    text = re.sub(r"\b\d+\b", _num, text)

    # 2) normalizar espacios
    text = re.sub(r"\s+", " ", text).strip()

    # 3) insertar pausas suaves después de cláusulas largas
    text = re.sub(r"([,;:])\s*", r"\1 ", text)
    # rompe en frases y agrega '…' si son muy largas
    chunks, out = re.split(r"(?<=[\.\?\!])\s+", text), []
    for c in chunks:
        c = c.strip()
        if not c: continue
        if len(c) > 80 and "," in c:
            c = c.replace(",", f",{pause}", 1)
        out.append(c)
    text = f" {pause} ".join(out)

    # 4) signos invertidos si faltan (heurístico básico)
    text = re.sub(r"(^|\s)(\?)", r"\1¿\2", text) if "?" in text and "¿" not in text else text
    text = re.sub(r"(^|\s)(\!)", r"\1¡\2", text) if "!" in text and "¡" not in text else text
    return text

def choose_speed(style: str | None, base=1.0):
    table = {
        "teacher": 0.94, "docente": 0.94,
        "casual": 0.98,
        "slow": 0.90,
        "fast": 1.02
    }
    return table.get((style or "").lower(), base)

@app.post("/speak")
async def speak(
    text: str = Form(...),
    speed: float = Form(1.0),            # 0.7 a 1.3 (si speaker_wav)
    lang: str = Form("es"),                 # es|en|pt|fr|it|zh
    speaker: str | None = Form(None),       # nombre de speaker integrado
    use_speaker_wav: bool = Form(True),     # usar clon si SPEAKER_WAV existe
):
    tts = get_tts()
    sr = 24000
    lang = _norm_lang(lang)
    spk_wav = os.getenv("SPEAKER_WAV") if use_speaker_wav else None
    print(f">> REQ speak: lang={lang} chars={len(text)} wav={'yes' if (spk_wav and os.path.exists(spk_wav)) else 'no'} speaker={speaker}")


    # preprocesar texto
    if lang == "es":
        text = preprocess_es(text)
    elif lang == "en":
        text = re.sub(r"\s+", " ", text).strip()
    # else: dejamos tal cual (zh, fr, it, pt)
    try:
        if spk_wav and os.path.exists(spk_wav):
            audio = tts.tts(text=text, language=lang, speaker_wav=spk_wav, speed=speed)  # sin speed para evitar incompat.
            used = f"wav:{os.path.basename(spk_wav)}"
        else:
            name = speaker or _default_speaker
            if not name:
                return JSONResponse({"error":"no speaker_wav y el modelo no trae speakers integrados"}, status_code=400)
            audio = tts.tts(text=text, language=lang, speaker=name)
            used = f"name:{name}"
    except Exception as e:
        # logueamos y devolvemos error legible
        print(">> ERROR tts.tts:", repr(e))
        return JSONResponse({"error": str(e)}, status_code=500)

    try:
        buf = io.BytesIO()
        sf.write(buf, np.array(audio), sr, format="WAV")
        buf.seek(0)
    except Exception as e:
        print(">> ERROR escribiendo WAV:", repr(e))
        return JSONResponse({"error": f"write wav: {e}"}, status_code=500)

    return StreamingResponse(buf, media_type="audio/wav",
                             headers={"X-Lang": lang, "X-Speaker": used})
