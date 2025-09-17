# tts2_app.py (CosyVoice2, API: /health, /speak)
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
import os, io, numpy as np, soundfile as sf, torch

# CosyVoice necesita tener Matcha-TTS en el path (según README)
import sys
sys.path.append('/workspace/CosyVoice/third_party/Matcha-TTS')
sys.path.append('/workspace/CosyVoice')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

app = FastAPI(title="TTS2 · CosyVoice2")

cosy = None
SR = 24000   # Cosy2 expone sample_rate, pero fijamos 24k como salida WAV

def get_cosy():
    global cosy
    if cosy is None:
        cosy = CosyVoice2('iic/CosyVoice2-0.5B',
                          load_jit=False, load_trt=False, load_vllm=False, fp16=False)
    return cosy

@app.get("/health")
def health():
    return {"ok": True}

def _cosy_lang_token(lang: str):
    # CosyVoice "oficial": zh/en/jp/yue/ko (el resto = cross-lingual, funciona pero no garantizan calidad)
    lang = (lang or "en").lower()
    if lang.startswith("zh"): return "zh"
    if lang.startswith("en"): return "en"
    if lang.startswith("ja") or lang == "jp": return "jp"
    if lang.startswith("ko"): return "ko"
    if lang in {"yue","cantonese"}: return "yue"
    # para es/pt/fr/it devolvemos 'en' como token básico (y que la prosodia la aporte el prompt)
    return "en"

@app.post("/speak")
async def speak(
    text: str = Form(...),
    lang: str = Form("en"),
    # si viene un clon, lo usamos; si no, probamos con uno default (opcional)
    speaker_wav_file: UploadFile | None = File(None),
    speaker_wav: str | None = Form(None),
):
    cv = get_cosy()
    token = _cosy_lang_token(lang)
    print(f"[Cosy] text={len(text)} chars, lang={lang}->{token}")

    # cargamos referencia 16k (requerido por cosy)
    ref_path = speaker_wav
    if speaker_wav_file:
        data = await speaker_wav_file.read()
        tmp = "/tmp/_cosy_ref.wav"
        with open(tmp, "wb") as f: f.write(data)
        ref_path = tmp

    prompt16 = None
    if ref_path and os.path.exists(ref_path):
        prompt16 = load_wav(ref_path, 16000)

    try:
        # Cosy2 devuelve un generador de chunks con 'tts_speech' (tensor [1, T])
        waves = []
        # zero-shot si tenemos referencia; si no, tiramos igual (usa estilo por defecto)
        if prompt16 is not None:
            g = cv.inference_zero_shot(text, '', prompt16, stream=False)
        else:
            # sin prompt: la prosodia sigue siendo buena, pero menos personalizada
            g = cv.inference_zero_shot(text, '', '', stream=False)
        for _, out in enumerate(g):
            wav = out['tts_speech'].squeeze(0).cpu().numpy()
            waves.append(wav)
        audio = np.concatenate(waves, axis=-1) if waves else np.zeros(1, dtype=np.float32)
    except Exception as e:
        print("[Cosy][ERROR]", repr(e))
        return JSONResponse({"error": str(e)}, status_code=500)

    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav",
                             headers={"X-Engine":"CosyVoice2","X-Lang":token})
