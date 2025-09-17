from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
import os, io, numpy as np, soundfile as sf, sys

# Rutas necesarias para CosyVoice
sys.path.append('/workspace/CosyVoice/third_party/Matcha-TTS')
sys.path.append('/workspace/CosyVoice')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

app = FastAPI(title="TTS2 · CosyVoice2")

cosy = None

def get_cosy():
    global cosy
    if cosy is None:
        cosy = CosyVoice2('iic/CosyVoice2-0.5B',
                          load_jit=False, load_trt=False, load_vllm=False, fp16=False)
        print("[Cosy] sample_rate:", cosy.sample_rate)
    return cosy

@app.get("/health")
def health():
    return {"ok": True}

def _cosy_lang_token(lang: str):
    lang = (lang or "en").lower()
    if lang.startswith("zh"): return "zh"
    if lang.startswith("en"): return "en"
    if lang.startswith("ja") or lang == "jp": return "jp"
    if lang.startswith("ko"): return "ko"
    if lang in {"yue","cantonese"}: return "yue"
    return "en"

@app.post("/speak")
async def speak(
    text: str = Form(...),
    lang: str = Form("en"),
    # referencia opcional (WAV 16k recomendado)
    speaker_wav_file: UploadFile | None = File(None),
    speaker_wav: str | None = Form(None),
    prompt_text: str | None = Form(""),   # texto aproximado de la referencia (opcional)
):
    cv = get_cosy()
    token = _cosy_lang_token(lang)
    print(f"[Cosy] text={len(text)} chars, lang={lang}->{token}")

    # 1) Resolver WAV de referencia (obligatorio para zero-shot si no hay spk_id)
    ref_path = None
    if speaker_wav_file:
        data = await speaker_wav_file.read()
        tmp = "/tmp/_cosy_ref.wav"
        with open(tmp, "wb") as f: f.write(data)
        ref_path = tmp
    elif speaker_wav and os.path.exists(speaker_wav):
        ref_path = speaker_wav
    else:
        # fallback: usa el prompt de ejemplo del repo (sirve para demo)
        fallback = "/workspace/CosyVoice/asset/zero_shot_prompt.wav"
        if os.path.exists(fallback):
            ref_path = fallback

    if not ref_path or not os.path.exists(ref_path):
        return JSONResponse(
            {"error": "Se requiere un speaker_wav (16 kHz) para zero-shot. Subí uno en speaker_wav_file o dejá /workspace/CosyVoice/asset/zero_shot_prompt.wav."},
            status_code=400
        )

    # 2) Cargar la referencia a 16k como espera Cosy
    try:
        prompt16 = load_wav(ref_path, 16000)
    except Exception as e:
        return JSONResponse({"error": f"no pude leer la referencia: {e}"}, status_code=400)

    # 3) Inference (no pasar strings vacíos donde van tensores)
    try:
        waves = []
        for _, out in enumerate(cv.inference_zero_shot(text, prompt_text or "", prompt16, stream=False)):
            wav = out['tts_speech'].squeeze(0).cpu().numpy()
            waves.append(wav)
        audio = np.concatenate(waves, axis=-1) if waves else np.zeros(1, dtype=np.float32)
    except Exception as e:
        print("[Cosy][ERROR]", repr(e))
        return JSONResponse({"error": str(e)}, status_code=500)

    buf = io.BytesIO()
    # usa el sample rate nativo de Cosy (expuesto por la instancia)
    sr = getattr(cv, "sample_rate", 24000)
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav",
                             headers={"X-Engine":"CosyVoice2","X-Lang":token})
