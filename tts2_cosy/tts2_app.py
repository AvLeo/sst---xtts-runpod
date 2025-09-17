# /workspace/sst---xtts-runpod/tts2_cosy/tts2_app.py
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
import os, io, numpy as np, soundfile as sf, sys, types, torch

# Rutas para CosyVoice
sys.path.append('/workspace/CosyVoice/third_party/Matcha-TTS')
sys.path.append('/workspace/CosyVoice')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

app = FastAPI(title="TTS2 · CosyVoice2 (robusto)")

cosy = None

def get_cosy():
    global cosy
    if cosy is None:
        cosy = CosyVoice2('iic/CosyVoice2-0.5B',
                          load_jit=False, load_trt=False, load_vllm=False, fp16=False)
        print("[Cosy] sample_rate:", cosy.sample_rate)

        # --- HOTFIX: aseguramos dtype long en el sampler del LLM ---
        try:
            llm = cosy.llm
            _orig = llm.sampling_ids
            def _patched_sampling_ids(self, logp, out_tokens, sampling, ignore_eos=False, max_trials=100):
                if not isinstance(out_tokens, torch.Tensor):
                    out_tokens = torch.tensor(out_tokens, device=logp.device)
                if out_tokens.dtype != torch.long:
                    out_tokens = out_tokens.long()
                return _orig(logp, out_tokens, sampling, ignore_eos, max_trials)
            llm.sampling_ids = types.MethodType(_patched_sampling_ids, llm)
            print("[Cosy] Patched llm.sampling_ids -> out_tokens.long()")
        except Exception as e:
            print("[Cosy] sampler patch skip:", repr(e))
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
    # referencia opcional (WAV; cualquier sr -> lo re-muestreamos a 16k)
    speaker_wav_file: UploadFile | None = File(None),
    speaker_wav: str | None = Form(None),
    prompt_text: str | None = Form(None),   # si viene vacío, usamos text
):
    cv = get_cosy()
    token = _cosy_lang_token(lang)
    # 1) prompt_text no vacío
    ptxt = (prompt_text if (prompt_text is not None and prompt_text.strip()) else text)
    print(f"[Cosy] text={len(text)} chars, lang={lang}->{token}, prompt_len={len(ptxt)}")

    # 2) resolver WAV de referencia 16k (archivo subido, ruta o fallback del repo)
    ref_path = None
    if speaker_wav_file:
        data = await speaker_wav_file.read()
        tmp = "/tmp/_cosy_ref.wav"
        with open(tmp, "wb") as f: f.write(data)
        ref_path = tmp
    elif speaker_wav and os.path.exists(speaker_wav):
        ref_path = speaker_wav
    else:
        fallback = "/workspace/CosyVoice/asset/zero_shot_prompt.wav"
        if os.path.exists(fallback):
            ref_path = fallback

    if not ref_path or not os.path.exists(ref_path):
        return JSONResponse({"error": "Falta referencia de voz. Subí speaker_wav_file o deja asset/zero_shot_prompt.wav."}, status_code=400)

    try:
        prompt16 = load_wav(ref_path, 16000)  # Cosy espera 16k
    except Exception as e:
        return JSONResponse({"error": f"no pude leer referencia: {e}"}, status_code=400)

    # 3) inferencia robusta
    try:
        waves = []
        # usar zero-shot con prompt_text y prompt16 (¡nada de strings vacíos en el 3er arg!)
        for _, out in enumerate(cv.inference_zero_shot(text, ptxt, prompt16, stream=False)):
            wav = out['tts_speech'].squeeze(0).cpu().numpy()
            waves.append(wav)
        audio = np.concatenate(waves, axis=-1) if waves else np.zeros(1, dtype=np.float32)
    except Exception as e:
        print("[Cosy][ERROR]", repr(e))
        return JSONResponse({"error": str(e)}, status_code=500)

    # 4) salida con el SR nativo de Cosy
    buf = io.BytesIO()
    sr = getattr(cv, "sample_rate", 24000)
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav",
                             headers={"X-Engine":"CosyVoice2","X-Lang":token})
