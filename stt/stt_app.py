from fastapi import FastAPI, UploadFile, Form, Query
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
from langdetect import detect_langs
import io

app = FastAPI(title="STT · faster-whisper large-v3")
ASR = WhisperModel("large-v3", device="cuda", compute_type="float16")  # GPU fp16

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/transcribe")
async def transcribe(
    audio: UploadFile,
    lang_hint: str | None = Form(default=None),
    return_segments: bool = Query(default=True),
    segment_langid: bool = Query(default=True),
):
    """
    - lang_hint: 'es','en','pt','fr','it','zh'... (opcional)
    - return_segments: incluir segmentos con timestamps
    - segment_langid: estimar idioma por segmento (heurístico, sobre texto)
    """
    data = await audio.read()
    segments, info = ASR.transcribe(
        io.BytesIO(data),
        vad_filter=True,
        language=lang_hint,    # None => auto
    )
    text = []
    seg_out = []
    for s in segments:
        t = s.text.strip()
        text.append(t)
        guess = []
        if segment_langid and t:
            try:
                # langdetect devuelve lista [lang:prob...]
                dl = detect_langs(t)
                if dl:
                    guess = [{"lang": g.lang, "prob": g.prob} for g in dl[:3]]
            except Exception:
                guess = []
        seg_out.append({
            "start": s.start, "end": s.end,
            "text": t,
            "lang_guess": guess
        })
    out = {
        "text": " ".join(text).strip(),
        "detected_lang": info.language,                # idioma dominante segun Whisper
        "language_probability": getattr(info, "language_probability", None),
        "duration": info.duration
    }
    if return_segments:
        out["segments"] = seg_out
    return JSONResponse(out)
