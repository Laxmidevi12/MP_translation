
import os, json, tempfile
from typing import Dict
from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from nlp.translator import Translator
from asr.transcribe import transcribe_audio
from translate_video import translate_video

app = FastAPI(title="Regional Chat & Video Translator (Emotion-Preserved)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

MODEL_MAP = {
    ("en","hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi","en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en","te"): "Helsinki-NLP/opus-mt-en-tel",
    ("te","en"): "Helsinki-NLP/opus-mt-tel-en",
    ("en","ta"): "Helsinki-NLP/opus-mt-en-ta",
    ("ta","en"): "Helsinki-NLP/opus-mt-ta-en",
    ("en","sat"): "ai4bharat/indictrans2-en-indic-1B",   # Santali via IndicTrans2 if available
    ("sat","en"): "ai4bharat/indictrans2-indic-en-1B",
    ("en","brx"): "ai4bharat/indictrans2-en-indic-1B",   # Bodo placeholder
    ("brx","en"): "ai4bharat/indictrans2-indic-en-1B",
    ("en","trp"): "ai4bharat/indictrans2-en-indic-1B",   # Kokborok placeholder
    ("trp","en"): "ai4bharat/indictrans2-indic-en-1B",
    ("en","gon"): "ai4bharat/indictrans2-en-indic-1B",   # Gondi placeholder
    ("gon","en"): "ai4bharat/indictrans2-indic-en-1B",
    ("en","kha"): "ai4bharat/indictrans2-en-indic-1B",   # Khasi placeholder
    ("kha","en"): "ai4bharat/indictrans2-indic-en-1B",
    ("en","lus"): "ai4bharat/indictrans2-en-indic-1B",   # Mizo placeholder
    ("lus","en"): "ai4bharat/indictrans2-indic-en-1B",
    ("en","hoc"): "ai4bharat/indictrans2-en-indic-1B",   # Ho placeholder
    ("hoc","en"): "ai4bharat/indictrans2-indic-en-1B",
    ("en","lep"): "ai4bharat/indictrans2-en-indic-1B",   # Lepcha placeholder
    ("lep","en"): "ai4bharat/indictrans2-indic-en-1B",

    # other regional pairs (examples)
    ("en","kn"): "Helsinki-NLP/opus-mt-en-kn",
    ("kn","en"): "Helsinki-NLP/opus-mt-kn-en",
    ("en","ml"): "Helsinki-NLP/opus-mt-en-ml",
    ("ml","en"): "Helsinki-NLP/opus-mt-ml-en",
    ("en","bn"): "Helsinki-NLP/opus-mt-en-bn",
    ("bn","en"): "Helsinki-NLP/opus-mt-bn-en",
    ("en","gu"): "Helsinki-NLP/opus-mt-en-gu",
    ("gu","en"): "Helsinki-NLP/opus-mt-gu-en",
    ("en","mr"): "Helsinki-NLP/opus-mt-en-mr",
    ("mr","en"): "Helsinki-NLP/opus-mt-mr-en",
    ("en","pa"): "Helsinki-NLP/opus-mt-en-pa",
    ("pa","en"): "Helsinki-NLP/opus-mt-pa-en",
    ("en","or"): "Helsinki-NLP/opus-mt-en-or",
    ("or","en"): "Helsinki-NLP/opus-mt-or-en",
    ("en","ur"): "Helsinki-NLP/opus-mt-en-ur",
    ("ur","en"): "Helsinki-NLP/opus-mt-ur-en",
}

def get_translator(src, tgt):
    name = MODEL_MAP.get((src, tgt))
    if not name:
        name = "Helsinki-NLP/opus-mt-en-hi"
    return Translator(name, src, tgt)

@app.get("/")
def root():
    return {"ok": True, "service": "Regional Chat & Video Translator (Emotion-Preserved)"}

@app.post("/translate_text")
async def translate_text(text: str = Form(...), src: str = Form(...), tgt: str = Form(...)):
    tr = get_translator(src, tgt)
    out = tr.translate(text)
    return {"text": text, "src": src, "tgt": tgt, "translation": out}

@app.websocket("/ws/chat")
async def chat_socket(ws: WebSocket):
    await ws.accept()
    translators: Dict[str, Translator] = {}
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            key = f"{data['src']}->{data['tgt']}"
            if key not in translators:
                translators[key] = get_translator(data["src"], data["tgt"])
            tr = translators[key]
            translation = tr.translate(data["text"])
            await ws.send_text(json.dumps({
                "msgId": data.get("msgId"),
                "translation": translation,
                "src": data["src"],
                "tgt": data["tgt"]
            }))
    except Exception:
        await ws.close()

@app.post("/transcribe_audio")
async def transcribe_audio_endpoint(file: UploadFile = File(...), lang: str = Form(None)):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        path = tmp.name
    result = transcribe_audio(path, language=lang)
    return JSONResponse(result)

@app.post("/translate_video")
async def translate_video_endpoint(file: UploadFile = File(...),
                                   src_lang_hint: str = Form(None),
                                   src: str = Form(...), tgt: str = Form(...)):
    model = MODEL_MAP.get((src,tgt), "Helsinki-NLP/opus-mt-en-hi")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(await file.read())
        vin = tmp.name
    vout = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    info = translate_video(vin, vout, src_lang_hint=src_lang_hint, translator_model=model)
    return {"translated_text": info["translated_text"], "video_url": vout}
