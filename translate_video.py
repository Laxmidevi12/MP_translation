
import os, tempfile, subprocess
from asr.transcribe import extract_audio, transcribe_audio
from nlp.translator import Translator
from ser.infer_ser import load_model, predict_emotion_segment
from tts.tts_coqui import synthesize

SER_WEIGHTS = os.getenv("SER_WEIGHTS", "ser/ser_cnn_bilstm.pt")
if os.path.exists(SER_WEIGHTS):
    load_model(SER_WEIGHTS)

def _concat_wavs(wavs, out_path):
    list_path = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt").name
    with open(list_path, "w", encoding="utf-8") as f:
        for w in wavs:
            f.write(f"file '{w}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_path

def _mux_audio_to_video(video_in: str, audio_wav: str, video_out: str):
    cmd = [
        "ffmpeg", "-y", "-i", video_in, "-i", audio_wav,
        "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", video_out
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return video_out

def translate_video(video_path: str, out_path: str, src_lang_hint=None, translator_model="Helsinki-NLP/opus-mt-en-hi"):
    wav_path = extract_audio(video_path)
    result = transcribe_audio(wav_path, language=src_lang_hint)
    segments = result.get("segments", [])
    full_text = result.get("text", "").strip()
    tr = Translator(model_name=translator_model)
    out_wavs = []
    for seg in segments if segments else [{"start":0,"end":0,"text":full_text}]:
        s = float(seg.get("start", 0.0)); e = float(seg.get("end", s + 1.0))
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        emo = predict_emotion_segment(wav_path, s, e)
        t_txt = tr.translate(txt)
        seg_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        synthesize(t_txt, seg_wav, emotion=emo)
        out_wavs.append(seg_wav)
    if not out_wavs:
        raise RuntimeError("No audio synthesized; transcription empty.")
    concat_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    _concat_wavs(out_wavs, concat_wav)
    _mux_audio_to_video(video_path, concat_wav, out_path)
    return {"translated_text": full_text, "output": out_path}
