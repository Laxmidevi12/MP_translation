
import os, tempfile, subprocess
from typing import Optional, Dict
import whisper

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
_MODEL = whisper.load_model(WHISPER_MODEL_NAME)

def transcribe_audio(path: str, language: Optional[str] = None) -> Dict:
    result = _MODEL.transcribe(path, language=language)
    return result

def extract_audio(in_video: str, out_wav: Optional[str] = None, sr: int = 16000) -> str:
    if out_wav is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out_wav = tmp.name
        tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", in_video, "-vn", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", str(sr), out_wav
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_wav
