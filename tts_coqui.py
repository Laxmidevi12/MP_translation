
import os
from TTS.api import TTS
MODEL_NAME = os.getenv("COQUI_TTS_MODEL", "tts_models/multilingual/multi-dataset/your_tts")
_tts = None
def _load():
    global _tts
    if _tts is None:
        _tts = TTS(MODEL_NAME)
EMO_PARAMS = {
    "neutral": {"speed": 1.0},
    "happy": {"speed": 1.08},
    "sad": {"speed": 0.92},
    "angry": {"speed": 1.05},
    "fear": {"speed": 0.98},
    "disgust": {"speed": 0.97},
    "surprise": {"speed": 1.12},
}
def synthesize(text: str, out_wav: str, emotion: str = "neutral", speaker: str = None):
    _load()
    params = EMO_PARAMS.get(emotion, EMO_PARAMS["neutral"])
    _tts.tts_to_file(text=text, file_path=out_wav, speaker=speaker, speed=params["speed"])
    return out_wav
