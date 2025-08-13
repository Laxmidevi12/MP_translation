
import os, torch, torchaudio
from ser_config import SAMPLE_RATE, EMOTIONS
# Minimalist infer: if no model, fallback neutral; otherwise load CNNBiLSTM from train_ser.py

_model = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(weights_path: str):
    global _model
    # lazy import to avoid heavy deps unless used
    from train_ser import CNNBiLSTM
    _model = CNNBiLSTM().to(DEVICE)
    _model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    _model.eval()

def _slice_wav(wav_path: str, start_s: float, end_s: float):
    wav, sr = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
    wav = wav.mean(dim=0, keepdim=True)
    start_t = int(max(0, start_s) * sr)
    end_t = int(max(start_t+1, end_s * sr))
    return wav[:, start_t:end_t]

def predict_emotion_segment(wav_path: str, start_s: float, end_s: float) -> str:
    if _model is None or not os.path.exists(wav_path):
        return "neutral"
    try:
        seg = _slice_wav(wav_path, start_s, end_s)
        from train_ser import melspec
        mel = melspec(seg)
        with torch.no_grad():
            logits = _model(mel.to(DEVICE))
            emo_id = logits.argmax(dim=1).item()
        return EMOTIONS[emo_id]
    except Exception:
        return "neutral"
