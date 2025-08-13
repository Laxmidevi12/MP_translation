
# Regional & Tribal Emotion-Preserved Translator (Chat + Video)

This project translates full video audio into target regional/tribal languages while preserving the speaker's emotion per segment.
It also includes a simple WebSocket chat translator UI.
See folder structure and quickstart below.

## Quickstart (local)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# install ffmpeg on your system (apt/yum/brew)
uvicorn server:app --reload --port 8000
```
Open `web/index.html` in browser for chat UI. Use `/translate_video` endpoint to upload a video file and get a translated output.

Note: This is a scaffold/demo. Replace placeholder models (esp. tribal languages) with better checkpoints for production.
