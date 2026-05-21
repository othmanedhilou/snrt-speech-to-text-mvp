# SNRT Speech-to-Text MVP (French-first, non-hardcoded)

This project turns audio or video into searchable timestamped text.

Core flow:
1. Ingest media file.
2. Convert to clean WAV using ffmpeg (16kHz mono).
3. Transcribe with Whisper.
4. Save transcript artifacts (JSON + TXT).
5. Search transcript segments by keyword/text with ranking.

## Project Files

- run_stt.py: CLI transcription pipeline (no hardcoded media).
- nlp_keywords.py: CLI transcript search (reads JSON output).
- pipeline.py: Shared reusable logic for conversion, transcription, and retrieval.
- app.py: Streamlit web app for testing.

## Requirements

System:
- Python 3.10+
- ffmpeg in PATH

Python packages:
- openai-whisper
- yt-dlp
- streamlit

## Windows Setup

Install system tools (PowerShell):

```powershell
winget install Python.Python.3.11
winget install Gyan.FFmpeg
```

Create environment and install project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## CLI Usage

1) Transcribe media to JSON/TXT artifacts:

```powershell
python run_stt.py --input path\to\your_video.mp4 --language fr --model small
```

Outputs are written to outputs/ with timestamped filenames.

2) Search a generated transcript JSON:

```powershell
python nlp_keywords.py --transcript outputs\your_file_YYYYMMDD_HHMMSS.json --query coupe
```

Optional tuning:

```powershell
python nlp_keywords.py --transcript outputs\file.json --query cup --min-score 0.65 --limit 30
```

## Web App (Local Test)

Run Streamlit:

```powershell
streamlit run app.py
```

In the UI:
1. Upload audio/video.
2. Click Transcribe.
3. Search with keyword/text.
4. Read ranked matches with timestamps.

## Free Live Demo Options

Recommended free deployment:
1. Hugging Face Spaces (Streamlit support, easy public URL).
2. Streamlit Community Cloud (also free, resource limits apply).

## Notes

- Start with language=fr for best French-first behavior.
- Later, switch to language=auto for multilingual processing.
- This is retrieval over Whisper output, not custom model training yet.
