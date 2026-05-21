# Free Deployment Guide

This project can be deployed for free for remote testing.

## Option 1: Hugging Face Spaces (Recommended)

1. Create a free account on Hugging Face.
2. Create a new Space and choose Streamlit SDK.
3. Push this repository to the Space.
4. Keep these files in repo root:
   - app.py
   - requirements.txt
   - packages.txt (contains ffmpeg)
5. Wait for build to complete.
6. Share the public Space URL with your monitor.

Notes:
- First run can be slow because Whisper model is downloaded.
- Free tier has limited CPU/RAM, so start with tiny/base/small models.

## Option 2: Streamlit Community Cloud

1. Push project to GitHub.
2. Create a free Streamlit Cloud app from repo.
3. Set main file to app.py.
4. Ensure requirements.txt is detected.
5. If ffmpeg is missing, use Hugging Face Spaces instead (packages.txt support is easier).

## Security and Data Notes

- Do not upload private or copyrighted streams without permission.
- Add a retention policy if handling sensitive media.
- For production, add authentication and rate limiting.
