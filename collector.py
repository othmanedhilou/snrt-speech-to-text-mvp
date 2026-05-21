"""24/7 daemon — captures live streams, transcribes, runs NLP, stores in DB.

Usage:
    python collector.py
    python collector.py --once          # single cycle then exit (for testing)
    python collector.py --channel al_aoula  # single channel
"""
import argparse
import logging
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

import config
import db
import nlp_pipeline
from pipeline import (
    INITIAL_PROMPTS,
    _get_model,
    capture_stream_chunk,
    denoise_audio,
    transcribe_wav,
)

# ── Logging ───────────────────────────────────────────────────────────────────
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOGS_DIR / "collector.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("collector")

_shutdown = False


def _on_signal(sig, frame):
    global _shutdown
    log.info("Shutdown signal received — finishing current cycle...")
    _shutdown = True


# ── Per-channel work ──────────────────────────────────────────────────────────

def process_channel(channel: dict) -> None:
    cid   = channel["id"]
    cname = channel["name"]
    url   = channel["url"]
    lang  = channel.get("language", "fr")

    if not url:
        log.warning(f"[{cname}] No URL — skipping.")
        return

    log.info(f"[{cname}] Capturing {config.CHUNK_DURATION}s...")

    chunk_dir = config.OUTPUTS_DIR / "chunks" / cid
    chunk_dir.mkdir(parents=True, exist_ok=True)
    ts_now = datetime.now(timezone.utc)
    wav_path = chunk_dir / f"{ts_now.strftime('%Y%m%d_%H%M%S')}.wav"

    try:
        # 1. Capture
        capture_stream_chunk(url, wav_path, duration=config.CHUNK_DURATION)

        # 2. Denoise
        denoise_audio(wav_path)

        # 3. Transcribe
        whisper_lang = None if lang == "auto" else lang
        result = transcribe_wav(
            wav_path,
            model_name=config.WHISPER_MODEL,
            language=whisper_lang,
            initial_prompt=INITIAL_PROMPTS.get(whisper_lang, ""),
            multi_pass=False,
        )

        text = result.get("text", "").strip()
        detected_lang = result.get("language") or lang

        if not text:
            log.info(f"[{cname}] No speech detected.")
            return

        segments = result.get("segments", [])
        avg_conf = (
            sum(s.get("avg_confidence", 0.0) for s in segments) / len(segments)
            if segments else 0.0
        )

        # 4. NLP
        nlp_result = nlp_pipeline.process(text, language=detected_lang)
        if nlp_result.get("skip"):
            log.info(f"[{cname}] Segment too short/noisy — skipped.")
            return

        # 5. Store
        nid = db.insert_news(
            channel_id=cid,
            channel_name=cname,
            captured_at=ts_now.isoformat(),
            text=text,
            summary=nlp_result.get("summary"),
            topic=nlp_result.get("topic"),
            language=detected_lang,
            avg_confidence=round(avg_conf, 4),
            entities=nlp_result.get("entities", []),
        )

        log.info(
            f"[{cname}] Saved #{nid} | topic={nlp_result.get('topic')} "
            f"| words={len(text.split())} | entities={len(nlp_result.get('entities', []))}"
        )

        # 6. Check alerts
        _check_alerts(nid, text)

    except Exception as exc:
        log.error(f"[{cname}] Error: {exc}", exc_info=True)
    finally:
        wav_path.unlink(missing_ok=True)


def _check_alerts(news_id: int, text: str) -> None:
    text_lower = text.lower()
    for alert in db.get_active_alerts():
        if alert["keyword"] in text_lower:
            db.record_alert_hit(alert["id"], news_id)
            log.warning(f"ALERT — keyword='{alert['keyword']}' found in news #{news_id}")


# ── Collection cycle ──────────────────────────────────────────────────────────

def run_cycle(only_channel: str | None = None) -> None:
    if _shutdown:
        return

    channels = [
        c for c in config.CHANNELS
        if c.get("active") and c.get("url")
        and (only_channel is None or c["id"] == only_channel)
    ]

    if not channels:
        log.warning("No active channels with URLs. Set them in .env or config.py")
        return

    workers = min(len(channels), config.MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_channel, ch): ch["name"] for ch in channels}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                log.error(f"[{name}] Unhandled: {exc}", exc_info=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SNRT News Collector daemon")
    parser.add_argument("--once", action="store_true", help="Run one cycle then exit")
    parser.add_argument("--channel", default=None, help="Only process this channel id")
    args = parser.parse_args()

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    log.info("=" * 60)
    log.info("SNRT News Collector starting")
    log.info(f"Whisper model : {config.WHISPER_MODEL} ({config.WHISPER_DEVICE})")
    log.info(f"Groq enabled  : {config.GROQ_ENABLED}")
    log.info(f"Chunk duration: {config.CHUNK_DURATION}s")
    log.info("=" * 60)

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    db.init_db()
    for ch in config.CHANNELS:
        db.upsert_channel(ch)

    log.info(f"Loading Whisper '{config.WHISPER_MODEL}'...")
    _get_model(config.WHISPER_MODEL, config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE)
    log.info("Whisper model ready.")

    if args.once:
        run_cycle(only_channel=args.channel)
        return

    # Scheduler — fires every CHUNK_DURATION seconds, max 1 concurrent instance
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        run_cycle,
        trigger="interval",
        seconds=config.CHUNK_DURATION,
        id="collect",
        max_instances=1,
        coalesce=True,
        kwargs={"only_channel": args.channel},
    )
    scheduler.start()
    log.info(f"Scheduler running — cycle every {config.CHUNK_DURATION}s. Press Ctrl+C to stop.")

    run_cycle(only_channel=args.channel)  # immediate first cycle

    try:
        while not _shutdown:
            time.sleep(1)
    finally:
        scheduler.shutdown(wait=False)
        log.info("Collector stopped cleanly.")


if __name__ == "__main__":
    main()
