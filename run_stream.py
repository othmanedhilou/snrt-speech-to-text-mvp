"""Live stream / IPTV transcription CLI.

Captures audio from a stream URL in chunks, transcribes each chunk
in near-real-time, and saves a merged transcript at the end.

Usage:
    python run_stream.py --url "http://example.com/stream.m3u8"
    python run_stream.py --url "rtsp://192.168.1.100/live" --duration 60
    python run_stream.py --url "http://iptv.example/al_aoula.m3u8" --chunks 10
"""

import argparse
import signal
import sys
import tempfile
import time
from pathlib import Path

from pipeline import (
    INITIAL_PROMPTS,
    default_model_for_cpu,
    probe_stream,
    save_stream_session,
    timestamp,
    transcribe_stream_chunk,
)

_stop_requested = False


def _signal_handler(sig, frame):
    global _stop_requested
    print("\n[!] Stop requested. Finishing current chunk and saving...")
    _stop_requested = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture and transcribe live streams (IPTV, m3u8, rtsp, etc.)"
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Stream URL (m3u8, rtsp, rtp, udp, http, or YouTube/Twitch URL).",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=0,
        help="Number of chunks to capture (0 = run until Ctrl+C).",
    )
    parser.add_argument(
        "--chunk-duration",
        type=int,
        default=30,
        help="Duration of each chunk in seconds (default: 30).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--language",
        default="fr",
        help="Language code (default: fr). Use auto for detection.",
    )
    parser.add_argument(
        "--model",
        default=default_model_for_cpu(),
        help="Whisper model (tiny, base, small, medium, large-v3).",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Skip noise reduction on chunks.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom initial prompt for domain vocabulary.",
    )
    return parser.parse_args()


def main() -> None:
    global _stop_requested
    args = parse_args()

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, _signal_handler)

    stream_url = args.url
    language = None if args.language.lower() == "auto" else args.language
    prompt = args.prompt or INITIAL_PROMPTS.get(language, "")
    output_dir = Path(args.output_dir)
    max_chunks = args.chunks  # 0 = unlimited

    print(f"Stream URL: {stream_url}")
    print(f"Chunk duration: {args.chunk_duration}s")
    print(f"Max chunks: {'unlimited (Ctrl+C to stop)' if max_chunks == 0 else max_chunks}")
    print(f"Model: {args.model}")
    print(f"Language: {args.language}")
    print()

    # Probe stream
    print("[*] Probing stream...")
    probe = probe_stream(stream_url)
    if not probe["reachable"]:
        print("[!] WARNING: Could not probe stream. Will try capturing anyway.")
    else:
        print("[+] Stream is reachable.")
    print()

    # Create temp dir for chunks
    chunk_dir = output_dir / "stream_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    chunk_index = 0
    total_start = time.time()

    try:
        while not _stop_requested:
            if max_chunks > 0 and chunk_index >= max_chunks:
                break

            chunk_start = time.time()
            print(f"--- Chunk {chunk_index + 1} {'/' + str(max_chunks) if max_chunks > 0 else ''} ---")

            try:
                print(f"  [1/2] Capturing {args.chunk_duration}s of audio...")
                result = transcribe_stream_chunk(
                    stream_url=stream_url,
                    chunk_dir=chunk_dir,
                    chunk_index=chunk_index,
                    duration=args.chunk_duration,
                    model_name=args.model,
                    language=language,
                    initial_prompt=prompt,
                    denoise=not args.no_denoise,
                )

                all_chunks.append(result)
                chunk_elapsed = time.time() - chunk_start

                # Print live results
                text = result.get("text", "").strip()
                seg_count = len(result.get("segments", []))
                time_start = timestamp(chunk_index * args.chunk_duration)
                time_end = timestamp((chunk_index + 1) * args.chunk_duration)

                print(f"  [2/2] Transcribed in {chunk_elapsed:.1f}s ({seg_count} segments)")
                print(f"  [{time_start} -> {time_end}]")
                if text:
                    # Show first 200 chars
                    preview = text[:200] + ("..." if len(text) > 200 else "")
                    print(f"  {preview}")
                else:
                    print("  (no speech detected)")
                print()

            except Exception as exc:
                print(f"  [!] Error on chunk {chunk_index + 1}: {exc}")
                print("  Retrying next chunk...")
                print()

            chunk_index += 1

    except KeyboardInterrupt:
        print("\n[!] Interrupted.")

    # Save merged results
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Session complete: {len(all_chunks)} chunks in {total_elapsed:.1f}s")

    if all_chunks:
        artifacts = save_stream_session(all_chunks, stream_url, output_dir)
        print(f"JSON: {artifacts.json_path}")
        print(f"TXT : {artifacts.txt_path}")

        # Print full transcript summary
        total_segments = sum(len(c.get("segments", [])) for c in all_chunks)
        total_text = " ".join(c.get("text", "").strip() for c in all_chunks if c.get("text"))
        word_count = len(total_text.split())
        print(f"\nTotal segments: {total_segments}")
        print(f"Total words: {word_count}")
        print(f"\nFull transcript preview (first 500 chars):")
        print(total_text[:500])
    else:
        print("No chunks were captured.")

    sys.exit(0)


if __name__ == "__main__":
    main()
