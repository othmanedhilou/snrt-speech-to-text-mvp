import argparse
import time
from pathlib import Path

from pipeline import (
    INITIAL_PROMPTS,
    apply_llm_correction,
    assign_speakers_to_segments,
    default_model_for_cpu,
    denoise_audio,
    diarization_available,
    diarize_audio,
    llm_available,
    prepare_audio,
    save_transcription_artifacts,
    timestamp,
    transcribe_wav,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert media to WAV, transcribe with Whisper, and save artifacts."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input media file (audio or video).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where JSON/TXT transcription files are written.",
    )
    parser.add_argument(
        "--language",
        default="fr",
        help="Language code for Whisper (default: fr). Use auto for detection.",
    )
    parser.add_argument(
        "--model",
        default=default_model_for_cpu(),
        help="Whisper model name (tiny, base, small, medium, large-v3).",
    )
    parser.add_argument(
        "--no-correct",
        action="store_true",
        help="Skip LLM post-correction even if Ollama is available.",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Skip audio noise reduction.",
    )
    parser.add_argument(
        "--no-multi-pass",
        action="store_true",
        help="Skip multi-pass retry of low-confidence segments.",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization (requires pyannote.audio + HF_TOKEN).",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Number of speakers for diarization (auto-detected if not set).",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom initial prompt to prime Whisper with domain vocabulary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    wav_path = output_dir / "prepared_audio.wav"
    print(f"[1/6] Preparing and normalizing audio from: {input_path}")
    prepare_audio(input_path, wav_path)

    if not args.no_denoise:
        print("[2/6] Applying noise reduction...")
        denoise_audio(wav_path)
    else:
        print("[2/6] Noise reduction skipped (--no-denoise)")

    language = None if args.language.lower() == "auto" else args.language
    prompt = args.prompt if args.prompt else INITIAL_PROMPTS.get(language, "")
    multi_pass = not args.no_multi_pass
    print(f"[3/6] Running Whisper model='{args.model}' language='{args.language}' multi_pass={multi_pass}")
    started = time.time()
    result = transcribe_wav(
        wav_path,
        model_name=args.model,
        language=language,
        initial_prompt=prompt,
        multi_pass=multi_pass,
    )
    elapsed = time.time() - started
    print(f"      Transcription finished in {elapsed:.2f}s")

    # Count low-confidence segments
    low_conf = sum(
        1 for s in result.get("segments", []) if s.get("avg_confidence", 1.0) < 0.55
    )
    total = len(result.get("segments", []))
    print(f"      Segments: {total} total, {low_conf} low-confidence")

    if args.diarize:
        if diarization_available():
            print("[4/6] Running speaker diarization...")
            diar_start = time.time()
            speaker_turns = diarize_audio(wav_path, num_speakers=args.num_speakers)
            if speaker_turns:
                result["segments"] = assign_speakers_to_segments(
                    result["segments"], speaker_turns
                )
                result["diarized"] = True
                speakers = set(t["speaker"] for t in speaker_turns)
                print(f"      Found {len(speakers)} speakers in {time.time() - diar_start:.2f}s")
            else:
                print("      Diarization returned no results")
        else:
            print("[4/6] Diarization skipped (pyannote.audio or HF_TOKEN not available)")
    else:
        print("[4/6] Diarization skipped (use --diarize to enable)")

    if not args.no_correct and llm_available():
        print("[5/6] Applying LLM post-correction...")
        correction_start = time.time()
        result = apply_llm_correction(result)
        correction_elapsed = time.time() - correction_start
        print(f"      LLM correction finished in {correction_elapsed:.2f}s")
    else:
        if args.no_correct:
            print("[5/6] LLM correction skipped (--no-correct)")
        else:
            print("[5/6] LLM correction skipped (Ollama not available)")

    print("[6/6] Saving artifacts")
    artifacts = save_transcription_artifacts(
        result, source_media_path=input_path, output_dir=output_dir
    )

    print(f"\nJSON: {artifacts.json_path}")
    print(f"TXT : {artifacts.txt_path}")
    print("\nTop segments preview:")
    for segment in result.get("segments", [])[:10]:
        start = timestamp(float(segment.get("start", 0.0)))
        end = timestamp(float(segment.get("end", 0.0)))
        text = str(segment.get("text", "")).strip()
        speaker = segment.get("speaker", "")
        conf = segment.get("avg_confidence")
        speaker_tag = f" [{speaker}]" if speaker else ""
        conf_tag = f" (conf={conf})" if conf is not None else ""
        print(f"[{start} -> {end}]{speaker_tag}{conf_tag} {text}")


if __name__ == "__main__":
    main()
