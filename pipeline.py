import json
import os
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import noisereduce as nr
import requests
import soundfile as sf
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# LLM configuration (Ollama by default, any OpenAI-compatible endpoint works)
# ---------------------------------------------------------------------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")

# ---------------------------------------------------------------------------
# Initial prompt – primes Whisper with expected vocabulary & domain terms.
# Reduces misrecognition of proper nouns, acronyms, and domain-specific words.
# ---------------------------------------------------------------------------
INITIAL_PROMPT_FR = (
    "SNRT, Société Nationale de Radiodiffusion et de Télévision, "
    "Al Aoula, Arryadia, Assadissa, Arrabia, Amazighie, Laâyoune, Tamazight, "
    "Rabat, Casablanca, Marrakech, Fès, Tanger, Agadir, Meknès, Oujda, Tétouan, "
    "Dakhla, Nador, Kénitra, Mohammedia, Béni Mellal, Khénifra, Safi, El Jadida, "
    "Sa Majesté le Roi Mohammed VI, le Maroc, marocain, marocaine, "
    "Parlement, gouvernement, ministre, premier ministre, "
    "dirham, PIB, croissance économique, inflation, "
    "football, Raja, Wydad, FAR, FRMF, Botola, CAN, "
    "COVID, OMS, ONU, UNESCO, Union Africaine, "
    "Ramadan, Aïd, mosquée, imam, "
    "bonjour, bonsoir, merci, bienvenue, "
)

INITIAL_PROMPT_AR = (
    "الشركة الوطنية للإذاعة والتلفزة، القناة الأولى، الرياضية، "
    "السادسة، الرابعة، الأمازيغية، العيون، الرباط، الدار البيضاء، "
    "مراكش، فاس، طنجة، أكادير، مكناس، وجدة، تطوان، الداخلة، الناظور، "
    "جلالة الملك محمد السادس، المغرب، مغربي، مغربية، "
    "البرلمان، الحكومة، الوزير الأول، رئيس الحكومة، "
    "الدرهم، الناتج الداخلي الخام، "
)

INITIAL_PROMPTS: dict[str | None, str] = {
    "fr": INITIAL_PROMPT_FR,
    "ar": INITIAL_PROMPT_AR,
}

# ---------------------------------------------------------------------------
# Model cache – avoids reloading the large model on every call
# ---------------------------------------------------------------------------
_model_cache: dict[str, WhisperModel] = {}


@dataclass
class ArtifactPaths:
    json_path: Path
    txt_path: Path


# ---------------------------------------------------------------------------
# Audio preparation
# ---------------------------------------------------------------------------
def ensure_ffmpeg_available() -> None:
    """Raise an error when ffmpeg is not available in PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError("ffmpeg is required but was not found in PATH.") from exc


def prepare_audio(input_path: str | Path, output_wav_path: str | Path) -> Path:
    """Convert any input media into a 16kHz mono WAV file for Whisper."""
    input_path = Path(input_path)
    output_wav_path = Path(output_wav_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input media not found: {input_path}")

    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_ffmpeg_available()

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        # Audio normalization — equalizes volume so quiet speech is not missed
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        str(output_wav_path),
    ]

    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {process.stderr.strip()}")

    return output_wav_path


def denoise_audio(wav_path: str | Path) -> Path:
    """Apply noise reduction to a WAV file in-place.

    Uses spectral gating to remove background noise, hum, and static
    while preserving speech clarity.
    """
    wav_path = Path(wav_path)
    data, sample_rate = sf.read(str(wav_path), dtype="float32")

    reduced = nr.reduce_noise(
        y=data,
        sr=sample_rate,
        prop_decrease=0.7,
        n_fft=2048,
        stationary=False,
    )

    sf.write(str(wav_path), reduced, sample_rate)
    return wav_path


# ---------------------------------------------------------------------------
# Whisper transcription (faster-whisper)
# ---------------------------------------------------------------------------
def _get_model(
    model_name: str = "large-v3",
    device: str = "cpu",
    compute_type: str = "int8",
) -> WhisperModel:
    """Return a cached WhisperModel instance."""
    key = f"{model_name}_{device}_{compute_type}"
    if key not in _model_cache:
        _model_cache[key] = WhisperModel(
            model_name, device=device, compute_type=compute_type
        )
    return _model_cache[key]


def _run_transcription(
    model: WhisperModel,
    wav_path: str,
    language: str | None,
    task: str,
    initial_prompt: str,
    temperature: float | list[float],
) -> tuple[list, Any]:
    """Run a single transcription pass with the given settings."""
    segments_gen, info = model.transcribe(
        wav_path,
        language=language,
        task=task,
        beam_size=10,
        best_of=5,
        temperature=temperature,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=300,
        ),
        word_timestamps=True,
        initial_prompt=initial_prompt,
        # Hallucination suppression
        hallucination_silence_threshold=1.0,
        # Repetition penalty
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        # Compression ratio filter
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
    )
    return list(segments_gen), info


def _filter_low_confidence_words(
    segments: list, min_word_confidence: float = 0.4
) -> list[dict[str, Any]]:
    """Build segment dicts, marking low-confidence words with [?].

    Words below min_word_confidence are flagged so the user knows
    which parts are uncertain.
    """
    segment_list = []
    for seg in segments:
        words = seg.words or []
        filtered_parts = []
        word_details = []
        for w in words:
            if w.probability >= min_word_confidence:
                filtered_parts.append(w.word)
            else:
                # Mark uncertain words so user can review
                filtered_parts.append(f"[{w.word.strip()}?]")
            word_details.append(
                {
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "confidence": round(w.probability, 4),
                }
            )

        clean_text = "".join(filtered_parts).strip()
        avg_confidence = (
            sum(w.probability for w in words) / len(words) if words else 0.0
        )

        segment_list.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": clean_text,
                "avg_confidence": round(avg_confidence, 4),
                "words": word_details,
            }
        )
    return segment_list


def transcribe_wav(
    wav_path: str | Path,
    model_name: str = "large-v3",
    language: str | None = "fr",
    task: str = "transcribe",
    initial_prompt: str | None = None,
    multi_pass: bool = True,
) -> dict[str, Any]:
    """Run faster-whisper on a WAV file and return the full transcription payload.

    Features:
    - VAD filtering (silence removal)
    - Beam-search tuning (beam_size=10, best_of=5)
    - Hallucination suppression
    - Repetition penalty
    - Initial prompt conditioning
    - Word-level confidence scores with low-confidence flagging
    - Multi-pass: retries low-confidence segments with higher temperature
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    model = _get_model(model_name)

    if initial_prompt is None:
        initial_prompt = INITIAL_PROMPTS.get(language, "")

    # --- Pass 1: deterministic (temperature=0.0) ---
    raw_segments, info = _run_transcription(
        model, str(wav_path), language, task, initial_prompt, temperature=0.0
    )

    segment_list = _filter_low_confidence_words(raw_segments)

    # --- Pass 2: retry low-confidence segments with temperature sampling ---
    if multi_pass:
        low_conf_threshold = 0.55
        for i, seg_dict in enumerate(segment_list):
            if seg_dict["avg_confidence"] < low_conf_threshold:
                # Re-transcribe with temperature fallback
                retry_segments, _ = _run_transcription(
                    model,
                    str(wav_path),
                    language,
                    task,
                    initial_prompt,
                    temperature=[0.2, 0.4, 0.6, 0.8, 1.0],
                )
                # Find the retry segment closest in time to the original
                best_retry = None
                best_overlap = 0.0
                for rs in retry_segments:
                    overlap_start = max(seg_dict["start"], rs.start)
                    overlap_end = min(seg_dict["end"], rs.end)
                    overlap = max(0, overlap_end - overlap_start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_retry = rs

                if best_retry and best_retry.words:
                    retry_conf = (
                        sum(w.probability for w in best_retry.words)
                        / len(best_retry.words)
                    )
                    if retry_conf > seg_dict["avg_confidence"]:
                        retry_filtered = _filter_low_confidence_words([best_retry])
                        if retry_filtered:
                            segment_list[i] = retry_filtered[0]

    full_text = " ".join(seg["text"].strip() for seg in segment_list)

    result = {
        "text": full_text,
        "language": info.language,
        "segments": segment_list,
    }
    return result


# ---------------------------------------------------------------------------
# Speaker diarization — identify who is speaking
# ---------------------------------------------------------------------------
def _try_import_diarization():
    """Try importing pyannote. Returns None if not installed."""
    try:
        from pyannote.audio import Pipeline as PyannotePipeline
        return PyannotePipeline
    except ImportError:
        return None


def diarize_audio(
    wav_path: str | Path,
    hf_token: str | None = None,
    num_speakers: int | None = None,
) -> list[dict[str, Any]]:
    """Run speaker diarization on a WAV file.

    Requires pyannote.audio and a HuggingFace token with access to
    pyannote/speaker-diarization-3.1 model.

    Returns list of {"speaker": str, "start": float, "end": float}.
    """
    PyannotePipeline = _try_import_diarization()
    if PyannotePipeline is None:
        return []

    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        return []

    pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )

    wav_path = Path(wav_path)
    diarization = pipeline(
        str(wav_path),
        num_speakers=num_speakers,
    )

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(
            {
                "speaker": speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
            }
        )
    return turns


def assign_speakers_to_segments(
    segments: list[dict[str, Any]],
    speaker_turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign speaker labels to transcription segments based on time overlap."""
    if not speaker_turns:
        return segments

    result = []
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_mid = (seg_start + seg_end) / 2.0

        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for turn in speaker_turns:
            overlap_start = max(seg_start, turn["start"])
            overlap_end = min(seg_end, turn["end"])
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

        # Fallback: if no overlap found, use closest turn to segment midpoint
        if best_overlap == 0:
            closest_dist = float("inf")
            for turn in speaker_turns:
                turn_mid = (turn["start"] + turn["end"]) / 2.0
                dist = abs(seg_mid - turn_mid)
                if dist < closest_dist:
                    closest_dist = dist
                    best_speaker = turn["speaker"]

        new_seg = dict(seg)
        new_seg["speaker"] = best_speaker
        result.append(new_seg)

    return result


def diarization_available() -> bool:
    """Check if pyannote and HF_TOKEN are available."""
    if _try_import_diarization() is None:
        return False
    token = os.getenv("HF_TOKEN")
    return bool(token)


# ---------------------------------------------------------------------------
# LLM post-correction (Ollama / OpenAI-compatible)
# ---------------------------------------------------------------------------
def _call_llm(prompt: str, timeout: int = 120) -> str | None:
    """Call Ollama generate endpoint. Returns None if unavailable."""
    try:
        resp = requests.post(
            f"{LLM_BASE_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception:
        return None


def llm_available() -> bool:
    """Check whether the configured LLM endpoint is reachable."""
    try:
        resp = requests.get(f"{LLM_BASE_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def correct_text_with_llm(text: str) -> str:
    """Send full transcription text to LLM for correction."""
    if not text.strip():
        return text

    prompt = (
        "You are a French language expert and professional proofreader. "
        "Correct the following speech-to-text transcription.\n\n"
        "Fix:\n"
        "- Spelling and grammar errors\n"
        "- Missing or misheard words (infer from context)\n"
        "- Punctuation and capitalization\n"
        "- Proper nouns and names\n\n"
        "Rules:\n"
        "- Keep the meaning exactly the same\n"
        "- Do NOT add commentary or explanations\n"
        "- Return ONLY the corrected text\n\n"
        f"Text:\n{text}"
    )
    result = _call_llm(prompt, timeout=180)
    return result if result else text


def correct_segments_with_llm(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Correct segment texts in batches via LLM."""
    if not segments:
        return segments

    batch_size = 20
    corrected: list[dict[str, Any]] = []

    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        numbered = "\n".join(
            f"{j + 1}: {seg['text'].strip()}" for j, seg in enumerate(batch)
        )

        prompt = (
            "You are a French language expert. Correct these speech-to-text segments.\n\n"
            "Fix spelling, grammar, punctuation, and misheard words using context.\n"
            "Return one corrected segment per line with the same numbering.\n"
            "Format: N: corrected text\n"
            "Return ONLY the corrected numbered lines, nothing else.\n\n"
            f"{numbered}"
        )

        result = _call_llm(prompt, timeout=180)
        if result:
            # Parse numbered lines back
            corrections: dict[int, str] = {}
            for line in result.strip().split("\n"):
                match = re.match(r"(\d+):\s*(.*)", line)
                if match:
                    idx = int(match.group(1)) - 1
                    corrections[idx] = match.group(2).strip()

            for j, seg in enumerate(batch):
                new_seg = dict(seg)
                if j in corrections and corrections[j]:
                    new_seg["text"] = corrections[j]
                corrected.append(new_seg)
        else:
            corrected.extend(batch)

    return corrected


def apply_llm_correction(result: dict[str, Any]) -> dict[str, Any]:
    """Apply LLM post-correction to a full transcription result."""
    corrected_text = correct_text_with_llm(result.get("text", ""))
    corrected_segments = correct_segments_with_llm(result.get("segments", []))

    return {
        **result,
        "text": corrected_text,
        "text_before_correction": result.get("text", ""),
        "segments": corrected_segments,
        "llm_corrected": True,
    }


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------
def save_transcription_artifacts(
    result: dict[str, Any],
    source_media_path: str | Path,
    output_dir: str | Path = "outputs",
) -> ArtifactPaths:
    """Save JSON and TXT artifacts from transcription output."""
    source_media_path = Path(source_media_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = source_media_path.stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"{stem}_{ts}.json"
    txt_path = output_dir / f"{stem}_{ts}.txt"

    # Build clean segments (strip word details for compact JSON if desired)
    segments_out = []
    for seg in result.get("segments", []):
        seg_out: dict[str, Any] = {
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": seg.get("text", ""),
        }
        if "speaker" in seg:
            seg_out["speaker"] = seg["speaker"]
        if "avg_confidence" in seg:
            seg_out["avg_confidence"] = seg["avg_confidence"]
        if "words" in seg:
            seg_out["words"] = seg["words"]
        segments_out.append(seg_out)

    payload = {
        "source": str(source_media_path),
        "created_at": ts,
        "language": result.get("language"),
        "text": result.get("text", "").strip(),
        "text_before_correction": result.get("text_before_correction", ""),
        "llm_corrected": result.get("llm_corrected", False),
        "diarized": result.get("diarized", False),
        "segments": segments_out,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(payload["text"])

    return ArtifactPaths(json_path=json_path, txt_path=txt_path)


# ---------------------------------------------------------------------------
# Transcript loading & search
# ---------------------------------------------------------------------------
def load_transcript_json(transcript_json_path: str | Path) -> dict[str, Any]:
    transcript_json_path = Path(transcript_json_path)
    if not transcript_json_path.exists():
        raise FileNotFoundError(f"Transcript JSON not found: {transcript_json_path}")

    with transcript_json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().strip()
    return re.sub(r"\s+", " ", value)


def timestamp(seconds: float) -> str:
    seconds = max(0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _segment_score(query_norm: str, segment_norm: str) -> float:
    if not segment_norm:
        return 0.0

    score = 0.0
    if query_norm in segment_norm:
        score += 1.0

    tokens = re.findall(r"[\w']+", segment_norm)
    if not tokens:
        return score

    fuzzy = max(SequenceMatcher(None, query_norm, token).ratio() for token in tokens)
    score += 0.6 * fuzzy
    return score


def search_transcript(
    transcript_payload: dict[str, Any],
    query: str,
    min_score: float = 0.8,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return ranked segment matches for a query."""
    query_norm = normalize_text(query)
    if not query_norm:
        return []

    matches: list[dict[str, Any]] = []
    for segment in transcript_payload.get("segments", []):
        text = str(segment.get("text", "")).strip()
        score = _segment_score(query_norm, normalize_text(text))
        if score >= min_score:
            entry: dict[str, Any] = {
                "score": round(score, 4),
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "text": text,
            }
            if "speaker" in segment:
                entry["speaker"] = segment["speaker"]
            if "avg_confidence" in segment:
                entry["avg_confidence"] = segment["avg_confidence"]
            matches.append(entry)

    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches[:limit] if limit > 0 else matches


def default_model_for_cpu() -> str:
    """Default to large-v3 for best accuracy."""
    return os.getenv("WHISPER_MODEL", "large-v3")


# ---------------------------------------------------------------------------
# IPTV / Live stream capture & transcription
# ---------------------------------------------------------------------------
def capture_stream_chunk(
    stream_url: str,
    output_wav_path: str | Path,
    duration: int = 30,
) -> Path:
    """Capture a chunk of audio from a live stream URL.

    Supports IPTV (m3u8, rtsp, rtp, udp), HTTP streams, and any URL
    that ffmpeg or yt-dlp can handle.
    """
    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_ffmpeg_available()

    # Try yt-dlp first for complex URLs (YouTube, HLS with auth, etc.)
    # Fall back to raw ffmpeg for direct IPTV streams
    yt_dlp_url = None
    if any(x in stream_url for x in ["youtube.com", "youtu.be", "twitch.tv"]):
        try:
            yt_result = subprocess.run(
                ["yt-dlp", "-g", "-f", "bestaudio", stream_url],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if yt_result.returncode == 0 and yt_result.stdout.strip():
                yt_dlp_url = yt_result.stdout.strip()
        except Exception:
            pass

    input_url = yt_dlp_url or stream_url

    cmd = [
        "ffmpeg",
        "-y",
        # Timeout for network streams
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5",
        "-i",
        input_url,
        "-t",
        str(duration),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        str(output_wav_path),
    ]

    process = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 30)
    if process.returncode != 0:
        raise RuntimeError(f"Stream capture failed: {process.stderr.strip()[:500]}")

    if not output_wav_path.exists() or output_wav_path.stat().st_size < 1000:
        raise RuntimeError("Captured audio file is too small or empty.")

    return output_wav_path


def probe_stream(stream_url: str) -> dict[str, Any]:
    """Probe a stream URL to check if it's reachable and get info."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                "-i", stream_url,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            info = json.loads(result.stdout)
            return {"reachable": True, "info": info}
    except Exception:
        pass
    return {"reachable": False, "info": {}}


def transcribe_stream_chunk(
    stream_url: str,
    chunk_dir: str | Path,
    chunk_index: int,
    duration: int = 30,
    model_name: str = "large-v3",
    language: str | None = "fr",
    initial_prompt: str | None = None,
    denoise: bool = True,
) -> dict[str, Any]:
    """Capture and transcribe a single chunk from a live stream.

    Returns a dict with chunk metadata, segments, and text.
    """
    chunk_dir = Path(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    wav_path = chunk_dir / f"chunk_{chunk_index:04d}.wav"

    # Capture
    capture_stream_chunk(stream_url, wav_path, duration=duration)

    # Denoise
    if denoise:
        denoise_audio(wav_path)

    # Transcribe (single pass for speed on live streams)
    result = transcribe_wav(
        wav_path,
        model_name=model_name,
        language=language,
        initial_prompt=initial_prompt,
        multi_pass=False,
    )

    # Add time offset based on chunk index
    time_offset = chunk_index * duration
    for seg in result.get("segments", []):
        seg["start"] = round(seg["start"] + time_offset, 3)
        seg["end"] = round(seg["end"] + time_offset, 3)
        if "words" in seg:
            for w in seg["words"]:
                w["start"] = round(w["start"] + time_offset, 3)
                w["end"] = round(w["end"] + time_offset, 3)

    result["chunk_index"] = chunk_index
    result["time_offset"] = time_offset
    result["stream_url"] = stream_url

    return result


def save_stream_session(
    all_chunks: list[dict[str, Any]],
    stream_url: str,
    output_dir: str | Path = "outputs",
) -> ArtifactPaths:
    """Merge all chunk results into a single transcript file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"stream_{ts}.json"
    txt_path = output_dir / f"stream_{ts}.txt"

    all_segments = []
    all_text_parts = []
    for chunk in all_chunks:
        all_segments.extend(chunk.get("segments", []))
        text = chunk.get("text", "").strip()
        if text:
            all_text_parts.append(text)

    full_text = " ".join(all_text_parts)

    payload = {
        "source": stream_url,
        "created_at": ts,
        "mode": "live_stream",
        "language": all_chunks[0].get("language") if all_chunks else None,
        "total_chunks": len(all_chunks),
        "text": full_text,
        "segments": all_segments,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(full_text)

    return ArtifactPaths(json_path=json_path, txt_path=txt_path)
