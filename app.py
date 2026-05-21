import tempfile
from pathlib import Path

import streamlit as st

from pipeline import (
    INITIAL_PROMPTS,
    _get_model,
    apply_llm_correction,
    assign_speakers_to_segments,
    default_model_for_cpu,
    denoise_audio,
    diarization_available,
    diarize_audio,
    llm_available,
    load_transcript_json,
    prepare_audio,
    probe_stream,
    save_stream_session,
    save_transcription_artifacts,
    search_transcript,
    timestamp,
    transcribe_stream_chunk,
    transcribe_wav,
)


@st.cache_resource
def load_whisper_model(model_name: str):
    """Cache the Whisper model so it is loaded only once."""
    return _get_model(model_name)


st.set_page_config(page_title="SNRT STT Search", page_icon="STT", layout="wide")

st.title("SNRT Speech-to-Text and Smart Search")

# ---- Sidebar (shared settings) ----
with st.sidebar:
    st.header("Transcription Settings")
    model_options = ["tiny", "base", "small", "medium", "large-v3"]
    default_model = default_model_for_cpu()
    default_index = (
        model_options.index(default_model) if default_model in model_options else 4
    )
    model_name = st.selectbox("Whisper model", model_options, index=default_index)

    language_mode = st.selectbox(
        "Language",
        ["fr", "auto", "en", "ar", "es"],
        index=0,
        help="Use fr for French-first mode, then switch to auto/multi-language later.",
    )

    st.divider()
    st.header("Audio & Accuracy")
    use_denoise = st.checkbox(
        "Noise reduction",
        value=True,
        help="Remove background noise, hum, and static before transcription.",
    )
    use_multi_pass = st.checkbox(
        "Multi-pass (retry low-confidence)",
        value=True,
        help="Re-transcribe low-confidence segments with different settings for better accuracy.",
    )

    default_prompt = INITIAL_PROMPTS.get(language_mode, "")
    custom_prompt = st.text_area(
        "Initial prompt (domain vocabulary)",
        value=default_prompt,
        height=100,
        help="Prime Whisper with names, terms, and vocabulary it should expect.",
    )

    st.divider()
    st.header("Speaker Diarization")
    has_diarization = diarization_available()
    if has_diarization:
        st.success("Diarization available")
    else:
        st.warning("Needs pyannote.audio + HF_TOKEN env var")
    use_diarization = st.checkbox(
        "Identify speakers",
        value=False,
        disabled=not has_diarization,
    )
    num_speakers = None
    if use_diarization:
        speaker_mode = st.radio("Number of speakers", ["Auto-detect", "Specify"])
        if speaker_mode == "Specify":
            num_speakers = st.number_input("How many speakers?", min_value=1, max_value=20, value=2)

    st.divider()
    st.header("LLM Post-Correction")
    has_llm = llm_available()
    if has_llm:
        st.success("Ollama connected")
    else:
        st.warning("Ollama not detected at localhost:11434")
    use_llm = st.checkbox(
        "Enable LLM correction",
        value=has_llm,
        disabled=not has_llm,
    )

    st.divider()
    st.header("Search")
    min_score = st.slider("Search sensitivity (min score)", 0.4, 1.5, 0.8, 0.05)
    limit = st.slider("Max results", 5, 50, 20, 1)

# Pre-load model
load_whisper_model(model_name)

# ---- Session state ----
if "last_transcript_json" not in st.session_state:
    st.session_state["last_transcript_json"] = None

# ---- Tabs ----
tab_file, tab_stream, tab_search = st.tabs(["Upload File", "Live Stream / IPTV", "Search Transcript"])

# ==========================================================================
# TAB 1: File upload
# ==========================================================================
with tab_file:
    st.subheader("Upload and Transcribe")
    uploaded = st.file_uploader(
        "Upload audio/video",
        type=["mp3", "wav", "m4a", "mp4", "mkv", "mov", "webm"],
    )

    run_file_btn = st.button("Transcribe File")

    if run_file_btn:
        if uploaded is None:
            st.error("Please upload a media file first.")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                source_path = tmpdir_path / uploaded.name
                source_path.write_bytes(uploaded.getvalue())

                output_dir = Path("outputs")
                wav_path = output_dir / "prepared_audio.wav"

                try:
                    with st.spinner("Converting and normalizing audio..."):
                        prepare_audio(source_path, wav_path)

                    if use_denoise:
                        with st.spinner("Removing background noise..."):
                            denoise_audio(wav_path)

                    with st.spinner(f"Transcribing with Whisper ({model_name})..."):
                        lang = None if language_mode == "auto" else language_mode
                        result = transcribe_wav(
                            wav_path,
                            model_name=model_name,
                            language=lang,
                            initial_prompt=custom_prompt.strip() or None,
                            multi_pass=use_multi_pass,
                        )

                    if use_diarization:
                        with st.spinner("Identifying speakers..."):
                            speaker_turns = diarize_audio(wav_path, num_speakers=num_speakers)
                            if speaker_turns:
                                result["segments"] = assign_speakers_to_segments(
                                    result["segments"], speaker_turns
                                )
                                result["diarized"] = True

                    if use_llm:
                        with st.spinner("Applying LLM correction..."):
                            result = apply_llm_correction(result)

                    artifacts = save_transcription_artifacts(
                        result, source_media_path=uploaded.name, output_dir=output_dir,
                    )
                except Exception as exc:
                    st.exception(exc)
                else:
                    st.session_state["last_transcript_json"] = str(artifacts.json_path)
                    st.success("Transcription completed.")
                    if result.get("llm_corrected"):
                        st.info("LLM post-correction applied.")
                    if result.get("diarized"):
                        st.info("Speaker diarization applied.")
                    st.write(f"JSON: {artifacts.json_path}")
                    st.write(f"TXT: {artifacts.txt_path}")

# ==========================================================================
# TAB 2: Live Stream / IPTV
# ==========================================================================
with tab_stream:
    st.subheader("Live Stream / IPTV Transcription")
    st.write("Capture audio from a live stream and transcribe in real-time chunks.")

    stream_url = st.text_input(
        "Stream URL",
        placeholder="http://example.com/stream.m3u8 or rtsp://...",
        help="Paste an IPTV m3u8 URL, RTSP URL, or any stream link.",
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        chunk_duration = st.number_input("Chunk duration (seconds)", 10, 120, 30, 5)
    with col_s2:
        num_chunks = st.number_input("Number of chunks (0 = unlimited)", 0, 100, 5, 1)

    col_probe, col_start = st.columns(2)

    with col_probe:
        probe_btn = st.button("Test Connection")

    with col_start:
        start_stream_btn = st.button("Start Transcription")

    if probe_btn and stream_url.strip():
        with st.spinner("Probing stream..."):
            result = probe_stream(stream_url.strip())
        if result["reachable"]:
            st.success("Stream is reachable!")
            info = result.get("info", {})
            fmt = info.get("format", {})
            if fmt:
                st.write(f"Format: {fmt.get('format_long_name', 'unknown')}")
                duration = fmt.get("duration")
                if duration:
                    st.write(f"Duration: {float(duration):.0f}s")
        else:
            st.error("Could not reach stream. Check the URL and your network.")

    if start_stream_btn:
        if not stream_url.strip():
            st.error("Please enter a stream URL.")
        else:
            output_dir = Path("outputs")
            chunk_dir = output_dir / "stream_chunks"
            all_chunks = []
            lang = None if language_mode == "auto" else language_mode
            prompt = custom_prompt.strip() or None

            progress_bar = st.progress(0)
            status_text = st.empty()
            live_transcript = st.empty()

            max_c = num_chunks if num_chunks > 0 else 999
            accumulated_text = []

            for i in range(max_c):
                progress = (i + 1) / max_c if num_chunks > 0 else 0
                progress_bar.progress(min(progress, 1.0))
                status_text.write(f"Capturing chunk {i + 1}{'/' + str(num_chunks) if num_chunks > 0 else ''}...")

                try:
                    chunk_result = transcribe_stream_chunk(
                        stream_url=stream_url.strip(),
                        chunk_dir=chunk_dir,
                        chunk_index=i,
                        duration=chunk_duration,
                        model_name=model_name,
                        language=lang,
                        initial_prompt=prompt,
                        denoise=use_denoise,
                    )
                    all_chunks.append(chunk_result)

                    text = chunk_result.get("text", "").strip()
                    if text:
                        accumulated_text.append(text)

                    # Update live transcript display
                    live_transcript.text_area(
                        "Live Transcript",
                        value="\n\n".join(accumulated_text),
                        height=300,
                        disabled=True,
                    )

                except Exception as exc:
                    status_text.write(f"Chunk {i + 1} failed: {exc}")
                    continue

            progress_bar.progress(1.0)
            status_text.write("Stream transcription complete!")

            if all_chunks:
                artifacts = save_stream_session(all_chunks, stream_url.strip(), output_dir)
                st.session_state["last_transcript_json"] = str(artifacts.json_path)
                st.success(f"Saved {len(all_chunks)} chunks.")
                st.write(f"JSON: {artifacts.json_path}")
                st.write(f"TXT: {artifacts.txt_path}")

# ==========================================================================
# TAB 3: Search
# ==========================================================================
with tab_search:
    st.subheader("Search Transcript")

    query = st.text_input("Search keyword or text", value="")

    if st.session_state.get("last_transcript_json"):
        transcript_json = Path(st.session_state["last_transcript_json"])
        st.write(f"Active transcript: {transcript_json}")

        if query.strip():
            try:
                payload = load_transcript_json(transcript_json)
                results = search_transcript(
                    payload, query=query, min_score=min_score, limit=limit
                )
            except Exception as exc:
                st.exception(exc)
                results = []

            st.write(f"Matches found: {len(results)}")
            for idx, item in enumerate(results, 1):
                start = timestamp(item["start"])
                end = timestamp(item["end"])
                speaker = item.get("speaker", "")
                speaker_tag = f" [{speaker}]" if speaker else ""
                conf = item.get("avg_confidence")
                conf_tag = f" conf={conf}" if conf is not None else ""
                st.markdown(
                    f"**{idx:02d}. [{start} -> {end}]{speaker_tag} score={item['score']}{conf_tag}**"
                )
                st.write(item["text"])
        else:
            st.info("Type a keyword or phrase to search.")
    else:
        st.info("Transcribe a file or stream first, then search here.")
