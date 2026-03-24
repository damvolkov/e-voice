"""Gradio UI for e-voice — STT/TTS playground + model management."""

import threading
from pathlib import Path

import gradio as gr

from e_voice.adapters.api_client import APIClient, create_stream, remove_stream, send_stream_chunk
from e_voice.core.logger import logger
from e_voice.core.settings import settings as st

##### THEME #####

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("JetBrains Mono"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#050508",
    body_background_fill_dark="#050508",
    block_background_fill="#0d1117",
    block_background_fill_dark="#0d1117",
    block_border_color="#1e3a4f",
    block_border_color_dark="#1e3a4f",
    block_label_text_color="#7dd3d8",
    block_label_text_color_dark="#7dd3d8",
    block_title_text_color="#7dd3d8",
    block_title_text_color_dark="#7dd3d8",
    body_text_color="#c8d6d8",
    body_text_color_dark="#c8d6d8",
    body_text_color_subdued="#5a7a7e",
    body_text_color_subdued_dark="#5a7a7e",
    button_primary_background_fill="#0891b2",
    button_primary_background_fill_dark="#0891b2",
    button_primary_background_fill_hover="#06b6d4",
    button_primary_text_color="#000",
    button_primary_text_color_dark="#000",
    button_secondary_background_fill="#0d1117",
    button_secondary_background_fill_dark="#0d1117",
    button_secondary_border_color="#1e3a4f",
    button_secondary_border_color_dark="#1e3a4f",
    button_secondary_text_color="#7dd3d8",
    button_secondary_text_color_dark="#7dd3d8",
    input_background_fill="#0a0f14",
    input_background_fill_dark="#0a0f14",
    input_border_color="#1e3a4f",
    input_border_color_dark="#1e3a4f",
    input_placeholder_color="#3a5a5e",
    input_placeholder_color_dark="#3a5a5e",
    border_color_accent="#0891b2",
    border_color_accent_dark="#0891b2",
    color_accent_soft="#0891b220",
    color_accent_soft_dark="#0891b220",
    slider_color="#0891b2",
    slider_color_dark="#0891b2",
)

_CSS = """
footer { display: none !important; }
.tab-nav button { color: #5a7a7e !important; border-color: transparent !important; }
.tab-nav button.selected { color: #06b6d4 !important; border-color: #0891b2 !important; }
.hide-device-select .audio-input-select { display: none !important; }
.hide-device-select select { display: none !important; }
"""


##### HELPERS #####


def _base_url() -> str:
    return f"http://127.0.0.1:{st.system.port}"


def _api() -> APIClient:
    return APIClient(_base_url())


##### LIVE MIC WRAPPERS #####


def _ws_url() -> str:
    return f"ws://127.0.0.1:{st.ws.port}"


def _on_start(language: str) -> dict | None:
    return create_stream(_ws_url(), language)


def _on_chunk(state: dict | None, audio_chunk) -> tuple[dict | None, str]:
    return send_stream_chunk(state, audio_chunk)


def _on_stop(state: dict | None) -> tuple[None, str]:
    return remove_stream(state)


##### STT WRAPPER #####


def _transcribe(audio_path: str, model: str, task: str, language: str, temperature: float, fmt: str, stream: bool):
    """Transcribe audio file. Generator for Gradio — yields text progressively or once."""
    if not audio_path:
        yield "Error: No audio file provided"
        return

    api = _api()
    try:
        if stream:
            gen = api.create_transcription_stream(
                Path(audio_path),
                model=model,
                task=task,
                language=language,
                temperature=temperature,
            )
            try:
                yield from gen
            finally:
                gen.close()
        else:
            yield api.create_transcription(
                Path(audio_path),
                model=model,
                task=task,
                language=language,
                response_format=fmt,
                temperature=temperature,
            )
    except Exception as e:
        error_msg = f"Error: {type(e).__name__}"
        if str(e):
            error_msg += f" — {e}"
        logger.error("transcription failed", error=error_msg, path=audio_path)
        yield error_msg


##### TTS WRAPPER #####


def _synthesize(text: str, voice: str, speed: float) -> str | None:
    """Synthesize and return temp file path."""
    if not text.strip():
        return None
    audio_bytes = _api().create_synthesis(text, voice=voice, speed=speed)
    tmp = Path("/tmp/evoice_tts_output.wav")
    tmp.write_bytes(audio_bytes)
    return str(tmp)


##### VOICE WRAPPERS #####

_LANG_LABELS: dict[str, str] = {
    "en-us": "🇺🇸 English (US)",
    "en-gb": "🇬🇧 English (UK)",
    "es": "🇪🇸 Spanish",
    "fr": "🇫🇷 French",
    "hi": "🇮🇳 Hindi",
    "it": "🇮🇹 Italian",
    "ja": "🇯🇵 Japanese",
    "pt-br": "🇧🇷 Portuguese (BR)",
    "zh": "🇨🇳 Chinese",
}


def _fetch_voice_choices() -> list[str]:
    """Fetch available voices from API, sorted by language then name."""
    voices = _api().get_voices()
    if not voices:
        return ["af_heart"]
    return [v["id"] for v in sorted(voices, key=lambda v: (v.get("language", ""), v["id"]))]


def _format_voices_display() -> str:
    """Format voices grouped by language as Markdown."""
    voices = _api().get_voices()
    if not voices:
        return "No voices available — TTS model not loaded."

    grouped: dict[str, list[str]] = {}
    for v in voices:
        lang = v.get("language", "unknown")
        grouped.setdefault(lang, []).append(v["id"])

    lines: list[str] = []
    for lang in sorted(grouped):
        label = _LANG_LABELS.get(lang, lang)
        names = ", ".join(f"`{n}`" for n in sorted(grouped[lang]))
        lines.append(f"**{label}** — {names}")
    return "\n\n".join(lines)


##### MODEL WRAPPERS #####


def _refresh_models() -> str:
    data = _api().get_downloaded_models()
    lines = ["**STT Models:**"]
    for m in data.get("stt", []):
        lines.append(f"- `{m['id']}` ({m['size_mb']} MB)")
    if not data.get("stt"):
        lines.append("- (none)")
    lines.append("\n**TTS Models:**")
    for m in data.get("tts", []):
        lines.append(f"- `{m['id']}` ({m['size_mb']} MB)")
    if not data.get("tts"):
        lines.append("- (none)")
    return "\n".join(lines)


def _download_model(model_id: str, service: str) -> str:
    return _api().download_model(model_id, service)


##### APP FACTORY #####


def _load_logo_html() -> str:
    logo_path = st.BASE_DIR / "assets" / "e-voice-landscape-front.svg"
    if not logo_path.exists():
        return ""
    svg = logo_path.read_text()
    return f'<div style="display:flex;justify-content:center;padding:20px 0 8px">{svg}</div>'


def create_app() -> gr.Blocks:
    """Build the Gradio Blocks UI."""
    api = _api()

    with gr.Blocks(title="e-voice") as app:
        gr.HTML(_load_logo_html())

        with gr.Tabs():
            ##### LIVE MIC TAB #####
            with gr.Tab("Live Mic"):
                with gr.Row():
                    with gr.Column(scale=1):
                        mic_lang = gr.Dropdown(
                            choices=["auto", "en", "es", "fr", "de", "it", "pt", "ja", "zh"],
                            value=st.stt.default_language or "auto",
                            label="Language",
                        )
                        mic_audio = gr.Audio(
                            sources=["microphone"],
                            streaming=True,
                            type="numpy",
                            elem_classes=["hide-device-select"],
                        )
                    with gr.Column(scale=1):
                        mic_output = gr.Textbox(label="Transcription", lines=12, interactive=False)

                mic_state = gr.State(value=None)
                mic_audio.start_recording(fn=_on_start, inputs=[mic_lang], outputs=[mic_state])
                mic_audio.stream(
                    fn=_on_chunk, inputs=[mic_state, mic_audio], outputs=[mic_state, mic_output], stream_every=0.5
                )
                mic_audio.stop_recording(fn=_on_stop, inputs=[mic_state], outputs=[mic_state, mic_output])

            ##### STT TAB #####
            with gr.Tab("Speech-to-Text"):
                with gr.Row():
                    with gr.Column(scale=1):
                        stt_audio = gr.Audio(type="filepath", label="Audio input")
                        stt_model = gr.Dropdown(
                            choices=api.get_models() or [st.stt.model], value=st.stt.model, label="Model"
                        )
                        stt_task = gr.Dropdown(choices=["transcribe", "translate"], value="transcribe", label="Task")
                        stt_language = gr.Textbox(value=st.stt.default_language or "auto", label="Language")
                        stt_format = gr.Dropdown(
                            choices=["text", "json", "verbose_json", "srt", "vtt"], value="text", label="Format"
                        )
                        stt_temp = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature")
                        stt_stream = gr.Checkbox(value=True, label="Stream (SSE)")
                        stt_btn = gr.Button("Transcribe", variant="primary")
                    with gr.Column(scale=1):
                        stt_output = gr.Textbox(label="Output", lines=14, interactive=False)

                stt_btn.click(
                    fn=_transcribe,
                    inputs=[stt_audio, stt_model, stt_task, stt_language, stt_temp, stt_format, stt_stream],
                    outputs=stt_output,
                    show_progress="full",
                )

            ##### TTS TAB #####
            with gr.Tab("Text-to-Speech"):
                with gr.Row():
                    with gr.Column(scale=1):
                        tts_text = gr.Textbox(label="Text", lines=4, placeholder="Enter text to synthesize...")
                        voice_choices = _fetch_voice_choices()
                        tts_voice = gr.Dropdown(
                            choices=voice_choices,
                            value=st.tts.default_voice if st.tts.default_voice in voice_choices else voice_choices[0],
                            label="Voice",
                        )
                        tts_speed = gr.Slider(0.5, 2.0, value=st.tts.default_speed, step=0.1, label="Speed")
                        tts_btn = gr.Button("Synthesize", variant="primary")
                    with gr.Column(scale=1):
                        tts_output = gr.Audio(label="Output", type="filepath")

                tts_btn.click(fn=_synthesize, inputs=[tts_text, tts_voice, tts_speed], outputs=tts_output)

            ##### VOICES TAB #####
            with gr.Tab("Voices"):
                voices_display = gr.Markdown(value=_format_voices_display)
                voices_refresh_btn = gr.Button("Refresh", variant="secondary")
                voices_refresh_btn.click(fn=_format_voices_display, outputs=voices_display)

            ##### MODELS TAB #####
            with gr.Tab("Models"):
                models_display = gr.Markdown(value=_refresh_models, label="Downloaded models")
                refresh_btn = gr.Button("Refresh", variant="secondary")
                refresh_btn.click(fn=_refresh_models, outputs=models_display)

                gr.Markdown("---\n### Download Model")
                with gr.Row():
                    dl_model_id = gr.Textbox(
                        label="Model ID", placeholder="mobiuslabsgmbh/faster-whisper-large-v3-turbo"
                    )
                    dl_service = gr.Dropdown(choices=["stt", "tts"], value="stt", label="Service")
                dl_btn = gr.Button("Download", variant="primary")
                dl_result = gr.Textbox(label="Result", interactive=False)

                dl_btn.click(fn=_download_model, inputs=[dl_model_id, dl_service], outputs=dl_result).then(
                    fn=_refresh_models, outputs=models_display
                )

    return app


##### LAUNCHER #####


def launch_background() -> None:
    """Launch Gradio in a background daemon thread."""
    if not st.front.enabled:
        logger.info("gradio UI disabled", step="START")
        return

    def _run() -> None:
        app = create_app()
        favicon = st.BASE_DIR / "assets" / "e-voice-icon-front.png"
        app.launch(
            theme=_THEME,
            css=_CSS,
            favicon_path=str(favicon) if favicon.exists() else None,
            server_name="0.0.0.0",
            server_port=st.front.port,
            share=st.front.share,
            quiet=True,
            show_error=st.system.debug,
        )

    thread = threading.Thread(target=_run, daemon=True, name="gradio-ui")
    thread.start()
    logger.info("gradio UI started", step="START", port=st.front.port)
