"""Gradio UI for e-voice — STT/TTS playground + model management."""

import threading
from collections.abc import Generator
from pathlib import Path

import gradio as gr
import httpx
from httpx_sse import connect_sse

from e_voice.core.logger import logger
from e_voice.core.settings import settings as st

_TIMEOUT = httpx.Timeout(timeout=180.0)

##### THEME #####

_CSS = """
.gradio-container { background-color: #1a1a2e !important; }
.gr-button-primary { background-color: #1e6091 !important; border-color: #1e6091 !important; }
.gr-button-primary:hover { background-color: #2980b9 !important; }
.gr-panel { border-color: #2c3e6b !important; }
footer { display: none !important; }
"""


##### HTTP CLIENT HELPERS #####


def _base_url() -> str:
    return f"http://127.0.0.1:{st.system.port}"


def _client() -> httpx.Client:
    return httpx.Client(base_url=_base_url(), timeout=_TIMEOUT)


def _fetch_stt_models() -> list[str]:
    """Fetch loaded STT model IDs from API."""
    try:
        with _client() as c:
            resp = c.get("/v1/models")
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
    except Exception:
        return [st.stt.model]


def _fetch_downloaded_models() -> dict:
    """Fetch all downloaded models from disk via API."""
    try:
        with _client() as c:
            resp = c.get("/v1/models/list")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"stt": [], "tts": []}


##### STT HANDLERS #####


def _transcribe(
    audio_path: str,
    model: str,
    task: str,
    language: str,
    temperature: float,
    response_format: str,
    stream: bool,
) -> Generator[str, None, None]:
    """Transcribe or translate audio via HTTP API."""
    endpoint = "/v1/audio/translations" if task == "translate" else "/v1/audio/transcriptions"

    with _client() as c, Path(audio_path).open("rb") as f:
        form_data = {
            "model": model,
            "response_format": response_format if not stream else "text",
            "temperature": str(temperature),
        }
        if language and language != "auto":
            form_data["language"] = language

        if stream:
            form_data["stream"] = "true"
            accumulated = ""
            with connect_sse(c, "POST", endpoint, files={"file": f}, data=form_data) as sse:
                for event in sse.iter_sse():
                    accumulated += event.data
                    yield accumulated
        else:
            resp = c.post(endpoint, files={"file": f}, data=form_data)
            resp.raise_for_status()
            yield resp.text


##### TTS HANDLERS #####


def _synthesize(text: str, voice: str, speed: float) -> str | None:
    """Synthesize speech and return path to audio file."""
    if not text.strip():
        return None
    with _client() as c:
        resp = c.post(
            "/v1/audio/speech",
            json={"input": text, "model": "kokoro", "voice": voice, "speed": speed, "response_format": "wav"},
        )
        resp.raise_for_status()
        tmp = Path("/tmp/evoice_tts_output.wav")
        tmp.write_bytes(resp.content)
        return str(tmp)


##### MODEL MANAGEMENT HANDLERS #####


def _download_model(model_id: str, service: str) -> str:
    """Download a model via the system API."""
    if not model_id.strip():
        return "No model ID provided"
    try:
        with _client() as c:
            resp = c.post("/v1/models/download", json={"model": model_id, "service": service}, timeout=600.0)
            resp.raise_for_status()
            data = resp.json()
            return f"Downloaded: {data['model']} → {data['path']}"
    except Exception as exc:
        return f"Error: {exc}"


def _refresh_models() -> str:
    """Refresh and format the downloaded models list."""
    data = _fetch_downloaded_models()
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


##### GRADIO APP FACTORY #####

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.gray,
).set(
    body_background_fill="#1a1a2e",
    body_background_fill_dark="#1a1a2e",
    block_background_fill="#16213e",
    block_background_fill_dark="#16213e",
    block_border_color="#2c3e6b",
    block_border_color_dark="#2c3e6b",
    button_primary_background_fill="#1e6091",
    button_primary_background_fill_dark="#1e6091",
    button_primary_background_fill_hover="#2980b9",
    input_background_fill="#0f3460",
    input_background_fill_dark="#0f3460",
    input_border_color="#2c3e6b",
    input_border_color_dark="#2c3e6b",
)


def create_app() -> gr.Blocks:
    """Build the Gradio Blocks UI."""
    logo_path = st.BASE_DIR / "assets" / "e-voice-landscape.svg"
    logo_html = ""
    if logo_path.exists():
        logo_html = f'<div style="text-align:center;padding:10px"><img src="/file={logo_path}" width="300"></div>'

    with gr.Blocks(title="e-voice") as app:
        if logo_html:
            gr.HTML(logo_html)
        gr.Markdown(f"<center><small>v{st.API_VERSION} · API at <code>{_base_url()}</code></small></center>")

        with gr.Tabs():
            ##### STT TAB #####
            with gr.Tab("Speech-to-Text"):
                with gr.Row():
                    with gr.Column(scale=1):
                        stt_audio = gr.Audio(type="filepath", label="Audio input")
                        stt_model = gr.Dropdown(
                            choices=_fetch_stt_models(),
                            value=st.stt.model,
                            label="Model",
                        )
                        stt_task = gr.Dropdown(
                            choices=["transcribe", "translate"],
                            value="transcribe",
                            label="Task",
                        )
                        stt_language = gr.Textbox(
                            value=st.stt.default_language or "auto",
                            label="Language (ISO 639-1 or 'auto')",
                        )
                        stt_format = gr.Dropdown(
                            choices=["text", "json", "verbose_json", "srt", "vtt"],
                            value="text",
                            label="Response format",
                        )
                        stt_temp = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature")
                        stt_stream = gr.Checkbox(value=True, label="Stream (SSE)")
                        stt_btn = gr.Button("Transcribe", variant="primary")
                    with gr.Column(scale=1):
                        stt_output = gr.Textbox(label="Output", lines=12, interactive=False)

                stt_btn.click(
                    fn=_transcribe,
                    inputs=[stt_audio, stt_model, stt_task, stt_language, stt_temp, stt_format, stt_stream],
                    outputs=stt_output,
                )

            ##### TTS TAB #####
            with gr.Tab("Text-to-Speech"):
                with gr.Row():
                    with gr.Column(scale=1):
                        tts_text = gr.Textbox(label="Text input", lines=4, placeholder="Enter text to synthesize...")
                        tts_voice = gr.Dropdown(
                            choices=[
                                "af_heart",
                                "af_bella",
                                "af_nicole",
                                "af_sarah",
                                "af_sky",
                                "am_adam",
                                "am_michael",
                                "bf_emma",
                                "bf_isabella",
                                "bm_george",
                                "bm_lewis",
                            ],
                            value=st.tts.default_voice,
                            label="Voice",
                        )
                        tts_speed = gr.Slider(0.5, 2.0, value=st.tts.default_speed, step=0.1, label="Speed")
                        tts_btn = gr.Button("Synthesize", variant="primary")
                    with gr.Column(scale=1):
                        tts_output = gr.Audio(label="Output", type="filepath")

                tts_btn.click(
                    fn=_synthesize,
                    inputs=[tts_text, tts_voice, tts_speed],
                    outputs=tts_output,
                )

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

                dl_btn.click(
                    fn=_download_model,
                    inputs=[dl_model_id, dl_service],
                    outputs=dl_result,
                ).then(fn=_refresh_models, outputs=models_display)

    return app


##### LAUNCHER #####


def launch_background() -> None:
    """Launch Gradio in a background daemon thread."""
    if not st.front.enabled:
        logger.info("gradio UI disabled", step="START")
        return

    def _run() -> None:
        app = create_app()
        app.launch(
            server_name="0.0.0.0",
            server_port=st.front.port,
            share=st.front.share,
            quiet=True,
            show_error=st.system.debug,
        )

    thread = threading.Thread(target=_run, daemon=True, name="gradio-ui")
    thread.start()
    logger.info("gradio UI started", step="START", url=f"http://localhost:{st.front.port}")
    logger.info("gradio UI started", step="START", port=st.front.port)
