"""Gradio UI for e-voice — STT/TTS playground + backend management."""

import base64
import threading
from collections.abc import Iterable
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
    body_background_fill="#0a0e14",
    body_background_fill_dark="#0a0e14",
    block_background_fill="#111820",
    block_background_fill_dark="#111820",
    block_border_color="#1a3a4a",
    block_border_color_dark="#1a3a4a",
    block_label_text_color="#7dd3d8",
    block_label_text_color_dark="#7dd3d8",
    block_title_text_color="#7dd3d8",
    block_title_text_color_dark="#7dd3d8",
    body_text_color="#c8d6d8",
    body_text_color_dark="#c8d6d8",
    body_text_color_subdued="#6b7f82",
    body_text_color_subdued_dark="#6b7f82",
    button_primary_background_fill="#0891b2",
    button_primary_background_fill_dark="#0891b2",
    button_primary_background_fill_hover="#06b6d4",
    button_primary_text_color="#000",
    button_primary_text_color_dark="#000",
    button_secondary_background_fill="#111820",
    button_secondary_background_fill_dark="#111820",
    button_secondary_border_color="#1a3a4a",
    button_secondary_border_color_dark="#1a3a4a",
    button_secondary_text_color="#7dd3d8",
    button_secondary_text_color_dark="#7dd3d8",
    input_background_fill="#0a0e14",
    input_background_fill_dark="#0a0e14",
    input_border_color="#1a3a4a",
    input_border_color_dark="#1a3a4a",
    input_placeholder_color="#3a5a5e",
    input_placeholder_color_dark="#3a5a5e",
    border_color_accent="#0891b2",
    border_color_accent_dark="#0891b2",
    color_accent_soft="#0891b220",
    color_accent_soft_dark="#0891b220",
    slider_color="#0891b2",
    slider_color_dark="#0891b2",
)

_CSS_PATH = st.STYLES_PATH / "front.css"
_CSS = _CSS_PATH.read_text() if _CSS_PATH.exists() else ""

_LOGO_HEIGHT = 160
_SPARK_W = 80
_SPARK_H = 20
_DEVICE_COLORS: dict[str, str] = {"gpu": "#66bb6a", "cpu": "#2196f3", "transitioning": "#ffb74d"}


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


##### VOICE HELPERS #####

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


def _fetch_voice_choices() -> list[tuple[str, str]]:
    """Fetch voices grouped by language. Returns (label, value) tuples for Dropdown."""
    voices = _api().get_voices()
    if not voices:
        return [("af_heart", "af_heart")]

    grouped: dict[str, list[str]] = {}
    for v in sorted(voices, key=lambda v: (v.get("language", ""), v["id"])):
        grouped.setdefault(v.get("language", "unknown"), []).append(v["id"])

    choices: list[tuple[str, str]] = []
    for lang in sorted(grouped):
        label = _LANG_LABELS.get(lang, lang)
        first = grouped[lang][0]
        choices.append((f"── {label} ──", first))
        choices.extend((vid, vid) for vid in grouped[lang])
    return choices


##### BACKEND TAB — STT SECTION #####


def _build_stt_backend_md() -> str:
    """Build markdown for STT backend section: active backend + downloaded models."""
    api = _api()
    backends = api.get_backends()
    active = backends.get("stt", {}).get("active", "unknown")

    data = api.get_downloaded_models()
    stt_models = data.get("stt", [])

    lines = [f"**Active backend:** `{active}`"]
    lines.append("\n**Downloaded models:**")
    if stt_models:
        for m in stt_models:
            lines.append(f"- `{m['id']}` ({m['size_mb']} MB)")
    else:
        lines.append("- (none)")
    return "\n".join(lines)


def _download_stt_model(model_id: str) -> tuple[str, str]:
    """Download STT model, return (result_text, refreshed_md)."""
    result = _api().download_model(model_id, "stt")
    return result, _build_stt_backend_md()


##### BACKEND TAB — TTS SECTION #####


def _build_tts_backend_md() -> str:
    """Build markdown for TTS backend section: active backend + models + voices."""
    api = _api()
    backends = api.get_backends()
    active = backends.get("tts", {}).get("active", "unknown")

    data = api.get_downloaded_models()
    tts_models = data.get("tts", [])

    lines = [f"**Active backend:** `{active}`"]
    lines.append("\n**Downloaded models:**")
    if tts_models:
        for m in tts_models:
            lines.append(f"- `{m['id']}` ({m['size_mb']} MB)")
    else:
        lines.append("- (none)")

    voices = api.get_voices()
    lines.append("\n**Voices:**")
    if voices:
        grouped: dict[str, list[str]] = {}
        for v in voices:
            grouped.setdefault(v.get("language", "unknown"), []).append(v["id"])
        for lang in sorted(grouped):
            label = _LANG_LABELS.get(lang, lang)
            names = ", ".join(f"`{n}`" for n in sorted(grouped[lang]))
            lines.append(f"{label} — {names}")
    else:
        lines.append("- No voices available — TTS model not loaded.")

    return "\n".join(lines)


def _download_tts_model(model_id: str) -> tuple[str, str]:
    """Download TTS model, return (result_text, refreshed_md)."""
    result = _api().download_model(model_id, "tts")
    return result, _build_tts_backend_md()


##### MONITOR — SVG SPARKLINES #####


def _bar_color(pct: float) -> str:
    if pct < 60:
        return "#66bb6a"
    return "#ffb74d" if pct < 85 else "#ef5350"


def _svg_sparkline(history: Iterable[float], color: str) -> str:
    """Generate an inline SVG sparkline with area fill from history data."""
    data = list(history)
    if len(data) < 2:
        return f'<svg width="{_SPARK_W}" height="{_SPARK_H}"></svg>'
    step = _SPARK_W / (len(data) - 1)
    points = [f"{i * step:.1f},{_SPARK_H - (v / 100 * _SPARK_H):.1f}" for i, v in enumerate(data)]
    pts = " ".join(points)
    area_pts = f"0,{_SPARK_H} {pts} {_SPARK_W},{_SPARK_H}"
    return (
        f'<svg width="{_SPARK_W}" height="{_SPARK_H}" style="flex-shrink:0">'
        f'<polygon points="{area_pts}" fill="{color}20" />'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.2" />'
        f"</svg>"
    )


##### MONITOR — HTML BUILDER #####


def _service_semaphore(label: str, info: dict) -> str:
    """Build a single service semaphore: colored dot + label + device name."""
    state = info.get("state", "unknown")
    device = info.get("device", "unknown")
    sem_color = _DEVICE_COLORS.get(state, "#666")
    pulse = "animation:pulse 1.4s ease-in-out infinite;" if state == "transitioning" else ""
    return (
        f'<div style="display:flex;align-items:center;gap:6px">'
        f'<div style="width:8px;height:8px;border-radius:50%;background:{sem_color};{pulse}"></div>'
        f'<span style="font-size:11px;color:#6b7f82;font-weight:600">{label}</span>'
        f'<span style="font-size:11px;font-weight:600;color:#c8d6d8">{device.upper()}</span>'
        f"</div>"
    )


def _build_monitor_html() -> str:
    """Build full monitor card: per-service semaphores + metric bars + sparklines."""
    api = _api()
    device_info = api.get_device()
    stt_info = device_info.get("stt", {})
    tts_info = device_info.get("tts", {})

    m = api.get_monitor()
    gpu_ok = m.get("gpu_available", False)
    history = m.get("history", {})

    metrics = [
        (
            "VRAM",
            m.get("vram_pct", 0) if gpu_ok else 0,
            f"{m.get('vram_used_mb', 0) // 1024}/{m.get('vram_total_mb', 0) // 1024}G" if gpu_ok else "n/a",
            history.get("vram", []),
        ),
        (
            "GPU",
            m.get("gpu_util_pct", 0) if gpu_ok else 0,
            f"{m.get('gpu_util_pct', 0):.0f}%" if gpu_ok else "n/a",
            history.get("gpu_util", []),
        ),
        ("CPU", m.get("cpu_pct", 0), f"{m.get('cpu_pct', 0):.0f}%", history.get("cpu", [])),
        (
            "RAM",
            m.get("ram_pct", 0),
            f"{int(m.get('ram_used_gb', 0))}/{int(m.get('ram_total_gb', 0))}G",
            history.get("ram", []),
        ),
    ]

    rows: list[str] = []
    for label, pct, detail, hist in metrics:
        bcolor = _bar_color(pct)
        sparkline = _svg_sparkline(hist, bcolor)
        rows.append(
            f'<div style="display:flex;align-items:center;gap:8px;height:22px">'
            f'<span style="font-size:11px;color:#6b7f82;width:38px;font-weight:600">{label}</span>'
            f'<span style="font-size:11px;color:#c8d6d8;width:58px;text-align:right;'
            f'font-variant-numeric:tabular-nums">{detail}</span>'
            f'<div style="height:4px;flex:1;min-width:60px;border-radius:2px;background:#1a2332;overflow:hidden">'
            f'<div style="height:100%;width:{pct:.0f}%;border-radius:2px;background:{bcolor};'
            f'transition:width .3s ease"></div></div>'
            f'<div style="margin-left:8px;flex-shrink:0">{sparkline}</div>'
            f"</div>"
        )

    semaphores = (
        f'<div style="display:flex;align-items:center;gap:16px;margin-bottom:8px">'
        f"{_service_semaphore('STT', stt_info)}"
        f"{_service_semaphore('TTS', tts_info)}"
        f"</div>"
    )
    return semaphores + "\n".join(rows)


##### MONITOR — CALLBACKS #####


def _toggle_stt_device() -> tuple[str, str, str]:
    """Toggle STT GPU↔CPU. Returns monitor HTML + STT button label + TTS button label."""
    device_info = _api().get_device()
    current = device_info.get("stt", {}).get("device", "gpu")
    target = "cpu" if current == "gpu" else "gpu"
    _api().switch_device(target, service="stt")
    html = _build_monitor_html()
    new_info = _api().get_device()
    stt_target = "cpu" if new_info.get("stt", {}).get("device") == "gpu" else "gpu"
    tts_target = "cpu" if new_info.get("tts", {}).get("device") == "gpu" else "gpu"
    return html, f"STT → {stt_target.upper()}", f"TTS → {tts_target.upper()}"


def _toggle_tts_device() -> tuple[str, str, str]:
    """Toggle TTS GPU↔CPU. Returns monitor HTML + STT button label + TTS button label."""
    device_info = _api().get_device()
    current = device_info.get("tts", {}).get("device", "gpu")
    target = "cpu" if current == "gpu" else "gpu"
    _api().switch_device(target, service="tts")
    html = _build_monitor_html()
    new_info = _api().get_device()
    stt_target = "cpu" if new_info.get("stt", {}).get("device") == "gpu" else "gpu"
    tts_target = "cpu" if new_info.get("tts", {}).get("device") == "gpu" else "gpu"
    return html, f"STT → {stt_target.upper()}", f"TTS → {tts_target.upper()}"


def _refresh_monitor() -> tuple[str, str, str]:
    """Timer tick — refresh monitor HTML + both button labels."""
    html = _build_monitor_html()
    device_info = _api().get_device()
    stt_target = "cpu" if device_info.get("stt", {}).get("device") == "gpu" else "gpu"
    tts_target = "cpu" if device_info.get("tts", {}).get("device") == "gpu" else "gpu"
    return html, f"STT → {stt_target.upper()}", f"TTS → {tts_target.upper()}"


##### LOGO #####


def _load_logo_html() -> str:
    logo_path = st.BASE_DIR / "assets" / "e-voice-landscape-front.svg"
    if not logo_path.exists():
        return ""
    b64 = base64.b64encode(logo_path.read_bytes()).decode()
    return (
        f'<div style="display:flex;align-items:center;justify-content:center;height:100%">'
        f'<img src="data:image/svg+xml;base64,{b64}" style="height:{_LOGO_HEIGHT}px" />'
        f"</div>"
    )


##### APP FACTORY #####


def create_app() -> gr.Blocks:
    """Build the Gradio Blocks UI."""
    api = _api()
    device_info = api.get_device()
    stt_target = "cpu" if device_info.get("stt", {}).get("device") == "gpu" else "gpu"
    tts_target = "cpu" if device_info.get("tts", {}).get("device") == "gpu" else "gpu"

    with gr.Blocks(title="e-voice") as app:
        ##### HEADER — logo left, monitor right #####
        with gr.Row(elem_id="header-row"):
            with gr.Column(scale=1):
                gr.HTML(_load_logo_html())
            with gr.Column(min_width=400, scale=0), gr.Group(elem_id="monitor-card"), gr.Row():
                monitor_html = gr.HTML(value=_build_monitor_html)
                with gr.Column(min_width=100, scale=0):
                    stt_switch_btn = gr.Button(
                        f"STT → {stt_target.upper()}",
                        size="sm",
                        variant="secondary",
                        elem_id="stt-switch-btn",
                    )
                    tts_switch_btn = gr.Button(
                        f"TTS → {tts_target.upper()}",
                        size="sm",
                        variant="secondary",
                        elem_id="tts-switch-btn",
                    )

        all_outputs = [monitor_html, stt_switch_btn, tts_switch_btn]
        stt_switch_btn.click(fn=_toggle_stt_device, outputs=all_outputs)
        tts_switch_btn.click(fn=_toggle_tts_device, outputs=all_outputs)

        mon_timer = gr.Timer(value=3)
        mon_timer.tick(fn=_refresh_monitor, outputs=all_outputs)

        ##### TABS #####
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
                        voice_values = [v for _, v in voice_choices]
                        default_voice = (
                            st.tts.default_voice if st.tts.default_voice in voice_values else voice_values[0]
                        )
                        tts_voice = gr.Dropdown(
                            choices=voice_choices,
                            value=default_voice,
                            label="Voice",
                        )
                        tts_speed = gr.Slider(0.5, 2.0, value=st.tts.default_speed, step=0.1, label="Speed")
                        tts_btn = gr.Button("Synthesize", variant="primary")
                    with gr.Column(scale=1):
                        tts_output = gr.Audio(label="Output", type="filepath")

                tts_btn.click(fn=_synthesize, inputs=[tts_text, tts_voice, tts_speed], outputs=tts_output)

            ##### BACKEND TAB #####
            with gr.Tab("Backend"):
                with gr.Accordion("Speech-to-Text (STT)", open=True):
                    stt_backend_md = gr.Markdown(value=_build_stt_backend_md)
                    stt_refresh_btn = gr.Button("Refresh", variant="secondary", size="sm")
                    stt_refresh_btn.click(fn=_build_stt_backend_md, outputs=stt_backend_md)

                    gr.Markdown("---")
                    with gr.Row():
                        stt_dl_model_id = gr.Textbox(
                            label="Model ID",
                            placeholder="mobiuslabsgmbh/faster-whisper-large-v3-turbo",
                            scale=3,
                        )
                        stt_dl_btn = gr.Button("Download", variant="primary", scale=1)
                    stt_dl_result = gr.Textbox(label="Result", interactive=False)

                    stt_dl_btn.click(
                        fn=_download_stt_model,
                        inputs=[stt_dl_model_id],
                        outputs=[stt_dl_result, stt_backend_md],
                    )

                with gr.Accordion("Text-to-Speech (TTS)", open=True):
                    tts_backend_md = gr.Markdown(value=_build_tts_backend_md)
                    tts_refresh_btn = gr.Button("Refresh", variant="secondary", size="sm")
                    tts_refresh_btn.click(fn=_build_tts_backend_md, outputs=tts_backend_md)

                    gr.Markdown("---")
                    with gr.Row():
                        tts_dl_model_id = gr.Textbox(
                            label="Model ID",
                            placeholder="kokoro",
                            scale=3,
                        )
                        tts_dl_btn = gr.Button("Download", variant="primary", scale=1)
                    tts_dl_result = gr.Textbox(label="Result", interactive=False)

                    tts_dl_btn.click(
                        fn=_download_tts_model,
                        inputs=[tts_dl_model_id],
                        outputs=[tts_dl_result, tts_backend_md],
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
