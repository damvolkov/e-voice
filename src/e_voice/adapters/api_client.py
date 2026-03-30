"""HTTP/WS client adapter for e-voice API — used by Gradio UI and CLI."""

import base64
from collections.abc import Generator
from contextlib import suppress
from math import gcd
from pathlib import Path

import httpx
import numpy as np
import orjson
from scipy.signal import resample_poly
from websockets.exceptions import WebSocketException
from websockets.sync.client import connect as ws_connect

from e_voice.core.settings import settings as st

_SSE_DONE = "[DONE]"


class APIClient:
    """Sync HTTP client for e-voice REST endpoints."""

    __slots__ = ("_base_url", "_timeout")

    def __init__(self, base_url: str, timeout: float = 180.0) -> None:
        self._base_url = base_url
        self._timeout = httpx.Timeout(timeout)

    def _http(self) -> httpx.Client:
        return httpx.Client(base_url=self._base_url, timeout=self._timeout)

    def _build_stt_form(
        self,
        *,
        model: str | None,
        language: str | None,
        response_format: str,
        temperature: float,
        stream: bool,
    ) -> dict[str, str]:
        """Build multipart form data for STT endpoints."""
        form: dict[str, str] = {"temperature": str(temperature)}
        if model:
            form["model"] = model
        if language and language != "auto":
            form["language"] = language
        form["response_format"] = "text" if stream else response_format
        if stream:
            form["stream"] = "true"
        return form

    @staticmethod
    def _stt_endpoint(task: str) -> str:
        return "/v1/audio/translations" if task == "translate" else "/v1/audio/transcriptions"

    @staticmethod
    def _read_file(audio_path: Path) -> tuple[str, bytes]:
        p = Path(audio_path)
        return p.name, p.read_bytes()

    ##### STT #####

    def create_transcription(
        self,
        audio_path: Path,
        *,
        model: str | None = None,
        task: str = "transcribe",
        language: str | None = None,
        response_format: str = "text",
        temperature: float = 0.0,
    ) -> str:
        """Transcribe/translate audio file via single HTTP POST. Returns full text."""
        endpoint = self._stt_endpoint(task)
        file_name, file_content = self._read_file(audio_path)
        form = self._build_stt_form(
            model=model,
            language=language,
            response_format=response_format,
            temperature=temperature,
            stream=False,
        )

        with self._http() as c:
            resp = c.post(endpoint, files={"file": (file_name, file_content)}, data=form)
            resp.raise_for_status()
            return resp.text

    def create_transcription_stream(
        self,
        audio_path: Path,
        *,
        model: str | None = None,
        task: str = "transcribe",
        language: str | None = None,
        temperature: float = 0.0,
    ) -> Generator[str, None, None]:
        """Transcribe/translate audio file via SSE streaming. Yields accumulated text."""
        endpoint = self._stt_endpoint(task)
        file_name, file_content = self._read_file(audio_path)
        form = self._build_stt_form(
            model=model,
            language=language,
            response_format="text",
            temperature=temperature,
            stream=True,
        )

        accumulated = ""
        with (
            self._http() as c,
            c.stream(
                "POST",
                endpoint,
                files={"file": (file_name, file_content)},
                data=form,
                headers={"Accept": "text/event-stream", "Connection": "close"},
            ) as resp,
        ):
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == _SSE_DONE:
                    return
                accumulated += data
                yield accumulated

    ##### TTS #####

    def create_synthesis(self, text: str, *, voice: str = "af_heart", speed: float = 1.0) -> bytes:
        """Synthesize speech via HTTP POST. Returns WAV bytes."""
        with self._http() as c:
            resp = c.post(
                "/v1/audio/speech",
                json={"input": text, "voice": voice, "speed": speed, "response_format": "wav", "stream": False},
            )
            resp.raise_for_status()
            return resp.content

    ##### VOICES #####

    def get_voices(self, lang: str | None = None) -> list[dict]:
        """Return available TTS voices, optionally filtered by BCP-47 language code."""
        try:
            with self._http() as c:
                params = {"lang": lang} if lang else {}
                resp = c.get("/v1/audio/voices", params=params)
                resp.raise_for_status()
                return resp.json().get("voices", [])
        except Exception:
            return []

    ##### MODELS #####

    def get_models(self) -> list[str]:
        """Return loaded model IDs."""
        try:
            with self._http() as c:
                resp = c.get("/v1/models")
                resp.raise_for_status()
                return [m["id"] for m in resp.json().get("data", [])]
        except Exception:
            return []

    def get_downloaded_models(self) -> dict:
        """Return all downloaded models from disk."""
        try:
            with self._http() as c:
                resp = c.get("/v1/models/list")
                resp.raise_for_status()
                return resp.json()
        except Exception:
            return {"stt": [], "tts": []}

    def download_model(self, model_id: str, service: str) -> str:
        """Download a model. Returns status string."""
        if not model_id.strip():
            return "No model ID provided"
        try:
            with self._http() as c:
                resp = c.post(
                    "/v1/models/download",
                    json={"model": model_id, "service": service},
                    timeout=600.0,
                )
                resp.raise_for_status()
                data = resp.json()
                return f"Downloaded: {data['model']} → {data['path']}"
        except Exception as exc:
            return f"Error: {exc}"

    def get_monitor(self) -> dict:
        """Poll system metrics snapshot with sparkline history."""
        try:
            with self._http() as c:
                resp = c.get("/v1/system/monitor")
                resp.raise_for_status()
                return resp.json()
        except Exception:
            return {
                "cpu_pct": 0,
                "ram_used_gb": 0,
                "ram_total_gb": 0,
                "ram_pct": 0,
                "gpu_util_pct": 0,
                "vram_used_mb": 0,
                "vram_total_mb": 0,
                "vram_pct": 0,
                "gpu_available": False,
                "history": {"cpu": [], "ram": [], "gpu_util": [], "vram": []},
            }

    def get_device(self) -> dict:
        """Return current device state."""
        try:
            with self._http() as c:
                resp = c.get("/v1/system/device")
                resp.raise_for_status()
                return resp.json()
        except Exception:
            return {"device": "unknown", "state": "unknown", "transitioning": False}

    def switch_device(self, device: str) -> dict:
        """Switch to target device (gpu/cpu). Returns result dict."""
        try:
            with self._http() as c:
                resp = c.post("/v1/system/device", json={"device": device}, timeout=120.0)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            return {"success": False, "device": device, "message": str(exc)}


##### LIVE STREAM (WebSocket STT) #####


def _resample_16k(y: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16kHz using polyphase filter."""
    if sr == st.stt.sample_rate:
        return y
    g = gcd(st.stt.sample_rate, sr)
    return resample_poly(y, st.stt.sample_rate // g, sr // g).astype(np.float32)


def _float32_to_b64_pcm16(y: np.ndarray) -> str:
    """Convert float32 audio to base64-encoded PCM16-LE."""
    pcm16 = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
    return base64.b64encode(pcm16.tobytes()).decode()


def create_stream(base_url: str, language: str) -> dict | None:
    """Open WebSocket STT session. Returns state dict or None on failure."""
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    lang = language if language and language != "auto" else ""
    params = "response_format=json"
    if lang:
        params = f"language={lang}&{params}"
    url = f"{ws_url}/v1/audio/transcriptions?{params}"

    try:
        ws = ws_connect(url)
        with suppress(TimeoutError):
            ws.recv(timeout=0.3)
        return {"ws": ws, "text": "", "partial": ""}
    except Exception:
        return None


def send_stream_chunk(state: dict | None, audio_chunk: tuple[int, np.ndarray] | None) -> tuple[dict | None, str]:
    """Send audio chunk over WS, return (state, display_text)."""
    if state is None or audio_chunk is None:
        return state, state.get("text", "") if state else ""

    sr, y = audio_chunk
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if (peak := np.max(np.abs(y))) > 0:
        y /= peak

    y = _resample_16k(y, sr)
    b64 = _float32_to_b64_pcm16(y)

    ws = state.get("ws")
    if ws is None:
        return None, state.get("text", "")

    try:
        ws.send(b64)
        response = ws.recv(timeout=5.0)

        if response and response.strip():
            with suppress(orjson.JSONDecodeError):
                obj = orjson.loads(response)
                if obj.get("text"):
                    state["text"] = obj["text"]
                state["partial"] = obj.get("partial", "")
    except (TimeoutError, WebSocketException):
        return remove_stream(state)

    display = state["text"]
    if state.get("partial"):
        display += f"\n···{state['partial']}"
    return state, display


def remove_stream(state: dict | None) -> tuple[None, str]:
    """Close WebSocket session. Returns (None, final_text)."""
    if state is None:
        return None, ""
    final_text = state.get("text", "")
    if (ws := state.get("ws")) is not None:
        with suppress(Exception):
            ws.close()
    return None, final_text
