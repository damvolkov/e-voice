"""Qwen3-TTS adapter — implements TTSBackend with GPU-only PyTorch model lifecycle.

Requires optional deps: qwen-tts, faster-qwen3-tts, torch, torchaudio.
Install with: uv add --group qwen qwen-tts faster-qwen3-tts torch torchaudio
"""

import asyncio
import gc
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from e_voice.adapters.base import TTSBackend
from e_voice.core.logger import logger
from e_voice.core.settings import DeviceType
from e_voice.core.settings import settings as st
from e_voice.models.error import BackendCapabilityError
from e_voice.models.tts import AudioChunk, SynthesisParams, TTSModelSpec, VoiceEntry, parse_voice_filename

_SENTINEL = object()

##### LANGUAGE MAPPING #####

_BCP47_TO_QWEN: dict[str, str] = {
    "en-us": "English",
    "en-gb": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt-br": "Portuguese",
    "zh": "Chinese",
    "ru": "Russian",
}


def _resolve_language(bcp47: str) -> str:
    """Map BCP-47 lang code to Qwen3 language name. Defaults to English."""
    return _BCP47_TO_QWEN.get(bcp47, "English")


##### ADAPTER #####


class QwenAdapter(TTSBackend):
    """Manages Qwen3-TTS model lifecycle and synthesis. GPU-only."""

    __slots__ = (
        "_batch_model",
        "_can_clone",
        "_config",
        "_loaded_spec",
        "_preset_voices",
        "_stream_model",
        "_voice_entries",
        "_voices",
    )

    def __init__(self) -> None:
        self._batch_model: object | None = None
        self._stream_model: object | None = None
        self._loaded_spec: TTSModelSpec | None = None
        self._voices: list[str] = []
        self._voice_entries: list[VoiceEntry] = []
        self._preset_voices: list[str] = []
        self._can_clone: bool = False
        self._config = st.tts

    # ── Capabilities ──────────────────────────────────────────────────

    @property
    def supported_devices(self) -> frozenset[DeviceType]:
        return frozenset({DeviceType.GPU})

    # ── Properties ────────────────────────────────────────────────────

    @property
    def voices(self) -> list[str]:
        """Available preset + cloned speaker IDs."""
        return self._voices

    @property
    def voice_entries(self) -> list[VoiceEntry]:
        """Voice metadata with language and clone status."""
        return self._voice_entries

    # ── Model Lifecycle ───────────────────────────────────────────────

    async def load(self, spec: TTSModelSpec | None = None) -> None:
        """Load Qwen3-TTS models (batch + streaming). GPU-only, thread-offloaded."""
        target = spec or TTSModelSpec(model_id="qwen", device=st.tts.device)

        if target.device != DeviceType.GPU:
            raise BackendCapabilityError("Qwen3-TTS requires GPU. CPU inference is not supported.")

        if self._loaded_spec == target:
            return

        model_name = self._config.qwen_model
        dtype_str = self._config.qwen_dtype
        attn = self._config.qwen_attn

        local_path = st.MODELS_PATH / "tts" / "qwen" / "models" / model_name.replace("/", "--")
        model_path = str(local_path) if local_path.exists() else model_name

        logger.info("🔄 MODEL_LOADING", extra={"model": model_path, "device": target.device.value})

        batch_model, stream_model, voices, can_clone = await asyncio.to_thread(
            self._qa_create_models, model_path, dtype_str, attn
        )

        self._batch_model = batch_model
        self._stream_model = stream_model
        self._preset_voices = voices
        self._can_clone = can_clone
        self._loaded_spec = target

        cloned_entries = self._qa_discover_voice_entries()
        cloned_ids = [e.id for e in cloned_entries]
        self._voices = voices + [v for v in cloned_ids if v not in voices]
        self._voice_entries = [VoiceEntry(id=v, language="multilingual") for v in voices] + list(cloned_entries)

        logger.info(
            "✅ MODEL_LOADED",
            extra={
                "model": model_name,
                "preset_voices": len(voices),
                "cloned_voices": len(cloned_entries),
            },
        )

    async def unload(self, spec: TTSModelSpec | None = None) -> bool:
        """Release GPU memory for both batch and streaming models."""
        if self._loaded_spec is None:
            return False

        import torch  # ty: ignore[unresolved-import]

        logger.info("🗑️ MODEL_UNLOADING", extra={"model": self._config.qwen_model})

        self._batch_model = None
        self._stream_model = None
        self._loaded_spec = None
        self._voices = []
        self._voice_entries = []
        self._preset_voices = []
        self._can_clone = False

        gc.collect()
        torch.cuda.empty_cache()

        logger.info("🗑️ MODEL_UNLOADED", extra={"model": self._config.qwen_model})
        return True

    async def is_loaded(self, spec: TTSModelSpec | None = None) -> bool:
        return self._loaded_spec is not None

    def loaded_models(self) -> list[TTSModelSpec]:
        return [self._loaded_spec] if self._loaded_spec else []

    async def download(self, model_id: str) -> Path:
        """Download model from HuggingFace Hub."""
        from huggingface_hub import snapshot_download

        models_dir = st.MODELS_PATH / "tts" / "qwen" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        logger.info("⬇️ MODEL_DOWNLOADING", extra={"model": model_id})
        path = await asyncio.to_thread(
            snapshot_download,
            repo_id=model_id,
            local_dir=str(models_dir / model_id.replace("/", "--")),
        )
        logger.info("✅ MODEL_DOWNLOADED", extra={"model": model_id, "path": path})
        return Path(str(path))

    # ── Batch Synthesis ───────────────────────────────────────────────

    async def synthesize(
        self,
        text: str,
        *,
        params: SynthesisParams | None = None,
    ) -> AudioChunk:
        """Full synthesis. Routes to custom_voice (preset) or voice_clone (cloned)."""
        model = self._qa_resolve_batch()
        p = params or SynthesisParams()
        language = _resolve_language(p.lang)

        clone_prompt = self._qa_load_clone_prompt(p.voice)
        is_preset = p.voice in (self._preset_voices or [])

        if clone_prompt is not None:
            wavs_list, sr = await asyncio.to_thread(
                model.generate_voice_clone,  # ty: ignore[unresolved-attribute]
                text,
                language,
                voice_clone_prompt=clone_prompt,
            )
        elif is_preset:
            wavs_list, sr = await asyncio.to_thread(
                model.generate_custom_voice,  # ty: ignore[unresolved-attribute]
                text,
                p.voice,
                language,
            )
        else:
            raise RuntimeError(
                f"Voice '{p.voice}' not found. Available: {self._voices}. "
                f"Clone a voice first or switch to a backend with preset speakers."
            )

        return self._qa_to_float32(wavs_list[0]), sr

    # ── Streaming Synthesis ───────────────────────────────────────────

    async def synthesize_stream(  # ty: ignore[invalid-method-override]
        self,
        text: str,
        *,
        params: SynthesisParams | None = None,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Yield audio chunks. Routes to custom_voice or voice_clone streaming."""
        model = self._qa_resolve_stream()
        p = params or SynthesisParams()
        language = _resolve_language(p.lang)
        chunk_size = self._config.qwen_chunk_size
        clone_prompt = self._qa_load_clone_prompt(p.voice)
        is_preset = p.voice in (self._preset_voices or [])

        if clone_prompt is None and not is_preset:
            raise RuntimeError(
                f"Voice '{p.voice}' not found. Available: {self._voices}. "
                f"Clone a voice first or switch to a backend with preset speakers."
            )

        queue: asyncio.Queue[AudioChunk | object] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        errors: list[Exception] = []

        def _produce() -> None:
            try:
                if clone_prompt is not None:
                    gen = model.generate_voice_clone_streaming(  # ty: ignore[unresolved-attribute]
                        text=text,
                        language=language,
                        ref_audio="",
                        ref_text="",
                        voice_clone_prompt=clone_prompt,
                        chunk_size=chunk_size,
                    )
                else:
                    gen = model.generate_custom_voice_streaming(  # ty: ignore[unresolved-attribute]
                        text=text,
                        speaker=p.voice,
                        language=language,
                        chunk_size=chunk_size,
                    )
                for audio_chunk, sr, _timing in gen:
                    f32 = self._qa_to_float32(audio_chunk)
                    loop.call_soon_threadsafe(queue.put_nowait, (f32, sr))
            except Exception as exc:
                errors.append(exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        producer = loop.run_in_executor(None, _produce)

        while (item := await queue.get()) is not _SENTINEL:
            yield item  # type: ignore[misc]

        await producer
        if errors:
            raise errors[0]

    # ── Capabilities ──────────────────────────────────────────────────

    @property
    def supports_voice_clone(self) -> bool:
        return self._can_clone

    async def clone_voice(
        self,
        voice_id: str,
        ref_audio: Path,
        ref_text: str,
        *,
        language: str | None = None,
    ) -> str:
        """Clone a voice from reference audio. Persists prompt for reuse."""
        import torch  # ty: ignore[unresolved-import]

        model = self._qa_resolve_batch()

        prompt = await asyncio.to_thread(
            model.create_voice_clone_prompt,  # ty: ignore[unresolved-attribute]
            str(ref_audio),
            ref_text,
        )

        lang_code = language or "en"
        voice_dir = st.MODELS_PATH / "tts" / "qwen" / "voices"
        voice_dir.mkdir(parents=True, exist_ok=True)
        save_path = voice_dir / f"{voice_id}_{lang_code}.pt"
        torch.save(prompt, save_path)

        if voice_id not in self._voices:
            self._voices.append(voice_id)
            self._voice_entries.append(VoiceEntry(id=voice_id, language=lang_code, cloned=True))

        logger.info("🎤 VOICE_CLONED", extra={"voice_id": voice_id, "path": str(save_path)})
        return voice_id

    # ── Private: Model Creation ───────────────────────────────────────

    @staticmethod
    def _qa_create_models(model_name: str, dtype_str: str, attn: str) -> tuple[object, object, list[str], bool]:
        """Create batch (qwen-tts) + streaming (faster-qwen3-tts) model instances."""
        import torch  # ty: ignore[unresolved-import]
        from faster_qwen3_tts import FasterQwen3TTS  # ty: ignore[unresolved-import]
        from qwen_tts import Qwen3TTSModel  # ty: ignore[unresolved-import]

        match dtype_str:
            case "bfloat16":
                dtype = torch.bfloat16
            case "float16":
                dtype = torch.float16
            case _:
                dtype = torch.float32

        batch_model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map="cuda",
            dtype=dtype,
            attn_implementation=attn,
        )

        stream_model = FasterQwen3TTS.from_pretrained(
            model_name,
            device="cuda",
            dtype=dtype_str,
            attn_implementation=attn,
        )

        voices = batch_model.get_supported_speakers() or []

        can_clone = False
        try:
            batch_model.create_voice_clone_prompt("__probe__", "probe")  # ty: ignore[unresolved-attribute]
        except FileNotFoundError:
            can_clone = True
        except ValueError:
            can_clone = False

        return batch_model, stream_model, voices, can_clone

    # ── Private: Resolution ───────────────────────────────────────────

    def _qa_resolve_batch(self) -> object:
        """Resolve loaded batch model (qwen-tts)."""
        if self._batch_model is None:
            raise RuntimeError("Qwen3-TTS batch model not loaded. Call load() first.")
        return self._batch_model

    def _qa_resolve_stream(self) -> object:
        """Resolve loaded streaming model (faster-qwen3-tts)."""
        if self._stream_model is None:
            raise RuntimeError("Qwen3-TTS streaming model not loaded. Call load() first.")
        return self._stream_model

    def _qa_load_clone_prompt(self, voice_id: str) -> object | None:
        """Load a persisted clone prompt by voice ID. Matches {voice_id}_*.pt pattern."""
        import torch  # ty: ignore[unresolved-import]

        voice_dir = st.MODELS_PATH / "tts" / "qwen" / "voices"
        if not voice_dir.exists():
            return None
        matches = list(voice_dir.glob(f"{voice_id}_*.pt"))
        if not matches:
            return None
        return torch.load(matches[0], weights_only=False, map_location="cuda")

    @staticmethod
    def _qa_discover_voice_entries() -> list[VoiceEntry]:
        """Scan voices/ directory for persisted voice prompts. Parses {name}_{lang}.pt convention."""
        voice_dir = st.MODELS_PATH / "tts" / "qwen" / "voices"
        if not voice_dir.exists():
            return []
        return [parse_voice_filename(p.name) for p in sorted(voice_dir.glob("*.pt"))]

    @staticmethod
    def _qa_to_float32(samples: NDArray | object) -> NDArray[np.float32]:
        """Normalize audio samples to float32 [-1, 1]. Handles torch Tensors and numpy arrays."""
        import torch  # ty: ignore[unresolved-import]

        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        arr: NDArray = np.asarray(samples).squeeze()
        if arr.dtype == np.float32:
            return arr
        return arr.astype(np.float32)
