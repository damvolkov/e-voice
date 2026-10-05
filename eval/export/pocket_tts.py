# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "beartype>=0.19",
#     "einops>=0.8",
#     "huggingface-hub>=0.34",
#     "numpy>=2",
#     "onnx>=1.17",
#     "onnxruntime>=1.22",
#     "onnxscript>=0.3",
#     "pydantic>=2.9",
#     "pyyaml>=6",
#     "requests>=2.32",
#     "safetensors>=0.4",
#     "scipy>=1.14",
#     "sentencepiece>=0.2",
#     "soundfile>=0.12",
#     "torch==2.8.0",
#     "transformers>=4.46",
#     "typer>=0.12",
#     "typing-extensions>=4.12",
# ]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Export Kyutai Pocket TTS checkpoints to the ONNX bundle e-voice-tts loads, plus a golden for its tests.

The gated checkpoint (kyutai/pocket-tts, pinned by commit) is fetched into `--data`, never the default
Hugging Face cache: run with HF_HOME pointing inside `data/` (the Makefile does). The graphs come from
lomotron/pocket-tts-onnx-export, pinned by commit and cloned next to the checkpoint; it splits FlowLM into
a stateful backbone and a stateless flow step and makes every streaming state an explicit tensor.
Run: `make export-tts`.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime
import scipy.signal
import sentencepiece
import soundfile
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError

CHECKPOINT = ("kyutai/pocket-tts", "3e82814a68665eec246ff649b14c71331f955c06")
EXPORTER = ("https://github.com/lomotron/pocket-tts-onnx-export", "8f52199533098963fe1cbd90bed35bb20d871b5e")
VARIANTS = ("spanish", "spanish_24l", "english_2026-09")
TEMPLATES = {"english_2026-09": "english_2026-04"}
MODELS = {"spanish": "pocket-es", "spanish_24l": "pocket-es-24l", "english_2026-09": "pocket-en"}
SERVED = (
    "bundle.json",
    "tokenizer.model",
    "bos_before_voice.npy",
    "text_conditioner.onnx",
    "mimi_encoder.onnx",
    "flow_lm_main.onnx",
    "flow_lm_main_int8.onnx",
    "flow_lm_flow.onnx",
    "flow_lm_flow_int8.onnx",
    "mimi_decoder.onnx",
    "mimi_decoder_int8.onnx",
)
GOLDEN = {
    "spanish": "Hola, soy la voz de e-voice. Esto es una prueba.",
    "spanish_24l": "Hola, soy la voz de e-voice. Esto es una prueba.",
    "english_2026-09": "Hello, I am the voice of e-voice. This is a test.",
}
EOS_THRESHOLD = -4.0
MIN_FRAMES = 6
VOICE_LIMIT = 30
ATTEMPTS = 8
FADE = 120
PAUSE_FRAME = 480
PAUSE_DB = 35.0
FILES = ("model.safetensors", "tokenizer.model", "tokenizer.json")


class Export:
    def __init__(self, data: Path) -> None:
        self.checkpoints = data / "checkpoints" / "pocket-tts"
        self.vendor = data / "vendor" / "pocket-tts-onnx-export"
        self.exports = data / "exports"

    def checkpoint(self, variants: list[str]) -> Path:
        """One file per request: the gated repo answers bursts of parallel requests with 403."""
        repo, revision = CHECKPOINT
        for variant, name in ((variant, name) for variant in variants for name in FILES):
            self.fetch(repo, revision, f"languages/{variant}/{name}")
        return self.checkpoints

    def fetch(self, repo: str, revision: str, filename: str) -> Path:
        for attempt in range(ATTEMPTS):
            try:
                return Path(hf_hub_download(repo, filename, revision=revision, local_dir=self.checkpoints))
            except GatedRepoError:
                time.sleep(2**attempt)
        raise SystemExit(
            f"{repo}/{filename}: access denied {ATTEMPTS} times; accept its terms with the token's account"
        )

    def exporter(self) -> Path:
        url, commit = EXPORTER
        self.vendor.exists() or subprocess.run(["git", "clone", "-q", url, str(self.vendor)], check=True)
        subprocess.run(["git", "-C", str(self.vendor), "checkout", "-q", commit], check=True)
        return self.vendor

    def config(self, variant: str) -> Path:
        """Registers a config resolving to the local checkpoint (the exporter's `--config` flag is broken)."""
        template = TEMPLATES.get(variant, variant)
        source = (self.vendor / "pocket_tts" / "config" / f"{template}.yaml").read_text(encoding="utf-8")
        local = source.replace(f"languages/{template}/", f"languages/{variant}/").replace(
            f"hf://{CHECKPOINT[0]}/", f"{self.checkpoints}/"
        )
        target = self.vendor / "pocket_tts" / "config" / f"pocket-{variant}.yaml"
        target.write_text(local, encoding="utf-8")
        return target

    def bundle(self, variant: str) -> Path:
        config = self.config(variant)
        env = {**os.environ, "PYTHONPATH": str(self.vendor)}
        command = [
            sys.executable,
            "export.py",
            "--language",
            config.stem,
            "--output_dir",
            str(self.exports),
            "--quantize",
        ]
        subprocess.run(command, check=True, cwd=self.vendor, env=env)
        return self.exports / config.stem

    def run(self, variants: list[str], golden: bool) -> None:
        exported = golden or (self.checkpoint(variants), self.exporter())
        for variant in variants:
            bundle = self.exports / f"pocket-{variant}" if exported is True else self.bundle(variant)
            Golden(bundle).write(GOLDEN[variant], self.vendor / "pocket_tts" / "config" / "jos.wav")
            print(self.manifest(variant, bundle))

    def manifest(self, variant: str, bundle: Path) -> str:
        """The `evoice/tts/models.toml` entry: only what the service loads, paths relative to the repository."""
        root = self.exports.parents[2]
        lines = [f'[[model]]\nid = "{MODELS[variant]}"\nfiles = [']
        for name in SERVED:
            path = bundle / name
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f'  {{ url = "file://{path.relative_to(root)}", sha256 = "{digest}" }},')
        return "\n".join([*lines, "]"])


class Golden:
    """Reference streaming loop over the bundle (numpy + onnxruntime): the Rust backend must reproduce it."""

    def __init__(self, bundle: Path) -> None:
        self.bundle = bundle
        self.meta = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 4
        self.graphs = {
            name: onnxruntime.InferenceSession(bundle / f"{name}.onnx", options, providers=["CPUExecutionProvider"])
            for name in ("text_conditioner", "mimi_encoder", "flow_lm_main", "flow_lm_flow", "mimi_decoder")
        }
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(bundle / self.meta["tokenizer_file"]))

    def state(self, manifest: str) -> dict[str, np.ndarray]:
        fill = {
            "zeros": np.zeros,
            "ones": np.ones,
            "empty": np.zeros,
            "nan": lambda shape, dtype: np.full(shape, np.nan, dtype),
        }
        return {
            entry["input_name"]: fill[entry["fill"]](entry["shape"], dtype=np.dtype(entry["dtype"]))
            for entry in self.meta[manifest]
        }

    def main(self, sequence: np.ndarray, text: np.ndarray, state: dict) -> tuple[np.ndarray, float, dict]:
        outputs = self.graphs["flow_lm_main"].run(None, {"sequence": sequence, "text_embeddings": text, **state})
        names = [output.name for output in self.graphs["flow_lm_main"].get_outputs()]
        result = dict(zip(names, outputs, strict=True))
        updated = {entry["input_name"]: result[entry["output_name"]] for entry in self.meta["flow_lm_state_manifest"]}
        return result["conditioning"], float(result["eos_logit"].reshape(-1)[0]), updated

    def pause(self, audio: np.ndarray) -> np.ndarray:
        """Upstream `end_on_pause`: trailing frames 35 dB under the loudest dropped, 20 ms fade, 80 ms of silence."""
        frame, rate = PAUSE_FRAME, self.meta["sample_rate"]
        rms = np.array(
            [np.sqrt(np.mean(np.square(audio[i : i + frame]))) for i in range(0, len(audio), frame)], np.float32
        )
        loud = np.nonzero(rms > rms.max() * np.float32(10 ** (-PAUSE_DB / 20)))[0]
        kept = audio[: min((int(loud[-1]) + 1 if loud.size else 0) * frame, len(audio))].copy()
        fade = min(len(kept), frame)
        kept[len(kept) - fade :] *= 1 - np.arange(fade, dtype=np.float32) / max(fade, 1)
        return np.concatenate([kept, np.zeros(rate * 2 // 25, np.float32)])

    def voice(self, wav: Path) -> tuple[np.ndarray, np.ndarray, dict]:
        audio, rate = soundfile.read(wav, dtype="float32", always_2d=True)
        target = self.meta["sample_rate"]
        common = math.gcd(rate, target)
        audio = scipy.signal.resample_poly(audio.mean(axis=1), target // common, rate // common).astype(np.float32)
        audio = audio[: VOICE_LIMIT * target]
        prompt = self.pause(audio)
        latents = self.graphs["mimi_encoder"].run(None, {"audio": prompt[None, None, :]})[0]
        frames = -(-prompt.shape[0] // self.meta["samples_per_frame"])
        bos = (
            np.load(self.bundle / self.meta["bos_before_voice_file"])
            if self.meta.get("insert_bos_before_voice")
            else None
        )
        latents = latents if bos is None or latents.shape[1] == frames + 1 else np.concatenate([bos, latents], axis=1)
        _, _, state = self.main(
            np.zeros((1, 0, self.meta["latent_dim"]), np.float32), latents, self.state("flow_lm_state_manifest")
        )
        return audio, latents, state

    def speak(self, text: str, state: dict) -> tuple[list[int], np.ndarray]:
        """Greedy (temperature 0): the flow starts from zeros, so the output is deterministic."""
        ids = self.tokenizer.encode(text)
        embeddings = self.graphs["text_conditioner"].run(None, {"token_ids": np.array([ids], np.int64)})[0]
        _, _, state = self.main(np.zeros((1, 0, self.meta["latent_dim"]), np.float32), embeddings, state)
        mimi = self.state("mimi_state_manifest")
        empty = np.zeros((1, 0, self.meta["conditioning_dim"]), np.float32)
        latent = np.full((1, 1, self.meta["latent_dim"]), np.nan, np.float32)
        after = self.meta.get("model_recommended_frames_after_eos") or ((3 if len(text.split()) <= 4 else 1) + 2)
        budget = int(np.ceil((len(ids) / 3 + 2) * self.meta["frame_rate"]))
        pcm, eos_at = [], None
        decoder = self.graphs["mimi_decoder"]
        decoder_names = [output.name for output in decoder.get_outputs()]
        for step in range(budget):
            conditioning, eos, state = self.main(latent, empty, state)
            eos_at = step if eos_at is None and eos > EOS_THRESHOLD and step >= MIN_FRAMES else eos_at
            if eos_at is not None and step >= eos_at + after:
                break
            x = np.zeros((1, self.meta["latent_dim"]), np.float32)
            flow = {"c": conditioning, "s": np.zeros((1, 1), np.float32), "t": np.ones((1, 1), np.float32), "x": x}
            x = x + self.graphs["flow_lm_flow"].run(None, flow)[0]
            latent = x[:, None, :]
            result = dict(zip(decoder_names, decoder.run(None, {"latent": latent, **mimi}), strict=True))
            mimi = {entry["input_name"]: result[entry["output_name"]] for entry in self.meta["mimi_state_manifest"]}
            pcm.append(result["audio_frame"].reshape(-1))
        audio = np.concatenate(pcm)
        audio[:FADE] *= np.arange(FADE, dtype=np.float32) / FADE
        return ids, audio

    def write(self, text: str, wav: Path) -> None:
        audio, latents, state = self.voice(wav)
        ids, pcm = self.speak(text, state)
        target = self.bundle / "golden"
        target.mkdir(exist_ok=True)
        np.savez(target / "golden.npz", ids=np.array(ids, np.int64), pcm=pcm, audio=audio, voice=latents)
        soundfile.write(target / "golden.wav", pcm, self.meta["sample_rate"])
        (target / "golden.txt").write_text(text, encoding="utf-8")
        frames, seconds = pcm.shape[0] // self.meta["samples_per_frame"], pcm.shape[0] / self.meta["sample_rate"]
        print(f"golden {self.bundle.name}: {len(ids)} tokens, {frames} frames, {seconds:.2f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, action="append")
    parser.add_argument("--golden", action="store_true", help="only regenerate goldens of exported bundles")
    args = parser.parse_args()
    Export(args.data.resolve()).run(args.variant or list(VARIANTS), args.golden)


if __name__ == "__main__":
    main()
