# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "huggingface-hub>=0.34",
#     "numpy>=2",
#     "onnx>=1.17",
#     "onnxruntime>=1.22",
#     "onnxruntime-genai>=0.9",
#     "torch==2.8.0",
#     "transformers>=4.46",
# ]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Export NeuTTS Nano (neuphonic, NeuTTS Open License 1.0: free under $5M annual revenue, outputs
included) to the bundle e-voice-tts loads: the backbone as an int4 GroupQueryAttention ONNX graph, its
tokenizer, and NeuCodec's ONNX decoder. Every neuphonic repo is gated: accept the terms with the account
of the token in data/ops/hf. Run: `make neutts`.

The onnxruntime-genai builder ignores the backbone's linear RoPE scaling (factor 32), so the exported
cos/sin caches are recomputed; without it the model emits noise.
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError
from onnx import numpy_helper

BACKBONES = {"es": "neuphonic/neutts-nano-spanish", "en": "neuphonic/neutts-nano"}
DECODER = "neuphonic/neucodec-onnx-decoder"
MODELS = {"es": "neutts-es", "en": "neutts-en"}
ATTEMPTS = 6
SCALING = 32.0
SERVED = ("model.onnx", "model.onnx.data", "tokenizer.json", "decoder.onnx")


class Export:
    def __init__(self, data: Path) -> None:
        self.checkpoints = data / "checkpoints"
        self.exports = data / "exports"

    def fetch(self, repo: str) -> Path:
        for attempt in range(ATTEMPTS):
            try:
                return Path(snapshot_download(repo, local_dir=self.checkpoints / repo.split("/")[1], max_workers=1))
            except GatedRepoError:
                time.sleep(2**attempt)
        raise SystemExit(f"{repo}: access denied; accept its terms with the token's account")

    def rope(self, model: Path) -> None:
        """cos/sin caches recomputed with positions divided by the linear scaling factor."""
        graph = onnx.load(model, load_external_data=True)
        theta, dim = 500_000.0, 64
        for tensor in graph.graph.initializer:
            if tensor.name.endswith(("cos_cache", "sin_cache")):
                positions = numpy_helper.to_array(tensor).shape[0]
                inverse = theta ** (-np.arange(0, dim, 2, dtype=np.float64) / dim)
                angles = np.outer(np.arange(positions) / SCALING, inverse)
                table = np.cos(angles) if tensor.name.endswith("cos_cache") else np.sin(angles)
                tensor.CopyFrom(numpy_helper.from_array(table.astype(numpy_helper.to_array(tensor).dtype), tensor.name))
        onnx.save(graph, model, save_as_external_data=True, location="model.onnx.data")

    def bundle(self, lang: str, decoder: Path) -> Path:
        source = self.fetch(BACKBONES[lang])
        target = self.exports / MODELS[lang]
        shutil.rmtree(target, ignore_errors=True)
        command = [sys.executable, "-m", "onnxruntime_genai.models.builder", "-i", str(source), "-o", str(target)]
        command += ["-p", "int4", "-e", "cpu", "--extra_options", "int4_accuracy_level=4"]
        subprocess.run(command, check=True)
        self.rope(target / "model.onnx")
        shutil.copy2(source / "tokenizer.json", target / "tokenizer.json")
        shutil.copy2(next(decoder.rglob("*.onnx")), target / "decoder.onnx")
        return target

    def manifest(self, lang: str, bundle: Path) -> str:
        root = self.exports.parents[2]
        lines = [f'[[model]]\nid = "{MODELS[lang]}"\nfiles = [']
        for name in SERVED:
            digest = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
            lines.append(f'  {{ url = "file://{(bundle / name).relative_to(root)}", sha256 = "{digest}" }},')
        return "\n".join([*lines, "]"])

    def run(self, langs: list[str]) -> None:
        decoder = self.fetch(DECODER)
        for lang in langs:
            print(self.manifest(lang, self.bundle(lang, decoder)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--lang", choices=sorted(BACKBONES), action="append")
    args = parser.parse_args()
    Export(args.data.resolve()).run(args.lang or sorted(BACKBONES))


if __name__ == "__main__":
    main()
