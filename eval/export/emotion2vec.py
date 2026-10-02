# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "funasr==1.2.7",
#     "huggingface-hub>=0.34",
#     "numpy<2",
#     "onnx>=1.17",
#     "onnxruntime>=1.22",
#     "torch==2.8.0",
#     "torchaudio==2.8.0",
# ]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchaudio = { index = "pytorch-cpu" }
# ///
"""Export an emotion2vec+ classifier to the layout e-voice loads: ONNX backbone + linear head JSON.

FunASR's exporter traces the backbone only (raw 16 kHz waveform -> frame features, normalization
folded in); the `proj` head is read from the checkpoint. Same recipe as the published base export
(github.com/ame700/emotion2vec, scripts/onnx). Run: `make export MODEL=emotion2vec-plus-large`.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from funasr import AutoModel
from huggingface_hub import snapshot_download

REVISIONS = {
    "emotion2vec/emotion2vec_plus_large": "6c303ba987b86b93193de93e34bb2b077a6bedc4",
}


class Export:
    def __init__(self, repo: str, out: Path) -> None:
        self.repo, self.out = repo, out
        self.name = repo.rsplit("/", maxsplit=1)[-1]

    def checkpoint(self) -> Path:
        return Path(snapshot_download(self.repo, revision=REVISIONS[self.repo]))

    def backbone(self, source: Path) -> Path:
        model = AutoModel(model=str(source), disable_update=True, device="cpu")
        model.export(type="onnx", quantize=False, opset_version=13)
        exported = next(path for path in source.iterdir() if path.name == "emotion2vec")
        target = self.out / f"{self.name}.onnx"
        shutil.move(exported, target)
        return target

    def head(self, source: Path) -> Path:
        state = torch.load(source / "model.pt", map_location="cpu")
        weights = state.get("model", state)
        tokens = [
            line.strip() for line in (source / "tokens.txt").read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        labels = ["unknown" if token == "<unk>" else token.split("/")[-1].strip().lower() for token in tokens]
        head = {"labels": labels, "weight": weights["proj.weight"].tolist(), "bias": weights["proj.bias"].tolist()}
        target = self.out / "emotion2vec_head.json"
        target.write_text(json.dumps(head), encoding="utf-8")
        return target

    def check(self, backbone: Path) -> tuple[int, ...]:
        session = onnxruntime.InferenceSession(backbone, providers=["CPUExecutionProvider"])
        features = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 16_000), dtype=np.float32)})[0]
        return tuple(features.shape)

    def run(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        source = self.checkpoint()
        backbone, head = self.backbone(source), self.head(source)
        print(f"backbone output for 1 s: {self.check(backbone)}")
        for path in (backbone, head):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            print(f'{{ url = "file://{path}", sha256 = "{digest}" }}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", choices=sorted(REVISIONS), default="emotion2vec/emotion2vec_plus_large")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    Export(args.repo, args.out).run()


if __name__ == "__main__":
    main()
