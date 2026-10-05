# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "librosa>=0.10",
#     "numpy>=2",
#     "onnxruntime>=1.22",
#     "soundfile>=0.12",
#     "tokenizers>=0.21",
#     "torch==2.8.0",
# ]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Golden of the Qwen3-TTS preprocessing the Rust backend must reproduce: the official speaker-encoder
mel recipe (QwenLM/Qwen3-TTS@022e286, `mel_spectrogram`), the x-vector, the speech-tokenizer codes, the
prompt token ids and the text projection. Reads an installed model directory (`e-voice-tts pull
qwen3-tts`); writes `golden.npz` beside nothing else. Run: `make qwen3`.
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import onnxruntime
import soundfile
import torch
from tokenizers import Tokenizer

RATE = 24_000
TEXT = "Hola, esta es una prueba de la voz."
REF = "Esto es lo que se dice en la referencia."
ENCODER_SAMPLES = 240_000


class Golden:
    def __init__(self, model: Path) -> None:
        self.model = model
        self.tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))

    def mels(self, audio: np.ndarray) -> np.ndarray:
        """Official `mel_spectrogram(n_fft=1024, num_mels=128, hop=256, win=1024, fmin=0, fmax=12000)`."""
        mel = torch.from_numpy(librosa.filters.mel(sr=RATE, n_fft=1024, n_mels=128, fmin=0, fmax=12000)).float()
        y = torch.nn.functional.pad(torch.from_numpy(audio)[None, None, :], (384, 384), mode="reflect")[:, 0]
        spec = torch.stft(
            y,
            1024,
            hop_length=256,
            win_length=1024,
            window=torch.hann_window(1024),
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        return np.ascontiguousarray(torch.log(torch.clamp(torch.matmul(mel, spec), min=1e-5))[0].T.numpy())

    def ids(self, template: str) -> np.ndarray:
        return np.array(self.tokenizer.encode(template, add_special_tokens=False).ids, np.int64)

    def projection(self, ids: np.ndarray) -> np.ndarray:
        load = lambda name: np.load(self.model / f"{name}.npy")  # noqa: E731
        hidden = load("text_embedding")[ids] @ load("text_projection_fc1_weight").T + load("text_projection_fc1_bias")
        hidden = hidden / (1 + np.exp(-hidden))
        return hidden @ load("text_projection_fc2_weight").T + load("text_projection_fc2_bias")

    def write(self, clip: Path, out: Path) -> None:
        audio, rate = soundfile.read(clip, dtype="float32", always_2d=True)
        audio = librosa.resample(audio.mean(axis=1), orig_sr=rate, target_sr=RATE).astype(np.float32)
        audio = audio[:ENCODER_SAMPLES]
        mels = self.mels(audio)
        speaker = onnxruntime.InferenceSession(self.model / "speaker_encoder.onnx").run(None, {"mels": mels[None]})[0][
            0
        ]
        padded = np.zeros((1, ENCODER_SAMPLES), np.float32)
        padded[0, : audio.shape[0]] = audio
        codes = onnxruntime.InferenceSession(self.model / "tokenizer_encoder.onnx").run(None, {"waveform": padded})[0][
            0
        ]
        codes = codes[:, : -(-audio.shape[0] // 1920)]
        target = self.ids(f"<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n")
        ref = self.ids(f"<|im_start|>assistant\n{REF}<|im_end|>\n")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            audio=audio,
            mels=mels,
            speaker=speaker,
            codes=codes.astype(np.int64),
            target=target,
            ref=ref,
            projection=self.projection(target[:8]).astype(np.float32),
        )
        print(f"golden: {audio.shape[0] / RATE:.2f} s, mels {mels.shape}, codes {codes.shape}, ids {target.shape[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    Golden(args.model).write(args.clip, args.out)


if __name__ == "__main__":
    main()
