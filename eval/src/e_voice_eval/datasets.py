"""Pinned evaluation sets fetched as 16-bit WAV files plus a JSON-lines manifest."""

import array
import io
import json
import wave
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import structlog

log = structlog.get_logger()

FLEURS = "https://huggingface.co/datasets/google/fleurs/resolve/168de341b3db6859a9bac1c50a2ef5e3b47647e0"
MESD = "https://huggingface.co/datasets/somosnlp-hackathon-2022/MESD/resolve/24d41c732de80b4b883f8e279d484a6d4b5eb017"
CREMA = "https://huggingface.co/datasets/confit/cremad-parquet/resolve/5a54051429170cc5b484cd7f715dfcb5fdf12195"

MESD_LABELS = {
    "Anger": "angry",
    "Disgust": "disgusted",
    "Fear": "fearful",
    "Happiness": "happy",
    "Neutral": "neutral",
    "Sadness": "sad",
}
CREMA_LABELS = {
    "anger": "angry",
    "disgust": "disgusted",
    "fear": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
}


@dataclass(frozen=True, slots=True)
class Dataset:
    """One pinned split: where it lives, its language and how a row becomes audio and references."""

    name: str
    url: str
    lang: str
    kind: str

    def _fetch_rows(self, limit: int | None) -> pl.DataFrame:
        frame = pl.scan_parquet(self.url)
        return (frame.head(limit) if limit else frame).collect()

    def _fetch_pcm(self, row: dict) -> bytes:
        match self.kind:
            case "fleurs" | "crema":
                return row["audio"]["bytes"]
            case "mesd":
                samples = array.array("h", (round(max(-1.0, min(1.0, x)) * 32_767) for x in row["audio_array"]))
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as out:
                    out.setnchannels(1)
                    out.setsampwidth(2)
                    out.setframerate(16_000)
                    out.writeframes(samples.tobytes())
                return buffer.getvalue()
            case other:
                raise ValueError(f"unknown dataset kind {other!r}")

    def _fetch_refs(self, row: dict) -> dict:
        match self.kind:
            case "fleurs":
                return {"text": row["raw_transcription"]}
            case "mesd":
                return {"text": row["word"], "emotion": MESD_LABELS[row["emotion"]]}
            case "crema":
                return {"emotion": CREMA_LABELS[row["emotion"]]}
            case other:
                raise ValueError(f"unknown dataset kind {other!r}")

    ############################################################

    def fetch(self, root: Path, limit: int | None = None) -> Path:
        """Writes `<root>/<name>/audio/*.wav` and `manifest.jsonl`; returns the manifest path."""
        target = root / self.name
        (target / "audio").mkdir(parents=True, exist_ok=True)
        rows = self._fetch_rows(limit)
        manifest = target / "manifest.jsonl"
        with manifest.open("w", encoding="utf-8") as out:
            for index, row in enumerate(rows.iter_rows(named=True)):
                audio = Path("audio") / f"{index:05d}.wav"
                (target / audio).write_bytes(self._fetch_pcm(row))
                item = {
                    "id": f"{self.name}-{index:05d}",
                    "audio": str(audio),
                    "lang": self.lang,
                    **self._fetch_refs(row),
                }
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
        log.info("dataset.fetched", name=self.name, items=rows.height, manifest=str(manifest))
        return manifest


DATASETS = {
    dataset.name: dataset
    for dataset in (
        Dataset("fleurs-es", f"{FLEURS}/es_419/test/0000.parquet", "es", "fleurs"),
        Dataset("fleurs-en", f"{FLEURS}/en_us/test/0000.parquet", "en", "fleurs"),
        Dataset("mesd", f"{MESD}/data/test-00000-of-00001.parquet", "es", "mesd"),
        Dataset("crema", f"{CREMA}/data/test-00000-of-00001.parquet", "en", "crema"),
    )
}
