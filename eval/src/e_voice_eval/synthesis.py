"""One table row per TTS bench results file: latency, cost, speaker similarity and intelligibility."""

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from e_voice_eval.scoring import Scoring

ROUNDTRIP = ".roundtrip.jsonl"


@dataclass(frozen=True, slots=True)
class Synthesis:
    """Reads `e-voice-tts bench` files; intelligibility comes from `<run>.roundtrip.jsonl`, the STT bench
    of the run's own audio (WER of what an ASR hears against the text that was spoken)."""

    files: tuple[Path, ...]

    def _rows_load(self, path: Path) -> tuple[pl.DataFrame, dict]:
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        summary = next((line for line in lines if line.get("kind") == "summary"), {})
        records = [line for line in lines if line.get("kind") == "item"]
        return pl.DataFrame(records, infer_schema_length=None), summary

    def _rows_heard(self, path: Path) -> dict[str, float | None]:
        heard = path.with_name(path.name.removesuffix(".jsonl") + ROUNDTRIP)
        if not heard.exists():
            return {}
        lines = [json.loads(line) for line in heard.read_text(encoding="utf-8").splitlines() if line.strip()]
        records = pl.DataFrame([line for line in lines if line.get("kind") != "summary"], infer_schema_length=None)
        scores = Scoring(fold=True).text(records)
        return {"wer": scores.get("wer"), "cer": scores.get("cer")}

    def _rows_one(self, path: Path) -> dict:
        records, summary = self._rows_load(path)
        ok = records.filter(pl.col("failure").is_null()) if records.height else records
        similarity = ok["similarity"].drop_nulls() if "similarity" in ok.columns else pl.Series([], dtype=pl.Float64)
        return {
            "run": path.stem,
            "streams": summary.get("streams"),
            "items": records.height,
            "failed": records.height - ok.height,
            **self._rows_heard(path),
            "similarity": similarity.mean() if similarity.len() else None,
            "ttfa_ms_p50": summary.get("ttfa_p50_ms"),
            "ttfa_ms_p95": summary.get("ttfa_p95_ms"),
            "rtf_p50": summary.get("rtf_p50"),
            "rtf_p95": summary.get("rtf_p95"),
            "throughput_x": summary.get("throughput_x"),
            "cpu_per_audio_s": summary.get("cpu_per_audio_s"),
            "peak_rss_mb": summary.get("peak_rss_mb"),
        }

    ############################################################

    def rows(self) -> pl.DataFrame:
        runs = [path for path in self.files if not path.name.endswith(ROUNDTRIP)]
        return pl.DataFrame([self._rows_one(path) for path in runs]).sort("run")

    def markdown(self) -> str:
        frame = self.rows().with_columns(pl.selectors.float().round(3))
        with pl.Config(
            tbl_formatting="MARKDOWN",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            tbl_rows=-1,
            tbl_cols=-1,
            tbl_width_chars=400,
            fmt_str_lengths=60,
        ):
            return str(frame)
