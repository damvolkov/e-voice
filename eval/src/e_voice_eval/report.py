"""One table row per bench results file: accuracy, latency and cost side by side."""

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from e_voice_eval.scoring import Scoring


@dataclass(frozen=True, slots=True)
class Report:
    """Reads `e-voice bench` JSON-lines files (items, then one summary line)."""

    files: tuple[Path, ...]

    def _rows_load(self, path: Path) -> tuple[pl.DataFrame, dict]:
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        summary = next((line for line in lines if line.get("kind") == "summary"), {})
        records = [line for line in lines if line.get("kind") != "summary"]
        return pl.DataFrame(records, infer_schema_length=None), summary

    def _rows_latency(self, records: pl.DataFrame) -> dict[str, object]:
        segments = records.select(pl.col("segments").explode()).unnest("segments")
        final, partial = segments["final_ms"].drop_nulls(), segments["partial_ms"].drop_nulls()
        return {
            "rtf_p50": records["rtf"].median(),
            "rtf_p95": records["rtf"].quantile(0.95),
            "final_ms_p50": final.median() if final.len() else None,
            "final_ms_p95": final.quantile(0.95) if final.len() else None,
            "partial_ms_p50": partial.median() if partial.len() else None,
        }

    def _rows_one(self, path: Path) -> dict:
        records, summary = self._rows_load(path)
        failures = records.filter(pl.col("failure").is_not_null()).height
        exact, folded = Scoring().text(records), Scoring(fold=True).text(records)
        return {
            "run": path.stem,
            "mode": summary.get("mode"),
            "items": records.height,
            "failed": failures,
            "wer": exact.get("wer"),
            "wer_folded": folded.get("wer"),
            "cer": exact.get("cer"),
            **Scoring().emotion(records),
            **self._rows_latency(records),
            "cpu_per_audio_s": summary.get("cpu_per_audio_s"),
            "throughput_x": summary.get("throughput_x"),
            "peak_rss_mb": summary.get("peak_rss_mb"),
        }

    ############################################################

    def rows(self) -> pl.DataFrame:
        return pl.DataFrame([self._rows_one(path) for path in self.files]).sort("run")

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
