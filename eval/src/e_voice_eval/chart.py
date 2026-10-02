"""Mermaid bar charts from a report frame, one metric per chart, for the documentation site."""

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, slots=True)
class Metric:
    """One chart: `column` of the runs whose name ends in `.<dataset>.<tag>`, scaled for display."""

    title: str
    column: str
    dataset: str
    tag: str
    axis: str
    scale: float = 1.0


METRICS: tuple[Metric, ...] = (
    Metric("WER · FLEURS es · files", "wer", "fleurs-es", "file", "WER %", 100.0),
    Metric("WER · FLEURS en · files", "wer", "fleurs-en", "file", "WER %", 100.0),
    Metric("CPU seconds per audio second · FLEURS es · files", "cpu_per_audio_s", "fleurs-es", "file", "CPU s / s"),
    Metric("Throughput · FLEURS es · files", "throughput_x", "fleurs-es", "file", "times real time"),
    Metric("Final after end of speech · p50 · live, 1 stream", "final_ms_p50", "fleurs-es", "live1", "ms"),
    Metric("Final after end of speech · p95 · live, 4 streams", "final_ms_p95", "fleurs-es", "live4", "ms"),
    Metric("Peak memory · FLEURS es · files", "peak_rss_mb", "fleurs-es", "file", "MB"),
    Metric("Emotion accuracy · CREMA-D en", "ser_accuracy", "crema", "file", "accuracy %", 100.0),
    Metric("Emotion accuracy · MESD es", "ser_accuracy", "mesd", "file", "accuracy %", 100.0),
)


@dataclass(frozen=True, slots=True)
class Chart:
    rows: pl.DataFrame

    def _series(self, metric: Metric) -> pl.DataFrame:
        suffix = f".{metric.dataset}.{metric.tag}"
        return (
            self.rows.filter(pl.col("run").str.ends_with(suffix) & pl.col(metric.column).is_not_null())
            .select(
                pl.col("run").str.strip_suffix(suffix).alias("config"),
                (pl.col(metric.column) * metric.scale).round(2).alias("value"),
            )
            .sort("value")
        )

    def mermaid(self, metric: Metric) -> str | None:
        series = self._series(metric)
        if series.is_empty():
            return None
        names = ", ".join(f'"{name}"' for name in series["config"])
        values = ", ".join(str(value) for value in series["value"])
        top = max(series["value"].to_list()) * 1.15
        return "\n".join(
            (
                "```mermaid",
                "xychart-beta horizontal",
                f'    title "{metric.title}"',
                f"    x-axis [{names}]",
                f'    y-axis "{metric.axis}" 0 --> {top:.2f}',
                f"    bar [{values}]",
                "```",
            )
        )

    def markdown(self, metrics: tuple[Metric, ...] = METRICS) -> str:
        charts = ((metric, self.mermaid(metric)) for metric in metrics)
        return "\n\n".join(f"### {metric.title}\n\n{chart}" for metric, chart in charts if chart) + "\n"
