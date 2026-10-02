import polars as pl

from e_voice_eval.chart import Chart, Metric


def test_mermaid_sorts_one_dataset_and_skips_missing_values() -> None:
    rows = pl.DataFrame(
        {
            "run": ["b.fleurs-es.file", "a.fleurs-es.file", "a.fleurs-en.file", "c.fleurs-es.file"],
            "wer": [0.081, 0.045, 0.2, None],
        }
    )
    chart = Chart(rows).mermaid(Metric("WER", "wer", "fleurs-es", "file", "WER %", 100.0))
    assert chart is not None
    assert 'x-axis ["a", "b"]' in chart
    assert "bar [4.5, 8.1]" in chart
    assert chart.startswith("```mermaid\nxychart-beta horizontal")


def test_markdown_omits_metrics_without_runs() -> None:
    rows = pl.DataFrame({"run": ["a.crema.file"], "wer": [None], "ser_accuracy": [0.66]})
    text = Chart(rows).markdown(
        (Metric("SER", "ser_accuracy", "crema", "file", "%", 100.0), Metric("WER", "wer", "fleurs-es", "file", "%"))
    )
    assert "### SER" in text
    assert "### WER" not in text
