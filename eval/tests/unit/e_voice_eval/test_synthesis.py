import json
from pathlib import Path

import pytest

from e_voice_eval.synthesis import Synthesis


def _write(path: Path, lines: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def run(tmp_path: Path) -> Path:
    items = [
        {"kind": "item", "id": "a", "text": "hola mundo", "similarity": 0.8, "failure": None},
        {"kind": "item", "id": "b", "text": "adiós", "similarity": 0.6, "failure": None},
        {"kind": "item", "id": "c", "text": "x", "similarity": None, "failure": "every pocket worker is busy"},
    ]
    summary = {"kind": "summary", "streams": 4, "ttfa_p50_ms": 90.0, "rtf_p50": 0.3, "throughput_x": 7.5}
    return _write(tmp_path / "base.fleurs-es.s4.jsonl", [*items, summary])


def test_rows_report_latency_cost_and_similarity(run: Path) -> None:
    row = Synthesis((run,)).rows().row(0, named=True)
    assert (row["run"], row["streams"], row["items"], row["failed"]) == ("base.fleurs-es.s4", 4, 3, 1)
    assert row["similarity"] == pytest.approx(0.7)
    assert (row["ttfa_ms_p50"], row["throughput_x"]) == (90.0, 7.5)
    assert "wer" not in row


def test_rows_join_the_stt_round_trip(run: Path) -> None:
    heard = [
        {"id": "a", "text": "Hola, mundo.", "ref_text": "hola mundo"},
        {"id": "b", "text": "adios", "ref_text": "adiós"},
        {"kind": "summary"},
    ]
    roundtrip = _write(run.with_name("base.fleurs-es.s4.roundtrip.jsonl"), heard)
    rows = Synthesis((run, roundtrip)).rows()
    assert rows.height == 1
    assert rows.row(0, named=True)["wer"] == pytest.approx(0.0)
