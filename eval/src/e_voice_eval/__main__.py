"""python -m e_voice_eval fetch|report|synthesis|annex|confusion"""

import argparse
from pathlib import Path

import polars as pl
import structlog

from e_voice_eval.chart import Chart
from e_voice_eval.datasets import DATASETS
from e_voice_eval.report import Report
from e_voice_eval.scoring import Scoring
from e_voice_eval.synthesis import Synthesis


def main() -> None:
    parser = argparse.ArgumentParser(prog="e_voice_eval")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="download a pinned dataset as WAV + manifest.jsonl")
    fetch.add_argument("name", choices=sorted(DATASETS))
    fetch.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3] / "data/stt/ops/datasets")
    fetch.add_argument("--limit", type=int)
    report = commands.add_parser("report", help="compare bench results files")
    report.add_argument("files", type=Path, nargs="+")
    report.add_argument("--out", type=Path)
    synthesis = commands.add_parser("synthesis", help="compare TTS bench results files (with their STT round trips)")
    synthesis.add_argument("files", type=Path, nargs="+")
    synthesis.add_argument("--out", type=Path)
    annex = commands.add_parser("annex", help="report.md, summary.csv and charts.md of bench results into a directory")
    annex.add_argument("files", type=Path, nargs="+")
    annex.add_argument("--out", type=Path, required=True)
    confusion = commands.add_parser("confusion", help="emotion confusion matrix of one results file")
    confusion.add_argument("file", type=Path)
    args = parser.parse_args()
    match args.command:
        case "fetch":
            DATASETS[args.name].fetch(args.root, args.limit)
        case "report":
            table = Report(tuple(args.files)).markdown()
            print(table)
            if args.out:
                args.out.write_text(table + "\n", encoding="utf-8")
        case "synthesis":
            table = Synthesis(tuple(args.files)).markdown()
            print(table)
            if args.out:
                args.out.write_text(table + "\n", encoding="utf-8")
        case "annex":
            args.out.mkdir(parents=True, exist_ok=True)
            report = Report(tuple(args.files))
            rows = report.rows()
            (args.out / "report.md").write_text(report.markdown() + "\n", encoding="utf-8")
            rows.with_columns(pl.selectors.float().round(4)).write_csv(args.out / "summary.csv")
            (args.out / "charts.md").write_text(Chart(rows).markdown(), encoding="utf-8")
        case "confusion":
            records = Report((args.file,))._rows_load(args.file)[0]
            with pl.Config(tbl_formatting="MARKDOWN", tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True):
                print(Scoring().confusion(records))


if __name__ == "__main__":
    structlog.configure(logger_factory=structlog.PrintLoggerFactory())
    main()
