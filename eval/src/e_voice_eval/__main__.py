"""python -m e_voice_eval fetch|report|confusion"""

import argparse
from pathlib import Path

import polars as pl
import structlog

from e_voice_eval.datasets import DATASETS
from e_voice_eval.report import Report
from e_voice_eval.scoring import Scoring


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
        case "confusion":
            records = Report((args.file,))._rows_load(args.file)[0]
            with pl.Config(tbl_formatting="MARKDOWN", tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True):
                print(Scoring().confusion(records))


if __name__ == "__main__":
    structlog.configure(logger_factory=structlog.PrintLoggerFactory())
    main()
