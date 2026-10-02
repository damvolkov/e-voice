"""Text normalization, WER/CER and emotion metrics over bench result lines."""

import re
import unicodedata
from dataclasses import dataclass

import jiwer
import polars as pl

PUNCTUATION = re.compile(r"[^\w\s']|_", re.UNICODE)
SPACES = re.compile(r"\s+")
LABELS = ("angry", "disgusted", "fearful", "happy", "neutral", "sad")


@dataclass(frozen=True, slots=True)
class Scoring:
    """Corpus-level scores of one results file; accents are kept unless `fold` is set."""

    fold: bool = False

    def _text_normalize(self, text: str) -> str:
        lowered = PUNCTUATION.sub(" ", unicodedata.normalize("NFC", text).lower())
        folded = (
            "".join(c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn")
            if self.fold
            else lowered
        )
        return SPACES.sub(" ", folded).strip()

    ############################################################

    def text(self, records: pl.DataFrame) -> dict[str, float]:
        """WER and CER over every record with a reference; empty references are skipped."""
        scored = records.filter(pl.col("ref_text").is_not_null() & (pl.col("ref_text") != ""))
        if scored.is_empty():
            return {}
        refs = [self._text_normalize(ref) for ref in scored["ref_text"]]
        hyps = [self._text_normalize(hyp) for hyp in scored["text"]]
        return {"wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps), "scored": float(scored.height)}

    def emotion(self, records: pl.DataFrame) -> dict[str, float]:
        """Accuracy, macro-F1 over the six reference labels, and recall of `angry` (the signal hooks need)."""
        scored = records.filter(pl.col("ref_emotion").is_not_null()).select(
            ref=pl.col("ref_emotion"), hyp=pl.col("emotion").struct.field("label")
        )
        if scored.is_empty():
            return {}
        f1 = []
        for label in LABELS:
            tp = scored.filter((pl.col("ref") == label) & (pl.col("hyp") == label)).height
            predicted = scored.filter(pl.col("hyp") == label).height
            actual = scored.filter(pl.col("ref") == label).height
            precision, recall = tp / predicted if predicted else 0.0, tp / actual if actual else 0.0
            f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        angry = scored.filter(pl.col("ref") == "angry")
        return {
            "ser_accuracy": scored.filter(pl.col("ref") == pl.col("hyp")).height / scored.height,
            "ser_macro_f1": sum(f1) / len(f1),
            "angry_recall": angry.filter(pl.col("hyp") == "angry").height / angry.height if angry.height else 0.0,
        }

    def confusion(self, records: pl.DataFrame) -> pl.DataFrame:
        """Counts per (reference label, predicted label) pair."""
        return (
            records.filter(pl.col("ref_emotion").is_not_null())
            .select(ref=pl.col("ref_emotion"), hyp=pl.col("emotion").struct.field("label"))
            .group_by("ref", "hyp")
            .len()
            .pivot(on="hyp", index="ref", values="len")
            .fill_null(0)
            .sort("ref")
        )
