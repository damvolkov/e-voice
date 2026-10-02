import polars as pl

from e_voice_eval.scoring import Scoring


def frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, infer_schema_length=None)


def test_text_ignores_case_and_punctuation_but_keeps_accents() -> None:
    records = frame([{"text": "No preguntes que puede.", "ref_text": "no preguntes, ¿qué puede?"}])
    assert Scoring().text(records)["wer"] == 0.25
    assert Scoring(fold=True).text(records)["wer"] == 0.0


def test_text_skips_records_without_reference() -> None:
    records = frame([{"text": "hola", "ref_text": None}, {"text": "adiós amigo", "ref_text": "adiós amigos"}])
    scores = Scoring().text(records)
    assert (scores["wer"], scores["scored"]) == (0.5, 1.0)


def test_emotion_metrics_and_angry_recall() -> None:
    rows = [
        {"emotion": {"label": "angry"}, "ref_emotion": "angry"},
        {"emotion": {"label": "neutral"}, "ref_emotion": "angry"},
        {"emotion": {"label": "neutral"}, "ref_emotion": "neutral"},
        {"emotion": {"label": "happy"}, "ref_emotion": None},
    ]
    scores = Scoring().emotion(frame(rows))
    assert scores["ser_accuracy"] == 2 / 3
    assert scores["angry_recall"] == 0.5
    matrix = Scoring().confusion(frame(rows))
    assert matrix.filter(pl.col("ref") == "angry")["neutral"].item() == 1
