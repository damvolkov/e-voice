"""Tests for VoiceEntry and parse_voice_filename."""

import pytest

from e_voice.models.tts import VoiceEntry, parse_voice_filename


@pytest.mark.parametrize(
    ("filename", "expected_id", "expected_lang"),
    [
        ("serena_en.pt", "serena", "en"),
        ("tatan_es.pt", "tatan", "es"),
        ("ono_anna_ja.pt", "ono_anna", "ja"),
        ("uncle_fu_zh.pt", "uncle_fu", "zh"),
    ],
    ids=["simple-en", "simple-es", "underscore-ja", "underscore-zh"],
)
async def test_parse_voice_filename(filename: str, expected_id: str, expected_lang: str) -> None:
    entry = parse_voice_filename(filename)
    assert entry.id == expected_id
    assert entry.language == expected_lang
    assert entry.cloned is True


async def test_parse_voice_filename_no_underscore() -> None:
    entry = parse_voice_filename("mystery.pt")
    assert entry.id == "mystery"
    assert entry.language == "multilingual"


async def test_voice_entry_frozen() -> None:
    entry = VoiceEntry(id="test", language="en")
    with pytest.raises(AttributeError):
        entry.id = "other"  # type: ignore[misc]


async def test_voice_entry_defaults() -> None:
    entry = VoiceEntry(id="test")
    assert entry.language == "multilingual"
    assert entry.cloned is False
