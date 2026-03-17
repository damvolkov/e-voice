"""Unit tests for streaming/text.py — word utilities and LocalAgreement primitives."""

import pytest

from e_voice.streaming.text import (
    StreamingWord,
    WordBuffer,
    canonicalize_word,
    common_prefix,
    is_eos,
    last_full_sentence_end,
    last_full_sentence_text,
    to_full_sentences,
    words_to_text,
)


def _w(word: str, start: float = 0.0, end: float = 0.0) -> StreamingWord:
    """Shortcut to create a StreamingWord."""
    return StreamingWord(word=word, start=start, end=end)


##### CANONICALIZE_WORD #####


@pytest.mark.parametrize(
    ("input_text", "expected"),
    [
        ("Hello!", "hello"),
        ("world.", "world"),
        ("it's", "its"),
        ("café", "caf"),
        ("123", "123"),
        ("", ""),
        ("¡Hola!", "hola"),
        ("  spaces  ", "spaces"),
    ],
    ids=["excl", "dot", "apostrophe", "accent", "digits", "empty", "spanish", "spaces"],
)
async def test_canonicalize_word(input_text: str, expected: str) -> None:
    assert canonicalize_word(input_text) == expected


##### COMMON_PREFIX #####


async def test_common_prefix_identical() -> None:
    a = [_w("hello"), _w("world")]
    b = [_w("hello"), _w("world")]
    result = common_prefix(a, b)
    assert len(result) == 2
    assert result[0].word == "hello"
    assert result[1].word == "world"


async def test_common_prefix_partial() -> None:
    a = [_w("hello"), _w("world"), _w("foo")]
    b = [_w("hello"), _w("bar")]
    result = common_prefix(a, b)
    assert len(result) == 1
    assert result[0].word == "hello"


async def test_common_prefix_none() -> None:
    a = [_w("hello")]
    b = [_w("goodbye")]
    assert common_prefix(a, b) == []


async def test_common_prefix_empty() -> None:
    assert common_prefix([], [_w("hello")]) == []
    assert common_prefix([_w("hello")], []) == []
    assert common_prefix([], []) == []


async def test_common_prefix_ignores_punctuation() -> None:
    a = [_w("Hello,"), _w("world!")]
    b = [_w("hello"), _w("world.")]
    result = common_prefix(a, b)
    assert len(result) == 2


async def test_common_prefix_returns_from_first_list() -> None:
    a = [_w("Hello", 1.0, 2.0)]
    b = [_w("hello", 5.0, 6.0)]
    result = common_prefix(a, b)
    assert result[0].start == 1.0


##### IS_EOS #####


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello.", True),
        ("hello?", True),
        ("hello!", True),
        ("hello", False),
        ("hello...", False),
        ("", False),
        ("  ", False),
        ("Dr.", True),
    ],
    ids=["period", "question", "excl", "none", "ellipsis", "empty", "whitespace", "abbreviation"],
)
async def test_is_eos(text: str, expected: bool) -> None:
    assert is_eos(text) == expected


##### TO_FULL_SENTENCES #####


async def test_to_full_sentences_single() -> None:
    words = [_w("Hello"), _w("world.")]
    sentences = to_full_sentences(words)
    assert len(sentences) == 1
    assert len(sentences[0]) == 2


async def test_to_full_sentences_multiple() -> None:
    words = [_w("Hi."), _w("How"), _w("are"), _w("you?"), _w("Fine")]
    sentences = to_full_sentences(words)
    assert len(sentences) == 2
    assert sentences[0][0].word == "Hi."
    assert sentences[1][-1].word == "you?"


async def test_to_full_sentences_excludes_trailing() -> None:
    words = [_w("Hello."), _w("World")]
    sentences = to_full_sentences(words)
    assert len(sentences) == 1


async def test_to_full_sentences_empty() -> None:
    assert to_full_sentences([]) == []


async def test_to_full_sentences_no_complete() -> None:
    words = [_w("Hello"), _w("world")]
    assert to_full_sentences(words) == []


##### LAST_FULL_SENTENCE_END #####


async def test_last_full_sentence_end_with_sentences() -> None:
    words = [_w("Hello.", 0.0, 1.0), _w("World.", 1.0, 2.0), _w("foo", 2.0, 3.0)]
    assert last_full_sentence_end(words) == 2.0


async def test_last_full_sentence_end_no_sentences() -> None:
    words = [_w("hello", 0.0, 1.0)]
    assert last_full_sentence_end(words) == 0.0


async def test_last_full_sentence_end_empty() -> None:
    assert last_full_sentence_end([]) == 0.0


##### LAST_FULL_SENTENCE_TEXT #####


async def test_last_full_sentence_text_returns_sentence() -> None:
    words = [_w("Hello."), _w("World."), _w("foo")]
    assert last_full_sentence_text(words) == "World."


async def test_last_full_sentence_text_returns_none() -> None:
    assert last_full_sentence_text([_w("hello")]) is None
    assert last_full_sentence_text([]) is None


##### WORDS_TO_TEXT #####


async def test_words_to_text() -> None:
    words = [_w("Hello"), _w("world")]
    assert words_to_text(words) == "Hello world"


async def test_words_to_text_empty() -> None:
    assert words_to_text([]) == ""


##### STREAMING_WORD #####


async def test_streaming_word_offset() -> None:
    w = StreamingWord(word="hello", start=1.0, end=2.0, probability=0.9)
    w.offset(5.0)
    assert w.start == 6.0
    assert w.end == 7.0


##### WORD_BUFFER #####


async def test_word_buffer_text() -> None:
    buf = WordBuffer([_w("Hello"), _w("world")])
    assert buf.text == "Hello world"


async def test_word_buffer_start_end() -> None:
    buf = WordBuffer([_w("a", 1.0, 2.0), _w("b", 3.0, 4.0)])
    assert buf.start == 1.0
    assert buf.end == 4.0


async def test_word_buffer_empty() -> None:
    buf = WordBuffer()
    assert buf.text == ""
    assert buf.start == 0.0
    assert buf.end == 0.0
    assert not buf
    assert len(buf) == 0


async def test_word_buffer_extend() -> None:
    buf = WordBuffer([_w("hello")])
    buf.extend([_w("world")])
    assert len(buf) == 2
    assert buf.text == "hello world"


async def test_word_buffer_after() -> None:
    buf = WordBuffer([_w("a", 0.0, 1.0), _w("b", 1.5, 2.0), _w("c", 3.0, 4.0)])
    after = buf.after(1.4)
    assert len(after) == 2
    assert after[0].word == "b"


async def test_word_buffer_bool() -> None:
    assert bool(WordBuffer([_w("x")])) is True
    assert bool(WordBuffer()) is False
