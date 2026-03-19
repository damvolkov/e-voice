"""Unit tests for models/core.py — BodyType enum and UploadFile."""

from e_voice.models.core import BodyType, UploadFile

##### BODY TYPE #####


async def test_body_type_values() -> None:
    assert BodyType.PYDANTIC == "pydantic"
    assert BodyType.JSONABLE == "jsonable"
    assert BodyType.RAW == "raw"
    assert BodyType.FILE == "file"


##### UPLOAD FILE #####


async def test_upload_file_empty() -> None:
    uf = UploadFile()
    assert not uf
    assert uf.get("missing") is None
    assert uf.keys() == []


async def test_upload_file_with_data() -> None:
    uf = UploadFile(files={"audio": b"\x00\x01", "meta": b"\x02"})
    assert uf
    assert uf.get("audio") == b"\x00\x01"
    assert uf.get("missing") is None
    assert sorted(uf.keys()) == ["audio", "meta"]


async def test_upload_file_iter() -> None:
    uf = UploadFile(files={"a": b"1", "b": b"2"})
    pairs = list(uf)
    assert ("a", b"1") in pairs
    assert ("b", b"2") in pairs
