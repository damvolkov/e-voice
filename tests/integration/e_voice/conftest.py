"""Shared fixtures for e-voice integration tests."""

import pytest
from pytest_audioeval.client import AudioEval


@pytest.fixture(scope="session")
def en_sample(audioeval: AudioEval):
    """English 'hello world' audio sample for STT tests."""
    return audioeval.samples.en_hello_world


@pytest.fixture(scope="session")
def es_sample(audioeval: AudioEval):
    """Spanish 'hola mundo' audio sample for STT tests."""
    return audioeval.samples.es_hola_mundo


@pytest.fixture(scope="session")
def en_counting_sample(audioeval: AudioEval):
    """English 'counting' audio sample for longer transcription tests."""
    return audioeval.samples.en_counting
