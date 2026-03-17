import pytest
from pytest_audioeval.client import AudioEval


@pytest.fixture(scope="session")
def en_sample(audioeval: AudioEval):
    return audioeval.samples.en_hello_world


@pytest.fixture(scope="session")
def es_sample(audioeval: AudioEval):
    return audioeval.samples.es_hola_mundo


@pytest.fixture(scope="session")
def en_counting_sample(audioeval: AudioEval):
    return audioeval.samples.en_counting
