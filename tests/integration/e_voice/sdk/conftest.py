"""Fixtures for OpenAI SDK compatibility tests."""

import pytest
from openai import AsyncOpenAI


@pytest.fixture(scope="session")
async def openai_client(e_voice_server: str) -> AsyncOpenAI:
    """AsyncOpenAI client pointed at the auto-started e-voice server."""
    return AsyncOpenAI(
        api_key="not-needed",
        base_url=f"{e_voice_server}/v1",
    )
