import pytest
from openai import AsyncOpenAI


@pytest.fixture(scope="session")
async def openai_client(e_voice_server: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="not-needed",
        base_url=f"{e_voice_server}/v1",
    )
