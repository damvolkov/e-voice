import pytest
from openai import AsyncOpenAI


@pytest.fixture(scope="session")
async def openai_client(base_url: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key="not-needed", base_url=f"{base_url}/v1")
