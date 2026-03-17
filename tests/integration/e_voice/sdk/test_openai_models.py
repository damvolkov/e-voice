"""OpenAI SDK compatibility tests — Models endpoint.

Validates that e-voice /v1/models is compatible with OpenAI's Models API.
"""

from openai import AsyncOpenAI

##### GET /v1/models (SDK) #####


async def test_list_models(openai_client: AsyncOpenAI) -> None:
    """SDK list models returns at least the default whisper model."""
    models = await openai_client.models.list()
    model_ids = [m.id for m in models.data]
    assert len(model_ids) > 0


async def test_list_models_has_whisper(openai_client: AsyncOpenAI) -> None:
    """SDK list models includes the configured whisper model."""
    models = await openai_client.models.list()
    model_ids = [m.id for m in models.data]
    assert any("whisper" in mid or "turbo" in mid for mid in model_ids)


async def test_model_object_schema(openai_client: AsyncOpenAI) -> None:
    """SDK model objects have required OpenAI fields."""
    models = await openai_client.models.list()
    assert len(models.data) > 0

    model = models.data[0]
    assert hasattr(model, "id")
    assert hasattr(model, "object")
    assert model.object == "model"
