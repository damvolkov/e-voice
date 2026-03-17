import httpx
import orjson

##### GET /v1/models #####


async def test_list_models_returns_list(http_client: httpx.AsyncClient) -> None:
    response = await http_client.get("/v1/models")
    assert response.status_code == 200

    body = orjson.loads(response.content)
    assert "data" in body
    assert "object" in body
    assert body["object"] == "list"
    assert isinstance(body["data"], list)


async def test_list_models_contains_loaded_models(http_client: httpx.AsyncClient) -> None:
    response = await http_client.get("/v1/models")
    assert response.status_code == 200

    body = orjson.loads(response.content)
    assert len(body["data"]) > 0

    model = body["data"][0]
    assert "id" in model
    assert "object" in model
    assert model["object"] == "model"
    assert "owned_by" in model


##### GET /v1/models/:model_id #####


async def test_get_model_loaded(http_client: httpx.AsyncClient) -> None:
    list_response = await http_client.get("/v1/models")
    models = orjson.loads(list_response.content)["data"]
    assert len(models) > 0

    model_id = models[0]["id"]
    response = await http_client.get(f"/v1/models/{model_id}")
    assert response.status_code == 200

    body = orjson.loads(response.content)
    assert body["id"] == model_id
    assert body["object"] == "model"


async def test_get_model_not_found(http_client: httpx.AsyncClient) -> None:
    response = await http_client.get("/v1/models/non-existent-model-id")
    assert response.status_code == 404

    body = orjson.loads(response.content)
    assert "error" in body


##### GET /v1/api/ps #####


async def test_list_loaded_models(http_client: httpx.AsyncClient) -> None:
    response = await http_client.get("/v1/api/ps")
    assert response.status_code == 200

    body = orjson.loads(response.content)
    assert "models" in body
    assert isinstance(body["models"], list)


##### POST /v1/api/ps/:model_id #####


async def test_load_model_already_loaded_returns_409(http_client: httpx.AsyncClient) -> None:
    list_response = await http_client.get("/v1/api/ps")
    models = orjson.loads(list_response.content)["models"]
    assert len(models) > 0

    model_id = models[0]
    response = await http_client.post(f"/v1/api/ps/{model_id}")
    assert response.status_code == 409

    body = orjson.loads(response.content)
    assert "error" in body


##### DELETE /v1/api/ps/:model_id #####


async def test_unload_model_not_found(http_client: httpx.AsyncClient) -> None:
    response = await http_client.delete("/v1/api/ps/non-existent-model-id")
    assert response.status_code == 404

    body = orjson.loads(response.content)
    assert "error" in body
