"""Tests for the Gemini API client."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from youtube_brain.llm.gemini import GeminiClient


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _mock_response(data: dict, status_code: int = 200) -> httpx.Response:
    """Build an httpx.Response from a dict."""
    return httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request("POST", "https://example.com"),
    )


def _generate_response(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def _embed_response(values: list[float]) -> dict:
    return {"embedding": {"values": values}}


def _batch_embed_response(vectors: list[list[float]]) -> dict:
    return {"embeddings": [{"values": v} for v in vectors]}


# ------------------------------------------------------------------
# Init
# ------------------------------------------------------------------


def test_client_init():
    client = GeminiClient(api_key="test-key")
    assert client.model == "gemini-2.5-flash"
    assert client.embed_model == "gemini-embedding-001"
    assert client.embed_dims == 768
    assert client.api_key == "test-key"


def test_client_init_custom_model():
    client = GeminiClient(api_key="test-key", model="gemini-2.0-flash")
    assert client.model == "gemini-2.0-flash"


# ------------------------------------------------------------------
# generate()
# ------------------------------------------------------------------


async def test_generate_basic():
    client = GeminiClient(api_key="test-key")
    mock_resp = _mock_response(_generate_response("Hello back!"))

    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp) as m:
        result = await client.generate("Hello")

    assert result == "Hello back!"

    # Verify request shape
    call_kwargs = m.call_args
    body = call_kwargs.kwargs["json"]
    assert body["contents"][0]["parts"][0]["text"] == "Hello"
    assert body["generationConfig"]["temperature"] == 0.7
    assert "systemInstruction" not in body
    assert "responseMimeType" not in body["generationConfig"]

    # Verify API key in query params
    assert call_kwargs.kwargs["params"] == {"key": "test-key"}
    await client.close()


async def test_generate_with_system():
    client = GeminiClient(api_key="test-key")
    mock_resp = _mock_response(_generate_response("sys response"))

    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp) as m:
        await client.generate("prompt", system="You are helpful")

    body = m.call_args.kwargs["json"]
    assert body["systemInstruction"]["parts"][0]["text"] == "You are helpful"
    await client.close()


async def test_generate_with_json_mode():
    client = GeminiClient(api_key="test-key")
    mock_resp = _mock_response(_generate_response('{"key": "value"}'))

    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp) as m:
        result = await client.generate("prompt", response_json=True)

    body = m.call_args.kwargs["json"]
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert result == '{"key": "value"}'
    await client.close()


async def test_generate_custom_temperature():
    client = GeminiClient(api_key="test-key")
    mock_resp = _mock_response(_generate_response("temp"))

    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp) as m:
        await client.generate("prompt", temperature=0.2)

    body = m.call_args.kwargs["json"]
    assert body["generationConfig"]["temperature"] == 0.2
    await client.close()


async def test_generate_url():
    client = GeminiClient(api_key="test-key", model="gemini-2.5-flash")
    mock_resp = _mock_response(_generate_response("ok"))

    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp) as m:
        await client.generate("hi")

    url = m.call_args.args[0]
    assert "gemini-2.5-flash:generateContent" in url
    await client.close()


# ------------------------------------------------------------------
# generate_json()
# ------------------------------------------------------------------


async def test_generate_json_dict():
    client = GeminiClient(api_key="test-key")
    payload = {"tags": ["science", "tech"], "score": 0.9}
    mock_resp = _mock_response(_generate_response(json.dumps(payload)))

    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp):
        result = await client.generate_json("classify this")

    assert result == payload
    await client.close()


async def test_generate_json_list():
    client = GeminiClient(api_key="test-key")
    payload = [{"id": 1}, {"id": 2}]
    mock_resp = _mock_response(_generate_response(json.dumps(payload)))

    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp):
        result = await client.generate_json("list items")

    assert result == payload
    await client.close()


async def test_generate_json_temperature():
    """generate_json defaults to temperature 0.3."""
    client = GeminiClient(api_key="test-key")
    mock_resp = _mock_response(_generate_response("{}"))

    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp) as m:
        await client.generate_json("prompt")

    body = m.call_args.kwargs["json"]
    assert body["generationConfig"]["temperature"] == 0.3
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    await client.close()


# ------------------------------------------------------------------
# embed_texts() routing
# ------------------------------------------------------------------


async def test_embed_texts_single_routes_to_embed_single():
    client = GeminiClient(api_key="test-key")
    vec = [0.1, 0.2, 0.3]

    with patch.object(client, "_embed_single", new_callable=AsyncMock, return_value=vec) as m:
        result = await client.embed_texts(["hello"])

    m.assert_called_once_with("hello")
    assert result == [vec]
    await client.close()


async def test_embed_texts_multiple_routes_to_embed_batch():
    client = GeminiClient(api_key="test-key")
    vecs = [[0.1, 0.2], [0.3, 0.4]]

    with patch.object(client, "_embed_batch", new_callable=AsyncMock, return_value=vecs) as m:
        result = await client.embed_texts(["a", "b"])

    m.assert_called_once_with(["a", "b"])
    assert result == vecs
    await client.close()


# ------------------------------------------------------------------
# _embed_single()
# ------------------------------------------------------------------


async def test_embed_single():
    client = GeminiClient(api_key="test-key")
    vec = [0.1] * 768
    mock_resp = _mock_response(_embed_response(vec))

    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp) as m:
        result = await client._embed_single("hello world")

    assert result == vec

    # Verify request
    url = m.call_args.args[0]
    assert "gemini-embedding-001:embedContent" in url
    body = m.call_args.kwargs["json"]
    assert body["content"]["parts"][0]["text"] == "hello world"
    assert body["outputDimensionality"] == 768
    await client.close()


# ------------------------------------------------------------------
# _embed_batch()
# ------------------------------------------------------------------


async def test_embed_batch_small():
    client = GeminiClient(api_key="test-key")
    texts = ["a", "b", "c"]
    vecs = [[0.1], [0.2], [0.3]]
    mock_resp = _mock_response(_batch_embed_response(vecs))

    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp) as m:
        result = await client._embed_batch(texts)

    assert result == vecs
    assert m.call_count == 1

    # Verify request body
    body = m.call_args.kwargs["json"]
    assert len(body["requests"]) == 3
    assert body["requests"][0]["model"] == "models/gemini-embedding-001"
    assert body["requests"][1]["content"]["parts"][0]["text"] == "b"
    assert body["requests"][0]["outputDimensionality"] == 768
    await client.close()


async def test_embed_batch_over_100():
    """Batches of >100 texts should be split into multiple requests."""
    client = GeminiClient(api_key="test-key", cooldown=0)
    texts = [f"text_{i}" for i in range(150)]

    vecs_batch1 = [[float(i)] for i in range(100)]
    vecs_batch2 = [[float(i)] for i in range(100, 150)]

    resp1 = _mock_response(_batch_embed_response(vecs_batch1))
    resp2 = _mock_response(_batch_embed_response(vecs_batch2))

    with patch.object(
        client._client, "post", new_callable=AsyncMock, side_effect=[resp1, resp2]
    ) as m:
        result = await client._embed_batch(texts)

    assert m.call_count == 2
    assert len(result) == 150

    # First call should have 100 requests
    first_body = m.call_args_list[0].kwargs["json"]
    assert len(first_body["requests"]) == 100

    # Second call should have 50 requests
    second_body = m.call_args_list[1].kwargs["json"]
    assert len(second_body["requests"]) == 50
    await client.close()


async def test_embed_batch_exactly_100():
    """Exactly 100 texts should require a single request."""
    client = GeminiClient(api_key="test-key")
    texts = [f"text_{i}" for i in range(100)]
    vecs = [[float(i)] for i in range(100)]
    mock_resp = _mock_response(_batch_embed_response(vecs))

    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp) as m:
        result = await client._embed_batch(texts)

    assert m.call_count == 1
    assert len(result) == 100
    await client.close()


async def test_embed_batch_201():
    """201 texts should require 3 requests (100 + 100 + 1)."""
    client = GeminiClient(api_key="test-key", cooldown=0)
    texts = [f"t{i}" for i in range(201)]

    vecs1 = [[float(i)] for i in range(100)]
    vecs2 = [[float(i)] for i in range(100, 200)]
    vecs3 = [[float(i)] for i in range(200, 201)]

    resp1 = _mock_response(_batch_embed_response(vecs1))
    resp2 = _mock_response(_batch_embed_response(vecs2))
    resp3 = _mock_response(_batch_embed_response(vecs3))

    with patch.object(
        client._client, "post", new_callable=AsyncMock, side_effect=[resp1, resp2, resp3]
    ) as m:
        result = await client._embed_batch(texts)

    assert m.call_count == 3
    assert len(result) == 201
    await client.close()


# ------------------------------------------------------------------
# close()
# ------------------------------------------------------------------


async def test_close():
    client = GeminiClient(api_key="test-key")
    with patch.object(client._client, "aclose", new_callable=AsyncMock) as m:
        await client.close()
    m.assert_called_once()


# ------------------------------------------------------------------
# HTTP error handling
# ------------------------------------------------------------------


async def test_generate_raises_on_http_error():
    client = GeminiClient(api_key="test-key")
    error_resp = httpx.Response(
        status_code=400,
        json={"error": {"message": "bad request"}},
        request=httpx.Request("POST", "https://example.com"),
    )

    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=error_resp):
        with pytest.raises(httpx.HTTPStatusError):
            await client.generate("bad prompt")

    await client.close()


# ------------------------------------------------------------------
# Integration tests (skipped without API key)
# ------------------------------------------------------------------


@pytest.mark.integration
async def test_embed_real():
    import os

    key = os.environ.get("YTBRAIN_GEMINI_API_KEY")
    if not key:
        pytest.skip("No Gemini API key")
    client = GeminiClient(api_key=key)
    vectors = await client.embed_texts(["Hello world"])
    assert len(vectors) == 1
    assert len(vectors[0]) > 0
    await client.close()


@pytest.mark.integration
async def test_generate_real():
    import os

    key = os.environ.get("YTBRAIN_GEMINI_API_KEY")
    if not key:
        pytest.skip("No Gemini API key")
    client = GeminiClient(api_key=key)
    result = await client.generate("Say hello in exactly one word.")
    assert isinstance(result, str)
    assert len(result) > 0
    await client.close()
