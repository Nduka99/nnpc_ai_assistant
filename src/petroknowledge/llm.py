"""A small client for the llama.cpp servers' OpenAI-compatible API."""

import base64
import json
import math
from collections.abc import Sequence
from typing import Any

import httpx

from petroknowledge import config, telemetry


class ModelServerError(RuntimeError):
    """A model server could not be reached or returned an unusable response."""


# One client for every call to the model servers. Building a client loads the
# certificate bundle, about 0.4 s per call on the laptop even for plain HTTP (X40), and
# a shared client reuses connections. httpx clients are thread-safe (encode/httpx
# discussion #1633); the streamed answer runs in a thread.
CLIENT = httpx.Client()


def auth() -> dict[str, str]:
    """The model servers' key as a bearer token, when one is configured."""
    if not config.MODEL_API_KEY:
        return {}
    return {"Authorization": f"Bearer {config.MODEL_API_KEY}"}


def embed(texts: list[str], url: str = "") -> list[list[float]]:
    """Return one embedding vector per text, in the same order. `url` names another
    embedding server (an experiment's); by default the configured one."""
    url = url or config.EMBEDDING_URL
    with telemetry.embeddings_span():
        try:
            response = CLIENT.post(
                url,
                json={"input": texts, "model": config.EMBEDDING_MODEL},
                headers=auth(),
                timeout=config.SEARCH_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ModelServerError(
                f"embedding server at {url} failed: {error}"
            ) from error
    try:
        items = response.json()["data"]
        if not isinstance(items, list):
            raise TypeError
        indices = [item["index"] for item in items]
        if any(type(index) is not int for index in indices):
            raise TypeError
        if sorted(indices) != list(range(len(texts))):
            raise ValueError
        vectors = [item["embedding"] for item in items]
        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise TypeError
            if any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in vector
            ):
                raise ValueError
        if len({len(vector) for vector in vectors}) > 1:
            raise ValueError
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        raise ModelServerError(
            "the embedding server returned a malformed response"
        ) from error
    # Match by index, never response order, so vectors stay attached to their text.
    ordered = [vector for _, vector in sorted(zip(indices, vectors, strict=True))]
    return [fitted(vector) for vector in ordered]


def fitted(vector: list[float]) -> list[float]:
    """The vector at the index's size: a longer one cut to its first
    `EMBEDDING_DIMENSIONS` values and scaled back to unit length."""
    if len(vector) <= config.EMBEDDING_DIMENSIONS:
        return vector
    cut = vector[: config.EMBEDDING_DIMENSIONS]
    length = math.sqrt(sum(value * value for value in cut)) or 1.0
    return [value / length for value in cut]


def rerank(query: str, documents: list[str], url: str = "") -> list[float]:
    """Return one relevance score per document, in the same order (higher is better).

    The reranker is a cross-encoder: it reads the question together with each
    document, so it can judge relevance that shares no words with the question.
    `url` names another reranking server (an experiment's); by default the configured
    one.
    """
    url = url or config.RERANK_URL
    with telemetry.rerank_span():
        try:
            response = CLIENT.post(
                url,
                json={"query": query, "documents": documents},
                headers=auth(),
                timeout=config.SEARCH_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ModelServerError(
                f"reranking server at {url} failed: {error}"
            ) from error
    try:
        results = response.json()["results"]
        if not isinstance(results, list):
            raise TypeError
        indices = [item["index"] for item in results]
        if any(type(index) is not int for index in indices):
            raise TypeError
        scores = [float(item["relevance_score"]) for item in results]
        if any(not math.isfinite(score) for score in scores):
            raise ValueError
    except (ValueError, KeyError, TypeError) as error:
        raise ModelServerError(
            "the reranking server returned a malformed response"
        ) from error
    if len(scores) != len(documents) or sorted(indices) != list(range(len(documents))):
        raise ModelServerError("the reranking server did not score every passage")
    return [score for _, score in sorted(zip(indices, scores, strict=True))]


def chat_payload(
    system: str, user: str, schema: dict[str, Any], images: Sequence[bytes] = ()
) -> dict[str, Any]:
    """The body of one chat request whose reply must follow `schema`, with any
    `images` (PNG bytes) after the user's text, in order (X43 B4).

    The request names no model: the server answers with whichever GGUF it has
    loaded, which lets the same client work with any model.
    """
    content: str | list[dict[str, Any]] = user
    if images:
        content = [{"type": "text", "text": user}]
        for png in images:
            address = "data:image/png;base64," + base64.b64encode(png).decode()
            content.append({"type": "image_url", "image_url": {"url": address}})
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "temperature": config.TEMPERATURE,
        "seed": config.SEED,
        "cache_prompt": config.CACHE_PROMPT,
        "max_tokens": config.MAX_ANSWER_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema},
        },
        # Reasoning is discarded, so it is switched off at the chat template:
        # a reasoning budget alone does not stop Qwen3.x from thinking. gpt-oss
        # cannot switch it off and reads only `reasoning_effort` (default medium);
        # no other template reads that name.
        "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"},
    }


def chat(system: str, user: str, schema: dict[str, Any]) -> object:
    """Send one chat request whose reply must follow `schema` to the generation
    server; return its JSON."""
    return chat_at(config.CHAT_URL, "generation", system, user, schema)


def judge(system: str, user: str, schema: dict[str, Any]) -> object:
    """The same request to the judging server (K28)."""
    return chat_at(config.JUDGE_URL, "judging", system, user, schema)


def chat_at(
    url: str, name: str, system: str, user: str, schema: dict[str, Any]
) -> object:
    """One chat request to the `name` server at `url`; its reply's JSON."""
    with telemetry.chat_span():
        try:
            response = CLIENT.post(
                url,
                json=chat_payload(system, user, schema),
                headers=auth(),
                timeout=config.CHAT_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ModelServerError(f"{name} server at {url} failed: {error}") from error
        try:
            body = response.json()
            choice = body["choices"][0]
            finish = choice["finish_reason"]
            content = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise ModelServerError(
                f"the {name} server returned a malformed response"
            ) from error
        telemetry.record_reply(body.get("model"), finish, body.get("usage"))
    # A reply cut off at max_tokens can still happen to be valid JSON, so only a
    # reply the model finished itself is used.
    if finish != "stop":
        raise ModelServerError(f"the model's reply did not finish ({finish})")
    try:
        reply: object = json.loads(content)
    except (json.JSONDecodeError, TypeError) as error:
        raise ModelServerError("the model's reply is not valid JSON") from error
    return reply


def healthy(url: str) -> bool:
    """Whether the llama.cpp server behind `url` has loaded its model: its /health
    answers 200 when ready and 503 while the model is still loading."""
    try:
        response = CLIENT.get(
            str(httpx.URL(url).join("/health")),
            timeout=config.HEALTH_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError:
        return False
    return response.status_code == 200
