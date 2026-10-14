import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from petroknowledge import config, llm

SCHEMA = {"type": "object"}


def chat_reply(content: str, finish_reason: str = "stop") -> dict[str, Any]:
    message = {"content": content}
    return {"choices": [{"message": message, "finish_reason": finish_reason}]}


def test_chat_sends_the_fixed_request_and_returns_parsed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: dict[str, Any] = {}

    def post(url: str, **kwargs: Any) -> httpx.Response:
        sent.update(url=url, **kwargs)
        body = chat_reply(json.dumps({"claims": []}))
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.CLIENT, "post", post)

    assert llm.chat("system text", "user text", SCHEMA) == {"claims": []}
    payload = sent["json"]
    assert sent["url"] == config.CHAT_URL
    assert payload["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]
    assert payload["response_format"]["json_schema"]["schema"] == SCHEMA
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": False,
        "reasoning_effort": "low",
    }
    assert (payload["temperature"], payload["seed"]) == (0.0, 42)
    assert payload["cache_prompt"] is False  # repeatable answers
    assert payload["max_tokens"] == config.MAX_ANSWER_TOKENS
    assert "model" not in payload


def test_pictures_follow_the_user_s_text_as_png_addresses() -> None:
    payload = llm.chat_payload("s", "user text", SCHEMA, [b"\x89PNG", b"x"])

    assert payload["messages"][1]["content"] == [
        {"type": "text", "text": "user text"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw=="}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}},
    ]


@pytest.mark.parametrize("content", ['{"claims": [{"evid', '{"claims": []}'])
def test_a_cut_off_reply_is_a_clear_error_even_if_it_is_valid_json(
    monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    def post(url: str, **kwargs: Any) -> httpx.Response:
        body = chat_reply(content, finish_reason="length")
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.CLIENT, "post", post)

    with pytest.raises(llm.ModelServerError, match="did not finish"):
        llm.chat("s", "u", SCHEMA)


def test_a_finished_reply_that_is_not_json_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def post(url: str, **kwargs: Any) -> httpx.Response:
        body = chat_reply("not json")
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.CLIENT, "post", post)

    with pytest.raises(llm.ModelServerError, match="not valid JSON"):
        llm.chat("s", "u", SCHEMA)


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"finish_reason": "stop", "message": {}}]},
    ],
)
def test_a_malformed_server_response_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, body: object
) -> None:
    def post(url: str, **kwargs: Any) -> httpx.Response:
        if body is None:
            return httpx.Response(
                200, text="not json", request=httpx.Request("POST", url)
            )
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.CLIENT, "post", post)

    with pytest.raises(llm.ModelServerError, match="malformed response"):
        llm.chat("s", "u", SCHEMA)


def test_a_non_text_reply_content_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def post(url: str, **kwargs: Any) -> httpx.Response:
        body = chat_reply("")
        body["choices"][0]["message"]["content"] = None
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.CLIENT, "post", post)

    with pytest.raises(llm.ModelServerError, match="not valid JSON"):
        llm.chat("s", "u", SCHEMA)


def test_an_unavailable_generation_model_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def post(url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(llm.CLIENT, "post", post)

    with pytest.raises(llm.ModelServerError, match=config.CHAT_URL):
        llm.chat("s", "u", SCHEMA)


def test_embed_returns_vectors_in_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    def post(url: str, **kwargs: Any) -> httpx.Response:
        data = [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ]
        return httpx.Response(
            200, json={"data": data}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(llm.CLIENT, "post", post)

    assert llm.embed(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_a_longer_vector_is_cut_to_the_index_size_and_made_unit_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "EMBEDDING_DIMENSIONS", 2)

    assert llm.fitted([3.0, 4.0, 12.0]) == [0.6, 0.8]
    assert llm.fitted([0.0, 0.0, 1.0]) == [0.0, 0.0]  # nothing to scale
    assert llm.fitted([0.6, 0.8]) == [0.6, 0.8]  # already the index's size


def test_rerank_sends_the_query_and_documents_and_returns_scores_in_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: dict[str, Any] = {}

    def post(url: str, **kwargs: Any) -> httpx.Response:
        sent.update(url=url, **kwargs)
        results = [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": -2.5},
        ]
        return httpx.Response(
            200, json={"results": results}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(llm.CLIENT, "post", post)

    assert llm.rerank("question", ["first", "second"]) == [-2.5, 0.9]
    assert sent["url"] == config.RERANK_URL
    assert sent["json"] == {"query": "question", "documents": ["first", "second"]}


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"data": {}},
        {"data": [None]},
        {"data": [{}]},
        {"data": [{"index": "0", "embedding": [1.0]}]},
        {"data": [{"index": False, "embedding": [1.0]}]},
        {"data": []},
        {"data": [{"index": 1, "embedding": [1.0]}]},
        {"data": [{"index": 0}, {"index": 1, "embedding": [1.0]}]},
        {"data": [{"index": 0, "embedding": [1.0]}] * 2},
        {
            "data": [
                {"index": 0, "embedding": [1.0]},
                {"index": 1, "embedding": [1.0, 0.0]},
            ]
        },
        *[
            {
                "data": [
                    {"index": 0, "embedding": vector},
                    {"index": 1, "embedding": [1.0]},
                ]
            }
            for vector in (None, [], "1.0", [True], ["nan"], [float("inf")])
        ],
    ],
)
def test_malformed_embeddings_fail_before_retrieval_or_indexing(
    monkeypatch: pytest.MonkeyPatch, body: object
) -> None:
    def post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            text="not json" if body is None else json.dumps(body),
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(llm.CLIENT, "post", post)

    with pytest.raises(llm.ModelServerError, match="embedding.*malformed response"):
        llm.embed(["first passage", "second passage"])


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"results": {}},
        {"results": [{"index": 0}]},
        {"results": [{"index": "0", "relevance_score": 1}]},
        {"results": [{"relevance_score": 1}]},
    ],
)
def test_a_malformed_rerank_response_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, body: object
) -> None:
    def post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.CLIENT, "post", post)

    with pytest.raises(llm.ModelServerError, match="malformed response"):
        llm.rerank("question", ["first"])


def test_a_rerank_response_missing_a_passage_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def post(url: str, **kwargs: Any) -> httpx.Response:
        results = [{"index": 0, "relevance_score": 1.0}]
        return httpx.Response(
            200, json={"results": results}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(llm.CLIENT, "post", post)

    with pytest.raises(llm.ModelServerError, match="did not score every passage"):
        llm.rerank("question", ["first", "second"])


@pytest.mark.parametrize(
    "results",
    [
        [
            {"index": 0, "relevance_score": 1.0},
            {"index": 0, "relevance_score": 0.5},
        ],
        [
            {"index": 0, "relevance_score": 1.0},
            {"index": 2, "relevance_score": 0.5},
        ],
    ],
)
def test_rerank_requires_each_passage_index_exactly_once(
    monkeypatch: pytest.MonkeyPatch, results: list[dict[str, float | int]]
) -> None:
    def post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200, json={"results": results}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(llm.CLIENT, "post", post)

    with pytest.raises(llm.ModelServerError, match="did not score every passage"):
        llm.rerank("question", ["first", "second"])


def test_a_non_finite_rerank_score_is_malformed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results": [{"index": 0, "relevance_score": "nan"}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(llm.CLIENT, "post", post)

    with pytest.raises(llm.ModelServerError, match="malformed response"):
        llm.rerank("question", ["first"])


def test_unreachable_reranking_server_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def post(url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(llm.CLIENT, "post", post)

    with pytest.raises(llm.ModelServerError, match=config.RERANK_URL):
        llm.rerank("question", ["first"])


def test_unreachable_embedding_server_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def post(url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(llm.CLIENT, "post", post)

    with pytest.raises(llm.ModelServerError, match=config.EMBEDDING_URL):
        llm.embed(["question"])


def test_a_server_is_healthy_only_when_its_health_route_answers_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []
    statuses = {
        "http://127.0.0.1:8081/health": 200,
        "http://127.0.0.1:8080/health": 503,
    }

    def get(url: str, **kwargs: Any) -> httpx.Response:
        asked.append(url)
        return httpx.Response(statuses[url], request=httpx.Request("GET", url))

    monkeypatch.setattr(llm.CLIENT, "get", get)

    assert llm.healthy("http://127.0.0.1:8081/v1/embeddings") is True
    assert llm.healthy("http://127.0.0.1:8080/v1/chat/completions") is False  # loading
    assert asked == list(statuses)


def test_an_unreachable_server_is_not_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    def get(url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(llm.CLIENT, "get", get)

    assert llm.healthy(config.RERANK_URL) is False


def test_a_chat_request_is_traced_with_its_tokens_but_no_text(
    monkeypatch: pytest.MonkeyPatch, spans: InMemorySpanExporter
) -> None:
    def post(url: str, **kwargs: Any) -> httpx.Response:
        body = chat_reply(json.dumps({"claims": []}))
        body.update(
            model="qwen.gguf", usage={"prompt_tokens": 812, "completion_tokens": 9}
        )
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.CLIENT, "post", post)

    llm.chat("secret system text", "secret user text", SCHEMA)

    [chat] = spans.get_finished_spans()
    assert (chat.name, chat.kind) == ("chat", SpanKind.CLIENT)
    assert chat.attributes is not None
    attributes = dict(chat.attributes)
    assert attributes["gen_ai.operation.name"] == "chat"
    assert attributes["gen_ai.provider.name"] == "llama.cpp"
    assert attributes["gen_ai.response.model"] == "qwen.gguf"
    assert attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert attributes["gen_ai.usage.input_tokens"] == 812
    assert attributes["gen_ai.usage.output_tokens"] == 9
    assert attributes["server.port"] == 8080
    assert not any("secret" in str(value) for value in attributes.values())


def test_search_requests_are_traced_as_embeddings_and_rerank(
    monkeypatch: pytest.MonkeyPatch, spans: InMemorySpanExporter
) -> None:
    def post(url: str, **kwargs: Any) -> httpx.Response:
        if url == config.EMBEDDING_URL:
            body: dict[str, Any] = {"data": [{"index": 0, "embedding": [1.0]}]}
        else:
            body = {"results": [{"index": 0, "relevance_score": 1.0}]}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.CLIENT, "post", post)

    llm.embed(["question"])
    llm.rerank("question", ["passage"])

    names = [span.name for span in spans.get_finished_spans()]
    assert names == [f"embeddings {config.EMBEDDING_MODEL}", "rerank"]


def test_a_failed_model_request_ends_its_span_with_an_error(
    monkeypatch: pytest.MonkeyPatch,
    spans: InMemorySpanExporter,
    metric_points: Callable[[str], list[Any]],
) -> None:
    def post(url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(llm.CLIENT, "post", post)

    with pytest.raises(llm.ModelServerError):
        llm.chat("s", "u", SCHEMA)

    [chat] = spans.get_finished_spans()
    assert chat.status.status_code is StatusCode.ERROR
    assert chat.attributes is not None
    assert chat.attributes["error.type"] == "ModelServerError"
    [duration] = metric_points("gen_ai.client.operation.duration")
    assert duration.attributes["error.type"] == "ModelServerError"


def test_model_requests_are_measured_by_operation_with_their_tokens(
    monkeypatch: pytest.MonkeyPatch, metric_points: Callable[[str], list[Any]]
) -> None:
    def post(url: str, **kwargs: Any) -> httpx.Response:
        if url == config.EMBEDDING_URL:
            body: dict[str, Any] = {"data": [{"index": 0, "embedding": [1.0]}]}
        else:
            body = chat_reply(json.dumps({"claims": []}))
            body["usage"] = {"prompt_tokens": 812, "completion_tokens": 9}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.CLIENT, "post", post)

    llm.embed(["question"])
    llm.chat("s", "u", SCHEMA)

    durations = metric_points("gen_ai.client.operation.duration")
    operations = sorted(p.attributes["gen_ai.operation.name"] for p in durations)
    assert operations == ["chat", "embeddings"]
    [inputs] = metric_points("gen_ai.client.inference.operation.input_tokens")
    [outputs] = metric_points("gen_ai.client.inference.operation.output_tokens")
    assert (inputs.sum, outputs.sum) == (812, 9)


def test_model_requests_carry_the_key_when_one_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[dict[str, str]] = []

    def post(url: str, **kwargs: Any) -> httpx.Response:
        sent.append(kwargs["headers"])
        if url == config.EMBEDDING_URL:
            body: dict[str, Any] = {"data": [{"index": 0, "embedding": [1.0]}]}
        elif url == config.RERANK_URL:
            body = {"results": [{"index": 0, "relevance_score": 1.0}]}
        else:
            body = chat_reply(json.dumps({"claims": []}))
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.CLIENT, "post", post)
    monkeypatch.setattr(config, "MODEL_API_KEY", "")

    llm.embed(["question"])
    monkeypatch.setattr(config, "MODEL_API_KEY", "k3y")
    llm.embed(["question"])
    llm.rerank("question", ["passage"])
    llm.chat("s", "u", SCHEMA)

    bearer = {"Authorization": "Bearer k3y"}
    assert sent == [{}, bearer, bearer, bearer]


def test_the_judge_gets_the_same_request_at_its_own_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[str] = []

    def post(url: str, **kwargs: Any) -> httpx.Response:
        sent.append(url)
        body = chat_reply(json.dumps({"verdict": "supported"}))
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.CLIENT, "post", post)

    assert llm.judge("s", "u", SCHEMA) == {"verdict": "supported"}
    assert sent == [config.JUDGE_URL]
