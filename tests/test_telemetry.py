"""Traces and metrics: export over OTLP/HTTP, errors on spans, reply details and
answer outcomes."""

import json
import logging
import sys
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from petroknowledge import config, telemetry
from petroknowledge.records import Answer

Received = list[tuple[str, str]]


@pytest.fixture
def collector() -> Iterator[tuple[str, Received]]:
    """A local OTLP/HTTP endpoint: its base URL, and the path and content type of
    every request it receives."""
    received: Received = []

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, self.headers["Content-Type"]))
            self.send_response(200)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            """Keep the test output quiet."""

    server = HTTPServer(("127.0.0.1", 0), Collector)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", received
    server.shutdown()


def test_spans_are_sent_over_otlp_http_when_an_endpoint_is_set(
    monkeypatch: pytest.MonkeyPatch, collector: tuple[str, Received]
) -> None:
    base, received = collector
    path = "/insert/opentelemetry/v1/traces"
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", base + path)

    provider = telemetry.tracer_provider()
    with provider.get_tracer("test").start_as_current_span("answer"):
        pass
    provider.shutdown()  # sends what is waiting

    assert received == [(path, "application/x-protobuf")]


def test_metrics_are_sent_over_otlp_http_when_an_endpoint_is_set(
    monkeypatch: pytest.MonkeyPatch, collector: tuple[str, Received]
) -> None:
    base, received = collector
    path = "/opentelemetry/v1/metrics"
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", base + path)

    provider = telemetry.meter_provider()
    provider.get_meter("test").create_counter("answers").add(1)
    provider.shutdown()  # sends what is waiting

    assert received == [(path, "application/x-protobuf")]


def test_an_answer_is_counted_by_status_with_its_claims(
    metric_points: Callable[[str], list[Any]], answered: Answer
) -> None:
    telemetry.record_answer(answered, 7, 4, 12.5)

    [answers] = metric_points("petroknowledge.answers")
    assert (dict(answers.attributes), answers.value) == (
        {"petroknowledge.answer.status": "answered"},
        1,
    )
    [duration] = metric_points("petroknowledge.answer.duration")
    assert duration.sum == 12.5
    claims = {}
    for point in metric_points("petroknowledge.claims"):
        claims[point.attributes["petroknowledge.claim.outcome"]] = point.value
    assert claims == {"kept": 2, "removed": 1}


def test_an_error_marks_the_span_with_its_type(spans: InMemorySpanExporter) -> None:
    with pytest.raises(ValueError), telemetry.span("searching"):
        raise ValueError("no passages")

    [searching] = spans.get_finished_spans()
    assert searching.status.status_code is StatusCode.ERROR
    assert searching.attributes is not None
    assert searching.attributes["error.type"] == "ValueError"


def test_reply_details_of_an_unexpected_type_are_skipped(
    spans: InMemorySpanExporter,
) -> None:
    with telemetry.chat_span():
        telemetry.record_reply(["not", "a", "name"], None, {"prompt_tokens": "12"})

    [chat] = spans.get_finished_spans()
    assert chat.attributes is not None
    for name in ["gen_ai.response.model", "gen_ai.response.finish_reasons"]:
        assert name not in chat.attributes
    assert "gen_ai.usage.input_tokens" not in chat.attributes


def test_an_https_model_server_defaults_to_port_443(
    monkeypatch: pytest.MonkeyPatch, spans: InMemorySpanExporter
) -> None:
    monkeypatch.setattr(config, "RERANK_URL", "https://rerank.test/v1/rerank")

    with telemetry.rerank_span():
        pass

    [rerank] = spans.get_finished_spans()
    assert rerank.attributes is not None
    assert (rerank.attributes["server.address"], rerank.attributes["server.port"]) == (
        "rerank.test",
        443,
    )


def log_line(message: str, exc_info: Any = None) -> dict[str, str]:
    record = logging.LogRecord(
        "petroknowledge", logging.ERROR, "", 0, message, (), exc_info
    )
    parsed: dict[str, str] = json.loads(telemetry.JsonFormatter().format(record))
    return parsed


def test_a_log_line_inside_a_span_carries_its_trace_id(
    spans: InMemorySpanExporter,
) -> None:
    with telemetry.span("answer") as step:
        line = log_line("model server slow")

    context = step.get_span_context()
    assert line["message"] == "model server slow"
    assert (line["level"], line["logger"]) == ("ERROR", "petroknowledge")
    assert line["trace_id"] == format(context.trace_id, "032x")
    assert line["span_id"] == format(context.span_id, "016x")
    assert line["trace_flags"] == format(context.trace_flags, "02x")
    assert line["time"].endswith("+00:00")


def test_a_log_line_outside_a_span_has_no_trace_id_but_keeps_the_error() -> None:
    try:
        raise ValueError("no index")
    except ValueError:
        line = log_line("startup failed", sys.exc_info())

    assert "trace_id" not in line
    assert line["exception"].endswith("ValueError: no index")


def test_an_answers_place_in_a_conversation_is_on_its_span(
    spans: InMemorySpanExporter,
) -> None:
    with telemetry.span("answer"):
        telemetry.record_turn("conversation-1", 3)

    [answer] = spans.get_finished_spans()
    assert answer.attributes is not None
    assert answer.attributes["gen_ai.conversation.id"] == "conversation-1"
    assert answer.attributes["petroknowledge.turn"] == 3


def test_requests_to_an_app_are_instrumented() -> None:
    from fastapi import FastAPI

    app = FastAPI()
    telemetry.instrument(app)

    # The instrumentor marks every app it wraps.
    assert getattr(app, "_is_instrumented_by_opentelemetry", False) is True
