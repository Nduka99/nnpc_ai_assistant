"""Traces, metrics and logs: a span for each answer stage and model call, metrics
for model-call durations, tokens and answer outcomes, and JSON log lines that carry
the current trace id.

Spans go over OTLP/HTTP to OTEL_EXPORTER_OTLP_TRACES_ENDPOINT and metrics to
OTEL_EXPORTER_OTLP_METRICS_ENDPOINT when they are set (deploy/compose.yaml runs
VictoriaTraces and VictoriaMetrics for this); otherwise nothing is sent. Every name is
in this module: the OpenTelemetry GenAI conventions are still "Development", so a
renamed attribute or metric is a one-place change. No question, passage or answer
text is recorded; that stays in the audit table, linked by the record id.

One answer's trace:

    POST /api/v1/answers          the request (FastAPI instrumentation)
      answer                      index version, status, record id, claims kept/removed,
                                  conversation id and turn
        understanding
          chat                    the emergency check: model, tokens, finish reason
        searching
          embeddings <model>
          rerank
        answering
          chat                    the answer request
"""

import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import FastAPI
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.metrics import Counter, Histogram, Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, SpanKind, get_current_span

from petroknowledge import config
from petroknowledge.records import Answer

Attributes = dict[str, str | int | float]  # the value types recorded here
RESOURCE = Resource.create({"service.name": "petroknowledge"})
PROVIDER = "llama.cpp"  # gen_ai.provider.name: every model server is llama-server
CHAT = {"gen_ai.operation.name": "chat", "gen_ai.provider.name": PROVIDER}
USAGE = {"prompt_tokens": "input_tokens", "completion_tokens": "output_tokens"}
# Histogram buckets the GenAI conventions recommend.
SECONDS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48,
           40.96, 81.92]  # fmt: skip
TOKENS = [4**n for n in range(14)]  # 1, 4, 16 … 67,108,864


@dataclass(frozen=True)
class Instruments:
    """Every metric this service records."""

    operation_duration: Histogram  # one request to a model server
    tokens: dict[str, Histogram]  # by USAGE kind, per chat request
    answers: Counter  # by status
    answer_duration: Histogram  # by status
    claims: Counter  # model-written claims, kept or removed by verification


def instruments(meter: Meter) -> Instruments:
    tokens = {}
    for kind in USAGE.values():
        tokens[kind] = meter.create_histogram(
            f"gen_ai.client.inference.operation.{kind}",
            unit="{token}",
            description=f"The {kind.replace('_', ' ')} of one model request.",
            explicit_bucket_boundaries_advisory=TOKENS,
        )
    return Instruments(
        operation_duration=meter.create_histogram(
            "gen_ai.client.operation.duration",
            unit="s",
            description="The duration of one request to a model server.",
            explicit_bucket_boundaries_advisory=SECONDS,
        ),
        tokens=tokens,
        answers=meter.create_counter(
            "petroknowledge.answers", unit="{answer}", description="Answers served."
        ),
        answer_duration=meter.create_histogram(
            "petroknowledge.answer.duration",
            unit="s",
            description="The time from a question to its recorded answer.",
            explicit_bucket_boundaries_advisory=SECONDS,
        ),
        claims=meter.create_counter(
            "petroknowledge.claims",
            unit="{claim}",
            description="Model-written claims, kept or removed by verification.",
        ),
    )


def tracer_provider() -> TracerProvider:
    """Spans for this service, sent on when an OTLP endpoint is configured."""
    provider = TracerProvider(resource=RESOURCE)
    if os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    return provider


def meter_provider() -> MeterProvider:
    """Metrics for this service, sent every minute (OTEL_METRIC_EXPORT_INTERVAL) when
    an OTLP endpoint is configured."""
    readers: list[MetricReader] = []
    if os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"):
        readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
    return MeterProvider(metric_readers=readers, resource=RESOURCE)


class JsonFormatter(logging.Formatter):
    """One JSON object per log line. Inside a span it carries `trace_id`, `span_id`
    and `trace_flags`, OpenTelemetry's stable fields for JSON logs, so a line leads
    to its trace."""

    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, str] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = get_current_span().get_span_context()
        if context.is_valid:
            line["trace_id"] = format(context.trace_id, "032x")
            line["span_id"] = format(context.span_id, "016x")
            line["trace_flags"] = format(context.trace_flags, "02x")
        if record.exc_info:
            line["exception"] = self.formatException(record.exc_info)
        return json.dumps(line)


# The server's logging: JSON lines on stdout. Where they are kept is the deployment's
# job (Twelve-Factor); in containers the engine collects stdout.
LOG_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"json": {"()": JsonFormatter}},
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "stream": "ext://sys.stdout",
        }
    },
    "root": {"handlers": ["stdout"], "level": "INFO"},
}

tracers = tracer_provider()
tracer = tracers.get_tracer("petroknowledge")
meters = meter_provider()
metrics = instruments(meters.get_meter("petroknowledge"))


def instrument(app: FastAPI) -> None:
    """A server span and duration metric for every request, parent of the answer's
    spans. Health probes and the ASGI per-message spans (one per streamed event) are
    left out as noise. So is the page: it carries the question in the query string,
    which a request span records (its answers are still traced)."""
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=tracers,
        meter_provider=meters,
        excluded_urls=r"healthz,readyz,://[^/]+/$",
        exclude_spans=["receive", "send"],
    )


@contextmanager
def span(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Attributes | None = None,
) -> Iterator[Span]:
    """A span around one step, current while it runs. An error ends it with the
    error status, the exception and `error.type`."""
    with tracer.start_as_current_span(name, kind=kind, attributes=attributes) as step:
        try:
            yield step
        except Exception as error:
            step.set_attribute("error.type", type(error).__qualname__)
            raise


@contextmanager
def model_span(
    name: str, operation: str, url: str, attributes: Attributes
) -> Iterator[Span]:
    """A client span for one request to a model server, and its duration."""
    server = httpx.URL(url)
    default_port = 443 if server.scheme == "https" else 80
    measured: Attributes = {
        "gen_ai.operation.name": operation,
        "gen_ai.provider.name": PROVIDER,
    }
    spanned: Attributes = {
        **measured,
        "server.address": server.host,
        "server.port": server.port or default_port,
        **attributes,
    }
    started = time.perf_counter()
    try:
        with span(name, SpanKind.CLIENT, spanned) as step:
            yield step
    except Exception as error:
        measured["error.type"] = type(error).__qualname__
        raise
    finally:
        metrics.operation_duration.record(time.perf_counter() - started, measured)


def chat_span() -> AbstractContextManager[Span]:
    # The request names no model: the server answers with whichever it loaded,
    # reported afterwards as gen_ai.response.model.
    return model_span(
        "chat",
        "chat",
        config.CHAT_URL,
        {
            "gen_ai.output.type": "json",
            "gen_ai.request.temperature": config.TEMPERATURE,
            "gen_ai.request.seed": config.SEED,
            "gen_ai.request.max_tokens": config.MAX_ANSWER_TOKENS,
        },
    )


def embeddings_span() -> AbstractContextManager[Span]:
    return model_span(
        f"embeddings {config.EMBEDDING_MODEL}",
        "embeddings",
        config.EMBEDDING_URL,
        {"gen_ai.request.model": config.EMBEDDING_MODEL},
    )


def rerank_span() -> AbstractContextManager[Span]:
    # Reranking has no GenAI operation name yet; the conventions allow a custom one.
    return model_span("rerank", "rerank", config.RERANK_URL, {})


def record_reply(model: object, finish_reason: object, usage: object) -> None:
    """What the server reports about a chat reply, on the current span, and its token
    counts as metrics. Values of an unexpected type are skipped: the reply itself is
    validated by the caller."""
    step = get_current_span()
    if isinstance(model, str):
        step.set_attribute("gen_ai.response.model", model)
    if isinstance(finish_reason, str):
        step.set_attribute("gen_ai.response.finish_reasons", [finish_reason])
    if isinstance(usage, dict):
        for key, kind in USAGE.items():
            count = usage.get(key)
            if type(count) is int:
                step.set_attribute(f"gen_ai.usage.{kind}", count)
                metrics.tokens[kind].record(count, CHAT)


def record_answer(
    result: Answer, answer_id: int, index_version: int, seconds: float
) -> None:
    """The outcome of an answer: on the current span with its audit record id, and
    as metrics by status."""
    get_current_span().set_attributes(
        {
            "petroknowledge.index.version": index_version,
            "petroknowledge.answer.id": answer_id,
            "petroknowledge.answer.status": result.status,
            "petroknowledge.claims.kept": len(result.claims),
            "petroknowledge.claims.removed": len(result.rejected),
        }
    )
    status = {"petroknowledge.answer.status": result.status}
    metrics.answers.add(1, status)
    metrics.answer_duration.record(seconds, status)
    metrics.claims.add(len(result.claims), {"petroknowledge.claim.outcome": "kept"})
    metrics.claims.add(
        len(result.rejected), {"petroknowledge.claim.outcome": "removed"}
    )


def record_turn(conversation_id: str, turn: int) -> None:
    """An answer's place in a conversation, on the current span. The conversation id
    is the GenAI convention's (`gen_ai.conversation.id`); no message text is sent."""
    get_current_span().set_attributes(
        {"gen_ai.conversation.id": conversation_id, "petroknowledge.turn": turn}
    )
