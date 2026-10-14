"""A stand-in web for the collection tests: an invented company site, a news feed
and hosts whose robots.txt is missing, broken or unreachable. No test reaches the
network or sleeps."""

import json
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import psycopg
import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from psycopg import sql

from petroknowledge import config, fetch, store, telemetry
from petroknowledge.records import Answer, Claim, Evidence, Passage, Rejection

SITE = "https://example.test/"
NEWS = "https://news.test/"
PDF = b"%PDF-1.7 an annual report"
SITEMAP = (
    '<?xml version="1.0"?>'
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    "<url><loc> https://example.test/about </loc><lastmod>2026-10-01</lastmod></url>"
    "<url><loc>https://example.test/contact</loc></url>"
    "<url><lastmod>2026-10-01</lastmod></url>"
    "</urlset>"
)
POSTS = (
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    "<url><loc>https://example.test/news/one</loc><lastmod>2026-10-01</lastmod></url>"
    "<url><loc>https://example.test/news/two</loc></url>"
    "<url><loc>https://example.test/news/three</loc></url>"
    "<url><loc>https://example.test/news/gone</loc></url>"
    "</urlset>"
)
ABOUT = (
    '<a href="/docs/report.pdf">Annual report</a>'
    '<a href="https://cdn.example.test/docs/b.PDF">b</a>'
    '<a href="https://other.test/x.pdf">elsewhere</a><a href="/contact">page</a>'
    '<img src="/img/a.png"><img data-src="/img/lazy.webp"><img alt="no source">'
    '<img src="/img/icon.svg"><img src="https://tracker.test/pixel.gif">'
    "<div style=\"background-image:url('/img/banner.jpg?v=1')\"></div>"
)
POST_ONE = (
    '<head><meta property="og:title" content="NNPC  signs deal">'
    '<meta name="description" content="A gas deal.">'
    '<meta property="article:published_time" content="2026-10-01T09:00:00Z"></head>'
)
POST_TWO = (
    "<head><title> Acme results </title>"
    '<meta property="og:description" content="Half-year figures."></head>'
    '<body><time datetime="2026-09-30">30 September</time></body>'
)
FEED = (
    "<rss><channel>"
    "<item><title>Acme Oil &amp; NNPC sign deal</title>"
    "<link>https://news.test/a</link>"
    "<pubDate>Fri, 09 Oct 2026 10:00:00 +0100</pubDate>"
    "<description><![CDATA[<p>The <b>gas</b> deal.</p>]]></description></item>"
    "<item><title>Football results</title><link>https://news.test/b</link>"
    "<pubDate>not a date</pubDate></item>"
    "<item><title>No link here, Acme Oil</title></item>"
    "<item><title>Beta Energy expands</title><link>https://news.test/c</link></item>"
    "</channel></rss>"
)


def handler(request: httpx.Request) -> httpx.Response:
    host, path = request.url.host, request.url.path
    if host == "down.test":
        raise httpx.ConnectError("refused")
    if path == "/robots.txt":
        if host == "broken.test":
            return httpx.Response(500)
        if host == "norules.test":
            return httpx.Response(404)
        if host == "slow.test":
            return httpx.Response(200, text="User-agent: *\nCrawl-delay: 5\n")
        return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
    pages = {
        "/sitemap-pages.xml": SITEMAP,
        "/sitemap-posts.xml": POSTS,
        "/about": ABOUT,
        "/contact": "<p>Contact us</p>",
        "/news/one": POST_ONE,
        "/news/two": POST_TWO,
        "/news/three": "<p>no metadata</p>",
        "/feed/": FEED,
    }
    if path in pages:
        return httpx.Response(200, text=pages[path])
    if path in ("/docs/report.pdf", "/docs/b.PDF"):
        headers = {"content-type": "application/pdf", "etag": '"v1"'}
        return httpx.Response(200, content=PDF, headers=headers)
    if path == "/docs/moved.pdf":
        return httpx.Response(301, headers={"location": SITE + "docs/report.pdf"})
    if path == "/docs/offsite.pdf":
        return httpx.Response(301, headers={"location": "https://elsewhere.test/x.pdf"})
    if path == "/x.pdf":
        return httpx.Response(200, content=PDF)
    if path == "/docs/huge.pdf":
        return httpx.Response(200, content=b"x" * 1001)  # states its length
    if path == "/docs/endless.pdf":
        return httpx.Response(200, content=iter([b"x" * 600, b"x" * 600]))  # does not
    if path == "/docs/boom.pdf":
        raise httpx.ConnectError("refused")
    if path in ("/img/a.png", "/img/unsized.webp"):
        return httpx.Response(200, content=b"")
    return httpx.Response(404)


def company(company_id: str = "acme", **changes: Any) -> dict[str, Any]:
    found: dict[str, Any] = {
        "id": company_id,
        "name": "Acme Limited",
        "aliases": ["Acme Oil"],
        "terms": "invented company for tests",
        "allowed_hosts": ["example.test"],
        "own_hosts": ["example.test"],
        "max_file_mb": 1,
        "core_sitemaps": [SITE + "sitemap-pages.xml", SITE + "private/sitemap.xml"],
        "post_sitemaps": [],
        "core_urls": [
            {
                "url": SITE + "docs/report.pdf",
                "kind": "document",
                "licence": "CC BY 4.0",
            }
        ],
        "link_sitemaps": [SITE + "sitemap-posts.xml", SITE + "private/posts.xml"],
    }
    found.update(changes)
    return found


@pytest.fixture
def web(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Point every store at tmp_path, answer every request from `handler`, never
    sleep; yields the requests made, as "METHOD host/path"."""
    requested: list[str] = []

    def record(request: httpx.Request) -> httpx.Response:
        requested.append(f"{request.method} {request.url.host}{request.url.path}")
        return handler(request)

    monkeypatch.setattr(config, "COMPANIES_DIR", tmp_path / "companies")
    monkeypatch.setattr(config, "NEWS_SOURCES", tmp_path / "companies" / "news.json")
    monkeypatch.setattr(config, "FILE_STORE", tmp_path / "files")
    monkeypatch.setattr(config, "LINK_REGISTRY", tmp_path / "links")
    monkeypatch.setattr(config, "MEASUREMENTS", tmp_path / "measurements")
    monkeypatch.setattr(config, "INBOX", tmp_path / "inbox")
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    real_client = fetch.polite_client
    monkeypatch.setattr(
        fetch,
        "polite_client",
        lambda transport=None, hosts=None: real_client(
            httpx.MockTransport(record), hosts
        ),
    )
    (tmp_path / "companies").mkdir()
    yield requested


@pytest.fixture(scope="session")
def span_store() -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    telemetry.tracers.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


@pytest.fixture
def spans(span_store: InMemorySpanExporter) -> InMemorySpanExporter:
    """Every span finished during the test, oldest first."""
    span_store.clear()
    return span_store


MetricPoints = Callable[[str], list[Any]]


@pytest.fixture
def metric_points(monkeypatch: pytest.MonkeyPatch) -> MetricPoints:
    """A function giving the data points recorded so far for one metric name."""
    reader = InMemoryMetricReader()
    meters = MeterProvider(metric_readers=[reader])
    monkeypatch.setattr(
        telemetry, "metrics", telemetry.instruments(meters.get_meter("t"))
    )

    def points(name: str) -> list[Any]:
        found: list[Any] = []
        data = reader.get_metrics_data()
        for resource in data.resource_metrics if data else []:
            for scope in resource.scope_metrics:
                for metric in scope.metrics:
                    if metric.name == name:
                        found.extend(metric.data.data_points)
        return found

    return points


# An invented answer for the telemetry tests: two kept claims and a removed one.
PROFILE = Passage(
    1,
    SITE + "about",
    "About Acme",
    "Profile",
    "Profile\nAcme Oil produces 5,000 barrels a day.",
    "2026-10-10T10:00:00+00:00",
)


@pytest.fixture
def answered() -> Answer:
    quote = "Acme Oil produces 5,000 barrels a day."
    evidence = Evidence(
        1,
        PROFILE.url,
        PROFILE.title,
        PROFILE.heading,
        quote,
        PROFILE.url,
        PROFILE.fetched_at,
    )
    return Answer(
        "How much does Acme produce?",
        "answered",
        "",
        (
            Claim("Acme produces 5,000 barrels a day.", (evidence,)),
            Claim("Acme Oil is a producer.", (evidence,)),
        ),
        (PROFILE,),
        (Rejection("Acme is the largest producer.", "quote not in S1"),),
    )


def write_company(folder: Path, found: dict[str, Any]) -> None:
    """companies/<id>/sources.json under `folder`."""
    (folder / found["id"]).mkdir(parents=True, exist_ok=True)
    (folder / found["id"] / "sources.json").write_text(json.dumps(found))


def write_sources(*companies: dict[str, Any]) -> None:
    for found in companies:
        write_company(config.COMPANIES_DIR, found)
    feeds = {"feeds": [{"name": "News", "url": NEWS + "feed/"}]}
    config.NEWS_SOURCES.write_text(json.dumps(feeds), encoding="utf-8")


# Postgres tests run against the development server started with
# `docker compose -f deploy/compose.yaml --profile dev up -d` (settings in
# deploy/.env), in a throwaway database, so development data is never touched.
REPOSITORY = Path(__file__).parent.parent
MIGRATIONS = REPOSITORY / "db" / "migrations"


def server_url(database: str) -> str:
    settings = {}
    for line in (
        (REPOSITORY / "deploy" / ".env").read_text(encoding="utf-8").splitlines()
    ):
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            settings[key] = value
    user, password = settings["POSTGRES_USER"], settings["POSTGRES_PASSWORD"]
    return f"postgresql://{user}:{password}@127.0.0.1:5433/{database}"


def migration_up(path: Path) -> str:
    """The SQL between a dbmate migration's `-- migrate:up` and `-- migrate:down`."""
    text = path.read_text(encoding="utf-8")
    return text.split("-- migrate:up", 1)[1].split("-- migrate:down", 1)[0]


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    """A fresh database with every migration applied, dropped after the session."""
    name = f"petroknowledge_test_{uuid.uuid4().hex[:8]}"
    try:
        admin = psycopg.connect(server_url("postgres"), autocommit=True)
    except psycopg.OperationalError as error:
        pytest.fail(
            "the development Postgres is not running; start it with "
            f"`docker compose -f deploy/compose.yaml --profile dev up -d` ({error})"
        )
    with admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    with psycopg.connect(server_url(name)) as conn:
        for path in sorted(MIGRATIONS.glob("*.sql")):
            conn.execute(migration_up(path).encode())
    yield server_url(name)
    with psycopg.connect(server_url("postgres"), autocommit=True) as admin:
        admin.execute(
            sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name))
        )


@pytest.fixture
def pg(postgres_url: str) -> Iterator[store.Connection]:
    """A connection to the test database, emptied after each test."""
    with psycopg.connect(postgres_url) as conn:
        yield conn
        conn.rollback()
        conn.execute(
            "TRUNCATE answer_feedback, question_failures, model_calls, answers, "
            "conversations, structure, unit_passages, units, role_statements, "
            "role_readings, "
            "passages, index_versions, documents, images RESTART IDENTITY"
        )
        # Each index version has its own BM25 indexes; the next test starts at 1.
        for (name,) in conn.execute(
            "SELECT indexname FROM pg_indexes WHERE indexname LIKE 'passages_bm25_v%'"
        ).fetchall():
            conn.execute(sql.SQL("DROP INDEX {}").format(sql.Identifier(name)))
