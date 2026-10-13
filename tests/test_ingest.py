"""An index version built from the corpus and written to a throwaway Postgres (the
`pg` fixture). Every source, passage, picture and vector here is invented."""

import json
from pathlib import Path
from typing import Any

import pytest

from petroknowledge import cli, config, corpus, ingest, llm, store

SITE = "https://acme.test/"
DOCUMENTS = [
    {"url": SITE + "about", "title": "About Acme",
     "fetched_at": "2026-10-10T10:00:00+00:00", "sha256": "a" * 64},
]  # fmt: skip
PICTURE = "c" * 64


def row(heading: str, text: str, **changes: Any) -> dict[str, Any]:
    found = {"url": SITE + "about", "title": "About Acme", "heading": heading,
             "text": text, "context": "", "page": None, "kind": "page",
             "channel": "company", "quotable": True, "published": "",
             "image": ""}  # fmt: skip
    return found | changes


ROWS = [
    row("Refining", "Acme refines crude oil."),
    row("Gas", "Acme sells gas."),
    row("Picture", "", context="Picture\nFlare stack", kind="picture",
        channel="picture", image=PICTURE),
]  # fmt: skip
UNITS = [{"name": "Acme Refining Limited", "acronym": "", "url": SITE + "about"}]


@pytest.fixture
def corpus_built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """The corpus above, its picture stored with a SigLIP2 vector; yields each
    embedding request's texts."""
    monkeypatch.setattr(corpus, "build", lambda company_id, compiled: (ROWS, DOCUMENTS))
    monkeypatch.setattr(corpus, "listed_units", lambda company_id: UNITS)
    monkeypatch.setattr(config, "IMAGES", tmp_path)
    (tmp_path / f"{PICTURE}.png").write_bytes(b"png")
    vectors = {PICTURE: [1.0] + [0.0] * 1151, "d" * 64: [0.0, 1.0]}
    (tmp_path / ingest.PICTURE_VECTORS).write_text(json.dumps(vectors))
    monkeypatch.setattr(config, "EMBEDDING_BATCH_SIZE", 2)
    return []


def fake_embed(requests: list[list[str]]) -> Any:
    def embed(texts: list[str]) -> list[list[float]]:
        requests.append(texts)
        return [[1.0, float(len(text))] + [0.0] * 1022 for text in texts]

    return embed


def test_a_build_embeds_each_new_passage_once_and_stores_its_pictures(
    pg: store.Connection, corpus_built: list[list[str]]
) -> None:
    first, manifest = ingest.build(pg, "acme", fake_embed(corpus_built))

    assert corpus_built == [
        ["About Acme\nAcme refines crude oil.", "About Acme\nAcme sells gas."],
        ["About Acme\nPicture\nFlare stack"],
    ]
    assert {key: manifest[key] for key in ("company", "sources", "passages",
            "embedded", "pictures", "picture_vectors")} == {
        "company": "acme", "sources": "1", "passages": "3", "embedded": "3",
        "pictures": "1", "picture_vectors": "1",
    }  # fmt: skip
    assert manifest["units"] == "1"
    (unit,) = store.version_units(pg, first)
    scope = pg.execute("SELECT count(*) FROM unit_passages WHERE unit_id = %s",
                       (unit["id"],)).fetchone()  # fmt: skip
    assert scope == (3,)  # every passage of the unit's own page
    assert store.live_version(pg) is None
    assert store.picture_ranking(pg, first, [1.0] + [0.0] * 1151, 5) != []

    second, again = ingest.build(pg, "acme", fake_embed(corpus_built), live=True)

    assert again["embedded"] == "0"  # every vector reused
    assert len(corpus_built) == 2
    assert again["passages_sha256"] == manifest["passages_sha256"]
    assert store.live_version(pg) == (second, config.EMBEDDING_MODEL)


def test_fingerprints_follow_the_sources_and_every_passage_field() -> None:
    other = {"url": SITE + "b", "title": "B", "fetched_at": "", "sha256": "b" * 64}

    assert ingest.corpus_hash([*DOCUMENTS, other]) == ingest.corpus_hash(
        [other, *DOCUMENTS]
    )
    changed = [ROWS[0] | {"quotable": False}, *ROWS[1:]]
    assert ingest.passages_hash(changed) != ingest.passages_hash(ROWS)


def test_pictures_without_vectors_are_stored_without_them(tmp_path: Path) -> None:
    (tmp_path / f"{PICTURE}.png").write_bytes(b"png")

    assert ingest.stored_pictures([PICTURE], tmp_path) == ({PICTURE: b"png"}, {})


def test_the_index_command_builds_a_version_and_can_make_it_live(
    postgres_url: str,
    pg: store.Connection,
    corpus_built: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(config, "DATABASE_URL", "")
    assert cli.main(["index", "acme"]) == 1
    assert "PETROKNOWLEDGE_DATABASE_URL" in capsys.readouterr().err
    monkeypatch.setattr(config, "DATABASE_URL", postgres_url)
    monkeypatch.setattr(llm, "healthy", lambda url: False)
    assert cli.main(["index", "acme"]) == 1
    assert "no embedding model ready" in capsys.readouterr().err
    monkeypatch.setattr(llm, "healthy", lambda url: True)
    monkeypatch.setattr(llm, "embed", fake_embed(corpus_built))

    monkeypatch.setattr(llm, "healthy", lambda url: url != config.CHAT_URL)
    assert cli.main(["index", "acme", "--read-roles"]) == 1
    assert "no chat model ready" in capsys.readouterr().err
    monkeypatch.setattr(llm, "healthy", lambda url: True)
    monkeypatch.setattr(llm, "chat", lambda s, u, schema: {"statements": []})

    built: list[bool] = []

    def build(company_id: str, compiled: bool) -> tuple[Any, Any]:
        built.append(compiled)
        return ROWS, DOCUMENTS

    monkeypatch.setattr(corpus, "build", build)
    assert cli.main(["index", "acme", "--live", "--read-roles"]) == 0

    out = capsys.readouterr().out
    assert "3 passages embedded" in out
    assert json.loads(out[out.index("{") : out.rindex("}") + 1])["passages"] == "3"
    assert out.rstrip().endswith("index version 1: live")
    assert store.live_version(pg) == (1, config.EMBEDDING_MODEL)
    assert cli.main(["index", "acme", "--compiled-units"]) == 0
    assert built == [False, True]  # the compiled list only on request (K27)


def test_role_statements_are_read_once_per_text_and_dated(
    pg: store.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = row("Board", "Ada Obi, Chief Executive Officer of Acme, spoke.")
    monkeypatch.setattr(
        corpus, "build", lambda company_id, compiled: ([board], DOCUMENTS)
    )
    monkeypatch.setattr(corpus, "listed_units", lambda company_id: [])
    monkeypatch.setattr(config, "IMAGES", tmp_path)
    said = {"person": "Ada Obi", "role": "Chief Executive Officer",
            "organisation": "Acme",
            "quote": "Ada Obi, Chief Executive Officer"}  # fmt: skip
    asked: list[str] = []

    def chat(system: str, user: str, schema: dict[str, Any]) -> object:
        asked.append(user)
        return {"statements": [said]}

    first, manifest = ingest.build(pg, "acme", fake_embed([]), chat=chat)

    assert (manifest["role_statements"], manifest["role_passages_read"]) == ("1", "1")
    (statement,) = store.version_role_statements(pg, first)
    assert (statement["person"], statement["stated_on"], statement["date_kind"]) == (
        "Ada Obi", "2026-10-10", "captured",
    )  # fmt: skip
    _, again = ingest.build(pg, "acme", fake_embed([]), chat=chat)
    assert (again["role_passages_read"], again["role_statements"]) == ("0", "1")
    assert len(asked) == 1  # the kept reading is reused
    added = row("Board", "Ben Eze, Chief Financial Officer, joined.")
    monkeypatch.setattr(
        corpus, "build", lambda c, compiled: ([board, added], DOCUMENTS)
    )
    _, unread = ingest.build(pg, "acme", fake_embed([]))
    assert unread["role_passages_unread"] == "1"
    _, broken = ingest.build(pg, "acme", fake_embed([]),
                             chat=lambda s, u, schema: {"rows": []})  # fmt: skip
    assert broken["role_replies_broken"] == "1"
