"""The Postgres index, tested against a real Postgres 18 with pgvector and
pg_textsearch in a throwaway database (the `pg` fixture). Every source, passage and
vector here is invented."""

import math
import uuid
from datetime import date
from typing import Any

import psycopg
import pytest

from petroknowledge import config, store

SITE = "https://acme.test/"
DOCUMENTS = [
    {"url": SITE + "about", "title": "About Acme",
     "fetched_at": "2026-10-10T10:00:00+00:00", "sha256": "sha-about"},
    {"url": SITE + "news/pipeline", "title": "Pipeline Opens",
     "fetched_at": "2026-10-10T11:30:00+01:00", "sha256": "sha-news"},
]  # fmt: skip
GUIDE = {"url": "https://wiki.test/Acme", "title": "Acme - Wikipedia",
         "fetched_at": "2026-10-10T10:00:00+00:00", "sha256": "sha-wiki"}  # fmt: skip


def row(
    url: str, title: str, heading: str, text: str, **changes: Any
) -> dict[str, Any]:
    """A corpus row as `corpus.build` makes it."""
    found = {"url": url, "title": title, "heading": heading, "text": text,
             "context": "", "page": None, "kind": "page", "channel": "company",
             "quotable": True, "published": "", "image": ""}  # fmt: skip
    return found | changes


ROWS = [
    row(SITE + "about", "About Acme", "Refining",
        "Refining\nAcme refines crude oil at Port Harcourt."),
    row(SITE + "news/pipeline", "Pipeline Opens", "Route",
        "Route\nThe pipeline runs 614 km.", kind="post", published="2024-03-12"),
    row(SITE + "news/pipeline", "Pipeline Opens", "Gas",
        "Gas\nThe line carries three gas grades.", kind="post",
        published="2024-03-12"),
]  # fmt: skip
# Two dimensions padded to the column's 1024; each passage points a different way.
VECTORS = [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]
MANIFEST = {
    "corpus_sha256": "corpus",
    "passages_sha256": "passages",
    "embedding_model": "test-embedder",
}


def padded(vector: list[float], size: int = 1024) -> list[float]:
    return vector + [0.0] * (size - len(vector))


def build(pg: store.Connection) -> int:
    return store.write_version(
        pg, DOCUMENTS, ROWS, [padded(v) for v in VECTORS], MANIFEST
    )


def headings(pg: store.Connection, ids: list[int]) -> list[str]:
    return [p.heading for p in store.load_passages(pg, ids)]


def test_vectors_are_sent_as_unit_length_text() -> None:
    text = store.vector_text([3.0, 4.0])

    assert text == "[0.6,0.8]"
    values = [float(x) for x in text.strip("[]").split(",")]
    assert math.isclose(sum(v * v for v in values), 1.0)


def test_a_new_version_keeps_each_passage_s_fields_and_is_not_live(
    pg: store.Connection,
) -> None:
    version = build(pg)

    rows = pg.execute(
        "SELECT heading, kind, channel, quotable, published::text, page "
        "FROM passages WHERE index_version_id = %s ORDER BY id",
        (version,),
    ).fetchall()
    assert rows == [
        ("Refining", "page", "company", True, None, None),
        ("Route", "post", "company", True, "2024-03-12", None),
        ("Gas", "post", "company", True, "2024-03-12", None),
    ]
    assert store.live_version(pg) is None
    assert store.index_version(pg, version) == (version, "test-embedder")
    assert store.index_version(pg, version + 1) is None
    assert store.passage_count(pg, version) == 3
    assert not store.has_pictures(pg, version)


def test_an_unchanged_source_is_stored_once_across_versions(
    pg: store.Connection,
) -> None:
    first = build(pg)
    second = build(pg)

    assert second != first
    assert pg.execute("SELECT count(*) FROM documents").fetchone() == (2,)
    assert pg.execute("SELECT count(*) FROM passages").fetchone() == (6,)


def test_one_version_is_live_at_a_time(pg: store.Connection) -> None:
    first = build(pg)
    second = build(pg)

    store.set_live(pg, first)
    assert store.live_version(pg) == (first, "test-embedder")
    store.set_live(pg, second)
    assert store.live_version(pg) == (second, "test-embedder")
    with pytest.raises(ValueError, match="no index version 999"):
        store.set_live(pg, 999)
    assert store.live_version(pg) == (second, "test-embedder")


def test_keyword_ranking_matches_any_stemmed_word(pg: store.Connection) -> None:
    version = build(pg)

    # "refining" stems to "refin", as "refines" does; "grades" to "grade".
    found = store.keyword_ranking(pg, version, "refining grade", 10)

    assert set(headings(pg, found)) == {"Refining", "Gas"}


def test_keyword_search_keeps_stop_words_such_as_not(pg: store.Connection) -> None:
    [(words,)] = pg.execute(
        "SELECT to_tsvector(%s, 'Do not flare gas')::text", (store.KEYWORD_CONFIG,)
    ).fetchall()

    # Stemmed like `english`, but "do" and "not" are kept: "not" carries meaning.
    assert words == "'do':1 'flare':3 'gas':4 'not':2"


def test_keyword_ranking_without_a_matching_word_finds_nothing(
    pg: store.Connection,
) -> None:
    version = build(pg)

    assert store.keyword_ranking(pg, version, "?!", 10) == []
    assert store.keyword_ranking(pg, version, "zebra xylophone", 10) == []


def test_keyword_ranking_reads_query_syntax_as_plain_words(
    pg: store.Connection,
) -> None:
    version = build(pg)

    found = store.keyword_ranking(pg, version, 'NOT "Route" OR (refining)?', 10)

    assert set(headings(pg, found)) == {"Route", "Refining"}


def bm25_names(pg: store.Connection) -> set[str]:
    rows = pg.execute(
        "SELECT indexname FROM pg_indexes WHERE indexname LIKE 'passages_bm25_v%'"
    ).fetchall()
    return {name for (name,) in rows}


def test_each_version_has_a_bm25_index_per_channel_removed_with_it(
    pg: store.Connection,
) -> None:
    first = build(pg)
    second = build(pg)
    store.set_live(pg, second)

    assert bm25_names(pg) == {
        store.bm25_index(version, channel)
        for version in (first, second)
        for channel in store.CHANNELS
    }
    store.delete_version(pg, first)
    assert pg.execute("SELECT count(*) FROM passages").fetchone() == (3,)
    assert bm25_names(pg) == {store.bm25_index(second, c) for c in store.CHANNELS}
    assert store.keyword_ranking(pg, second, "pipeline", 10) != []
    with pytest.raises(ValueError, match="not live"):
        store.delete_version(pg, second)


def test_guidance_passages_do_not_move_the_company_keyword_scores(
    pg: store.Connection,
) -> None:
    # Words the company passages share, repeated, change BM25's word statistics.
    extra = [
        row(GUIDE["url"], GUIDE["title"], f"Part {n}", "pipeline gas pipeline gas",
            channel="guidance")
        for n in range(6)
    ]  # fmt: skip
    alone = build(pg)
    beside = store.write_version(
        pg, [*DOCUMENTS, GUIDE], [*ROWS, *extra],
        [padded(v) for v in [*VECTORS, *[[0.5, 0.5]] * 6]], MANIFEST,
    )  # fmt: skip

    def scores(version: int) -> list[tuple[str, float]]:
        rows = pg.execute(
            "SELECT heading, (title || ' ' || context || ' ' || text) "
            "<@> to_bm25query('pipeline gas', %s) FROM passages "
            "WHERE index_version_id = %s AND channel = 'company' ORDER BY heading",
            (store.bm25_index(version, "company"), version),
        ).fetchall()
        return [(heading, round(score, 9)) for heading, score in rows]

    assert scores(beside) == scores(alone)
    assert store.keyword_ranking(pg, beside, "pipeline", 10, channel="guidance")


def test_search_fuses_both_rankings_then_reranks(pg: store.Connection) -> None:
    version = build(pg)
    asked = []

    def embed(texts: list[str]) -> list[list[float]]:
        asked.extend(texts)
        return [padded([0.0, 1.0]) for _ in texts]  # nearest: Route, then Gas

    def rerank(query: str, documents: list[str]) -> list[float]:
        return [1.0 if "three gas grades" in d else 0.0 for d in documents]

    found = store.search(pg, version, "Where does the pipeline run?", embed, rerank)

    # The reranker's pick first; the rest keep their fused order (Route matched
    # both the keyword and the vector ranking, so it leads the fused list).
    assert [p.heading for p in found] == ["Gas", "Route", "Refining"]
    assert asked == [config.QUERY_INSTRUCTION + "Where does the pipeline run?"]


def test_vector_ranking_orders_by_cosine_within_one_version(
    pg: store.Connection,
) -> None:
    first = build(pg)
    second = build(pg)

    found = store.vector_ranking(pg, second, padded([0.0, 2.0]), 3)

    assert headings(pg, found) == ["Route", "Gas", "Refining"]
    older = store.vector_ranking(pg, first, padded([0.0, 2.0]), 3)
    assert set(found).isdisjoint(older)


def test_loaded_passages_keep_the_given_order_utc_capture_time_and_quoting(
    pg: store.Connection,
) -> None:
    scan = row(SITE + "about", "About Acme", "Accounts", "Revenue rose.", page=4,
               kind="scan", quotable=False, context="Accounts › Note 3")  # fmt: skip
    version = store.write_version(
        pg, DOCUMENTS, [*ROWS, scan], [padded(v) for v in [*VECTORS, [0.8, 0.6]]],
        MANIFEST,
    )  # fmt: skip
    ids = store.vector_ranking(pg, version, padded([1.0, 0.0]), 4)

    passages = store.load_passages(pg, list(reversed(ids)))

    assert [p.id for p in passages] == list(reversed(ids))
    route = next(p for p in passages if p.heading == "Route")
    assert (route.url, route.title) == (SITE + "news/pipeline", "Pipeline Opens")
    assert route.fetched_at == "2026-10-10T10:30:00+00:00"  # 11:30 at +01:00
    assert route.quotable and route.image == ""
    assert route.published == "2024-03-12"
    accounts = next(p for p in passages if p.heading == "Accounts")
    assert (accounts.page, accounts.context, accounts.quotable) == (
        4, "Accounts › Note 3", False,
    )  # fmt: skip
    assert accounts.published == ""


def test_a_build_finds_the_vectors_the_newest_version_of_its_embedder_stored(
    pg: store.Connection,
) -> None:
    assert store.known_vectors(pg, "test-embedder") == {}
    build(pg)
    newest = build(pg)
    pg.execute(
        "UPDATE passages SET text = 'changed' WHERE index_version_id <> %s", (newest,)
    )

    known = store.known_vectors(pg, "test-embedder")

    key = ("Pipeline Opens", "", "Route\nThe pipeline runs 614 km.")
    assert len(known) == 3 and known[key][:2] == [0.0, 1.0]
    assert store.known_vectors(pg, "another-embedder") == {}


def test_each_channel_is_searched_apart_and_a_picture_needs_words_of_its_own(
    pg: store.Connection,
) -> None:
    rows = [
        *ROWS,
        row(GUIDE["url"], GUIDE["title"], "Wells", "Acme drilled 40 wells.",
            channel="guidance"),
        row(SITE + "about", "About Acme", "Picture", "", context="Picture › Refining",
            kind="picture", channel="picture", image="a" * 64),
        row(SITE + "about", "About Acme", "Picture", "",
            context="Picture › Refining\nwells at dawn", kind="picture",
            channel="picture", image="b" * 64),
    ]  # fmt: skip
    vectors = [padded(v) for v in [*VECTORS, [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]]
    pictures = {"a" * 64: b"a", "b" * 64: b"b"}
    version = store.write_version(
        pg, [*DOCUMENTS, GUIDE], rows, vectors, MANIFEST, pictures
    )

    def channel_of(passage_id: int) -> str:
        found = pg.execute("SELECT channel FROM passages WHERE id = %s", (passage_id,))
        return str(found.fetchone()[0])  # type: ignore[index]

    wells = store.keyword_ranking(pg, version, "wells", 10, channel="guidance")
    # Both pictures' section paths hold "Refining"; only one has words of its own.
    found = store.keyword_ranking(pg, version, "refining", 10, channel="picture")

    assert [channel_of(n) for n in wells] == ["guidance"]
    assert store.keyword_ranking(pg, version, "wells", 10) == []
    assert len(found) == 1  # the picture with no words is not searched by words
    assert store.load_passages(pg, found)[0].image == "b" * 64
    well_pictures = store.keyword_ranking(pg, version, "wells", 10, channel="picture")
    assert store.load_passages(pg, well_pictures)[0].image == "b" * 64
    fused = store.fused(pg, version, "wells", lambda texts: [padded([0.0, 1.0])])
    assert fused and all(channel_of(n) == "company" for n in fused)


def test_a_picture_is_stored_once_and_ranked_by_what_it_shows(
    pg: store.Connection,
) -> None:
    picture = "a" * 64
    shown = row(SITE + "about", "About Acme", "Picture", "Flare stack at Port Harcourt",
                context="Picture › Refining", kind="picture", channel="picture",
                image=picture)  # fmt: skip
    vectors = [padded(v) for v in [*VECTORS, [0.8, 0.6]]]
    siglip = padded([1.0], 1152)
    for _ in range(2):  # a second version reuses the stored picture
        version = store.write_version(
            pg, DOCUMENTS, [*ROWS, shown], vectors, MANIFEST, {picture: b"png"},
            {picture: siglip},
        )  # fmt: skip

    by_words = store.keyword_ranking(pg, version, "flare", 10, channel="picture")
    by_sight = store.picture_ranking(pg, version, siglip, 5)

    assert by_sight == by_words
    assert headings(pg, by_words) == ["Picture"]
    assert store.keyword_ranking(pg, version, "flare", 10) == []  # company only
    assert pg.execute("SELECT count(*) FROM images").fetchone() == (1,)
    assert store.has_pictures(pg, version)
    fused = store.fused(pg, version, "stack", lambda texts: [padded([0.0, 1.0])],
                        also=[by_sight], channel="picture")  # fmt: skip
    assert fused == by_sight


def test_an_answer_record_round_trips_as_plain_data(pg: store.Connection) -> None:
    version = build(pg)
    shown = {"status": "answered", "claims": [{"text": "The line runs 614 km."}]}
    removed = [{"text": "The line is new.", "reason": "quote not in S1"}]

    answer_id = store.record_answer(
        pg,
        question="How long is the pipeline?",
        status="answered",
        shown=shown,
        removed=removed,
        passage_ids=[3, 1],
        index_version_id=version,
        embedding_model="test-embedder",
        prompt_sha256="abc",
        seconds=1.5,
    )

    found = pg.execute(
        "SELECT question, status, shown, removed, passage_ids, index_version_id, "
        "embedding_model, prompt_sha256, seconds, created_at IS NOT NULL "
        "FROM answers WHERE id = %s",
        (answer_id,),
    ).fetchone()
    assert found == (
        "How long is the pipeline?", "answered", shown, removed, [3, 1], version,
        "test-embedder", "abc", 1.5, True,
    )  # fmt: skip


def test_an_unknown_status_is_refused_by_the_database(pg: store.Connection) -> None:
    with pytest.raises(psycopg.errors.CheckViolation):
        store.record_answer(
            pg, question="q", status="maybe", shown={}, removed=[], passage_ids=[],
            index_version_id=1, embedding_model="m", prompt_sha256="p", seconds=0.1,
        )  # fmt: skip


LIMITS = (30, 8, 20)  # idle minutes, hours since the start, turns


def record_turn(
    pg: store.Connection, version: int, conversation_id: str, turn: int, text: str
) -> int:
    """Record an answer for one turn whose shown view is a single claim `text`."""
    shown = {"claims": [{"text": text}], "notice": ""}
    return store.record_answer(
        pg, f"question {turn}", "answered", shown, [], [], version, "e", "p", 1.0,
        conversation_id=conversation_id, turn=turn,
    )  # fmt: skip


def test_a_question_without_a_conversation_starts_one_at_turn_one(
    pg: store.Connection,
) -> None:
    opened = store.open_turn(pg, None, *LIMITS)

    assert opened is not None
    conversation_id, turn = opened
    assert uuid.UUID(conversation_id).version == 4  # random, made by the server
    assert turn == 1


def test_each_turn_of_a_conversation_takes_the_next_number(
    pg: store.Connection,
) -> None:
    opened = store.open_turn(pg, None, *LIMITS)
    assert opened is not None
    conversation_id = opened[0]

    numbers = [store.open_turn(pg, conversation_id, *LIMITS) for _ in range(2)]

    assert numbers == [(conversation_id, 2), (conversation_id, 3)]
    assert store.open_turn(pg, str(uuid.uuid4()), *LIMITS) is None  # unknown


@pytest.mark.parametrize(
    ("change", "open_again"),
    [
        ("last_active_at = now() - interval '29 minutes'", True),
        ("last_active_at = now() - interval '31 minutes'", False),
        ("created_at = now() - interval '9 hours'", False),  # however active
        ("turns = 20", False),
    ],
)
def test_a_conversation_ends_when_idle_too_old_or_full(
    pg: store.Connection, change: str, open_again: bool
) -> None:
    opened = store.open_turn(pg, None, *LIMITS)
    assert opened is not None
    pg.execute(f"UPDATE conversations SET {change}")  # the test's own literal SQL

    assert (store.open_turn(pg, opened[0], *LIMITS) is not None) == open_again


def test_earlier_turns_are_the_last_answers_before_the_turn_oldest_first(
    pg: store.Connection,
) -> None:
    version = build(pg)
    ids = []
    for conversation in range(2):
        opened = store.open_turn(pg, None, *LIMITS)
        assert opened is not None
        ids.append(opened[0])
        for turn in range(1, 6):
            if turn > 1:
                store.open_turn(pg, opened[0], *LIMITS)
            record_turn(pg, version, opened[0], turn, f"reply {conversation}.{turn}")

    earlier = store.earlier_turns(pg, ids[0], 6, 4)
    first_two = store.earlier_turns(pg, ids[0], 3, 4)

    assert earlier == [
        (f"question {turn}", {"claims": [{"text": f"reply 0.{turn}"}], "notice": ""})
        for turn in range(2, 6)
    ]
    assert [question for question, _ in first_two] == ["question 1", "question 2"]


def test_an_answer_in_a_conversation_keeps_its_turn_and_search_questions(
    pg: store.Connection,
) -> None:
    version = build(pg)
    opened = store.open_turn(pg, None, *LIMITS)
    assert opened is not None

    answer_id = store.record_answer(
        pg, "and the gas?", "answered", {}, [], [], version, "e", "p", 1.0,
        search_questions=["What gas does the line carry?"],
        conversation_id=opened[0], turn=1,
    )  # fmt: skip

    found = pg.execute(
        "SELECT conversation_id::text, turn, search_questions FROM answers "
        "WHERE id = %s",
        (answer_id,),
    ).fetchone()
    assert found == (opened[0], 1, ["What gas does the line carry?"])
    with pytest.raises(psycopg.errors.UniqueViolation):
        record_turn(pg, version, opened[0], 1, "the same turn twice")


def test_an_answer_keeps_its_model_calls_and_history_reads_them_back(
    pg: store.Connection,
) -> None:
    version = build(pg)
    calls = [
        {"purpose": "understanding", "system": "Understand.", "user": "Q: gas?",
         "reply": {"search_questions": []}, "seconds": 0.5},
        {"purpose": "answering", "system": "Answer.", "user": "<passage>…",
         "reply": {"claims": []}, "seconds": 2.0},
    ]  # fmt: skip
    opened = store.open_turn(pg, None, *LIMITS)
    assert opened is not None
    in_conversation = store.record_answer(
        pg, "and the gas?", "answered", {"claims": []}, [], [4], version, "e", "p",
        2.5, search_questions=["What gas?"], conversation_id=opened[0], turn=1,
        calls=calls,
    )  # fmt: skip
    alone = store.record_answer(
        pg, "a single question", "insufficient_evidence", {}, [], [], version, "e",
        "p", 1.0,
    )  # fmt: skip

    turns = store.answer_records(pg, opened[0])
    everything = store.answer_records(pg)
    listed = store.recent_conversations(pg, 10)

    assert [record["id"] for record in turns] == [in_conversation]
    assert turns[0]["model_calls"] == calls
    assert turns[0]["search_questions"] == ["What gas?"]
    assert (turns[0]["conversation_id"], turns[0]["turn"]) == (opened[0], 1)
    assert [record["id"] for record in everything] == [in_conversation, alone]
    assert everything[1]["model_calls"] == []
    assert everything[1]["conversation_id"] is None
    assert [(c["id"], c["turns"], c["first_question"]) for c in listed] == [
        (opened[0], 1, "and the gas?")
    ]
    # Deleting an answer deletes its provenance with it.
    pg.execute("DELETE FROM answers WHERE id = %s", (in_conversation,))
    assert pg.execute("SELECT count(*) FROM model_calls").fetchone() == (0,)


def test_a_question_that_got_no_answer_is_kept_with_its_cause(
    pg: store.Connection,
) -> None:
    ended = str(uuid.uuid4())
    store.record_failure(pg, "And NGIC?", "ended", "conversation ended", ended)
    store.record_failure(pg, "Who?", "unavailable", "generation server failed", None, 2)

    assert [f["question"] for f in store.question_failures(pg)] == ["And NGIC?", "Who?"]
    found = store.question_failures(pg, ended)
    assert [(f["reason"], f["detail"], f["conversation_id"]) for f in found] == [
        ("ended", "conversation ended", ended)
    ]
    assert store.question_failures(pg)[1]["conversation_id"] is None
    with pytest.raises(psycopg.errors.CheckViolation):
        store.record_failure(pg, "q", "lost", "no such reason")


def test_a_turn_needs_its_conversation_and_a_conversation_its_turn(
    pg: store.Connection,
) -> None:
    version = build(pg)

    with pytest.raises(psycopg.errors.CheckViolation):
        store.record_answer(pg, "q", "answered", {}, [], [], version, "e", "p", 1.0,
                            turn=1)  # fmt: skip


UNITS = [
    {"name": "Acme Refining Limited", "acronym": "ARL", "url": SITE + "about"},
    {"name": "Acme Pipelines", "acronym": "", "url": SITE + "pipelines"},
]


def test_a_version_keeps_its_units_and_each_unit_s_passages(
    pg: store.Connection,
) -> None:
    version = build(pg)
    passages = store.passage_texts(pg, version)
    assert [(url, title) for _, url, title, _ in passages] == [
        (SITE + "about", "About Acme"),
        (SITE + "news/pipeline", "Pipeline Opens"),
        (SITE + "news/pipeline", "Pipeline Opens"),
    ]
    ids = [passage_id for passage_id, *_ in passages]

    unit_ids = store.write_units(pg, version, UNITS, [(0, ids[0]), (1, ids[1])])

    found = store.version_units(pg, version)
    assert found == [unit | {"id": n} for unit, n in zip(UNITS, unit_ids, strict=True)]
    assert store.version_units(pg, build(pg)) == []  # a version has its own
    store.delete_version(pg, version)
    assert pg.execute("SELECT count(*) FROM unit_passages").fetchone() == (0,)


def test_a_search_inside_units_ranks_only_the_passages_in_their_scope(
    pg: store.Connection,
) -> None:
    version = build(pg)
    ids = [passage_id for passage_id, *_ in store.passage_texts(pg, version)]
    refining, pipelines = store.write_units(
        pg, version, UNITS, [(0, ids[0]), (1, ids[1]), (1, ids[2])]
    )

    def embed(texts: list[str]) -> list[list[float]]:
        return [padded([1.0, 0.0]) for _ in texts]  # nearest: Refining

    def rerank(query: str, documents: list[str]) -> list[float]:
        return [0.0 for _ in documents]

    asked = "Which gas grades does the pipeline carry from the refinery?"
    everywhere = store.search(pg, version, asked, embed, rerank)
    inside = store.search(pg, version, asked, embed, rerank, units=[pipelines])

    assert "Refining" in [p.heading for p in everywhere]
    assert sorted(p.heading for p in inside) == ["Gas", "Route"]
    words = store.keyword_ranking(pg, version, "pipeline", 5, units=[refining])
    assert words == []  # the refining unit's one passage has no such word
    nearest = store.vector_ranking(pg, version, padded([1.0, 0.0]), 5,
                                   units=[pipelines])  # fmt: skip
    assert headings(pg, nearest) == ["Gas", "Route"]


def test_role_readings_are_kept_by_text_and_statements_by_version(
    pg: store.Connection,
) -> None:
    version = build(pg)
    dated = store.dated_passages(pg, version)
    assert [(title, published) for _, title, _, published, _ in dated] == [
        ("About Acme", None),
        ("Pipeline Opens", date(2024, 3, 12)),
        ("Pipeline Opens", date(2024, 3, 12)),
    ]
    assert dated[0][4] == "2026-10-10"  # the capture date, in UTC
    said = {"person": "Ada Obi", "role": "Chief Executive Officer",
            "organisation": "Acme", "quote": "Ada Obi"}  # fmt: skip

    store.write_role_reading(pg, "a" * 64, [said])
    store.write_role_reading(pg, "a" * 64, [])  # a text is read once

    assert store.role_readings(pg, ["a" * 64, "b" * 64]) == {"a" * 64: [said]}
    dates = {"passage_id": dated[1][0], "stated_on": "2024-03-12",
             "date_kind": "published"}  # fmt: skip
    store.write_role_statements(pg, version, [said | dates])
    found = store.version_role_statements(pg, version)
    assert found == [said | dates | {"title": "Pipeline Opens"}]
    store.delete_version(pg, version)
    assert pg.execute("SELECT count(*) FROM role_statements").fetchone() == (0,)


def test_feedback_is_kept_only_from_the_answer_s_conversation_and_read_back(
    pg: store.Connection,
) -> None:
    version = build(pg)
    opened = store.open_turn(pg, None, *LIMITS)
    assert opened is not None
    answer_id = record_turn(pg, version, opened[0], 1, "NGML markets gas.")

    assert store.record_feedback(pg, answer_id, opened[0], False, "wrong year")
    assert not store.record_feedback(pg, answer_id, str(uuid.uuid4()), True, "")
    (record,) = store.answer_records(pg, opened[0])
    assert [(f["helpful"], f["comment"]) for f in record["feedback"]] == [
        (False, "wrong year")
    ]
