"""The one call behind every interface, end to end on a throwaway Postgres index with
fake model servers. Every source, passage and vector here is invented."""

import uuid
from typing import Any

import pytest

from petroknowledge import assistant, config, llm, siglip, store, units
from petroknowledge.answer import HISTORY_PROMPT, PROMPT_SHA256, REFERRAL
from petroknowledge.records import Passage

SITE = "https://acme.test/"
DOCUMENTS = [
    {"url": SITE + "gas", "title": "Gas Plant", "fetched_at": "2026-10-10T10:00:00Z",
     "sha256": "a" * 64},
    {"url": "https://wiki.test/Gas", "title": "Gas - Wikipedia",
     "fetched_at": "2026-10-10T10:00:00Z", "sha256": "b" * 64},
]  # fmt: skip
WORDS = ["plant", "commission", "days", "flare", "gas", "pipeline", "valve"]


def row(text: str, **changes: Any) -> dict[str, Any]:
    found = {"url": SITE + "gas", "title": "Gas Plant", "heading": "Plant",
             "text": text, "context": "", "page": None, "kind": "page",
             "channel": "company", "quotable": True, "published": "",
             "image": ""}  # fmt: skip
    return found | changes


ROWS = [
    row("The plant takes about two days to commission."),
    row("The pipeline has a valve every 30 km.", heading="Pipeline"),
    row("Gas is flared at the plant only in emergencies.", url="https://wiki.test/Gas",
        title="Gas - Wikipedia", heading="Flaring", channel="guidance"),
    row("A flare stack at the gas plant", heading="Picture", kind="picture",
        channel="picture", context="Picture", image="c" * 64),
]  # fmt: skip


def embed(texts: list[str]) -> list[list[float]]:
    """Counts of a few words, plus a constant so no vector is zero."""
    vectors = []
    for text in texts:
        counts = [float(text.casefold().count(word)) for word in WORDS]
        vectors.append([1.0, *counts] + [0.0] * (1023 - len(WORDS)))
    return vectors


def rerank(query: str, documents: list[str]) -> list[float]:
    asked = set(query.casefold().split())
    return [float(len(asked & set(d.casefold().split()))) for d in documents]


def live_index(pg: store.Connection, embedding_model: str = config.EMBEDDING_MODEL
               ) -> int:  # fmt: skip
    manifest = {"corpus_sha256": "c", "passages_sha256": "p",
                "embedding_model": embedding_model}  # fmt: skip
    vectors = embed([r["text"] for r in ROWS])
    version = store.write_version(pg, DOCUMENTS, ROWS, vectors, manifest,
                                  {"c" * 64: b"png"},
                                  {"c" * 64: [1.0] + [0.0] * 1151})  # fmt: skip
    store.set_live(pg, version)
    pg.commit()
    return version


class Chat:
    """A stand-in model: reads `advice` for the first request, then quotes the
    commissioning time; records every request."""

    def __init__(self, advice: bool = False) -> None:
        self.advice = advice
        self.requests: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str, schema: dict[str, Any]) -> object:
        self.requests.append((system, user))
        if "verdict" in schema["properties"]:  # the grounding check
            return {"verdict": "supported"}
        if "asks_for_advice" in schema["properties"]:
            asked = user.rsplit("Question: ", 1)[1]
            return {"asks_for_advice": self.advice,
                    "search_questions": [{"question": asked, "items": []}]}  # fmt: skip
        blocks = user.split('<passage id="')[1:]
        source = next(b.split('"')[0] for b in blocks if "two days" in b)
        evidence = [{"source_id": source, "quote": "about two days"}]
        text = "The plant takes about two days to commission."
        return {"claims": [{"part": 1, "text": text, "evidence": evidence}]}


@pytest.fixture
def served(
    postgres_url: str, pg: store.Connection, monkeypatch: pytest.MonkeyPatch
) -> Chat:
    """The live test index served with fake model servers; yields the fake chat."""
    live_index(pg)
    chat = Chat()
    monkeypatch.setattr(config, "DATABASE_URL", postgres_url)
    monkeypatch.setattr(config, "INDEX_VERSION", "")
    monkeypatch.setattr(llm, "embed", embed)
    monkeypatch.setattr(llm, "rerank", rerank)
    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "judge", chat)
    monkeypatch.setattr(siglip, "text_vector", lambda q: [1.0] + [0.0] * 1151)
    return chat


def test_an_answer_comes_from_the_live_index_and_is_recorded(
    served: Chat, pg: store.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every claim judged, so the record holds each kind of model request (the
    # default judges only claims stating a figure, and this one states none).
    monkeypatch.setattr(config, "GROUNDING", "remove")
    result, answer_id = assistant.ask_and_record("How long to commission the plant?")

    assert result.status == "answered"
    assert [c.text for c in result.claims] == [
        "The plant takes about two days to commission."
    ]
    record = pg.execute(
        "SELECT question, status, shown, passage_ids, prompt_sha256, embedding_model, "
        "index_version_id, conversation_id FROM answers WHERE id = %s",
        (answer_id,),
    ).fetchone()
    assert record is not None
    question, status, shown, passage_ids, prompt, model, version, conversation = record
    assert (question, status) == ("How long to commission the plant?", "answered")
    assert shown["claims"][0]["text"] == result.claims[0].text
    assert passage_ids == [p.id for p in result.passages]
    assert (prompt, model, conversation) == (PROMPT_SHA256, config.EMBEDDING_MODEL,
                                             None)  # fmt: skip
    assert version == store.live_version(pg)[0]  # type: ignore[index]
    # Its provenance: each model request exactly as sent, with the model's reply.
    calls = pg.execute(
        "SELECT purpose, system, user_message, reply FROM model_calls "
        "WHERE answer_id = %s ORDER BY position",
        (answer_id,),
    ).fetchall()
    assert [call[0] for call in calls] == ["understanding", "answering", "grounding"]
    assert [(system, user) for _, system, user, _ in calls] == served.requests
    assert calls[1][3]["claims"][0]["text"] == result.claims[0].text


def test_a_question_without_an_answer_is_kept_and_a_lost_record_only_logged(
    served: Chat,
    pg: store.Connection,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    assistant.record_failure("Who?", "unavailable", "generation server failed")
    assert [(f["question"], f["reason"]) for f in store.question_failures(pg)] == [
        ("Who?", "unavailable")
    ]

    monkeypatch.setattr(config, "DATABASE_URL", "")
    assistant.record_failure("Who?", "unavailable", "again")

    assert caplog.messages == [
        "failed question not recorded: PETROKNOWLEDGE_DATABASE_URL is not set; run "
        "with `uv run --env-file .env petroknowledge ...`"
    ]


def test_a_stored_conversation_reads_what_its_reader_was_shown(served: Chat) -> None:
    first = assistant.open_turn(None)
    assistant.ask_and_record("How long to commission the plant?", turn=first)
    second = assistant.open_turn(first.conversation_id)

    assistant.ask_and_record("and the pipeline?", turn=second)

    assert (first.number, first.history, second.number) == (1, [], 2)
    reply = "The plant takes about two days to commission."
    assert second.history == [("How long to commission the plant?", reply)]
    # The second turn's first request: the one sent with the conversation so far.
    system, user = next(r for r in served.requests if r[1].startswith("Conversation"))
    assert system.endswith(HISTORY_PROMPT)
    assert user.startswith("Conversation so far:\nUser: How long to commission")
    with pytest.raises(assistant.ConversationEnded):
        assistant.open_turn(str(uuid.uuid4()))


def test_each_stage_is_reported_as_it_starts_and_a_referral_only_understands(
    served: Chat,
) -> None:
    stages: list[str] = []
    assistant.ask_and_record("How long to commission the plant?", stages.append)
    served.advice = True
    referred: list[str] = []

    result, _ = assistant.ask_and_record("Should I buy shares?", referred.append)

    assert stages == ["understanding", "searching", "answering", "checking"]
    assert (result.status, result.notice, referred) == (
        "safety_referral", REFERRAL, ["understanding"],
    )  # fmt: skip


def test_a_search_adds_its_best_pictures_then_guidance_on_merit(
    served: Chat, pg: store.Connection
) -> None:
    version = store.live_version(pg)[0]  # type: ignore[index]

    # The reranker reads each passage with its title: the guidance passage shares
    # every word of the question, the company's passages three of them.
    found = assistant.searched(pg, version, "gas flared at the plant")

    kinds = [("picture" if p.image else "guidance" if "wiki" in p.url else "company")
             for p in found]  # fmt: skip
    assert kinds[:2] == ["company", "company"]
    assert set(kinds[2:]) == {"picture", "guidance"}


def test_a_version_without_pictures_or_guidance_adds_nothing(
    postgres_url: str, pg: store.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {"corpus_sha256": "c", "passages_sha256": "p", "embedding_model": "e"}
    version = store.write_version(pg, DOCUMENTS[:1], ROWS[:1], embed([ROWS[0]["text"]]),
                                  manifest)  # fmt: skip
    monkeypatch.setattr(llm, "embed", embed)
    monkeypatch.setattr(siglip, "text_vector", lambda q: pytest.fail("loaded"))
    models = assistant.answer_models()

    assert assistant.extras(pg, version, models)("gas", []) == []
    assert assistant.above_company("gas", [], [], models, 2) == []


def test_another_channel_joins_only_above_the_company_s_last_place(
    served: Chat,
) -> None:
    models = assistant.answer_models()
    query = "gas flared at the plant"
    company = [Passage(1, "u", "Gas Plant", "Plant", "the plant takes two days", "d")]
    good = Passage(2, "w", "Gas", "Flaring", "gas flared at the plant", "d")
    # A picture whose words do not match: no better than the company's passages.
    poor = Passage(3, "p", "Speakers", "Picture", "", "d", context="Picture\nspeakers",
                   image="x")  # fmt: skip

    assert assistant.above_company(query, company, [poor, good], models, 2) == [good]
    assert assistant.above_company(query, [], [poor, good], models, 1) == [good]


def test_one_answer_embeds_and_scores_each_pair_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedded: list[list[str]] = []
    scored: list[list[str]] = []
    pictured: list[str] = []

    def embed_once(texts: list[str]) -> list[list[float]]:
        embedded.append(texts)
        return embed(texts)

    def rerank_once(query: str, documents: list[str]) -> list[float]:
        scored.append(documents)
        return rerank(query, documents)

    def picture_once(query: str) -> list[float]:
        pictured.append(query)
        return [1.0]

    monkeypatch.setattr(llm, "embed", embed_once)
    monkeypatch.setattr(llm, "rerank", rerank_once)
    monkeypatch.setattr(siglip, "text_vector", picture_once)
    models = assistant.answer_models()

    gas, flare = models.embed(["gas", "flare"])
    assert models.embed(["flare", "gas", "gas"]) == [flare, gas, gas]
    models.rerank("q", ["a", "b"])
    models.rerank("q", ["b", "c"])
    first = models.picture("gas")

    assert models.picture("gas") is first and first.result() == [1.0]
    assert embedded == [["gas", "flare"]]  # each text once
    assert scored == [["a", "b"], ["c"]]  # each pair once
    assert pictured == ["gas"]


def test_the_index_served_must_exist_and_match_the_embedder(
    postgres_url: str, pg: store.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "DATABASE_URL", postgres_url)
    monkeypatch.setattr(config, "INDEX_VERSION", "")
    with pytest.raises(assistant.Unavailable, match="no live index"):
        assistant.served_index(pg)
    assert not assistant.database_ready()
    version = live_index(pg)
    assert assistant.database_ready()

    monkeypatch.setattr(config, "INDEX_VERSION", str(version))
    assert assistant.served_index(pg) == version  # a chosen candidate is served
    for chosen, message in (("x", "must be an index version number"),
                            ("999", "no index version 999")):  # fmt: skip
        monkeypatch.setattr(config, "INDEX_VERSION", chosen)
        with pytest.raises(assistant.Unavailable, match=message):
            assistant.served_index(pg)
    monkeypatch.setattr(config, "INDEX_VERSION", "")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "another-embedder")
    with pytest.raises(assistant.Unavailable, match="was embedded with"):
        assistant.served_index(pg)


def test_a_missing_or_unreachable_database_is_explained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "DATABASE_URL", "")
    with pytest.raises(assistant.Unavailable, match="DATABASE_URL is not set"):
        assistant.connect()
    monkeypatch.setattr(config, "DATABASE_URL", "postgresql://u:p@127.0.0.1:9/none")
    with pytest.raises(assistant.Unavailable, match="cannot reach"):
        assistant.connect()


def test_the_retriever_starts_each_query_s_picture_vector(
    served: Chat, pg: store.Connection
) -> None:
    version = store.live_version(pg)[0]  # type: ignore[index]
    asked: list[str] = []
    models = assistant.answer_models()

    def picture(query: str) -> Any:
        asked.append(query)
        return models.picture(query)

    started = assistant.Models(models.embed, models.rerank, picture)

    found = assistant.retriever(pg, version, started)("pipeline valve")

    assert asked == ["pipeline valve"]
    assert isinstance(found[0], Passage) and "valve" in found[0].text


def test_the_units_a_question_names_scope_a_second_search(
    served: Chat, pg: store.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = store.live_version(pg)
    assert live is not None
    version = live[0]
    ids = [passage_id for passage_id, *_ in store.passage_texts(pg, version)]
    pipes = {"name": "Acme Pipelines", "acronym": "APL", "url": SITE + "pipes"}
    unit_ids = store.write_units(pg, version, [pipes], [(0, ids[1])])
    pg.commit()
    models = assistant.answer_models()

    found = assistant.asked_units(pg, version, "Where does APL run?", [])
    assert found == [pipes | {"aliases": "", "id": unit_ids[0]}]
    earlier = [("Where does APL run?", "Along the coast.")]
    assert assistant.asked_units(pg, version, "and its valves?", earlier) == found
    assert assistant.asked_units(pg, version, "Who runs Acme?", []) == []
    assert assistant.scoped_retriever(pg, version, models, []) is None
    scoped = assistant.scoped_retriever(pg, version, models, unit_ids)
    assert scoped is not None
    assert [p.heading for p in scoped("plant")] == ["Pipeline"]
    monkeypatch.setattr(config, "TOP_K", 1)  # the plant alone without the scope

    asked = "How long does APL's plant take to commission?"
    answer, _ = assistant.ask_and_record(asked)

    assert [p.heading for p in answer.passages] == ["Plant", "Pipeline"]
    told = "named in the user's messages: Acme Pipelines (APL)."
    assert told not in served.requests[0][1] and told in served.requests[1][1]


def test_the_roles_a_question_asks_about_come_with_their_latest_statement(
    served: Chat, pg: store.Connection
) -> None:
    live = store.live_version(pg)
    assert live is not None
    version = live[0]
    ids = [passage_id for passage_id, *_ in store.passage_texts(pg, version)]
    assert assistant.role_lookup(pg, version) is None  # no statements yet
    said = {"role": "Plant Manager", "organisation": "Acme", "quote": "q",
            "date_kind": "published"}  # fmt: skip
    store.write_role_statements(pg, version, [
        said | {"passage_id": ids[1], "person": "Ada Obi", "stated_on": "2026-09-10"},
        said | {"passage_id": ids[0], "person": "Ben Eze", "stated_on": "2024-01-01"},
    ])  # fmt: skip
    pg.commit()
    lookup = assistant.role_lookup(pg, version)
    assert lookup is not None

    line, passages = lookup("Who is the plant manager?")

    assert (
        line
        == '- Plant Manager (Acme): Ada Obi, in "Gas Plant"; earlier holders: Ben Eze.'
    )
    assert [p.id for p in passages] == [ids[0], ids[1]]  # oldest first
    assert lookup("What does gas cost?") == ("", [])
    assistant.ask_and_record("Who is the plant manager?")
    assert "Who holds the roles asked about" in served.requests[1][1]


def test_the_structure_is_told_when_the_words_or_passages_touch_it(
    served: Chat, pg: store.Connection
) -> None:
    live = store.live_version(pg)
    assert live is not None
    assert assistant.structure_lookup(pg, live[0]) is None  # no structure yet
    compiled = row(f"{units.COMPILED}\nUpstream, 1 unit: Acme Pipelines.",
                   heading=units.COMPILED)  # fmt: skip
    rows = [*ROWS, compiled]
    manifest = {"corpus_sha256": "c", "passages_sha256": "q",
                "embedding_model": config.EMBEDDING_MODEL}  # fmt: skip
    version = store.write_version(pg, DOCUMENTS, rows, embed([r["text"] for r in rows]),
                                  manifest, {"c" * 64: b"png"},
                                  {"c" * 64: [1.0] + [0.0] * 1151})  # fmt: skip
    pipes = {"name": "Acme Pipelines", "acronym": "APL", "url": SITE + "pipes"}
    (pipes_id,) = store.write_units(pg, version, [pipes], [])
    store.write_structure(pg, version, [("Upstream", "Acme Pipelines", pipes_id),
                                        ("Downstream", "Acme Retail", None),
                                        ("Gas", "Acme Gas", None)])  # fmt: skip
    pg.commit()
    lookup = assistant.structure_lookup(pg, version)
    assert lookup is not None

    passages = lookup("How do Upstream and Downstream differ?", [])

    assert [p.heading for p in passages] == [units.COMPILED]
    assert lookup("Is APL part of Upstream?", []) == passages  # a group and a unit
    assert lookup("How long does commissioning take?", []) == []
    listing = Passage(1, SITE, "Old", "Areas", "Upstream, Downstream and Gas", "2026")
    assert lookup("How long?", [listing]) == passages  # an older grouping retrieved


def test_feedback_goes_to_the_answer_in_its_conversation(served: Chat) -> None:
    turn = assistant.open_turn(None)
    _, answer_id = assistant.ask_and_record(
        "How long to commission the plant?", turn=turn
    )

    assert assistant.record_feedback(answer_id, turn.conversation_id, True, "")
    assert not assistant.record_feedback(answer_id + 1, turn.conversation_id, True, "")
