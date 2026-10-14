"""The one call behind every interface: a question answered from the served Postgres
index and recorded for audit (Lime Green's assistant, carried over for NNPC).

The command line and the API both call `ask_and_record`, so they cannot differ in
what they retrieve, which model requests they make or what they verify.
"""

import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import psycopg

from petroknowledge import config, llm, roles, siglip, store, telemetry, units
from petroknowledge.answer import (
    GROUND_SCHEMA,
    PROMPT_SHA256,
    UNDERSTAND_SCHEMA,
    Chat,
    Extra,
    History,
    Retrieve,
    RoleLookup,
    StaleCheck,
    StructureLookup,
    answer,
    with_extras,
)
from petroknowledge.records import Answer, Passage
from petroknowledge.retrieve import Embed, Rerank, rerank_top
from petroknowledge.view import reply_text, view


class Unavailable(Exception):
    """The index cannot be served: no database address, Postgres unreachable, no
    version to serve, or one embedded by another model. The message says what to do."""


class ConversationEnded(Exception):
    """The conversation is unknown, idle too long, too old or at its turn limit: the
    reader starts a new one."""


@dataclass(frozen=True)
class Turn:
    """A reserved turn of a stored conversation, with the earlier turns its first
    request reads: each message and the reply its reader saw (`view.reply_text`)."""

    conversation_id: str
    number: int
    history: History


logger = logging.getLogger(__name__)


def record_failure(
    question: str,
    reason: str,
    detail: str,
    conversation_id: str | None = None,
    turn: int | None = None,
) -> None:
    """Keep a question that got no answer, with its cause (history). If it cannot be
    kept, that is logged and never raised: the reader's own error comes first."""
    try:
        with connect() as conn:
            store.record_failure(conn, question, reason, detail, conversation_id, turn)
    except (Unavailable, psycopg.Error) as error:
        logger.warning("failed question not recorded: %s", error)


def record_feedback(
    answer_id: int, conversation_id: str, helpful: bool, comment: str
) -> bool:
    """Keep a reader's judgement of an answer from its own conversation;
    whether it was kept."""
    with connect() as conn:
        return store.record_feedback(conn, answer_id, conversation_id, helpful, comment)


def open_turn(conversation_id: str | None) -> Turn:
    """Reserve the next turn of a conversation, or the first turn of a new one, and
    rebuild its history from the stored answers: never from the client, since a
    forged assistant turn would be an injection (Lime Green D60)."""
    with connect() as conn:
        opened = store.open_turn(
            conn,
            conversation_id,
            config.CONVERSATION_IDLE_MINUTES,
            config.CONVERSATION_MAX_HOURS,
            config.CONVERSATION_MAX_TURNS,
        )
        if opened is None:
            raise ConversationEnded(f"conversation {conversation_id} has ended")
        found, number = opened
        earlier = store.earlier_turns(conn, found, number, config.HISTORY_TURNS)
    history = [(question, reply_text(shown)) for question, shown in earlier]
    return Turn(found, number, history)


def no_stage(stage: str) -> None:
    """The default stage report: nobody is listening."""


def with_stages(
    retrieve: Retrieve, chat: Chat, on_stage: Callable[[str], None]
) -> tuple[Retrieve, Chat]:
    """The same retrieval and model calls, reporting each stage as it starts and
    tracing it: understanding (the first model request), searching, answering and
    checking (verification runs once the answer request returns)."""

    searched = False

    def staged_retrieve(query: str) -> list[Passage]:
        # A question with several parts is searched several times: one stage event.
        nonlocal searched
        if not searched:
            on_stage("searching")
            searched = True
        with telemetry.span("searching"):
            return retrieve(query)

    def staged_chat(system: str, user: str, schema: dict[str, Any]) -> object:
        if schema is UNDERSTAND_SCHEMA:
            on_stage("understanding")
            with telemetry.span("understanding"):
                return chat(system, user, schema)
        on_stage("answering")
        with telemetry.span("answering"):
            reply = chat(system, user, schema)
        on_stage("checking")
        return reply

    return staged_retrieve, staged_chat


def recording(chat: Chat, calls: list[dict[str, Any]]) -> Chat:
    """The same model requests, each kept in `calls` with its reply and time: the
    answer's provenance, stored with its audit record (`history` reads it back)."""

    def recorded(system: str, user: str, schema: dict[str, Any]) -> object:
        started = time.perf_counter()
        reply = chat(system, user, schema)
        seconds = time.perf_counter() - started
        purpose = "answering"
        if schema is UNDERSTAND_SCHEMA:
            purpose = "understanding"
        elif schema is GROUND_SCHEMA:
            purpose = "grounding"
        call = {"purpose": purpose, "system": system, "user": user, "reply": reply}
        calls.append({**call, "seconds": seconds})
        return reply

    return recorded


# SigLIP2's text vectors are computed here while the search's other requests run.
PICTURE_WORK = ThreadPoolExecutor(max_workers=2)


@dataclass(frozen=True)
class Models:
    """One answer's model calls, each remembering its results: one answer searches
    the same query in several channels and several searches (Lime Green X44 F2, X45
    T1). `picture` starts a query's SigLIP2 vector in a thread and returns its
    future, so it is computed while the query is embedded (X45 T2). Made for each
    answer, so nothing is kept between answers."""

    embed: Embed
    rerank: Rerank
    picture: Callable[[str], "Future[list[float]]"]


def answer_models() -> Models:
    """Fresh `Models` for one answer, calling the configured model servers."""
    vectors: dict[str, list[float]] = {}
    scores: dict[tuple[str, str], float] = {}
    started: dict[str, Future[list[float]]] = {}

    def embed(texts: list[str]) -> list[list[float]]:
        missing = [text for text in dict.fromkeys(texts) if text not in vectors]
        if missing:
            vectors.update(zip(missing, llm.embed(missing), strict=True))
        return [vectors[text] for text in texts]

    def rerank(query: str, documents: list[str]) -> list[float]:
        missing = [d for d in dict.fromkeys(documents) if (query, d) not in scores]
        if missing:
            found = llm.rerank(query, missing)
            scores.update({(query, d): v for d, v in zip(missing, found, strict=True)})
        return [scores[(query, document)] for document in documents]

    def picture(query: str) -> Future[list[float]]:
        if query not in started:
            started[query] = PICTURE_WORK.submit(siglip.text_vector, query)
        return started[query]

    return Models(embed, rerank, picture)


def retriever(conn: store.Connection, version_id: int, models: Models) -> Retrieve:
    """One search of an index version's company channel. It returns its whole
    reranked pool; `answer.gather` takes the top 8, or interleaves several searches
    up to 12 (Lime Green S2b C2). The query's picture vector starts first."""

    def retrieve(query: str) -> list[Passage]:
        models.picture(query)
        return store.search(conn, version_id, query, models.embed, models.rerank,
                            config.RERANK_CANDIDATES)  # fmt: skip

    return retrieve


def asked_units(
    conn: store.Connection, version_id: int, question: str, history: History
) -> list[dict[str, Any]]:
    """The units the user's own words name (`units.asked`), never the model's rewrite
    of them: each with its id, name, acronym and url."""
    found = store.version_units(conn, version_id)
    places = units.asked(units.naming(found), question, history)
    return [found[place] for place in places]


def scoped_retriever(
    conn: store.Connection, version_id: int, models: Models, unit_ids: Sequence[int]
) -> Retrieve | None:
    """A search of an index version inside the units' scope: its best
    `config.SCOPED_TOP` passages (Lime Green E5 B4); none for no units."""
    if not unit_ids:
        return None

    def retrieve(query: str) -> list[Passage]:
        return store.search(conn, version_id, query, models.embed, models.rerank,
                            config.SCOPED_TOP, units=unit_ids)  # fmt: skip

    return retrieve


def current_roles(conn: store.Connection, version_id: int) -> list[dict[str, Any]]:
    """A version's roles, each with its latest holder and earlier ones, after each
    statement's organisation is resolved to the unit it is about."""
    statements = store.version_role_statements(conn, version_id)
    listed = store.version_units(conn, version_id)
    return roles.latest(roles.resolve(statements, listed))


def stale_check(conn: store.Connection, version_id: int) -> StaleCheck | None:
    """The check removing a claim that names an earlier holder of a role it names;
    None for a version without role statements."""
    found = current_roles(conn, version_id)
    if not found:
        return None

    def check(text: str, whole: str) -> str:
        return roles.earlier_holder(text, found, whole)

    return check


def structure_lookup(conn: store.Connection, version_id: int) -> StructureLookup | None:
    """The company's current structure from the graph: the compiled
    passage, which the answer request is pointed to, when the user's words name
    two of its groups or units, or a passage found names three of its groups (a
    competing classification). None for a version without a structure or a compiled
    passage."""
    groups = store.version_structure(conn, version_id)
    compiled = store.compiled_passage(conn, version_id)
    if not groups or compiled is None:
        return None
    named = units.naming(store.version_units(conn, version_id))
    names = [group["name"] for group in groups]

    def lookup(words: str, passages: Sequence[Passage]) -> list[Passage]:
        asked = units.structure_named(words, names) + len(named(words))
        told = any(units.structure_named(p.text, names) >= 3 for p in passages)
        return [compiled] if asked >= 2 or told else []

    return lookup


def role_lookup(conn: store.Connection, version_id: int) -> RoleLookup | None:
    """Who holds the roles a question's words ask about: the latest dated
    statement of each (`roles.latest`, `roles.naming`) and its earlier holders, as the
    line for the answer request and their passages, oldest first. None for a version
    without role statements."""
    found = current_roles(conn, version_id)
    if not found:
        return None
    named = roles.naming(found)

    def lookup(text: str) -> tuple[str, list[Passage]]:
        asked = [found[place] for place in named(text)]
        ids = []
        for role in asked:
            for statement in reversed(role["earlier"][: roles.EARLIER]):
                ids.append(statement["passage_id"])
            ids.append(role["current"]["passage_id"])
        passages = store.load_passages(conn, list(dict.fromkeys(ids)))
        return roles.described(asked), passages

    return lookup


def extras(conn: store.Connection, version_id: int, models: Models) -> Extra:
    """What each search adds after the company's top places (Lime Green X44 F2): its
    best pictures, then its best guidance, each kept only where the reranker places
    it above the company's last place (`above_company`). NNPC's pictures are read
    for their words only, which the reranker reads in their context, so a picture no
    better than the company's passages is not pointed to. A version with no
    pictures never loads the picture model."""
    has_pictures = store.has_pictures(conn, version_id)

    def add(query: str, company: Sequence[Passage]) -> list[Passage]:
        found = best_pictures(conn, version_id, query, models) if has_pictures else []
        ranked = store.fused(conn, version_id, query, models.embed, channel="guidance")
        guidance = store.load_passages(conn, ranked[: config.TOP_K])
        pictures = above_company(query, company, found, models,
                                 config.PICTURES_PER_SEARCH)  # fmt: skip
        kept = above_company(query, company, guidance, models,
                             config.GUIDANCE_PER_SEARCH)  # fmt: skip
        return [*pictures, *kept]

    return add


def best_pictures(
    conn: store.Connection, version_id: int, query: str, models: Models
) -> list[Passage]:
    """A search's best pictures: its best `config.PICTURES_PER_SEARCH` by their own
    words, then its best by what they show (SigLIP2) not already among them. Each
    retriever keeps its own places: fused into one ranking, a picture both rank
    moderately displaced each one's best (Lime Green X44 amendment 1)."""
    best = config.PICTURES_PER_SEARCH
    limit = config.CANDIDATES_PER_METHOD
    vector = models.picture(query)
    words = store.fused(conn, version_id, query, models.embed, channel="picture")
    shown = store.picture_ranking(conn, version_id, vector.result(), limit)
    return store.load_passages(
        conn, list(dict.fromkeys([*words[:best], *shown[:best]]))
    )


def above_company(
    query: str,
    company: Sequence[Passage],
    candidates: Sequence[Passage],
    models: Models,
    limit: int,
) -> list[Passage]:
    """Another channel's candidates (guidance, pictures) that the reranker puts above
    the company's last place: they are reranked with the company's top places, and
    at most `limit` of those ahead of every company passage's last are kept, best
    first. With no company passage, the best candidates stand alone. The company's
    passages keep the scores their own search gave them (`Models.rerank`)."""
    if not candidates:
        return []
    pool = [*company, *candidates]
    ranked = rerank_top(query, pool, models.rerank, len(pool))
    own = {passage.id for passage in company}
    places = [n for n, passage in enumerate(ranked) if passage.id in own]
    last = places[-1] if places else len(ranked)
    above = [passage for passage in ranked[:last] if passage.id not in own]
    return above[:limit]


def searched(conn: store.Connection, version_id: int, query: str) -> list[Passage]:
    """One search as an answer makes it, in every channel: the company's top places,
    then the pictures and guidance it adds, within an answer's limits. The
    evaluation sets score it."""
    models = answer_models()
    company = retriever(conn, version_id, models)(query)[: config.TOP_K]
    added = extras(conn, version_id, models)(query, company)
    return list(with_extras(company, added))


def ask_and_record(
    question: str,
    on_stage: Callable[[str], None] = no_stage,
    history: History = (),
    turn: Turn | None = None,
) -> tuple[Answer, int]:
    """Answer from the served Postgres index and store the audit record; return the
    answer and the record's id. `on_stage` hears each stage as it starts. `history`
    is the conversation so far, given in process only (evaluation): the API never
    takes it from a client, since a forged assistant turn would be an injection. A
    `turn` of a stored conversation brings its own history instead, and the record
    keeps the turn."""
    if turn is not None:
        history = turn.history
    with telemetry.span("answer"), connect() as conn:
        version_id = served_index(conn)
        started = time.perf_counter()
        models = answer_models()
        calls: list[dict[str, Any]] = []
        retrieve, chat = with_stages(
            retriever(conn, version_id, models), recording(llm.chat, calls), on_stage
        )
        extra = extras(conn, version_id, models)
        asked = asked_units(conn, version_id, question, history)
        scoped = scoped_retriever(conn, version_id, models, [u["id"] for u in asked])
        lookup = role_lookup(conn, version_id)
        judge = recording(llm.judge, calls)
        structure = structure_lookup(conn, version_id)
        stale = stale_check(conn, version_id)
        result = answer(question, retrieve, chat, extra, history, scoped, asked, lookup,
                        judge, structure, stale)  # fmt: skip
        seconds = time.perf_counter() - started
        removed = [{"text": r.text, "reason": r.reason} for r in result.rejected]
        answer_id = store.record_answer(
            conn,
            question=question,
            status=result.status,
            shown=dict(view(result)),
            removed=removed,
            passage_ids=[p.id for p in result.passages],
            index_version_id=version_id,
            embedding_model=config.EMBEDDING_MODEL,
            prompt_sha256=PROMPT_SHA256,
            seconds=seconds,
            search_questions=result.understood,
            calls=calls,
            conversation_id=turn.conversation_id if turn else None,
            turn=turn.number if turn else None,
        )
        telemetry.record_answer(result, answer_id, version_id, seconds)
        if turn is not None:
            telemetry.record_turn(turn.conversation_id, turn.number)
    return result, answer_id


def connect() -> store.Connection:
    """A connection to the Postgres index, or a clear error saying what is wrong."""
    if not config.DATABASE_URL:
        raise Unavailable(
            "PETROKNOWLEDGE_DATABASE_URL is not set; run with "
            "`uv run --env-file .env petroknowledge ...`"
        )
    try:
        return psycopg.connect(
            config.DATABASE_URL, connect_timeout=config.DATABASE_CONNECT_TIMEOUT_SECONDS
        )
    except psycopg.OperationalError as error:
        raise Unavailable(f"cannot reach the Postgres index: {error}") from error


def database_ready() -> bool:
    """Whether answers can be served and recorded: Postgres is reachable and the
    index to serve exists, embedded with the configured model."""
    try:
        with connect() as conn:
            served_index(conn)
    except Unavailable:
        return False
    return True


def served_index(conn: store.Connection) -> int:
    """The index version to serve: `PETROKNOWLEDGE_INDEX_VERSION` when set (a
    candidate evaluated before it goes live), otherwise the live one. Either must
    have been embedded with the configured model: question vectors from another
    model would not match its passage vectors."""
    if config.INDEX_VERSION:
        if not config.INDEX_VERSION.isdigit():
            raise Unavailable(
                "PETROKNOWLEDGE_INDEX_VERSION must be an index version number"
            )
        served = store.index_version(conn, int(config.INDEX_VERSION))
        if served is None:
            raise Unavailable(f"no index version {config.INDEX_VERSION} to serve")
    else:
        served = store.live_version(conn)
        if served is None:
            raise Unavailable(
                "no live index; build one with `petroknowledge index <company> --live`"
            )
    version_id, embedding_model = served
    if embedding_model != config.EMBEDDING_MODEL:
        raise Unavailable(
            f"index version {version_id} was embedded with {embedding_model}, but the "
            f"configured model is {config.EMBEDDING_MODEL}; rebuild it with "
            "`petroknowledge index`"
        )
    return version_id
