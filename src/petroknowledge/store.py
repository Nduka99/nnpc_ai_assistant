"""The Postgres index (Lime Green's store, carried over): write an index version,
choose which version is live, rank its passages for a question (keywords by BM25 with
pg_textsearch, meaning by pgvector), and keep the answer log.

All SQL lives here. Retrieval and answering stay plain functions that receive the
rankings. An index version is written beside the live one and made live in one
transaction, so a failed build never touches what is served.

Lime Green's price filter (X16) is not carried over: per-unit prices are never
answered from the index (`prices`). Its product scope returns as business units: a
version keeps its units and the passages in each one's scope, and a search
can be kept inside them.
"""

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, date
from typing import Any

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from petroknowledge import config, units
from petroknowledge.records import Passage
from petroknowledge.retrieve import Embed, Rerank, fuse, rerank_top, unit_vector

Connection = psycopg.Connection[tuple[Any, ...]]
# A corpus row (`corpus.build`): url, title, heading, text, context, page, kind,
# channel, quotable, published ("" for none) and image ("" for none).
Row = Mapping[str, Any]

# English stemming that keeps stop words (the migration). pg_textsearch finds a
# custom configuration only by its schema-qualified name.
KEYWORD_CONFIG = "public.english_keep_stop"
# What keyword search reads: the title, the context and the text.
KEYWORD_TEXT = sql.SQL("(title || ' ' || context || ' ' || text)")
# Search channels (Lime Green X44 F2): the company's own content, guidance, pictures.
CHANNELS = ("company", "guidance", "picture")


def bm25_index(version_id: int, channel: str) -> str:
    """A version's BM25 index for one channel. Each keeps its own word statistics, so
    adding guidance or pictures to a version cannot move the company's keyword scores
    (Lime Green X48)."""
    return f"passages_bm25_v{version_id}_{channel}"


def keyword_indexes(conn: Connection, version_id: int) -> None:
    """Create a version's BM25 index for each channel."""
    for name in CHANNELS:
        # The version and channel are literals: the planner must see the predicate.
        conn.execute(
            sql.SQL(
                "CREATE INDEX {} ON passages USING bm25 ({}) "
                "WITH (text_config = {}) "
                "WHERE index_version_id = {} AND channel = {}"
            ).format(
                sql.Identifier(bm25_index(version_id, name)),
                KEYWORD_TEXT,
                sql.Literal(KEYWORD_CONFIG),
                sql.Literal(version_id),
                sql.Literal(name),
            )
        )


def vector_text(vector: Sequence[float]) -> str:
    """pgvector's text form, "[x,y,...]", of the unit-length vector.

    Passing vectors as text keeps the code free of an adapter package; Postgres
    parses it with a ::vector cast.
    """
    return "[" + ",".join(repr(float(x)) for x in unit_vector(vector)) + "]"


def write_version(
    conn: Connection,
    documents: Sequence[Mapping[str, str]],
    rows: Sequence[Row],
    vectors: Sequence[Sequence[float]],
    manifest: Mapping[str, str],
    pictures: Mapping[str, bytes] | None = None,
    picture_vectors: Mapping[str, Sequence[float]] | None = None,
) -> int:
    """Store one index version (not yet live) with its BM25 indexes; return its id.

    `documents` are the sources (url, title, fetched_at, sha256); one captured with
    the same bytes before is reused, not stored twice. `rows` are the passages, one
    vector each. `pictures` holds the PNG of each picture a row names, stored once
    across versions, and `picture_vectors` their SigLIP2 vectors.
    """
    with conn.transaction():
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO images (id, png) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                list((pictures or {}).items()),
            )
            cursor.executemany(
                "UPDATE images SET siglip = %s::vector WHERE id = %s",
                [
                    (vector_text(vector), identity)
                    for identity, vector in (picture_vectors or {}).items()
                ],
            )
        document_ids = {}
        for document in documents:
            row = conn.execute(
                "INSERT INTO documents (url, title, fetched_at, sha256, file_date) "
                "VALUES (%s, %s, %s, %s, %s::date) "
                "ON CONFLICT (url, sha256) DO UPDATE SET title = EXCLUDED.title, "
                "file_date = EXCLUDED.file_date RETURNING id",
                (document["url"], document["title"], document["fetched_at"],
                 document["sha256"], document.get("file_date") or None),
            ).fetchone()  # fmt: skip
            assert row is not None  # RETURNING always yields the row
            document_ids[document["url"]] = row[0]
        version = conn.execute(
            "INSERT INTO index_versions "
            "(corpus_sha256, passages_sha256, embedding_model) "
            "VALUES (%s, %s, %s) RETURNING id",
            (
                manifest["corpus_sha256"],
                manifest["passages_sha256"],
                manifest["embedding_model"],
            ),
        ).fetchone()
        assert version is not None
        values = []
        for passage, vector in zip(rows, vectors, strict=True):
            values.append(
                (
                    version[0],
                    document_ids[passage["url"]],
                    passage["title"],
                    passage["heading"],
                    passage["text"],
                    passage["context"],
                    passage["page"],
                    passage["kind"],
                    passage["channel"],
                    passage["quotable"],
                    passage["published"] or None,
                    passage["image"] or None,
                    vector_text(vector),
                )
            )
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO passages (index_version_id, document_id, title, heading, "
                "text, context, page, kind, channel, quotable, published, image, "
                "embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "%s::date, %s, %s::vector)",
                values,
            )
        keyword_indexes(conn, version[0])
    return int(version[0])


def known_vectors(
    conn: Connection, embedding_model: str
) -> dict[tuple[str, str, str], list[float]]:
    """The newest version embedded by `embedding_model`: each passage's (title,
    context, text) and its stored vector, so a new build embeds only new text."""
    row = conn.execute(
        "SELECT max(id) FROM index_versions WHERE embedding_model = %s",
        (embedding_model,),
    ).fetchone()
    if row is None or row[0] is None:
        return {}
    found = {}
    for title, context, text, vector in conn.execute(
        "SELECT title, context, text, embedding::text FROM passages "
        "WHERE index_version_id = %s",
        (row[0],),
    ):
        found[(title, context, text)] = [
            float(x) for x in vector.strip("[]").split(",")
        ]
    return found


def delete_version(conn: Connection, version_id: int) -> None:
    """Remove a version that is not live: its passages and its BM25 indexes."""
    with conn.transaction():
        deleted = conn.execute(
            "DELETE FROM index_versions WHERE id = %s AND NOT live", (version_id,)
        )
        if deleted.rowcount != 1:
            raise ValueError(f"no index version {version_id} that is not live")
        for name in CHANNELS:
            conn.execute(
                sql.SQL("DROP INDEX {}").format(
                    sql.Identifier(bm25_index(version_id, name))
                )
            )


def set_live(conn: Connection, version_id: int) -> None:
    """Make this version the one that is served, in a single transaction."""
    with conn.transaction():
        conn.execute("UPDATE index_versions SET live = false WHERE live")
        updated = conn.execute(
            "UPDATE index_versions SET live = true WHERE id = %s", (version_id,)
        )
        if updated.rowcount != 1:
            raise ValueError(f"no index version {version_id}")


def live_version(conn: Connection) -> tuple[int, str] | None:
    """The live version's id and embedding model, or None before the first build."""
    row = conn.execute(
        "SELECT id, embedding_model FROM index_versions WHERE live"
    ).fetchone()
    return (int(row[0]), str(row[1])) if row else None


def index_version(conn: Connection, version_id: int) -> tuple[int, str] | None:
    """One version's id and embedding model, or None if there is no such version."""
    row = conn.execute(
        "SELECT id, embedding_model FROM index_versions WHERE id = %s", (version_id,)
    ).fetchone()
    return (int(row[0]), str(row[1])) if row else None


def in_channel(name: str) -> sql.Composable:
    """A filter keeping one search channel's passages. A picture is searched by words
    only when it has words of its own: text, or a context line after its section
    path (`corpus.picture_rows`)."""
    found = sql.SQL(" AND channel = {}").format(sql.Literal(name))
    if name == "picture":
        found += sql.SQL(" AND (text <> '' OR strpos(context, chr(10)) > 0)")
    return found


def in_units(unit_ids: Sequence[int]) -> sql.Composable:
    """A filter keeping the passages in the scope of any of the units; no filter
    when there are none."""
    if not unit_ids:
        return sql.SQL("")
    return sql.SQL(
        " AND id IN (SELECT passage_id FROM unit_passages WHERE unit_id = ANY({}))"
    ).format(sql.Literal(list(unit_ids)))


def keyword_ranking(
    conn: Connection,
    version_id: int,
    question: str,
    limit: int,
    channel: str = "company",
    units: Sequence[int] = (),
) -> list[int]:
    """Passage ids of one channel ranked by BM25 over the question's words, inside
    the units' scope if any are given (`in_units`).

    Passages that share no word with the question score 0 and are left out, as a
    full-text match leaves them out. Ties keep passage id order.
    """
    if not re.search(r"\w", question):
        return []
    score = sql.SQL("{} <@> to_bm25query({}, {})").format(
        KEYWORD_TEXT,
        sql.Literal(question),
        sql.Literal(bm25_index(version_id, channel)),
    )
    query = sql.SQL(
        "SELECT id FROM passages WHERE index_version_id = {} AND {} < 0{}{} "
        "ORDER BY {}, id LIMIT {}"
    ).format(
        sql.Literal(version_id),
        score,
        in_channel(channel),
        in_units(units),
        score,
        sql.Literal(limit),
    )
    return [row[0] for row in conn.execute(query).fetchall()]


def vector_ranking(
    conn: Connection,
    version_id: int,
    query_vector: Sequence[float],
    limit: int,
    channel: str = "company",
    units: Sequence[int] = (),
) -> list[int]:
    """Passage ids of one channel ranked by cosine similarity (inner product of unit
    vectors), inside the units' scope if any are given."""
    query = sql.SQL(
        "SELECT id FROM passages WHERE index_version_id = %s{}{} "
        "ORDER BY embedding <#> %s::vector, id LIMIT %s"
    ).format(in_channel(channel), in_units(units))
    rows = conn.execute(
        query, (version_id, vector_text(query_vector), limit)
    ).fetchall()
    return [row[0] for row in rows]


def search(
    conn: Connection,
    version_id: int,
    question: str,
    embed: Embed,
    rerank: Rerank,
    top: int | None = None,
    also: Sequence[list[int]] = (),
    channel: str = "company",
    units: Sequence[int] = (),
) -> list[Passage]:
    """The top passages of one index version's channel for a question, best first:
    `top` of them, or `config.TOP_K`, inside the units' scope if any are given. The
    fused rankings' best candidates ordered by the reranker (`retrieve.rerank_top`)."""
    ranking = fused(conn, version_id, question, embed, also, channel, units)
    candidates = load_passages(conn, ranking[: config.RERANK_CANDIDATES])
    return rerank_top(question, candidates, rerank, top)


def fused(
    conn: Connection,
    version_id: int,
    question: str,
    embed: Embed,
    also: Sequence[list[int]] = (),
    channel: str = "company",
    units: Sequence[int] = (),
) -> list[int]:
    """One channel's keyword and vector rankings for a question, inside the units'
    scope if any are given, and any rankings `also` given (passage ids, best first,
    such as `picture_ranking`), fused: passage ids, best first."""
    query_vector = embed([config.QUERY_INSTRUCTION + question])[0]
    limit = config.CANDIDATES_PER_METHOD
    return fuse(
        [
            keyword_ranking(conn, version_id, question, limit, channel, units),
            vector_ranking(conn, version_id, query_vector, limit, channel, units),
            *also,
        ]
    )


def picture_ranking(
    conn: Connection, version_id: int, query_vector: Sequence[float], limit: int
) -> list[int]:
    """Picture passage ids ranked by what their pictures show: SigLIP2's cosine
    between the question and each stored picture (Lime Green X44 F2)."""
    rows = conn.execute(
        "SELECT p.id FROM passages p JOIN images i ON i.id = p.image "
        "WHERE p.index_version_id = %s AND p.channel = 'picture' "
        "AND i.siglip IS NOT NULL ORDER BY i.siglip <#> %s::vector, p.id LIMIT %s",
        (version_id, vector_text(query_vector), limit),
    ).fetchall()
    return [row[0] for row in rows]


def passage_texts(conn: Connection, version_id: int) -> list[tuple[int, str, str, str]]:
    """Every passage of a version as (id, document url, title, text), in id order:
    what the units' scopes are read from (`units.members`)."""
    rows = conn.execute(
        "SELECT p.id, d.url, p.title, p.text FROM passages p "
        "JOIN documents d ON d.id = p.document_id "
        "WHERE p.index_version_id = %s ORDER BY p.id",
        (version_id,),
    ).fetchall()
    return [(row[0], row[1], row[2], row[3]) for row in rows]


def write_units(
    conn: Connection,
    version_id: int,
    units: Sequence[Mapping[str, str]],
    members: Sequence[tuple[int, int]],
) -> list[int]:
    """Store a version's units (name, acronym, url, former names) and the passages
    in each one's scope, as (place in `units`, passage id); return the units' ids,
    in order."""
    with conn.transaction():
        ids = []
        for unit in units:
            row = conn.execute(
                "INSERT INTO units (index_version_id, name, acronym, url, aliases) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (version_id, unit["name"], unit["acronym"], unit["url"],
                 unit.get("aliases", "")),
            ).fetchone()  # fmt: skip
            assert row is not None  # RETURNING always yields the row
            ids.append(int(row[0]))
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO unit_passages (unit_id, passage_id) VALUES (%s, %s)",
                [(ids[place], passage_id) for place, passage_id in members],
            )
    return ids


def write_structure(
    conn: Connection, version_id: int, rows: Sequence[tuple[str, str, int | None]]
) -> None:
    """Store a version's structure: (group, member, unit id or None) in the
    units page's order."""
    with conn.transaction(), conn.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO structure (index_version_id, position, group_name, member, "
            "unit_id) VALUES (%s, %s, %s, %s, %s)",
            [(version_id, n, group, member, unit_id)
             for n, (group, member, unit_id) in enumerate(rows, start=1)],
        )  # fmt: skip


def version_structure(conn: Connection, version_id: int) -> list[dict[str, Any]]:
    """A version's structure as groups (name and entries), in the page's order."""
    rows = conn.execute(
        "SELECT group_name, member FROM structure WHERE index_version_id = %s "
        "ORDER BY position",
        (version_id,),
    ).fetchall()
    groups: list[dict[str, Any]] = []
    for group, member in rows:
        if not groups or groups[-1]["name"] != group:
            groups.append({"name": group, "entries": []})
        groups[-1]["entries"].append(member)
    return groups


def compiled_passage(conn: Connection, version_id: int) -> Passage | None:
    """A version's compiled units passage, if it holds one."""
    row = conn.execute(
        "SELECT id FROM passages WHERE index_version_id = %s AND heading = %s "
        "ORDER BY id LIMIT 1",
        (version_id, units.COMPILED),
    ).fetchone()
    return load_passages(conn, [row[0]])[0] if row else None


def version_units(conn: Connection, version_id: int) -> list[dict[str, Any]]:
    """A version's units (name, acronym, url, former names, id), in the order they
    were listed."""
    rows = conn.execute(
        "SELECT id, name, acronym, url, aliases FROM units "
        "WHERE index_version_id = %s ORDER BY id",
        (version_id,),
    ).fetchall()
    return [{"name": row[1], "acronym": row[2], "url": row[3], "aliases": row[4],
             "id": row[0]} for row in rows]  # fmt: skip


def dated_passages(
    conn: Connection, version_id: int
) -> list[tuple[int, str, str, date | None, str]]:
    """A version's quotable company posts and pages as (id, title, text, publication
    date or None, capture date YYYY-MM-DD): the passages role statements are read
    from."""
    rows = conn.execute(
        "SELECT p.id, p.title, p.text, p.published, d.fetched_at FROM passages p "
        "JOIN documents d ON d.id = p.document_id WHERE p.index_version_id = %s "
        "AND p.kind IN ('post', 'page') AND p.channel = 'company' AND p.quotable "
        "ORDER BY p.id",
        (version_id,),
    ).fetchall()
    found = []
    for passage_id, title, text, published, fetched_at in rows:
        captured = fetched_at.astimezone(UTC).date().isoformat()
        found.append((passage_id, title, text, published, captured))
    return found


def role_readings(
    conn: Connection, keys: Sequence[str]
) -> dict[str, list[dict[str, str]]]:
    """The statements read before from each passage text (`roles.text_sha256`)."""
    rows = conn.execute(
        "SELECT passage_sha256, statements FROM role_readings "
        "WHERE passage_sha256 = ANY(%s)",
        (list(keys),),
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def write_role_reading(
    conn: Connection, key: str, statements: Sequence[Mapping[str, str]]
) -> None:
    """Keep one passage text's statements, so no later build reads it again."""
    conn.execute(
        "INSERT INTO role_readings (passage_sha256, statements) VALUES (%s, %s) "
        "ON CONFLICT (passage_sha256) DO NOTHING",
        (key, Jsonb([dict(s) for s in statements])),
    )
    conn.commit()


def write_role_statements(
    conn: Connection, version_id: int, statements: Sequence[Mapping[str, Any]]
) -> None:
    """Store a version's role statements, each with its passage and its date."""
    with conn.transaction(), conn.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO role_statements (index_version_id, passage_id, person, role, "
            "organisation, quote, stated_on, date_kind) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s::date, %s)",
            [
                (
                    version_id,
                    s["passage_id"],
                    s["person"],
                    s["role"],
                    s["organisation"],
                    s["quote"],
                    s["stated_on"],
                    s["date_kind"],
                )
                for s in statements
            ],  # fmt: skip
        )


def version_role_statements(conn: Connection, version_id: int) -> list[dict[str, Any]]:
    """A version's role statements, each with its passage's title and address and its
    date as YYYY-MM-DD."""
    rows = conn.execute(
        "SELECT s.passage_id, s.person, s.role, s.organisation, s.quote, s.stated_on, "
        "s.date_kind, p.title, d.url FROM role_statements s "
        "JOIN passages p ON p.id = s.passage_id "
        "JOIN documents d ON d.id = p.document_id "
        "WHERE s.index_version_id = %s ORDER BY s.id",
        (version_id,),
    ).fetchall()
    found = []
    for row in rows:
        passage_id, person, role, organisation, quote, stated_on, kind = row[:7]
        title, url = row[7:]
        found.append({"passage_id": passage_id, "person": person, "role": role,
                      "organisation": organisation, "quote": quote,
                      "stated_on": stated_on.isoformat(), "date_kind": kind,
                      "title": title, "url": url})  # fmt: skip
    return found


def load_passages(conn: Connection, passage_ids: Sequence[int]) -> list[Passage]:
    """The passages with these ids, in the order given."""
    rows = conn.execute(
        "SELECT passages.id, documents.url, passages.title, passages.heading, "
        "passages.text, documents.fetched_at, passages.page, passages.context, "
        "coalesce(passages.image, ''), passages.quotable, "
        "coalesce(passages.published::text, ''), passages.channel, "
        "coalesce(documents.file_date::text, '') "
        "FROM passages JOIN documents ON documents.id = passages.document_id "
        "WHERE passages.id = ANY(%s)",
        (list(passage_ids),),
    ).fetchall()
    by_id = {}
    for row in rows:
        passage_id, url, title, heading, text, fetched_at, page, context = row[:8]
        image, quotable, published, channel, file_date = row[8:]
        captured = fetched_at.astimezone(UTC).isoformat(timespec="seconds")
        by_id[passage_id] = Passage(
            passage_id, url, title, heading, text, captured, page, context, image,
            quotable, published, channel, file_date,
        )  # fmt: skip
    return [by_id[passage_id] for passage_id in passage_ids]


def has_pictures(conn: Connection, version_id: int) -> bool:
    """Whether an index version holds picture passages: without any, a search never
    loads the picture model."""
    row = conn.execute(
        "SELECT EXISTS (SELECT 1 FROM passages WHERE index_version_id = %s "
        "AND channel = 'picture')",
        (version_id,),
    ).fetchone()
    assert row is not None  # EXISTS always yields a row
    return bool(row[0])


def passage_count(conn: Connection, version_id: int) -> int:
    """How many passages an index version holds."""
    row = conn.execute(
        "SELECT count(*) FROM passages WHERE index_version_id = %s", (version_id,)
    ).fetchone()
    assert row is not None  # count(*) always yields a row
    return int(row[0])


def record_answer(
    conn: Connection,
    question: str,
    status: str,
    shown: dict[str, Any],
    removed: list[dict[str, str]],
    passage_ids: Sequence[int],
    index_version_id: int,
    embedding_model: str,
    prompt_sha256: str,
    seconds: float,
    search_questions: Sequence[str] = (),
    conversation_id: str | None = None,
    turn: int | None = None,
    calls: Sequence[Mapping[str, Any]] = (),
) -> int:
    """Store one answer's audit record (plain, JSON-compatible data); return its id.
    An answer in a conversation keeps its conversation and turn (`open_turn`). Its
    model `calls` (purpose, system, user, reply, seconds), in order, are its
    provenance, written in the same transaction."""
    row = conn.execute(
        "INSERT INTO answers (question, status, shown, removed, passage_ids, "
        "index_version_id, embedding_model, prompt_sha256, seconds, "
        "search_questions, conversation_id, turn) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
        (
            question,
            status,
            Jsonb(shown),
            Jsonb(removed),
            list(passage_ids),
            index_version_id,
            embedding_model,
            prompt_sha256,
            seconds,
            Jsonb(list(search_questions)),
            conversation_id,
            turn,
        ),
    ).fetchone()
    assert row is not None  # RETURNING always yields the row
    answer_id = int(row[0])
    for position, call in enumerate(calls, 1):
        conn.execute(
            "INSERT INTO model_calls (answer_id, position, purpose, system, "
            "user_message, reply, seconds) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (answer_id, position, call["purpose"], call["system"], call["user"],
             Jsonb(call["reply"]), call["seconds"]),
        )  # fmt: skip
    conn.commit()
    return answer_id


def record_failure(
    conn: Connection,
    question: str,
    reason: str,
    detail: str,
    conversation_id: str | None = None,
    turn: int | None = None,
) -> None:
    """Keep a question that got no answer, with its server-side cause: the
    conversation had ended, or the index or a model could not answer."""
    conn.execute(
        "INSERT INTO question_failures (question, conversation_id, turn, reason, "
        "detail) VALUES (%s, %s, %s, %s, %s)",
        (question, conversation_id, turn, reason, detail),
    )
    conn.commit()


def record_feedback(
    conn: Connection, answer_id: int, conversation_id: str, helpful: bool, comment: str
) -> bool:
    """Keep a reader's judgement of an answer, only from the conversation the answer
    belongs to; whether it was kept."""
    kept = conn.execute(
        "INSERT INTO answer_feedback (answer_id, helpful, comment) "
        "SELECT id, %s, %s FROM answers WHERE id = %s AND conversation_id = %s",
        (helpful, comment, answer_id, conversation_id),
    ).rowcount
    conn.commit()
    return kept == 1


def question_failures(
    conn: Connection, conversation_id: str | None = None
) -> list[dict[str, Any]]:
    """The questions that got no answer, oldest first: one conversation's, or all."""
    where = sql.SQL("WHERE conversation_id = %s" if conversation_id else "")
    rows = conn.execute(
        sql.SQL(
            "SELECT created_at, conversation_id, turn, question, reason, detail "
            "FROM question_failures {} ORDER BY id"
        ).format(where),
        (conversation_id,) if conversation_id else (),
    ).fetchall()
    return [
        {
            "created": created.isoformat(),
            "conversation_id": str(conversation) if conversation else None,
            "turn": turn,
            "question": question,
            "reason": reason,
            "detail": detail,
        }
        for created, conversation, turn, question, reason, detail in rows
    ]


def recent_conversations(conn: Connection, limit: int) -> list[dict[str, Any]]:
    """The latest conversations, newest first: id, start, last activity, turns and
    first question (history; read on this machine only)."""
    rows = conn.execute(
        "SELECT c.id, c.created_at, c.last_active_at, c.turns, "
        "(SELECT question FROM answers a WHERE a.conversation_id = c.id "
        "ORDER BY a.turn LIMIT 1) FROM conversations c "
        "ORDER BY c.created_at DESC LIMIT %s",
        (limit,),
    ).fetchall()
    return [
        {
            "id": str(found),
            "started": started.isoformat(),
            "last_active": active.isoformat(),
            "turns": turns,
            "first_question": first or "",
        }
        for found, started, active, turns, first in rows
    ]


def answer_records(
    conn: Connection, conversation_id: str | None = None
) -> list[dict[str, Any]]:
    """Every recorded answer with its provenance, oldest first: one conversation's
    turns when `conversation_id` is given, otherwise all answers (an export)."""
    where = sql.SQL("WHERE conversation_id = %s" if conversation_id else "")
    rows = conn.execute(
        sql.SQL(
            "SELECT id, created_at, conversation_id, turn, question, status, shown, "
            "removed, search_questions, passage_ids, index_version_id, "
            "embedding_model, prompt_sha256, seconds FROM answers {} ORDER BY id"
        ).format(where),
        (conversation_id,) if conversation_id else (),
    ).fetchall()
    records = []
    for row in rows:
        (answer_id, created, conversation, turn, question, status, shown, removed,
         searched, passage_ids, version, model, prompt, seconds) = row  # fmt: skip
        calls = conn.execute(
            "SELECT purpose, system, user_message, reply, seconds FROM model_calls "
            "WHERE answer_id = %s ORDER BY position",
            (answer_id,),
        ).fetchall()
        feedback = conn.execute(
            "SELECT created_at, helpful, comment FROM answer_feedback "
            "WHERE answer_id = %s ORDER BY id",
            (answer_id,),
        ).fetchall()
        records.append(
            {
                "id": answer_id,
                "created": created.isoformat(),
                "conversation_id": str(conversation) if conversation else None,
                "turn": turn,
                "question": question,
                "status": status,
                "shown": shown,
                "removed": removed,
                "search_questions": searched,
                "passage_ids": list(passage_ids),
                "index_version_id": version,
                "embedding_model": model,
                "prompt_sha256": prompt,
                "seconds": seconds,
                "model_calls": [
                    {
                        "purpose": purpose,
                        "system": system,
                        "user": user,
                        "reply": reply,
                        "seconds": took,
                    }
                    for purpose, system, user, reply, took in calls
                ],  # fmt: skip
                "feedback": [
                    {"created": when.isoformat(), "helpful": helpful, "comment": text}
                    for when, helpful, text in feedback
                ],
            }
        )
    return records


def open_turn(
    conn: Connection,
    conversation_id: str | None,
    idle_minutes: int,
    max_hours: int,
    max_turns: int,
) -> tuple[str, int] | None:
    """Reserve the next turn: (conversation id, turn number), or None when the
    conversation is unknown, idle for `idle_minutes`, `max_hours` old or at
    `max_turns`. Without an id, a new conversation starts at turn 1.

    One UPDATE checks the limits and takes the number: under Read Committed a
    concurrent turn waits for it and re-checks the row, so no two turns share a
    number and no lock is held while the answer is written."""
    if conversation_id is None:
        row = conn.execute(
            "INSERT INTO conversations (turns) VALUES (1) RETURNING id, turns"
        ).fetchone()
    else:
        row = conn.execute(
            "UPDATE conversations SET turns = turns + 1, last_active_at = now() "
            "WHERE id = %s AND last_active_at > now() - make_interval(mins => %s) "
            "AND created_at > now() - make_interval(hours => %s) AND turns < %s "
            "RETURNING id, turns",
            (conversation_id, idle_minutes, max_hours, max_turns),
        ).fetchone()
    conn.commit()
    return (str(row[0]), int(row[1])) if row else None


def earlier_turns(
    conn: Connection, conversation_id: str, turn: int, count: int
) -> list[tuple[str, Any]]:
    """The last `count` answered turns before `turn`, oldest first: each question
    and the view its reader was shown. A turn that failed recorded nothing."""
    rows = conn.execute(
        "SELECT question, shown FROM answers WHERE conversation_id = %s "
        "AND turn < %s ORDER BY turn DESC LIMIT %s",
        (conversation_id, turn, count),
    ).fetchall()
    return [(question, shown) for question, shown in reversed(rows)]
