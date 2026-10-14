"""Build an index version from what was read (Lime Green's ingest, the index build
only).

Every passage of a company (`corpus.build`) is embedded once: a passage whose title,
context and text the newest version of the same embedder already holds keeps its
stored vector (Lime Green X43 A4). The version is fingerprinted and written to
Postgres beside the live one (`store.write_version`), with the pictures its passages
show and their SigLIP2 vectors, and with the business units its company's units
page lists and the passages in each one's scope.
"""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from petroknowledge import config, corpus, roles, store, units
from petroknowledge.records import described
from petroknowledge.retrieve import Embed

Chat = Callable[[str, str, dict[str, Any]], object]

PICTURE_VECTORS = "vectors.json"  # SigLIP2 vectors beside the pictures (read-images)


def corpus_hash(documents: Sequence[Mapping[str, str]]) -> str:
    """One fingerprint for the sources, from their sorted (url, sha256) pairs."""
    pairs = sorted((document["url"], document["sha256"]) for document in documents)
    lines = "".join(f"{url} {sha256}\n" for url, sha256 in pairs)
    return hashlib.sha256(lines.encode()).hexdigest()


def passages_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    """One fingerprint for the passages as indexed, every field counted: two builds
    with the same fingerprint search the same passages."""
    lines = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    return hashlib.sha256(lines.encode()).hexdigest()


def vectors_for(
    rows: Sequence[Mapping[str, Any]],
    embed: Embed,
    known: Mapping[tuple[str, str, str], list[float]],
) -> tuple[list[list[float]], int]:
    """One vector per passage, and how many were embedded now: a passage `known`
    by its (title, context, text) keeps its vector; the rest are embedded in
    batches, each as the search models read it (`records.described`)."""
    vectors = [
        known.get((row["title"], row["context"], row["text"]), []) for row in rows
    ]
    missing = [number for number, vector in enumerate(vectors) if not vector]
    for start in range(0, len(missing), config.EMBEDDING_BATCH_SIZE):
        batch = missing[start : start + config.EMBEDDING_BATCH_SIZE]
        texts = []
        for number in batch:
            row = rows[number]
            texts.append(described(row["title"], row["context"], row["text"]))
        for number, vector in zip(batch, embed(texts), strict=True):
            vectors[number] = vector
    return vectors, len(missing)


def stored_pictures(
    ids: Sequence[str], folder: Path
) -> tuple[dict[str, bytes], dict[str, list[float]]]:
    """The PNG of each picture and the SigLIP2 vectors made for them, by id."""
    path = folder / PICTURE_VECTORS
    found = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    pngs = {identity: (folder / f"{identity}.png").read_bytes() for identity in ids}
    vectors = {identity: found[identity] for identity in ids if identity in found}
    return pngs, vectors


def role_statements(
    conn: store.Connection, version_id: int, chat: Chat | None
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """A version's role statements: each dated post or page passage that names
    a role title is read once (`roles.extract`); its reading is kept by its text, so
    only new passages are read. With no `chat`, only kept readings are used. Returns
    the statements, each dated, and how many passages were read, left unread or
    answered outside the schema."""
    passages = [p for p in store.dated_passages(conn, version_id)
                if roles.states_roles(p[2])]  # fmt: skip
    keys = [roles.text_sha256(title, text) for _, title, text, _, _ in passages]
    known = store.role_readings(conn, keys)
    counts = {
        "role_passages_read": 0,
        "role_passages_unread": 0,
        "role_replies_broken": 0,
    }
    found: list[dict[str, Any]] = []
    for (passage_id, title, text, published, captured), key in zip(
        passages, keys, strict=True
    ):
        if key not in known:
            if chat is None:
                counts["role_passages_unread"] += 1
                continue
            read = roles.extract(title, text, chat)
            if read is None:
                counts["role_replies_broken"] += 1
                continue
            store.write_role_reading(conn, key, read)
            known[key] = read
            counts["role_passages_read"] += 1
        for statement in known[key]:
            stated_on = published.isoformat() if published else captured
            kind = "published" if published else "captured"
            named = roles.as_named(statement, f"{title}\n{text}")
            found.append(named | {"passage_id": passage_id,
                                  "stated_on": stated_on,
                                  "date_kind": kind})  # fmt: skip
    return found, counts


def structure_rows(
    company_id: str, listed: list[units.Unit], unit_ids: list[int]
) -> list[tuple[str, str, int | None]]:
    """The units page's groups as structure edges: (group, member, unit id),
    a member linked to the unit it names, if any."""
    named = units.naming(listed)
    rows = []
    for group in corpus.listed_groups(company_id):
        for member in group["entries"]:
            places = named(member)
            unit_id = unit_ids[places[0]] if places else None
            rows.append((group["name"], member, unit_id))
    return rows


def build(
    conn: store.Connection,
    company_id: str,
    embed: Embed,
    live: bool = False,
    chat: Chat | None = None,
    compiled_units: bool = False,
) -> tuple[int, dict[str, str]]:
    """Build a new index version of a company's passages and, if `live`, make it
    the one served; return its id and manifest. The version is written beside the
    live one, so a failed build leaves the served index untouched. With
    `compiled_units`, the units page's list is one compiled passage too."""
    rows, documents = corpus.build(company_id, compiled_units)
    listed = corpus.listed_units(company_id)
    known = store.known_vectors(conn, config.EMBEDDING_MODEL)
    vectors, embedded = vectors_for(rows, embed, known)
    ids = sorted({row["image"] for row in rows if row["image"]})
    pngs, picture_vectors = stored_pictures(ids, config.IMAGES)
    manifest = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "company": company_id,
        "corpus_sha256": corpus_hash(documents),
        "passages_sha256": passages_hash(rows),
        "embedding_model": config.EMBEDDING_MODEL,
        "sources": str(len(documents)),
        "passages": str(len(rows)),
        "embedded": str(embedded),
        "pictures": str(len(ids)),
        "picture_vectors": str(len(picture_vectors)),
        "units": str(len(listed)),
    }
    version = store.write_version(
        conn, documents, rows, vectors, manifest, pngs, picture_vectors
    )
    scope = units.members(listed, store.passage_texts(conn, version))
    unit_ids = store.write_units(conn, version, listed, scope)
    store.write_structure(conn, version, structure_rows(company_id, listed, unit_ids))
    statements, counts = role_statements(conn, version, chat)
    store.write_role_statements(conn, version, statements)
    manifest |= {"role_statements": str(len(statements))}
    manifest |= {name: str(count) for name, count in counts.items()}
    if live:
        store.set_live(conn, version)
    return version, manifest
