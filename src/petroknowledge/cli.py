"""Command line: acquire collects a company's core sources, download-listed and add
bring in files from crawler-forbidden hosts, links and news keep often-updated pages
as links, read-pages and read-documents read what was collected, transcribe reads
scanned PDF pages, index builds an index version, ask answers a question from it,
serve runs the API, openapi writes its schema, history reads the recorded
conversations, browse names the stored files, models checks the model files."""

import argparse
import json
import sys
import uuid
from pathlib import Path

import psycopg

from petroknowledge import (
    config,
    documents,
    fetch,
    inbox,
    links,
    llm,
    pages,
    sources,
    transcribe,
    weights,
)
from petroknowledge.fetch import Record, Robots
from petroknowledge.view import INDEPENDENT, AnswerView, view


def core_wanted(company: sources.Company) -> tuple[list[Record], list[Record]]:
    """Every page and post its sitemaps list, then its listed URLs, as wanted
    records; and the sitemaps that could not be read."""
    wanted: list[Record] = []
    failures: list[Record] = []
    known: Robots = {}
    sitemaps = [(url, "page") for url in company["core_sitemaps"]]
    sitemaps += [(url, "post") for url in company["post_sitemaps"]]
    with fetch.polite_client() as client:
        for sitemap, kind in sitemaps:
            xml = fetch.get_text(client, known, sitemap)
            if isinstance(xml, dict):
                failures.append(xml)
                continue
            for entry in fetch.sitemap_entries(xml):
                wanted.append(
                    {
                        "url": entry["loc"],
                        "company": company["id"],
                        "kind": kind,
                        "listed_lastmod": entry.get("lastmod", ""),
                    }
                )
    for item in company["core_urls"]:
        wanted.append({**item, "company": company["id"]})
    return wanted, failures


def linked_wanted(company: sources.Company, with_pictures: bool) -> list[Record]:
    """The documents linked from the company's stored pages, and their pictures
    when asked for (most are logos, banners and photos)."""
    documents, pictures = fetch.stored_links(company["id"], company["allowed_hosts"])
    wanted = [
        {"url": url, "company": company["id"], "kind": "document"} for url in documents
    ]
    if with_pictures:
        wanted += [
            {"url": url, "company": company["id"], "kind": "picture"}
            for url in pictures
        ]
    return wanted


def save_measurement(company_id: str, what: str, results: list[Record]) -> None:
    config.MEASUREMENTS.mkdir(parents=True, exist_ok=True)
    path = config.MEASUREMENTS / f"{company_id}-{what}.json"
    path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(f"sizes saved to {path}")


def report(results: list[Record]) -> None:
    fine = [record for record in results if "error" not in record]
    sizes = [record["bytes"] for record in fine if record.get("bytes") is not None]
    print(f"{len(fine)} of {len(results)} fine, {sum(sizes) / 1e6:.1f} MB")
    for record in results:
        if "error" in record:
            print(f"  failed: {record['url']}: {record['error']}")


def run_acquire(
    company_id: str, linked: bool, measure_only: bool, with_pictures: bool = False
) -> None:
    """The company's core sources, or the files its stored pages link, not yet
    stored; `measure_only` sizes them without downloading."""
    company = sources.load(company_id)
    failures: list[Record] = []
    if linked:
        wanted = linked_wanted(company, with_pictures)
    else:
        wanted, failures = core_wanted(company)
    stored = {record["url"] for record in fetch.read_manifest()}
    missing = [item for item in wanted if item["url"] not in stored]
    if measure_only:
        results = fetch.measure([item["url"] for item in missing])
        report(failures + results)
        save_measurement(company_id, "linked" if linked else "core", results)
        return
    max_bytes = company["max_file_mb"] * 1_000_000
    report(failures + fetch.acquire(missing, company["allowed_hosts"], max_bytes))


def run_download_listed(company_id: str) -> None:
    """Download the files data/inbox/<company>-to-download.txt names into the
    company's inbox, on the owner's instruction (see `inbox.download_listed`)."""
    company = sources.load(company_id)
    listed = inbox.listed_urls(config.INBOX / f"{company_id}-to-download.txt")
    report(
        inbox.download_listed(
            listed,
            config.INBOX / company_id,
            company["allowed_hosts"],
            company["max_file_mb"] * 1_000_000,
        )
    )


def run_add(company_id: str) -> None:
    folder = config.INBOX / company_id
    results = inbox.add_folder(sources.load(company_id), folder)
    print(f"from {folder}:")
    report(results)


def run_links(company_id: str) -> None:
    added, failures = links.collect_posts(sources.load(company_id))
    print(f"{added} links added or updated")
    report(failures)


def run_read_documents(vlm: str) -> int:
    """Read every stored PDF and spreadsheet; print the run's totals, failures and
    flagged pages."""
    if vlm and not llm.healthy(vlm):
        print(f"error: no vision model ready at {vlm}", file=sys.stderr)
        return 1
    files = documents.document_files(fetch.read_manifest())
    report = documents.run_all(files, vlm, documents.OUT)
    print(
        f"{len(files)} documents: {len(report['read'])} read, "
        f"{len(report['skipped'])} already read, {len(report['failed'])} failed"
    )
    for sha256, failure in report["failed"].items():
        print(f"  failed {failure['urls'][0]} ({sha256[:12]}): {failure['errors'][-1]}")
    for key, value in report["summary"]["totals"].items():
        print(f"{key}: {value:g}" if isinstance(value, float) else f"{key}: {value}")
    for page in report["summary"]["flagged"]:
        name = page["url"].rsplit("/", 1)[-1]
        print(f"  flagged {name} p{page['page']}: {', '.join(page['flags'])} "
              f"({page['kept']}/{page['words']} words, {page['grade']})")  # fmt: skip
    print(f"report: {documents.OUT / 'report.json'}")
    return 1 if report["failed"] else 0


def listed_pages(path: Path) -> set[tuple[str, int]]:
    """The (sha256, page) pairs a JSON file lists under "pages", as an evaluation
    set's truth.json does."""
    listed = json.loads(path.read_text(encoding="utf-8"))["pages"]
    return {(str(page["sha256"]), int(page["page"])) for page in listed}


def run_transcribe(vlm: str, pages_file: Path | None, out: Path) -> int:
    """Transcribe the scanned pages of every read PDF, or only those a file lists;
    print the run's counts and the pages that failed (tried again next run)."""
    if not llm.healthy(vlm):
        print(f"error: no vision model ready at {vlm}", file=sys.stderr)
        return 1
    only = listed_pages(pages_file) if pages_file else None
    report = transcribe.transcribe_all(vlm, config.READINGS, out, only=only)
    print(
        f"{report['documents']} documents with scanned pages: {report['pages']} "
        f"pages transcribed, {report['skipped']} already transcribed, "
        f"{len(report['failed'])} failed"
    )
    for page in report["failed"]:
        name = page["url"].rsplit("/", 1)[-1]
        print(f"  failed {name} p{page['page']}: {page['error']}")
    print(f"transcripts: {out}")
    return 1 if report["failed"] else 0


def run_read_images(vlm: str, vectors: bool) -> int:
    """Store every picture of the posts and documents once, then read the text in
    each with a vision model (`vlm`) and give each its SigLIP2 vector (`vectors`);
    what is already saved the same way is kept."""
    from petroknowledge import images, siglip

    if not vlm and not vectors:
        print("error: give --vlm, --vectors or both", file=sys.stderr)
        return 1
    if vlm and not llm.healthy(vlm):
        print(f"error: no vision model ready at {vlm}", file=sys.stderr)
        return 1
    records = [r for r in fetch.read_manifest() if "sha256" in r]
    stored = {r["url"]: fetch.store_path(r["sha256"]) for r in records}
    posts = [
        (r["url"], stored[r["url"]].read_text(encoding="utf-8", errors="replace"),
         sources.host_rules(r["url"]).get("media_base", ""))
        for r in records
        if r["kind"] == "post"
    ]  # fmt: skip
    readings = []
    for path in sorted(documents.OUT.glob("*.json")):
        if len(path.stem) != 64:
            continue  # the run's report, which carries the reader's fingerprint too
        reading = json.loads(path.read_text(encoding="utf-8"))
        if "pypdfium2" in reading.get("fingerprint", {}).get("packages", {}):
            readings.append(reading)  # a PDF's reading, not a spreadsheet's
    places = images.collect(posts, readings, stored, config.IMAGES)
    text = json.dumps(places, indent=1, ensure_ascii=False) + "\n"
    (config.IMAGES / "places.json").write_text(text, encoding="utf-8", newline="\n")
    read = 0
    if vlm:
        found = transcribe.fingerprint(vlm)
        reader = {key: found[key] for key in ("vision_model", "prompt", "max_tokens",
                                              "sampling")}  # fmt: skip
        read = images.read_all(places, config.IMAGES, vlm, reader)
    ids = list(dict.fromkeys(place["id"] for place in places))
    made = 0
    if vectors:
        path = config.IMAGES / "vectors.json"
        known = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        embed = siglip.picture_model(config.SIGLIP)
        vectors_found = siglip.picture_vectors(ids, config.IMAGES, embed, known)
        path.write_text(json.dumps(vectors_found), encoding="utf-8")
        made = len(set(vectors_found) - set(known))
    print(f"{len(places)} places of {len(ids)} pictures; {read} read and {made} "
          "vectors made now")  # fmt: skip
    return 0


def run_read_pages(company_id: str) -> None:
    """Read every stored page of the company; print totals and thin pages."""
    sources.load(company_id)  # an unknown company is an error, not an empty run
    report = pages.read_all(company_id)
    for key, value in report["totals"].items():
        print(f"{key}: {value}")
    for page in report["thin"]:
        print(f"  thin: {page['url']} ({page['words']} words)")
    print(f"readings: {config.PAGE_READINGS}")


def run_index(
    company_id: str, live: bool, read_roles: bool = False, compiled_units: bool = False
) -> int:
    """Build a company's index version from what was read and print its manifest;
    with `live`, make it the version served. With `read_roles`, passages whose role
    statements were never read are read by the chat model (K24); otherwise only kept
    readings are used. With `compiled_units`, the units page's list is one compiled
    passage too (K27)."""
    from petroknowledge import ingest

    if not config.DATABASE_URL:
        print("error: PETROKNOWLEDGE_DATABASE_URL is not set (.env)", file=sys.stderr)
        return 1
    if not llm.healthy(config.EMBEDDING_URL):
        print(f"error: no embedding model ready at {config.EMBEDDING_URL}",
              file=sys.stderr)  # fmt: skip
        return 1
    if read_roles and not llm.healthy(config.CHAT_URL):
        print(f"error: no chat model ready at {config.CHAT_URL}", file=sys.stderr)
        return 1
    done = 0

    def embed(texts: list[str]) -> list[list[float]]:
        nonlocal done
        found = llm.embed(texts)
        done += len(texts)
        print(f"{done} passages embedded", flush=True)
        return found

    with psycopg.connect(config.DATABASE_URL) as conn:
        chat = llm.chat if read_roles else None
        version, manifest = ingest.build(
            conn, company_id, embed, live, chat, compiled_units
        )
    print(json.dumps(manifest, indent=1))
    print(f"index version {version}: {'live' if live else 'not live'}")
    return 0


def render_text(shown: AnswerView) -> str:
    """The answer as plain text: its claims, any notice, the numbered sources with
    their dates, the closest pages of a refusal, the search-only pages it points to,
    and the independence notice."""
    lines = ["Answer:"]
    for number, claim in enumerate(shown["claims"], 1):
        refs = " ".join(f"[{source}]" for source in claim["sources"])
        lines.append(f"{number}. {claim['text']} {refs}")
    if shown["notice"]:
        if shown["claims"]:
            lines += ["", "Note:"]
        lines += shown["notice"].splitlines()
    if shown["sources"]:
        lines += ["", "Sources:"]
        for source in shown["sources"]:
            published = (
                f"published {source['published']}, " if source["published"] else ""
            )
            lines += [
                f"[{source['number']}] {source['title']} › {source['heading']} "
                f"({published}captured {source['captured']})",
                f'    "{source["quote"]}"',
                f"    {source['link']}",
            ]
    if shown["closest_pages"]:
        lines += ["", "Closest pages:"]
        lines += [
            f"- {page['title']}: {page['url']}" for page in shown["closest_pages"]
        ]
    if shown["pointers"]:
        lines += ["", "Also on these pages (not quoted):"]
        lines += [
            f"- {p['title']} ({p['kind']}): {p['link']}" for p in shown["pointers"]
        ]
    return "\n".join([*lines, "", INDEPENDENT])


def run_ask(question: str) -> int:
    """Answer one question from the served index, record it and print it."""
    from petroknowledge import assistant

    try:
        result, _ = assistant.ask_and_record(question)
    except (assistant.Unavailable, llm.ModelServerError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(render_text(view(result)))
    return 0


def run_serve(port: int) -> None:
    """Run the API on this machine (Lime Green's serve). uvicorn logs the address
    once it is listening, or why it could not start. Its access lines are off:
    request spans and metrics record every request."""
    import uvicorn

    from petroknowledge import telemetry
    from petroknowledge.app import app

    uvicorn.run(
        app,
        host=config.APP_HOST,
        port=port,
        log_config=telemetry.LOG_CONFIG,
        access_log=False,
    )


def run_history(conversation: str | None, export: Path | None, limit: int) -> int:
    """Read the recorded conversations back, with each answer's model calls: the
    latest conversations, one conversation in full, or every answer as JSON Lines.
    Only this command reads them: the API never serves them, since the public link
    would show anyone other people's questions."""
    from petroknowledge import assistant, store

    if conversation is not None:
        try:
            uuid.UUID(conversation)
        except ValueError:
            print(f"error: {conversation} is not a conversation id", file=sys.stderr)
            return 1
    try:
        with assistant.connect() as conn:
            if export is not None:
                answers = store.answer_records(conn)
                failures = store.question_failures(conn)
                lines = [{"record": "answer", **one} for one in answers]
                lines += [{"record": "failure", **one} for one in failures]
                with export.open("w", encoding="utf-8", newline="\n") as out:
                    for line in lines:
                        out.write(json.dumps(line, ensure_ascii=False) + "\n")
                print(f"{len(answers)} answers with their model calls and "
                      f"{len(failures)} unanswered questions in {export}")  # fmt: skip
            elif conversation is not None:
                found = {
                    "answers": store.answer_records(conn, conversation),
                    "failures": store.question_failures(conn, conversation),
                }
                print(json.dumps(found, indent=1, ensure_ascii=False))
            else:
                for one in store.recent_conversations(conn, limit):
                    print(f"{one['started'][:16]}  {one['turns']:>2} turns  "
                          f"{one['id']}  {one['first_question']}")  # fmt: skip
                failures = store.question_failures(conn)[-limit:]
                if failures:
                    print("\nQuestions that got no answer:")
                for one in failures:
                    print(f"{one['created'][:16]}  {one['reason']:<11}  "
                          f"{one['question']}  ({one['detail']})")  # fmt: skip
    except assistant.Unavailable as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def run_news() -> None:
    companies = [sources.load(company_id) for company_id in sources.company_ids()]
    counts, failures = links.collect_news(companies, sources.news_feeds())
    for company_id, count in counts.items():
        print(f"{company_id}: {count} news links added or updated")
    report(failures)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="petroknowledge", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    acquire_parser = commands.add_parser(
        "acquire", help="collect a company's core pages and documents"
    )
    acquire_parser.add_argument("company")
    acquire_parser.add_argument(
        "--linked",
        action="store_true",
        help="the documents and pictures its stored pages link",
    )
    acquire_parser.add_argument(
        "--measure", action="store_true", help="only size them (HEAD requests)"
    )
    acquire_parser.add_argument(
        "--pictures",
        action="store_true",
        help="with --linked, the linked pictures too (mostly logos and banners)",
    )
    add_parser = commands.add_parser(
        "add", help="store documents saved by hand in data/inbox/<company>/"
    )
    add_parser.add_argument("company")
    download_parser = commands.add_parser(
        "download-listed",
        help="download the files in data/inbox/<company>-to-download.txt into the "
        "inbox, on the owner's instruction",
    )
    download_parser.add_argument("company")
    links_parser = commands.add_parser(
        "links", help="keep a company's posts as links (metadata only)"
    )
    links_parser.add_argument("company")
    commands.add_parser("news", help="keep news items naming each company as links")
    read_parser = commands.add_parser(
        "read-documents",
        help="read every stored PDF and spreadsheet into elements, each in its own "
        "process (needs the ingest group)",
    )
    read_parser.add_argument(
        "--vlm", default="", help="a vision model server that reads tables again"
    )
    images_parser = commands.add_parser(
        "read-images",
        help="store the posts' and documents' pictures, read the text in each and "
        "give each its SigLIP2 vector (needs the ingest group)",
    )
    images_parser.add_argument("--vlm", default="", help="the vision model server")
    images_parser.add_argument(
        "--vectors", action="store_true", help="give each picture its SigLIP2 vector"
    )
    transcribe_parser = commands.add_parser(
        "transcribe",
        help="read the scanned pages of every read PDF with a vision model (needs "
        "the ingest group)",
    )
    transcribe_parser.add_argument(
        "--vlm", required=True, help="the vision model server, e.g. GLM-OCR"
    )
    transcribe_parser.add_argument(
        "--pages", type=Path, help='only the pages a JSON file lists under "pages"'
    )
    transcribe_parser.add_argument("--out", type=Path, default=config.TRANSCRIPTS)
    pages_parser = commands.add_parser(
        "read-pages", help="read a company's stored web pages into elements"
    )
    pages_parser.add_argument("company")
    one_parser = commands.add_parser(
        "read-document",
        help="read one stored document into a JSON file (used by read-documents)",
    )
    one_parser.add_argument("sha256")
    one_parser.add_argument("out", type=Path)
    one_parser.add_argument("--vlm", default="")
    ask_parser = commands.add_parser(
        "ask",
        help="answer a question with its sources (needs Postgres and the model "
        "servers)",
    )
    ask_parser.add_argument("question")
    serve_parser = commands.add_parser(
        "serve",
        help="run the API on this machine (needs Postgres and the model "
        "servers to answer)",
    )
    serve_parser.add_argument("--port", type=int, default=config.APP_PORT)
    openapi_parser = commands.add_parser(
        "openapi", help="write the API's OpenAPI schema (the web app's types)"
    )
    openapi_parser.add_argument("out", type=Path)
    history_parser = commands.add_parser(
        "history",
        help="the recorded conversations, with each answer's model calls (read on "
        "this machine only)",
    )
    history_parser.add_argument(
        "conversation", nargs="?", help="one conversation's id: its turns as JSON"
    )
    history_parser.add_argument(
        "--export", type=Path, help="every answer with its model calls, as JSON Lines"
    )
    history_parser.add_argument("--limit", type=int, default=50)
    index_parser = commands.add_parser(
        "index",
        help="build a company's index version from what was read (needs Postgres "
        "and the embedding server)",
    )
    index_parser.add_argument("company")
    index_parser.add_argument(
        "--live", action="store_true", help="make the new version the one served"
    )
    index_parser.add_argument(
        "--read-roles",
        action="store_true",
        help="read role statements from new passages with the chat model (K24)",
    )
    index_parser.add_argument(
        "--compiled-units",
        action="store_true",
        help="add the units page's list as one compiled passage (K27, not adopted)",
    )
    commands.add_parser(
        "browse", help="give the stored files readable names in data/browse/"
    )
    models_parser = commands.add_parser(
        "models", help="check the files in models/ against models.json"
    )
    models_parser.add_argument(
        "--quick", action="store_true", help="compare sizes only, not SHA-256"
    )
    args = parser.parse_args(argv)
    if args.command == "read-documents":
        return run_read_documents(args.vlm)
    if args.command == "transcribe":
        return run_transcribe(args.vlm, args.pages, args.out)
    if args.command == "read-images":
        return run_read_images(args.vlm, args.vectors)
    if args.command == "read-document":
        reading = documents.read_one(args.sha256, args.vlm)
        text = json.dumps(reading, indent=1, ensure_ascii=False) + "\n"
        args.out.write_text(text, encoding="utf-8", newline="\n")
        return 0
    if args.command == "models":
        entries = weights.read(weights.MANIFEST)
        found = weights.problems(entries, weights.FOLDER, args.quick)
        for problem in found:
            print(problem)
        print(f"{len(entries)} model files checked, {len(found)} problems")
        return 1 if found else 0
    if args.command == "ask":
        return run_ask(args.question)
    if args.command == "history":
        return run_history(args.conversation, args.export, args.limit)
    if args.command == "serve":
        run_serve(args.port)
        return 0
    if args.command == "openapi":
        from petroknowledge.app import app

        text = json.dumps(app.openapi(), indent=1, ensure_ascii=False) + "\n"
        args.out.write_text(text, encoding="utf-8", newline="\n")
        return 0
    try:
        if args.command == "index":
            return run_index(
                args.company, args.live, args.read_roles, args.compiled_units
            )
        if args.command == "acquire":
            run_acquire(args.company, args.linked, args.measure, args.pictures)
        elif args.command == "add":
            run_add(args.company)
        elif args.command == "download-listed":
            run_download_listed(args.company)
        elif args.command == "read-pages":
            run_read_pages(args.company)
        elif args.command == "links":
            run_links(args.company)
        elif args.command == "news":
            run_news()
        else:
            print(f"{fetch.browse()} new readable names in data/browse/")
    except sources.SourceError as error:
        print(f"error: {error}")
        return 1
    return 0
