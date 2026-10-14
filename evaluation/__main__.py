"""The harness command line. Run from the repository root:

    uv run python -m evaluation verify                       # every set's hashes
    uv run python -m evaluation register SET --description "..."
    uv run --group ingest python -m evaluation scan-sample SET --seed N \
        [--per-stratum 4] [--reserve "financial statements=8"]   # scanned pages
    uv run --group ingest python -m evaluation scan-transcribe SET --vlm URL \
        --prompt "Text Recognition:" --out RUN.json      # one arm on the set's pages
    uv run python -m evaluation scan-score SET --out FILE [--run RUN.json]  # scan gate
    uv run --group ingest python -m evaluation pdf-candidates SET --seed N --count K
    uv run --group ingest python -m evaluation pdf-score SET --out FILE  # PDF gate
    uv run --group ingest python -m evaluation sheet-sample SET --seed N --count K
    uv run python -m evaluation sheet-score SET --out FILE    # spreadsheet gate
    uv run python -m evaluation web-sample SET --seed N [--held-out-from SET]
    uv run --group evaluation python -m evaluation web-render SET  # web page gate
    uv run python -m evaluation web-truth SET                 # outlines -> truth
    uv run python -m evaluation web-score SET --out FILE
    uv run --group ingest python -m evaluation picture-candidates SET --seed N --count K
    uv run --group ingest python -m evaluation picture-score SET --out FILE  # pictures
    uv run --env-file .env python -m evaluation ask-keys SET --file KEY --out RUN
    uv run python -m evaluation reach SET --file KEY --run RUN --out FILE
    uv run --env-file .env python -m evaluation replay SET --file KEY --out FILE \
        [--history RUN] [--keep REPLAY] [--scoped] [--roles]  # no answers written
    uv run python -m evaluation grading-items SET --file KEY --run RUN --out FILE
    uv run python -m evaluation grade-counts SET --file KEY --grades G --run RUN --out F
    uv run python -m evaluation claims-items --spec S --labels L --out FILE  # claims

A set lives in git-ignored data/eval/<SET>/ and is registered by SHA-256 in
evaluation/sets.json before it is used (`sets`).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from evaluation import (
    claims,
    grading,
    parsing,
    pdfpages,
    pictures,
    replay,
    rescore,
    runs,
    scans,
    sets,
    sheetvalues,
    webpages,
)

# Runs whose answers stay unread until the release run, so nothing is tuned on them.
SEALED_RUNS = ("nnpc-wording-check-v1", "heldout")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=1, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def run_verify(args: argparse.Namespace) -> int:
    names = args.names or list(sets.read_registry(args.registry)["sets"])
    found = [p for name in names for p in sets.problems(name, args.root, args.registry)]
    for problem in found:
        print("PROBLEM", problem)
    print(f"{len(names)} sets checked, {len(found)} problems")
    return 1 if found else 0


def run_scan_sample(args: argparse.Namespace) -> int:
    """Draw a scan set's pages from its strata and save their images, once."""
    folder = args.root / args.name
    out = folder / "pages.json"
    if out.exists():
        raise ValueError(f"{out} exists: a set's pages are drawn once")
    strata = read_json(folder / "strata.json")
    wanted = {sha256 for documents in strata["strata"].values() for sha256 in documents}
    readings = {
        sha256: read_json(args.readings / f"{sha256}.json") for sha256 in wanted
    }
    reserve = {}
    for item in args.reserve:
        stratum, _, count = item.rpartition("=")
        reserve[stratum] = int(count)
    found = scans.candidates(strata, readings)
    drawn = scans.draw(found, args.seed, args.per_stratum, reserve)
    images = folder / "images"
    images.mkdir(parents=True, exist_ok=True)
    for page in drawn["pages"] + drawn["reserve"]:
        prefix = images / f"p{page['number']:02d}"
        scans.save_images(args.store / page["sha256"], page["page"], prefix)
    write_json(out, {"seed": args.seed, **drawn})
    for page in drawn["pages"] + drawn["reserve"]:
        name = page["url"].rsplit("/", 1)[-1].split("?")[0]
        print(f"{page['number']:3} {page['stratum']:24} p{page['page']:<4} {name}")
    print(f"{len(drawn['pages'])} pages, {len(drawn['reserve'])} reserve, in {folder}")
    return 0


def run_scan_transcribe(args: argparse.Namespace) -> int:
    """Read a registered scan set's pages with one prompt (an arm); save the run."""
    folder = sets.require(args.name, args.root, args.registry)
    truth = read_json(folder / "truth.json")["pages"]
    answers = {}
    for page in truth:
        image = scans.page_image(args.store / page["sha256"], page["page"])
        answers[str(page["number"])] = scans.ask(image, args.vlm, args.prompt)
    unfinished = [n for n, found in answers.items() if not found["finished"]]
    run = {"prompt": args.prompt, "sampling": scans.SAMPLING, "pages": answers}
    write_json(args.out, run)
    print(f"{len(answers)} pages read, {len(unfinished)} unfinished, in {args.out}")
    return 0


def saved_answer(page: dict[str, Any], args: argparse.Namespace) -> str | None:
    """A page's answer from the run file when one is given, else from the engine's
    saved transcripts; None when there is none."""
    if args.run:
        found = read_json(args.run)["pages"].get(str(page["number"]))
        return None if found is None else str(found["answer"])
    path = args.transcripts / f"{page['sha256']}.json"
    saved = read_json(path)["pages"] if path.exists() else {}
    found = saved.get(str(page["page"]))
    return None if found is None else str(found["answer"])


def run_scan_score(args: argparse.Namespace) -> int:
    """Score the transcriptions of a registered scan set's pages: an arm's run, or
    the engine's saved transcripts."""
    folder = sets.require(args.name, args.root, args.registry)
    truth = read_json(folder / "truth.json")["pages"]
    scores = {}
    missing = []
    for page in truth:
        answer = saved_answer(page, args)
        if answer is None:
            missing.append(f"{page['sha256'][:12]} p{page['page']}")
            continue
        scores[page["number"]] = scans.score_page(page, answer)
    if missing:
        raise ValueError("not transcribed yet: " + ", ".join(missing))
    shares = scans.pooled(list(scores.values()))
    gate = scans.passed(shares)
    write_json(args.out, {"pages": scores, "pooled": shares, "gate": gate})
    for measure, bar in scans.GATE.items():
        print(f"{measure}: {shares[measure]:.3f} (bar {bar})")
    print("pass" if all(gate.values()) else "FAIL")
    return 0 if all(gate.values()) else 1


def pdf_readings(folder: Path) -> list[dict[str, Any]]:
    """Every PDF reading saved in `folder` (a spreadsheet's reading is not one)."""
    found = []
    for path in sorted(folder.glob("*.json")):
        if len(path.stem) != 64:  # the run's report, or a part file
            continue
        reading = read_json(path)
        if "pypdfium2" in reading["fingerprint"]["packages"]:
            found.append(reading)
    return found


def run_pdf_candidates(args: argparse.Namespace) -> int:
    """List a PDF set's candidate pages in their seeded order and save the first
    `count` pages' images, once, for the truth writer to pick table pages from."""
    folder = args.root / args.name
    out = folder / "candidates.json"
    if out.exists():
        raise ValueError(f"{out} exists: a set's candidates are drawn once")
    used: set[tuple[str, int]] = set()
    for name in args.exclude_set:
        earlier = sets.require(name, args.root, args.registry)
        used |= {
            (p["sha256"], p["page"]) for p in read_json(earlier / "truth.json")["pages"]
        }
    found = pdfpages.candidates(pdf_readings(args.readings), args.seed, used)
    images = folder / "images"
    images.mkdir(parents=True, exist_ok=True)
    for page in found[: args.count]:
        prefix = images / f"c{page['number']:03d}"
        scans.save_images(args.store / page["sha256"], page["page"], prefix)
        name = page["url"].rsplit("/", 1)[-1].split("?")[0]
        print(f"{page['number']:3} p{page['page']:<4} {name}")
    write_json(out, {"seed": args.seed, "pages": found})
    print(f"{len(found)} candidates, images of the first {args.count}, in {folder}")
    return 0


def run_pdf_score(args: argparse.Namespace) -> int:
    """Score the saved readings of a registered PDF set's pages with round 3's gate;
    the text layer is scored beside them as the reference."""
    folder = sets.require(args.name, args.root, args.registry)
    truth = read_json(folder / "truth.json")["pages"]
    scores: dict[str, list[dict[str, Any]]] = {"pypdf": [], "reading": []}
    seconds = []
    for page in truth:
        reading = read_json(args.readings / f"{page['sha256']}.json")
        layer = pdfpages.reference(args.store / page["sha256"], page["page"])
        scored = pdfpages.score_page(page, reading, layer)
        for arm, result in scored.items():
            scores[arm].append(result)
        seconds.append(float(reading["seconds"]))
    result = parsing.summarise(scores, seconds, "reading")
    write_json(args.out, {"pages": scores, **result})
    print(parsing.markdown(result))
    return 0 if all(result["gate"].values()) else 1


def sheet_readings(folder: Path) -> dict[str, dict[str, Any]]:
    """Every spreadsheet reading saved in `folder`, by its file's SHA-256."""
    found = {}
    for path in sorted(folder.glob("*.json")):
        if len(path.stem) != 64:  # the run's report, or a part file
            continue
        reading = read_json(path)
        if "openpyxl" in reading["fingerprint"]["packages"]:
            found[path.stem] = reading
    return found


def run_sheet_sample(args: argparse.Namespace) -> int:
    """Draw a spreadsheet set's values from the workbooks as stored, once, with the
    raw grid around each for the truth writer."""
    folder = args.root / args.name
    out = folder / "values.json"
    if out.exists():
        raise ValueError(f"{out} exists: a set's values are drawn once")
    workbooks = {
        sha256: sheetvalues.sheets(args.store / sha256)
        for sha256 in sheet_readings(args.readings)
    }
    drawn = sheetvalues.draw(workbooks, args.seed, args.count)
    shown = folder / "context"
    shown.mkdir(parents=True, exist_ok=True)
    for item in drawn:
        grid = workbooks[item["sha256"]][item["sheet"]]
        text = sheetvalues.context(grid, item["row"], item["column"]) + "\n"
        target = shown / f"v{item['number']:02d}.txt"
        target.write_text(text, encoding="utf-8", newline="\n")
    write_json(out, {"seed": args.seed, "values": drawn})
    print(f"{len(drawn)} values from {len(workbooks)} workbooks, in {folder}")
    return 0


def run_sheet_score(args: argparse.Namespace) -> int:
    """Score the saved spreadsheet readings against a registered set's truth: each
    value in its row and under its header."""
    folder = sets.require(args.name, args.root, args.registry)
    truth = read_json(folder / "truth.json")["values"]
    readings = sheet_readings(args.readings)
    excluded = [str(item["number"]) for item in truth if "exclude" in item]
    found = {
        str(item["number"]): sheetvalues.placed(item, readings[item["sha256"]])
        for item in truth
        if "exclude" not in item
    }
    share = sum(found.values()) / len(found)
    passed = share >= parsing.GRID_GATE["grid"]
    write_json(args.out, {"values": found, "excluded": excluded, "placed": share,
                          "pass": passed})  # fmt: skip
    missed = [number for number, hit in found.items() if not hit]
    print(f"values placed: {share:.3f} of {len(found)} (bar "
          f"{parsing.GRID_GATE['grid']}); missed: {', '.join(missed) or 'none'}; "
          f"excluded: {', '.join(excluded) or 'none'}")  # fmt: skip
    print("pass" if passed else "FAIL")
    return 0 if passed else 1


def web_readings(folder: Path) -> dict[str, dict[str, Any]]:
    """Every saved web reading in `folder`, by its page's SHA-256."""
    return {
        path.stem: read_json(path)
        for path in sorted(folder.glob("*.json"))
        if len(path.stem) == 64  # not a run's report
    }


def run_web_sample(args: argparse.Namespace) -> int:
    """Draw a web set's pages by seed and stratum, once; a held-out set draws its own
    quotas from pages outside the set named."""
    out = args.root / args.name / "pages.json"
    if out.exists():
        raise ValueError(f"{out} exists: a set's pages are drawn once")
    readings = web_readings(args.readings)
    exclude: set[str] = set()
    quotas = webpages.QUOTAS
    if args.held_out_from:
        drawn = read_json(args.root / args.held_out_from / "pages.json")["pages"]
        exclude = {page["sha256"] for page in drawn}
        quotas = webpages.HELD_OUT
    pages = []
    for sha256 in webpages.draw(readings, quotas, args.seed, exclude):
        reading = readings[sha256]
        stratum = webpages.stratum(reading)
        pages.append({"url": reading["url"], "sha256": sha256, "stratum": stratum})
    write_json(out, {"seed": args.seed, "pages": pages})
    print(f"{len(pages)} pages in {out}")
    return 0


def run_web_render(args: argparse.Namespace) -> int:
    """Draw a web set's pages from their stored HTML into its rendering folder."""
    folder = args.root / args.name
    pages = read_json(folder / "pages.json")["pages"]
    for page in pages:
        page["html"] = (
            (args.store / page["sha256"]).read_bytes().decode("utf-8", "replace")
        )
    with webpages.chromium() as browser:
        counts = webpages.render(pages, folder / args.folder, browser)
    print(f"{len(counts)} pages rendered to {folder / args.folder}")
    return 0


def run_web_truth(args: argparse.Namespace) -> int:
    """Each judged outline over its page's rendered text becomes the page's truth."""
    folder = args.root / args.name
    truth = {}
    problems = []
    for path in sorted((folder / "outlines").glob("*.json")):
        text = (folder / args.folder / f"{path.stem}.txt").read_text(encoding="utf-8")
        lines = webpages.visible_lines(text)
        outline = read_json(path)
        found = webpages.outline_problems(outline, len(lines))
        problems += [f"{path.stem[:12]}: {problem}" for problem in found]
        if not found:
            truth[path.stem] = webpages.build_truth(outline, lines)
    if problems:
        print("\n".join(problems))
        return 1
    write_json(folder / "truth.json", truth)
    blocks = sum(len(page["blocks"]) for page in truth.values())
    print(f"{len(truth)} pages, {blocks} blocks in {folder / 'truth.json'}")
    return 0


def run_web_score(args: argparse.Namespace) -> int:
    """Score the saved web readings against a registered web set's truth."""
    truth = read_json(sets.require(args.name, args.root, args.registry) / "truth.json")
    readings = {
        sha256: webpages.reading_of(read_json(args.readings / f"{sha256}.json"))
        for sha256 in truth
    }
    found = webpages.scores(truth, readings)
    rates = found["rates"]
    passed = all(rates[name] >= bar for name, bar in webpages.GATE.items())
    write_json(args.out, found | {"pass": passed})
    print(" ".join(f"{name} {value:.3f}" for name, value in rates.items()))
    print("pass" if passed else "FAIL")
    return 0 if passed else 1


def run_picture_candidates(args: argparse.Namespace) -> int:
    """List a picture set's candidates by source in their seeded order and copy the
    first `count` of each as stored, once. No reading of any picture is opened."""
    folder = args.root / args.name
    out = folder / "candidates.json"
    if out.exists():
        raise ValueError(f"{out} exists: a set's candidates are drawn once")
    found = pictures.candidates(read_json(args.images / "places.json"), args.seed)
    shown = folder / "images"
    shown.mkdir(parents=True, exist_ok=True)
    for source, drawn in found.items():
        for number, picture in enumerate(drawn, 1):
            picture["number"] = number
            if number > args.count:
                continue
            picture["png"] = f"{source}-{number:02d}.png"
            stored = args.images / f"{picture['id']}.png"
            shutil.copyfile(stored, shown / picture["png"])
            name = picture["url"].rsplit("/", 1)[-1].split("?")[0]
            page = f"p{picture['page']}" if picture["page"] else ""
            print(f"{picture['png']:15} {page:6} {name}")
    write_json(out, {"seed": args.seed, **found})
    sizes = ", ".join(f"{len(drawn)} {source}" for source, drawn in found.items())
    print(f"candidates: {sizes}; images of the first {args.count} each, in {folder}")
    return 0


def run_picture_score(args: argparse.Namespace) -> int:
    """Score the saved picture readings against a registered picture set's truth."""
    truth = read_json(sets.require(args.name, args.root, args.registry) / "truth.json")
    readings = {}
    for picture in truth["pictures"] + truth["skipped"]:
        saved = read_json(args.images / f"{picture['id']}.json")
        readings[picture["id"]] = str(saved["ocr"])
    found = pictures.score(truth, readings)
    write_json(args.out, found)
    for name, rates in found["rates"].items():
        recall, precision = rates["recall"], rates["precision"]
        print(f"{name}: recall {recall:.3f}, precision {precision:.3f}")
    print(f"words read from skipped pictures with no words: {found['invented']}")
    passed = all(found["gate"].values())
    print("pass" if passed else "FAIL")
    return 0 if passed else 1


def run_ask_keys(args: argparse.Namespace) -> int:
    """Ask every question of a keyed set through the answer path, resuming a run.
    A held-out set is sealed until the release run."""
    if "heldout" in args.name and not args.release:
        raise ValueError(f"{args.name} is sealed until the release run (--release)")
    from petroknowledge import assistant

    key = read_json(sets.require(args.name, args.root, args.registry) / args.file)

    def ask(message: str, conversation: str | None, in_one: bool) -> Any:
        turn = assistant.open_turn(conversation) if in_one else None
        answer, answer_id = assistant.ask_and_record(message, turn=turn)
        return answer, answer_id, turn.conversation_id if turn else None

    def save(saved: list[dict[str, Any]]) -> None:
        write_json(args.out, saved)
        print(f"{len(saved)} answered", flush=True)

    saved = read_json(args.out) if args.out.exists() else []
    found = runs.run(runs.items(key), ask, saved, save)
    print(f"{len(found)} answers in {args.out}")
    return 0


def run_reach(args: argparse.Namespace) -> int:
    """Status agreement and evidence reach of a run against its keyed set."""
    from petroknowledge import fetch

    key = read_json(sets.require(args.name, args.root, args.registry) / args.file)
    files = {r["url"]: r["sha256"] for r in fetch.read_manifest() if "sha256" in r}
    found = runs.measure(runs.items(key), read_json(args.run), files)
    write_json(args.out, found)
    for expected, got in found["status"].items():
        print(f"expected {expected}: {got}")
    for form, counts in found["forms"].items():
        print(f"evidence {form}: {counts}")
    return 0


def run_replay(args: argparse.Namespace) -> int:
    """The first request and the searches of a keyed set, no answers written, scored
    by evidence reach (Lime Green E5). A conversation's turns read the history a run
    showed (`--history`); `--keep` reuses an earlier replay's search questions;
    `--scoped` rewrites again with the units the user's words name when the first
    request dropped them, and searches again inside them; `--roles` adds
    the statements of the roles asked about."""
    if "heldout" in args.name and not args.release:
        raise ValueError(f"{args.name} is sealed until the release run (--release)")
    from petroknowledge import answer, assistant, config, fetch, llm

    key = read_json(sets.require(args.name, args.root, args.registry) / args.file)
    asked = runs.items(key)
    if key.get("conversations") and args.history is None:
        raise ValueError("a set with conversations needs --history RUN")
    history = {}
    if args.history is not None:
        history = replay.histories(asked, read_json(args.history), config.HISTORY_TURNS)
    kept = None
    if args.keep is not None:
        kept = {e["id"]: e["searched"] for e in read_json(args.keep)["replayed"]}
    with assistant.connect() as conn:
        version_id = assistant.served_index(conn)
        models = assistant.answer_models()
        retrieve = assistant.retriever(conn, version_id, models)
        extra = assistant.extras(conn, version_id, models)
        lookup = assistant.role_lookup(conn, version_id) if args.roles else None

        def named_units(message: str, earlier: answer.History) -> list[Any]:
            if not args.scoped:
                return []
            return assistant.asked_units(conn, version_id, message, earlier)

        def understand(message: str, earlier: answer.History) -> Any:
            named = named_units(message, earlier)
            return answer.understand_for(message, llm.chat, earlier, named)

        def find(message: str, earlier: answer.History, understood: Any) -> Any:
            ids = [unit["id"] for unit in named_units(message, earlier)]
            scoped = assistant.scoped_retriever(conn, version_id, models, ids)
            dated: list[Any] = []
            if lookup is not None:
                parts = [part.question for part in understood]
                _, dated = lookup(" ".join([message, *parts]))
            return answer.find_passages(understood, retrieve, extra, scoped, dated)

        entries = replay.replay(asked, understand, find, history, kept)
    files = {r["url"]: r["sha256"] for r in fetch.read_manifest() if "sha256" in r}
    found = replay.reach(asked, entries, files)
    write_json(args.out, {"reach": found, "replayed": entries})
    print(f"parts reached: {found['parts_reached']} of {found['parts']} "
          f"({len(entries)} questions)")  # fmt: skip
    return 0


def run_rescore(args: argparse.Namespace) -> int:
    """A run's recorded replies checked again by today's post-generation checks:
    the judge's recorded verdicts (asking the live judge only for statements
    never judged), the live judge for every statement, or no judge, on every claim or
    on figure claims only (`--grounding`); the earlier-holder check with an index
    version's roles if given. Removals are applied, as served."""
    if any(name in str(args.run) for name in SEALED_RUNS) and not args.release:
        raise ValueError(f"{args.run} is sealed until the release run (--release)")
    from petroknowledge import assistant, config, llm

    config.GROUNDING = args.grounding  # the judge's removals applied, as served
    judges = {
        "recorded": lambda verdicts: rescore.recorded(verdicts, llm.judge),
        "live": lambda verdicts: llm.judge,
        "none": lambda verdicts: rescore.supporting,
    }
    with assistant.connect() as conn:
        stale = None
        if args.stale_version is not None:
            stale = assistant.stale_check(conn, args.stale_version)
        entries = []
        for entry in read_json(args.run):
            recorded = rescore.recorded_answer(conn, entry)
            entries.append(
                rescore.rescored(entry, *recorded, judges[args.judge], stale)
            )
    write_json(args.out, entries)
    answered = sum(e["shown"]["status"] == "answered" for e in entries)
    print(f"{len(entries)} answers rescored ({answered} answered) in {args.out}")
    return 0


def run_grading_items(args: argparse.Namespace) -> int:
    """Write a run's answers as items for a grader (`briefs/grading-guide.md`)."""
    if "heldout" in args.name and not args.release:
        raise ValueError(f"{args.name} is sealed until the release run (--release)")
    key = read_json(sets.require(args.name, args.root, args.registry) / args.file)
    found = grading.items(key, read_json(args.run))
    write_json(args.out, found)
    print(f"{len(found)} items in {args.out}")
    return 0


def run_second_sample(args: argparse.Namespace) -> int:
    """Write the items a second grader grades, without the primary's verdicts."""
    key = read_json(sets.require(args.name, args.root, args.registry) / args.file)
    types = {item["id"]: item["type"] for item in runs.items(key)}
    found = grading.second_sample(read_json(args.items), types,
                                  read_json(args.grades), 0.3, args.seed)  # fmt: skip
    write_json(args.out, found)
    print(f"{len(found)} items for a second grader in {args.out}")
    return 0


def run_claims_items(args: argparse.Namespace) -> int:
    """Write the claim benchmark: every claim of the graded runs a spec lists,
    labelled by its answer's verdict, or by the labels read for wrong answers."""
    graded = []
    for source in read_json(args.spec):
        if "heldout" in source["set"]:
            raise ValueError(f"{source['set']} is sealed: no claims are taken from it")
        key = read_json(sets.require(source["set"], args.root, args.registry)
                        / source["file"])  # fmt: skip
        verdicts = {}
        for path in source["grades"]:
            for item_id, grade in read_json(Path(path)).items():
                verdicts[item_id] = grade["verdict"]
        graded.append(
            {
                "entries": read_json(Path(source["run"])),
                "verdicts": verdicts,
                "asked": claims.asked(key),
                "version": source["version"],
            }
        )
    found = claims.collect(graded, read_json(args.labels))
    write_json(args.out, {"claims": found})
    wrong = sum(item["label"] == "wrong" for item in found)
    print(f"{len(found)} claims ({wrong} wrong) in {args.out}")
    return 0


def run_grade_agreement(args: argparse.Namespace) -> int:
    """Cohen's κ between the primary grader and a second grader."""
    found = grading.agreement(read_json(args.primary), read_json(args.second))
    write_json(args.out, found)
    print(f"{found['agree']} of {found['items']} agree; kappa {found['kappa']:.3f}")
    print(f"differ: {', '.join(found['differ']) or 'none'}")
    return 0


def run_grade_counts(args: argparse.Namespace) -> int:
    """Count a grader's verdicts against release bar R1."""
    key = read_json(sets.require(args.name, args.root, args.registry) / args.file)
    types = {item["id"]: item["type"] for item in runs.items(key)}
    statuses = {entry["id"]: entry["shown"]["status"] for entry in read_json(args.run)}
    found = grading.counts(types, statuses, read_json(args.grades))
    write_json(args.out, found)
    low, high = found["wrong_interval"]
    print(f"{found['graded']} graded: {found['verdicts']}")
    print(f"wrong among {found['answered']} answered: {found['wrong_rate']:.3f} "
          f"(95% {low:.3f}-{high:.3f}); "
          f"R1 (upper bound <= {grading.R1}): "
          f"{'pass' if found['r1_pass'] else 'FAIL'}")  # fmt: skip
    return 0 if found["r1_pass"] else 1


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(prog="python -m evaluation", description=__doc__)
    main.add_argument("--root", type=Path, default=sets.ROOT)
    main.add_argument("--registry", type=Path, default=sets.REGISTRY)
    commands = main.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="check registered hashes")
    verify.add_argument("names", nargs="*", help="sets to check (default: all)")
    register = commands.add_parser("register", help="hash a set's new files")
    register.add_argument("name")
    register.add_argument("--description", default="")
    sample = commands.add_parser(
        "scan-sample", help="draw a scan set's pages and save their images"
    )
    sample.add_argument("name")
    sample.add_argument("--seed", type=int, required=True)
    sample.add_argument("--per-stratum", type=int, default=4)
    sample.add_argument(
        "--reserve", action="append", default=[], help="STRATUM=N, repeatable"
    )
    sample.add_argument("--readings", type=Path, default=Path("data/elements"))
    sample.add_argument("--store", type=Path, default=Path("data/files"))
    transcribe = commands.add_parser(
        "scan-transcribe", help="read a scan set's pages with one prompt (an arm)"
    )
    transcribe.add_argument("name")
    transcribe.add_argument("--vlm", required=True, help="the model server's URL")
    transcribe.add_argument("--prompt", required=True)
    transcribe.add_argument("--out", type=Path, required=True)
    transcribe.add_argument("--store", type=Path, default=Path("data/files"))
    score = commands.add_parser(
        "scan-score", help="score saved transcriptions against a scan set's truth"
    )
    score.add_argument("name")
    score.add_argument("--transcripts", type=Path, default=Path("data/transcripts"))
    score.add_argument("--run", type=Path, help="an arm's run (scan-transcribe)")
    score.add_argument("--out", type=Path, required=True)
    candidates = commands.add_parser(
        "pdf-candidates", help="list a PDF set's candidate pages and save images"
    )
    candidates.add_argument("name")
    candidates.add_argument("--seed", type=int, required=True)
    candidates.add_argument("--count", type=int, required=True)
    candidates.add_argument(
        "--exclude-set", action="append", default=[], help="a set whose pages to skip"
    )
    candidates.add_argument("--readings", type=Path, default=Path("data/elements"))
    candidates.add_argument("--store", type=Path, default=Path("data/files"))
    pdf_score = commands.add_parser(
        "pdf-score", help="score saved PDF readings against a PDF set's truth"
    )
    pdf_score.add_argument("name")
    pdf_score.add_argument("--out", type=Path, required=True)
    pdf_score.add_argument("--readings", type=Path, default=Path("data/elements"))
    pdf_score.add_argument("--store", type=Path, default=Path("data/files"))
    sheet_sample = commands.add_parser(
        "sheet-sample", help="draw a spreadsheet set's values with their context"
    )
    sheet_sample.add_argument("name")
    sheet_sample.add_argument("--seed", type=int, required=True)
    sheet_sample.add_argument("--count", type=int, required=True)
    sheet_sample.add_argument("--readings", type=Path, default=Path("data/elements"))
    sheet_sample.add_argument("--store", type=Path, default=Path("data/files"))
    sheet_score = commands.add_parser(
        "sheet-score", help="score saved spreadsheet readings against a set's truth"
    )
    sheet_score.add_argument("name")
    sheet_score.add_argument("--out", type=Path, required=True)
    sheet_score.add_argument("--readings", type=Path, default=Path("data/elements"))
    web_sample = commands.add_parser("web-sample", help="draw a web set's pages")
    web_sample.add_argument("name")
    web_sample.add_argument("--seed", type=int, required=True)
    web_sample.add_argument("--held-out-from", help="a set whose pages to skip")
    web_sample.add_argument("--readings", type=Path, default=Path("data/pages"))
    web_render = commands.add_parser(
        "web-render", help="draw a web set's pages as a visitor sees them"
    )
    web_render.add_argument("name")
    web_render.add_argument("--store", type=Path, default=Path("data/files"))
    web_render.add_argument("--folder", default="rendered", help="a folder in the set")
    web_truth = commands.add_parser(
        "web-truth", help="a web set's truth from its judged outlines"
    )
    web_truth.add_argument("name")
    web_truth.add_argument("--folder", default="rendered", help="its rendering folder")
    web_score = commands.add_parser(
        "web-score", help="score saved web readings against a web set's truth"
    )
    web_score.add_argument("name")
    web_score.add_argument("--out", type=Path, required=True)
    web_score.add_argument("--readings", type=Path, default=Path("data/pages"))
    picture_candidates = commands.add_parser(
        "picture-candidates", help="draw a picture set's candidates and copy images"
    )
    picture_candidates.add_argument("name")
    picture_candidates.add_argument("--seed", type=int, required=True)
    picture_candidates.add_argument("--count", type=int, required=True)
    picture_candidates.add_argument("--images", type=Path, default=Path("data/images"))
    picture_score = commands.add_parser(
        "picture-score", help="score saved picture readings against a set's truth"
    )
    picture_score.add_argument("name")
    picture_score.add_argument("--out", type=Path, required=True)
    picture_score.add_argument("--images", type=Path, default=Path("data/images"))
    ask_keys = commands.add_parser(
        "ask-keys", help="ask a keyed set's questions through the answer path"
    )
    ask_keys.add_argument("name")
    ask_keys.add_argument("--file", required=True, help="the key file in the set")
    ask_keys.add_argument("--out", type=Path, required=True, help="the run (resumes)")
    ask_keys.add_argument(
        "--release", action="store_true", help="unseal a held-out set"
    )
    replayed = commands.add_parser(
        "replay", help="the first request and the searches, scored by reach"
    )
    replayed.add_argument("name")
    replayed.add_argument("--file", required=True, help="the key file in the set")
    replayed.add_argument("--out", type=Path, required=True)
    replayed.add_argument(
        "--history", type=Path, help="a run whose shown replies are each turn's history"
    )
    replayed.add_argument(
        "--keep", type=Path, help="an earlier replay whose search questions are kept"
    )
    replayed.add_argument(
        "--scoped", action="store_true", help="search again inside named units"
    )
    replayed.add_argument(
        "--roles", action="store_true", help="add the asked roles' statements"
    )
    replayed.add_argument(
        "--release", action="store_true", help="unseal a held-out set"
    )
    rescored = commands.add_parser(
        "rescore", help="a run's recorded replies checked again"
    )
    rescored.add_argument("--run", type=Path, required=True, help="the run file")
    rescored.add_argument("--out", type=Path, required=True)
    rescored.add_argument(
        "--judge", choices=("recorded", "live", "none"), default="recorded"
    )
    rescored.add_argument(
        "--grounding", choices=("remove", "figures"), default="remove",
        help="the judge on every claim, or on figure claims only",
    )  # fmt: skip
    rescored.add_argument(
        "--stale-version", type=int, help="earlier-holder check with these roles"
    )
    rescored.add_argument("--release", action="store_true", help="unseal a sealed run")
    reach = commands.add_parser(
        "reach", help="status agreement and evidence reach of a run"
    )
    reach.add_argument("name")
    reach.add_argument("--file", required=True, help="the key file in the set")
    reach.add_argument("--run", type=Path, required=True)
    reach.add_argument("--out", type=Path, required=True)
    items = commands.add_parser("grading-items", help="a run's answers for a grader")
    items.add_argument("name")
    items.add_argument("--file", required=True, help="the key file in the set")
    items.add_argument("--run", type=Path, required=True)
    items.add_argument("--out", type=Path, required=True)
    items.add_argument("--release", action="store_true", help="unseal a held-out set")
    second = commands.add_parser(
        "second-sample", help="the items a second grader grades (30%% plus disputes)"
    )
    second.add_argument("name")
    second.add_argument("--file", required=True, help="the key file in the set")
    second.add_argument("--items", type=Path, required=True)
    second.add_argument("--grades", type=Path, required=True, help="the primary's")
    second.add_argument("--seed", type=int, required=True)
    second.add_argument("--out", type=Path, required=True)
    agreed = commands.add_parser(
        "grade-agreement", help="Cohen's kappa between two graders' verdicts"
    )
    agreed.add_argument("--primary", type=Path, required=True)
    agreed.add_argument("--second", type=Path, required=True)
    agreed.add_argument("--out", type=Path, required=True)
    claimed = commands.add_parser(
        "claims-items", help="the claims of graded runs, labelled"
    )
    claimed.add_argument("--spec", type=Path, required=True, help="the graded runs")
    claimed.add_argument(
        "--labels", type=Path, required=True, help="wrong answers' claims"
    )
    claimed.add_argument("--out", type=Path, required=True)
    counted = commands.add_parser(
        "grade-counts", help="count a grader's verdicts against release bar R1"
    )
    counted.add_argument("name")
    counted.add_argument("--file", required=True, help="the key file in the set")
    counted.add_argument("--grades", type=Path, required=True)
    counted.add_argument("--run", type=Path, required=True, help="the graded run")
    counted.add_argument("--out", type=Path, required=True)
    return main


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "verify":
            return run_verify(args)
        if args.command == "register":
            added = sets.register(args.name, args.description, args.root, args.registry)
            print(f"{args.name}: {len(added)} files registered")
            return 0
        if args.command == "scan-sample":
            return run_scan_sample(args)
        if args.command == "scan-transcribe":
            return run_scan_transcribe(args)
        if args.command == "pdf-candidates":
            return run_pdf_candidates(args)
        if args.command == "pdf-score":
            return run_pdf_score(args)
        if args.command == "sheet-sample":
            return run_sheet_sample(args)
        if args.command == "sheet-score":
            return run_sheet_score(args)
        if args.command == "web-sample":
            return run_web_sample(args)
        if args.command == "web-render":
            return run_web_render(args)
        if args.command == "web-truth":
            return run_web_truth(args)
        if args.command == "web-score":
            return run_web_score(args)
        if args.command == "picture-candidates":
            return run_picture_candidates(args)
        if args.command == "picture-score":
            return run_picture_score(args)
        if args.command == "ask-keys":
            return run_ask_keys(args)
        if args.command == "replay":
            return run_replay(args)
        if args.command == "rescore":
            return run_rescore(args)
        if args.command == "reach":
            return run_reach(args)
        if args.command == "grading-items":
            return run_grading_items(args)
        if args.command == "grade-counts":
            return run_grade_counts(args)
        if args.command == "second-sample":
            return run_second_sample(args)
        if args.command == "grade-agreement":
            return run_grade_agreement(args)
        if args.command == "claims-items":
            return run_claims_items(args)
        return run_scan_score(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
