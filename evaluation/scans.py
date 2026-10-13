"""Scanned pages' transcriptions scored against truth written from the page images
(K12's gate; Lime Green's X8 transcription check, with numbers added).

A set's `strata.json`, written before the draw, names the documents in each stratum
and the pages already seen (their transcriptions were read), which are left out. A
seeded draw takes pages from each stratum, plus an ordered reserve used only by the
set's written rules. Each drawn page is saved as two images, its top and bottom
halves with an overlap, at 200 DPI, the resolution the model read, for the truth
writer. The truth (`truth.json`) holds every legible word on the page in reading
order and each table as a grid, written from the images before any transcription
of those pages is read, and registered with the set.

A transcription is scored as saved (data/transcripts) against the truth, both with
Markdown and inline LaTeX marks removed: words, and numbers (words holding a digit), as
multisets, recall and precision each, pooled over the set. The harness reads saved
files only; it never imports the engine it measures.
"""

import base64
import io
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from evaluation import parsing

DPI = 200  # GLM-OCR's setting for PDF pages, and the resolution transcribed
OVERLAP = 0.05  # each half image reaches this share of the page past the middle
SCANNED = "little or no text"  # the flag a reading gives a page with no text layer
# Text read from images and quoted: X43 B2's bar for words, and every number right.
GATE = {
    "recall": 0.95,
    "precision": 0.98,
    "number recall": 0.99,
    "number precision": 0.99,
}
MARKS = str.maketrans(dict.fromkeys("|#*`$\\", " "))
TAG = re.compile(r"<[^>]+>")
# A dash between two numbers splits them, however it is spaced ("6- 10", "6-10"):
# joining instead turned a nil cell ("2022 - 1,686,102") into a false token.
RANGE = re.compile(r"(\d)\s*[-–—]\s*(?=\d)")
# GLM-OCR's published request settings (its SDK's config.yaml); the engine sends the
# same, but the harness never imports the engine.
SAMPLING = {"temperature": 0.0, "top_p": 0.00001, "top_k": 1, "repeat_penalty": 1.1}
MAX_TOKENS = 8192
TIMEOUT_SECONDS = 600.0

Page = dict[str, Any]


def scanned_pages(reading: dict[str, Any]) -> list[int]:
    pages = reading["validation"]["pages"]
    return [page["page"] for page in pages if SCANNED in page["flags"]]


def candidates(
    strata: dict[str, Any], readings: dict[str, dict[str, Any]]
) -> dict[str, list[Page]]:
    """Each stratum's scanned pages not yet seen, in document then page order."""
    seen = {(page["sha256"], page["page"]) for page in strata["exclude"]}
    found: dict[str, list[Page]] = {}
    for stratum, documents in strata["strata"].items():
        pages = []
        for sha256 in documents:
            reading = readings[sha256]
            for number in scanned_pages(reading):
                if (sha256, number) not in seen:
                    pages.append(
                        {"stratum": stratum, "sha256": sha256, "page": number,
                         "url": reading["urls"][0]}
                    )  # fmt: skip
        found[stratum] = pages
    return found


def draw(
    found: dict[str, list[Page]], seed: int, per_stratum: int, reserve: dict[str, int]
) -> dict[str, list[Page]]:
    """`per_stratum` pages from each stratum in a seeded order, then the next pages
    in that order as each stratum's reserve; pages numbered in drawing order."""
    rng = random.Random(seed)
    drawn: dict[str, list[Page]] = {"pages": [], "reserve": []}
    for stratum in sorted(found):
        order = found[stratum][:]
        rng.shuffle(order)
        drawn["pages"] += order[:per_stratum]
        drawn["reserve"] += order[per_stratum : per_stratum + reserve.get(stratum, 0)]
    for number, page in enumerate(drawn["pages"] + drawn["reserve"], 1):
        page["number"] = number
    return drawn


def page_image(source: Path, page: int) -> Any:
    import pypdfium2

    document = pypdfium2.PdfDocument(str(source))
    try:
        image = document[page - 1].render(scale=DPI / 72).to_pil()
    finally:
        document.close()
    return image.convert("RGB")


def ask(image: Any, url: str, prompt: str) -> dict[str, Any]:
    """One arm's reading of a page image: the model's answer, whether it finished,
    and the seconds it took. An unfinished answer scores as no answer, as the engine
    keeps none."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
        {"type": "text", "text": prompt},
    ]
    body = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": MAX_TOKENS,
        **SAMPLING,
    }
    started = time.perf_counter()
    response = httpx.post(
        f"{url}/v1/chat/completions", json=body, timeout=TIMEOUT_SECONDS
    )
    response.raise_for_status()
    choice = response.json()["choices"][0]
    finished = choice["finish_reason"] == "stop"
    return {
        "answer": str(choice["message"]["content"]) if finished else "",
        "finished": finished,
        "seconds": round(time.perf_counter() - started, 1),
    }


def save_images(source: Path, page: int, prefix: Path) -> list[Path]:
    """The page's top and bottom halves at 200 DPI, each reaching OVERLAP past the
    middle, so a line on the fold is whole in one of them."""
    image = page_image(source, page)
    width, height = image.size
    reach = round(height * (0.5 + OVERLAP))
    halves = {"a": (0, 0, width, reach), "b": (0, height - reach, width, height)}
    saved = []
    for name, box in halves.items():
        path = prefix.with_name(f"{prefix.name}-{name}.png")
        image.crop(box).save(path)
        saved.append(path)
    return saved


def unmarked(text: str) -> str:
    """Text without Markdown and LaTeX marks: table bars, heading and emphasis
    marks, dollar signs and backslashes, tags; a dash between numbers as a space.
    Both sides are compared this way, so a printed "US$" and a transcribed "US$"
    still match."""
    return RANGE.sub(r"\1 ", TAG.sub(" ", text).translate(MARKS))


def truth_text(page: Page) -> str:
    """Every word the truth holds for a page: its text and every table cell."""
    cells = [cell for table in page["tables"] for row in table for cell in row]
    return " ".join([page["text"], *cells])


def numbers(text: str) -> Counter[str]:
    return Counter({w: n for w, n in parsing.words(text).items() if any(
        c.isdigit() for c in w)})  # fmt: skip


def score_page(page: Page, answer: str) -> dict[str, tuple[int, int]]:
    """Word and number recall and precision of one transcription, as (found, of)."""
    truth, written = unmarked(truth_text(page)), unmarked(answer)
    scored = parsing.transcription(truth, written)
    wanted, have = numbers(truth), numbers(written)
    common = sum((wanted & have).values())
    scored["number recall"] = (common, sum(wanted.values()))
    scored["number precision"] = (common, sum(have.values()))
    return scored


def pooled(scores: list[dict[str, tuple[int, int]]]) -> dict[str, float]:
    """Each measure over the whole set: what was found, out of all there was."""
    shares = {}
    for measure in GATE:
        found = sum(score[measure][0] for score in scores)
        total = sum(score[measure][1] for score in scores)
        shares[measure] = found / total if total else 0.0
    return shares


def passed(shares: dict[str, float]) -> dict[str, bool]:
    return {measure: shares[measure] >= bar for measure, bar in GATE.items()}
