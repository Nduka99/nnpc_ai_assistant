"""Pages without a text layer (scans) transcribed by a vision model.

The PDF reader runs no OCR (`pdf`): a page whose text layer holds almost no words is
flagged "little or no text" in its reading's validation. `transcribe_all` draws each
such page at 200 DPI (GLM-OCR's setting for PDFs). The layout model finds its regions
(`regions`), and the regions the GLM-OCR SDK leaves out are painted white: page
headers and footers, and pictures, whose words the picture reader reads. The model
then reads the rest of the page whole with its text-recognition prompt. Read whole
with nothing painted out, pages gained chart labels and furniture; read region by
region, they lost words in tables and doubled lines at region edges (measured on
scans-v2). The answers are saved in data/transcripts/<sha256>.json with
what produced them. A run resumes page by page. An answer the model did not finish
(a repetition loop that reached the token limit) is reported and tried again next
run, never kept.

The model answers in Markdown, and now and then writes a table in HTML.
`transcript_elements` turns saved answers into elements in the reader's shape, so
changing that conversion needs no new model run.
Unlike a table read again (`tables`), a transcription keeps the model's own text:
the page has no text layer to spell it with. So its elements stay apart from the
document's reading, and wherever they are quoted they are labelled as transcribed.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import time
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from petroknowledge import fetch, llm, regions
from petroknowledge.elements import Element, grid_text, row_text
from petroknowledge.tables import MATH, SAMPLING, Cell, parse, unlatex

if TYPE_CHECKING:
    from PIL.Image import Image

PROMPT = "Text Recognition:"  # GLM-OCR's prompt for the text in an image
# Painted out as the SDK leaves them out: running furniture, and pictures, which
# are read for their words by the picture reader, not as page text.
MASKED = ("page_header", "page_footer", "picture")
MAX_TOKENS = 8192  # GLM-OCR's published limit
TIMEOUT_SECONDS = 600.0
DPI = 200  # GLM-OCR's setting for PDF pages
LAYOUT_DPI = 72  # the layout model sees the page as Docling draws it: boxes in points
SCANNED = "little or no text"  # the flag `documents.validate` gives such a page
RULE = re.compile(r"^:?-{3,}:?$")  # a cell of a Markdown table's header rule
CAPTION_WORDS = 15  # a longer line above a table is prose, not the table's title

ReadPage = Callable[[Path, int, str], "str | None"]


def scanned_pages(reading: dict[str, Any]) -> list[int]:
    """The pages a document's reading flags as having little or no text layer."""
    pages = reading["validation"]["pages"]
    return [page["page"] for page in pages if SCANNED in page["flags"]]


def page_image(path: Path, number: int, dpi: int = DPI) -> Image:
    import pypdfium2

    document = pypdfium2.PdfDocument(str(path))
    try:
        image: Image = document[number - 1].render(scale=dpi / 72).to_pil()
    finally:
        document.close()
    return image.convert("RGB")


@cache
def layout() -> tuple[Any, Any]:
    """The layout model, loaded once per process."""
    return regions.load()


def masked(image: Image, boxes: list[regions.Box], scale: float) -> Image:
    """The page with each box, given at the layout's resolution, painted white."""
    from PIL import ImageDraw

    painted = image.copy()
    draw = ImageDraw.Draw(painted)
    for left, top, right, bottom in boxes:
        draw.rectangle(
            (left * scale, top * scale, right * scale, bottom * scale), fill="white"
        )
    return painted


def read_page(path: Path, number: int, url: str) -> str | None:
    """The model's reading of the page with its furniture and pictures painted out,
    or None when it did not finish its answer."""
    found = regions.find(page_image(path, number, LAYOUT_DPI), *layout())
    hidden = [region.box for region in found if region.label in MASKED]
    image = masked(page_image(path, number), hidden, DPI / LAYOUT_DPI)
    return ask(image, url, PROMPT)


def ask(image: Image, url: str, prompt: str) -> str | None:
    """The model's answer for one image, or None when it did not finish it.
    `tables.recognise` asks the same way but cannot tell an unfinished answer, and
    changing it would change the PDF reader's fingerprint and reread every PDF."""
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
    try:
        response = llm.CLIENT.post(
            f"{url}/v1/chat/completions",
            json=body,
            headers=llm.auth(),
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        answer = str(choice["message"]["content"])
        finished = choice["finish_reason"] == "stop"
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
        raise llm.ModelServerError(f"vision model at {url} failed: {error}") from error
    return answer if finished else None


def plain(text: str) -> str:
    """A line of the answer as printed: the inline LaTeX the model writes for a
    symbol ($^{\\circ}$C, $10\\%$) as characters. A span between dollar signs is
    math only when it holds LaTeX markup, so the dollar signs of amounts ("$5
    million and $6 million") stay as written."""

    def convert(match: re.Match[str]) -> str:
        span = match.group(0)
        return unlatex(span) if any(mark in span for mark in "\\^_{") else span

    return " ".join(MATH.sub(convert, text).split())


def cells(line: str) -> list[str]:
    """A Markdown table line's cells."""
    inner = line.removeprefix("|").removesuffix("|")
    return [plain(cell) for cell in inner.split("|")]


def table_elements(
    rows: list[list[str]],
    page: int,
    number: int,
    caption: str,
    section: tuple[str, ...],
) -> list[Element]:
    """A Markdown table as the PDF reader's table elements: the whole table, its
    header, and each row with every value paired with its column's header. The
    model's first row is the header; its header rule is not a row."""
    rows = [row for row in rows if not all(RULE.match(cell) for cell in row)]
    if not rows:
        return []
    width = max(len(row) for row in rows)
    grid = tuple(tuple(row + [""] * (width - len(row))) for row in rows)
    headers = grid[0]
    header = " | ".join(text for text in headers if text)
    found = [
        Element(
            page, "table", grid_text(grid, caption), section, table=number, grid=grid
        ),
        Element(page, "table_header", header, section, table=number),
    ]
    for index, row in enumerate(grid[1:]):
        pairs = tuple(zip(headers, row, strict=True))
        found.append(
            Element(page, "table_row", row_text(pairs, caption), section,
                    table=number, row=index, cells=pairs)
        )  # fmt: skip
    return found


def cell_grid(cells: list[Cell]) -> list[list[str]]:
    """An HTML table answer's cells as rows, a cell spanning rows or columns
    repeated in each, so every value keeps its row label and column header."""
    height = max(cell.row + cell.rows for cell in cells)
    width = max(cell.column + cell.columns for cell in cells)
    grid = [[""] * width for _ in range(height)]
    for cell in cells:
        for row in range(cell.row, cell.row + cell.rows):
            for column in range(cell.column, cell.column + cell.columns):
                grid[row][column] = plain(cell.text)
    return grid


def html_table(
    lines: list[str], page: int, number: int, caption: str, section: tuple[str, ...]
) -> list[Element]:
    """A table the model wrote in HTML, as table elements; none when no cell
    reads."""
    cells = parse(" ".join(lines))
    if not cells:
        return []
    return table_elements(cell_grid(cells), page, number, caption, section)


def caption_above(found: list[Element]) -> str:
    """The line directly above a table when it is short enough to be its title, as
    a printed table's title sits above it."""
    if not found or found[-1].kind not in ("heading", "paragraph"):
        return ""
    text = found[-1].text
    return text if len(text.split()) <= CAPTION_WORDS else ""


def markdown_elements(answer: str, page: int, first_table: int = 1) -> list[Element]:
    """One page's Markdown answer as elements: `#` lines are headings that set the
    section of what follows, "- " lines are list items, consecutive table lines are
    one table (numbered from `first_table`), other lines are paragraphs. A table row
    not yet closed by "|" goes on over the next lines, as the model writes a cell
    holding several lines. A table written in HTML runs from `<table` to `</table>`."""
    found: list[Element] = []
    headings: list[tuple[int, str]] = []  # the level and text of each heading in force
    table: list[str] = []  # the current table's rows as written
    html: list[str] = []  # the current HTML table's lines as written
    number = first_table
    for line in [*answer.splitlines(), ""]:  # the last, empty line ends a table
        line = line.strip()
        if line.startswith("|"):
            table.append(line)
            continue
        if table and line and not table[-1].endswith("|"):
            table[-1] += " " + line  # a cell written over several lines
            continue
        section = tuple(text for _, text in headings)
        if table:
            rows = [cells(row) for row in table]
            made = table_elements(rows, page, number, caption_above(found), section)
            if made:
                number += 1
            found += made
            table = []
        if html or line.casefold().startswith("<table"):
            html.append(line)
            if "</table>" in line.casefold():
                made = html_table(html, page, number, caption_above(found), section)
                number += 1 if made else 0
                found += made
                html = []
            continue
        if not line:
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            text = plain(line[level:])
            headings = [(n, heading) for n, heading in headings if n < level]
            section = tuple(heading for _, heading in headings)
            found.append(Element(page, "heading", text, section))
            headings.append((level, text))
        elif line.startswith(("- ", "* ")):
            found.append(Element(page, "list", plain(line[2:]), section))
        else:
            found.append(Element(page, "paragraph", plain(line), section))
    if html:  # a table the answer never closed still holds its cells
        section = tuple(text for _, text in headings)
        found += html_table(html, page, number, caption_above(found), section)
    return found


def transcript_elements(transcript: dict[str, Any]) -> list[Element]:
    """Every transcribed page's elements in page order, with tables numbered
    through the document."""
    found: list[Element] = []
    for number in sorted(transcript["pages"], key=int):
        tables = sum(1 for element in found if element.kind == "table")
        answer = transcript["pages"][number]["answer"]
        found += markdown_elements(answer, int(number), tables + 1)
    return found


def fingerprint(url: str) -> dict[str, Any]:
    """What an answer depends on: the model the server has loaded, the layout model
    and the code that finds regions, what is painted out, the prompt, the token
    limit, the sampling settings and the resolutions. The Markdown conversion is not
    part of it: it runs on saved answers."""
    response = httpx.get(f"{url}/v1/models", headers=llm.auth(), timeout=30.0)
    response.raise_for_status()
    model = response.json()["data"][0]["id"]
    code = Path(regions.__file__).read_bytes()
    return {
        "vision_model": model,
        "layout_model": f"{regions.MODEL}@{regions.REVISION}",
        "regions_sha256": hashlib.sha256(code).hexdigest(),
        "masked": list(MASKED),
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "sampling": SAMPLING,
        "dpi": DPI,
        "layout_dpi": LAYOUT_DPI,
    }


def save(target: Path, transcript: dict[str, Any]) -> None:
    """Written whole, then moved into place, so an interrupted run never leaves a
    broken transcript."""
    part = target.with_suffix(".part")
    text = json.dumps(transcript, indent=1, ensure_ascii=False) + "\n"
    part.write_text(text, encoding="utf-8", newline="\n")
    part.replace(target)


def transcribe_all(
    url: str,
    readings: Path,
    out: Path,
    read: ReadPage = read_page,
    only: set[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    """Transcribe each scanned page of every document read into `readings` that is
    not yet transcribed the same way, or only the (sha256, page) pairs in `only`;
    the run's counts and the pages that failed."""
    out.mkdir(parents=True, exist_ok=True)
    found = fingerprint(url)
    report: dict[str, Any] = {"documents": 0, "pages": 0, "skipped": 0, "failed": []}
    for path in sorted(readings.glob("*.json")):
        if len(path.stem) != 64:  # not a reading: the run's report, or a part file
            continue
        reading = json.loads(path.read_text(encoding="utf-8"))
        pages = scanned_pages(reading)
        if only is not None:
            pages = [n for n in pages if (reading["sha256"], n) in only]
        if not pages:
            continue
        report["documents"] += 1
        target = out / path.name
        transcript = {"sha256": reading["sha256"], "fingerprint": found, "pages": {}}
        if target.exists():
            saved = json.loads(target.read_text(encoding="utf-8"))
            if saved["fingerprint"] == found:
                transcript = saved
        source = fetch.store_path(reading["sha256"])
        for number in pages:
            if str(number) in transcript["pages"]:
                report["skipped"] += 1
                continue
            failure = {"url": reading["urls"][0], "page": number}
            started = time.perf_counter()
            try:
                answer = read(source, number, url)
            except llm.ModelServerError as error:
                report["failed"].append({**failure, "error": str(error)})
                continue
            if answer is None:
                report["failed"].append({**failure, "error": "answer not finished"})
                continue
            seconds = round(time.perf_counter() - started, 1)
            transcript["pages"][str(number)] = {"seconds": seconds, "answer": answer}
            save(target, transcript)
            report["pages"] += 1
    return report
