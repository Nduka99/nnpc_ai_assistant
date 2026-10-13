"""Tables read again from their image by a small vision-language model (X8, round 2).

Docling finds each table and its box. A model such as GLM-OCR or PaddleOCR-VL reads
the table's image and answers with its structure: rows, columns, spans and, in HTML,
header cells. The model's own text is never kept. Each cell must be spelt by words of
the PDF inside the table's box, each word used once, and those words become the
cell's text, so every quote stays the document's own characters. A cell they cannot
spell is left empty and counted.
"""

from __future__ import annotations

import base64
import io
import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import TYPE_CHECKING

import httpx

from petroknowledge import llm

if TYPE_CHECKING:
    from PIL.Image import Image

PROMPT = "Table Recognition:"  # both models' prompt for a table
MAX_TOKENS = 8192  # GLM-OCR's published limit
# GLM-OCR's published request settings (its SDK's config.yaml, the settings X8
# measured), sent with every request so a reading never depends on how the server
# was started. Without them the server's own defaults sample, and long answers loop.
SAMPLING = {"temperature": 0.0, "top_p": 0.00001, "top_k": 1, "repeat_penalty": 1.1}
TIMEOUT_SECONDS = 600.0  # a large table can take minutes
OTSL = re.compile(r"(<fcel>|<ecel>|<lcel>|<ucel>|<xcel>|<nl>)")
MATH = re.compile(r"\$(.+?)\$|\\\((.+?)\\\)")  # inline LaTeX: $...$ or \(...\)
TYPOGRAPHY = str.maketrans(dict.fromkeys('‘’“”"', "'") | dict.fromkeys("–—", "-"))
PUNCTUATION = re.compile(r"[\W_]")  # anything but a letter or a digit
# A cell holding a value rather than words: a number, an amount, a percentage or a
# dash for nil ("2,310", "(1)", "N625,000", "25%", "-").
FIGURE = re.compile(r"[(\-]?[$£€₦N]?\d[\d,.]*%?\)?|-")


@dataclass(frozen=True)
class Cell:
    """One cell of the model's table: where it starts, how far it spans, its text."""

    row: int
    column: int
    text: str
    rows: int = 1
    columns: int = 1
    header: bool = False


def recognise(image: Image, url: str, prompt: str = PROMPT) -> str:
    """The model's reading of an image, from the llama.cpp server at `url`, with
    the model's published sampling settings."""
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
        return str(response.json()["choices"][0]["message"]["content"])
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
        raise llm.ModelServerError(f"vision model at {url} failed: {error}") from error


def parse(answer: str) -> list[Cell]:
    """The cells of a table answer written as HTML (GLM-OCR) or OTSL (PaddleOCR-VL);
    none when the answer is neither."""
    if "<table" in answer.casefold():
        return parse_html(answer)
    if OTSL.search(answer):
        return parse_otsl(answer)
    return []


class _Table(HTMLParser):
    """Collects each row's cells as (text, rows spanned, columns spanned, header)."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[tuple[str, int, int, bool]]] = []
        self.text: list[str] | None = None  # the open cell's text
        self.span = (1, 1)
        self.header = False
        self.head = False  # inside <thead>

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "thead":
            self.head = True
        elif tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th") and self.rows:
            values = dict(attrs)
            self.span = (_span(values.get("rowspan")), _span(values.get("colspan")))
            self.header = tag == "th" or self.head
            self.text = []
        elif tag == "br" and self.text is not None:
            self.text.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "thead":
            self.head = False
        elif tag in ("td", "th") and self.text is not None:
            text = " ".join("".join(self.text).split())
            self.rows[-1].append((text, *self.span, self.header))
            self.text = None

    def handle_data(self, data: str) -> None:
        if self.text is not None:
            self.text.append(data)


def _span(value: str | None) -> int:
    return int(value) if value and value.isdigit() and int(value) > 0 else 1


def parse_html(answer: str) -> list[Cell]:
    """An HTML table's cells at their grid positions: a position already covered by a
    cell spanning down from above is skipped, as browsers lay tables out."""
    table = _Table()
    table.feed(answer)
    taken: set[tuple[int, int]] = set()
    cells = []
    for row, found in enumerate(table.rows):
        column = 0
        for text, down, across, header in found:
            while (row, column) in taken:
                column += 1
            cells.append(Cell(row, column, text, down, across, header))
            taken |= {(row + i, column + j) for i in range(down) for j in range(across)}
            column += across
    return cells


def parse_otsl(answer: str) -> list[Cell]:
    """An OTSL table's cells: `<nl>` ends a row, `<fcel>` starts a cell with text and
    `<ecel>` an empty one; `<lcel>`, `<ucel>` and `<xcel>` extend the cell on the left,
    above, or both ([OTSL](https://arxiv.org/abs/2305.03393)). PaddleOCR-VL writes a
    line break inside a cell as the two characters `\\n`, read here as a space."""
    grid: list[list[tuple[str, str]]] = [[]]
    parts = OTSL.split(answer)
    for index in range(1, len(parts), 2):
        tag, text = parts[index], parts[index + 1].replace("\\n", " ")
        if tag == "<nl>":
            grid.append([])
        else:
            grid[-1].append((tag, " ".join(text.split())))
    grid = [row for row in grid if row]
    cells = []
    for row, tags in enumerate(grid):
        for column, (tag, text) in enumerate(tags):
            if tag not in ("<fcel>", "<ecel>"):
                continue
            across = 1
            while column + across < len(tags) and tags[column + across][0] in (
                "<lcel>",
                "<xcel>",
            ):
                across += 1
            down = 1
            while (
                row + down < len(grid)
                and column < len(grid[row + down])
                and grid[row + down][column][0] in ("<ucel>", "<xcel>")
            ):
                down += 1
            cells.append(Cell(row, column, text, down, across))
    return cells


def unlatex(text: str) -> str:
    """The model's inline LaTeX ($\\lambda$, $^{\\circ}$C, $^2$) as the characters a
    PDF prints (λ, °C, 2), so a cell can be matched to the PDF's words. Only math is
    converted: outside it, % and & are ordinary characters, not LaTeX markup."""
    from pylatexenc.latex2text import LatexNodes2Text

    converter = LatexNodes2Text()

    def plain(match: re.Match[str]) -> str:
        latex = match.group(1) or match.group(2)
        latex = latex.replace("^{\\circ}", "°").replace("^\\circ", "°")
        return re.sub(r"[\^_{}]", "", converter.latex_to_text(latex))

    return MATH.sub(plain, text)


def fold(text: str) -> str:
    """Text for comparison only: no whitespace, case, typographic or compatibility
    variants (a superscript ² is a 2)."""
    plain = unicodedata.normalize("NFKC", text).translate(TYPOGRAPHY)
    return "".join(plain.casefold().split())


def bare(text: str) -> str:
    """Folded text with its punctuation removed: letters and digits only."""
    return PUNCTUATION.sub("", fold(text))


def spell(text: str, words: list[str], used: set[int]) -> list[int] | None:
    """The unused words, by index, that spell `text` once folded and joined, taken in
    reading order where possible; None when the words cannot spell it. When they
    cannot, punctuation is ignored on both sides and they are tried again: a bracket
    or hyphen the model adds or leaves out ("LENGTH(...)" for "LENGTH (...", NNPC's
    pdfs-v1) is no reason to drop a cell whose words the PDF holds."""
    found = _spell(fold(text), [fold(word) for word in words], frozenset(used), 0)
    if found is None:
        found = _spell(bare(text), [bare(word) for word in words], frozenset(used), 0)
    return found


def _spell(
    rest: str, words: list[str], used: frozenset[int], start: int
) -> list[int] | None:
    if not rest:
        return []
    tried = set()
    for index in [*range(start, len(words)), *range(start)]:
        word = words[index]
        if index in used or not word or word in tried or not rest.startswith(word):
            continue
        tried.add(word)  # the same word elsewhere spells the same rest
        tail = _spell(rest[len(word) :], words, used | {index}, index + 1)
        if tail is not None:
            return [index, *tail]
    return None


@dataclass(frozen=True)
class Structure:
    """A table as the model reads it, in the PDF's own words."""

    headers: list[str]  # each column's header, levels joined by " › "
    # (row number, each column's value, each column's header): a row takes the
    # headers of the latest header block above it.
    rows: list[tuple[int, list[str], list[str]]]
    dropped: int  # cells the PDF's words could not spell, left empty
    leftover: list[str]  # the PDF's words no cell used, in the PDF's order
    # Every row, header rows included, a spanning cell repeated in each position.
    grid: list[list[str]]


def repeated_rows(cells: list[Cell]) -> set[int]:
    """Rows the model wrote twice: with no label of their own, and the same values as
    the next row, which has one. Spelt first, a repeat would use up the PDF words the
    labelled row needs (NNPC's pdfs-v1)."""
    by_row: dict[int, dict[int, str]] = {}
    for cell in cells:
        by_row.setdefault(cell.row, {})[cell.column] = fold(unlatex(cell.text))
    found = set()
    for row, values in by_row.items():
        following = by_row.get(row + 1, {})
        others = {column: text for column, text in values.items() if column}
        if values.get(0) or not following.get(0) or not any(others.values()):
            continue
        if others == {column: text for column, text in following.items() if column}:
            found.add(row)
    return found


def has_figure(values: list[str]) -> bool:
    """Whether a row holds a value: a number, an amount, a percentage or a dash."""
    return any(FIGURE.fullmatch(value.strip()) for value in values if value)


def names_columns(candidate: list[str], following: list[str]) -> bool:
    """Whether a text-only row heads the row of values after it: it names every
    column that row fills, after the label column, with at least two distinct names
    (a section's name or a title written across the columns is not a header)."""
    filled = [column for column in range(1, len(following)) if following[column]]
    named = [candidate[column] for column in filled]
    return all(named) and len(set(named)) >= 2


def joined(block: list[list[str]], width: int) -> list[str]:
    """One header per column: a header block's texts top to bottom, each once."""
    headers = []
    for column in range(width):
        parts: list[str] = []
        for values in block:
            if values[column] and values[column] not in parts:
                parts.append(values[column])
        headers.append(" › ".join(parts))
    return headers


def structure(cells: list[Cell], words: list[str], header_words: set[str]) -> Structure:
    """The table's column headers and data rows, each cell's text spelt by the PDF's
    `words`. Header rows are those the model marks; if it marks none, those whose
    words all lie in `header_words` (the rows Docling flagged: the two can split a
    header differently, a label spanning two header rows sitting in either). Lower
    down, a run of text-only rows naming the columns of the row of values after it
    starts a new header block, as when stacked tables are read as one (NNPC's
    pdfs-v1). A cell spanning columns or rows applies to each of them and is
    repeated, except that a value spanning the whole width (a title row) is written
    once. A row the model wrote twice is read once."""
    repeated = repeated_rows(cells)
    cells = [cell for cell in cells if cell.row not in repeated]
    used: set[int] = set()
    texts = {}
    dropped = 0
    for cell in cells:
        found = spell(unlatex(cell.text), words, used)
        if found is None:
            dropped += 1
        used.update(found or [])
        texts[cell] = " ".join(words[index] for index in found or [])
    height = max(cell.row + cell.rows for cell in cells)
    width = max(cell.column + cell.columns for cell in cells)
    starting = {
        row: [cell for cell in cells if cell.row == row] for row in range(height)
    }
    if any(cell.header for cell in cells):
        heads = {
            row
            for row, found in starting.items()
            if found and all(c.header for c in found)
        }
    else:
        heads = set()
        for row, begun in starting.items():
            said = {fold(word) for cell in begun for word in texts[cell].split()}
            if said and said <= header_words:
                heads.add(row)
    headers = []
    for column in range(width):
        parts: list[str] = []
        for cell in cells:
            if cell.row in heads and _covers(cell, cell.row, column):
                if texts[cell] and texts[cell] not in parts:
                    parts.append(texts[cell])
        headers.append(" › ".join(parts))
    rows: list[tuple[int, list[str], list[str]]] = []
    current = headers
    pending: list[tuple[int, list[str]]] = []  # text-only rows: a block, or rows
    for row in range(height):
        if row in heads:
            continue
        values = [""] * width
        for cell in cells:
            if not _covers(cell, row, cell.column):
                continue
            last = cell.column + (1 if cell.columns == width else cell.columns)
            for column in range(cell.column, last):
                values[column] = texts[cell]
        if not any(values):
            continue
        if not has_figure(values):
            pending.append((row, values))
            continue
        if pending and names_columns(pending[-1][1], values):
            current = joined([found for _, found in pending], width)
        else:
            rows += [(number, found, current) for number, found in pending]
        pending = []
        rows.append((row, values, current))
    rows += [(number, found, current) for number, found in pending]
    leftover = [word for index, word in enumerate(words) if index not in used]
    grid = [[""] * width for _ in range(height)]
    for cell in cells:
        for row in range(cell.row, cell.row + cell.rows):
            for column in range(cell.column, cell.column + cell.columns):
                grid[row][column] = texts[cell]
    return Structure(headers, rows, dropped, leftover, grid)


def _covers(cell: Cell, row: int, column: int) -> bool:
    return (
        cell.row <= row < cell.row + cell.rows
        and cell.column <= column < cell.column + cell.columns
    )
