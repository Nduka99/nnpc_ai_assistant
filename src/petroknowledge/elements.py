"""The element model every parser returns (PLAN §0f): one piece of a document, in
reading order, with where it sits and what kind of text it is.

Everything after parsing (validation, passages, citations) reads only elements, so
a new parser only has to produce them. Plain, JSON-compatible data.
"""

from dataclasses import dataclass

# What an element is: a heading, running text, a list item, a whole table, a table's
# header row or one of its rows, a caption, a figure (its caption), page furniture
# (running headers and footers), or a text-layer line the parser's reading lost.
KINDS = (
    "heading",
    "paragraph",
    "list",
    "table",
    "table_header",
    "table_row",
    "caption",
    "figure",
    "furniture",
    "recovered",
)


@dataclass(frozen=True)
class Element:
    page: int  # 1-based page number
    kind: str  # one of KINDS
    text: str
    section: tuple[str, ...] = ()  # the headings above it, outermost first
    # Left, top, right, bottom in points from the page's top-left corner.
    bbox: tuple[float, float, float, float] | None = None
    table: int | None = None  # the table's number in the document, for table rows
    row: int | None = None  # the row's number in its table
    # A table row's cells as (column header, value), the row's label first.
    cells: tuple[tuple[str, str], ...] = ()
    # A whole table's rows as read, header rows included; a cell spanning several
    # positions is repeated in each (so every row keeps its label).
    grid: tuple[tuple[str, ...], ...] = ()


def grid_text(grid: tuple[tuple[str, ...], ...], caption: str = "") -> str:
    """A whole table as text: its caption, then one line per row with its cells
    joined by " | ". The passage format itself is experiment X9's question."""
    lines = [caption] if caption else []
    lines += [" | ".join(row) for row in grid]
    return "\n".join(lines)


def row_text(cells: tuple[tuple[str, str], ...], caption: str = "") -> str:
    """A table row as one quotable line: `caption › label — header: value; ...`, so
    each value keeps its row and column in the text that is searched and quoted."""
    label = cells[0][1] if cells else ""
    values = [
        f"{header}: {value}" if header else value
        for header, value in cells[1:]
        if value  # an empty cell adds nothing, not a bare "header:"
    ]
    text = label + (" — " if label and values else "") + "; ".join(values)
    return f"{caption} › {text}" if caption else text
