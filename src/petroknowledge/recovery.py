"""Words a page shows that Docling's reading lacks, recovered from pdfium.

docling-parse extracts no characters for some text pdfium reads: Type 3 fonts,
some running headers (measured in X8's follow-ups: 930 visible words over the
corpus). Each pdfium line holding a visible word the page's elements lack is
recovered with only the words a reader sees (petroknowledge.visibility). A recovered
line goes after the element nearest above it in its column, in that element's
section.
"""

from collections import Counter

from petroknowledge.elements import Element
from petroknowledge.pdf import squash
from petroknowledge.visibility import Box, Line, Page, key


def around(boxes: list[Box]) -> Box:
    """The box around several boxes."""
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def anchor(placed: list[Element], box: Box) -> tuple[int, tuple[str, ...]]:
    """Where a recovered line goes: after the element nearest above it that overlaps it
    horizontally, else before the nearest such element below it, else at the page's
    end. Its section is that element's."""
    left, top, right, bottom = box
    middle = (top + bottom) / 2
    above = []
    below = []
    for index, element in enumerate(placed):
        if element.bbox is None:
            continue
        e_left, e_top, e_right, e_bottom = element.bbox
        if e_left >= right or left >= e_right:
            continue  # another column
        centre = (e_top + e_bottom) / 2
        if centre <= middle:
            above.append((centre, index))
        else:
            below.append((centre, index))
    if above:
        _, index = max(above)  # the lowest above, the last of equals (a table's rows)
        return index + 1, placed[index].section
    if below:
        _, index = min(below)  # the highest below, the first of equals
        return index, placed[index].section
    return len(placed), ()


def recover_page(
    number: int, elements: list[Element], lines: list[Line]
) -> tuple[list[Element], dict[str, int]]:
    """The page's elements with the visible words of every line they lack placed
    where the line stands, and the page's check: pdfium words a reader sees, how many
    of them the parser's own reading holds, words not shown, and lines recovered."""
    have = squash(" ".join(element.text for element in elements))
    seen = kept = hidden = 0
    lost = []
    for line in lines:
        compared = [word for word in line if key(word[0])]
        shown = [word for word in compared if word[2]]
        present = [word for word in shown if key(word[0]) in have]
        seen += len(shown)
        kept += len(present)
        hidden += len(compared) - len(shown)
        if len(present) < len(shown):
            lost.append([word for word in line if word[2]])
    placed = list(elements)
    recovered = []
    for words in lost:
        box = around([word[1] for word in words])
        recovered.append((box[1], box[0], " ".join(word[0] for word in words), box))
    for _, _, text, box in sorted(recovered):  # top to bottom, then left to right
        index, section = anchor(placed, box)
        placed.insert(index, Element(number, "recovered", text, section, box))
    check = {"words": seen, "kept": kept, "hidden": hidden, "recovered": len(lost)}
    return placed, check


def recover(
    found: list[Element],
    pages: dict[int, Page],
    removed: Counter[int],
    checks: dict[int, dict[str, int]] | None = None,
) -> list[Element]:
    """The elements in reading order, page by page, each page with its lost visible
    words recovered; `checks` receives each page's check, with the words `removed`
    from the parser's reading as hidden."""
    ordered = []
    for number in sorted(set(pages) | {element.page for element in found}):
        _, lines = pages.get(number, ((0.0, 0.0), []))
        on_page = [element for element in found if element.page == number]
        placed, check = recover_page(number, on_page, lines)
        ordered += placed
        if checks is not None:
            checks[number] = check | {"removed": removed[number]}
    return ordered
