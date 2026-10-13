"""What a reader sees on a PDF page, read by pdfium (X8 report: "index only what the
page shows").

A word is visible when its centre lies on the page and drawing the page's text changes
the page inside its box (the page drawn with and without its text objects). Text that
draws nothing itself is a layer over another rendition of it (outlines, a scan) and is
visible where the page has ink. Clipped, covered, white and off-page text is not
visible. Docling reads such text, so its items keep only the words the page shows
(`keep_visible`); visible words Docling lacks are recovered (petroknowledge.recovery).

Imported only when a PDF is read, so the API never needs the `ingest` group.
"""

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pypdfium2
import pypdfium2.raw as pdfium_c
from docling.utils.locks import pypdfium2_lock
from PIL import ImageChops

from petroknowledge.pdf import EDGES, squash

Box = tuple[float, float, float, float]  # left, top, right, bottom from the top-left
Found = tuple[str, Box, bool]  # a word, its box, and whether its text is a layer
Word = tuple[str, Box, bool]  # a word, its box, and whether a reader sees it
Line = list[Word]
Page = tuple[tuple[float, float], list[Line]]  # a page's size and its lines of words
Images = tuple[Any, Any]  # the drawn page, and what its text draws (`drawings`)
INK = 200  # a grey level (0 black, 255 white) darker than this is ink
SCALE = 2.0  # pages are drawn at 144 DPI
LINE_END_HYPHEN = "￾"  # how pdfium writes a hyphen that ends a line it joins
MARGIN = 1.0  # points around an item's box within which its words are looked for


def key(word: str) -> str:
    """A word as compared with the reading: no spacing, case, typography or edge
    marks. Empty unless it holds a letter or a digit, so bullets are never compared."""
    folded = squash(word).strip(EDGES)
    return folded if any(char.isalnum() for char in folded) else ""


def layer(textpage: Any, index: int) -> bool:
    """Whether the character's text draws nothing itself, being a layer over another
    rendition of it (outlines, a scan): invisible text, or a Type 3 font, the only
    font type with no BaseFont (PDF 32000-1, 9.6.5)."""
    text_object = pdfium_c.FPDFText_GetTextObject(textpage, index)
    mode = pdfium_c.FPDFTextObj_GetTextRenderMode(text_object)
    if mode == pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE:
        return True
    font = pdfium_c.FPDFTextObj_GetFont(text_object)
    return bool(pdfium_c.FPDFFont_GetBaseFontName(font, None, 0) <= 1)  # "" and NUL


def text_lines(textpage: Any, height: float) -> list[list[Found]]:
    """The page's words as pdfium reads them, line by line, each with the box around
    its characters (from the page's top-left corner) and whether it is a layer.
    pdfium joins a line ending in a hyphen to the next; the hyphen is written back as
    the page shows it."""
    text = textpage.get_text_range().replace(LINE_END_HYPHEN, "-")
    lines = []
    for line in re.finditer(r"[^\r\n]+", text):
        found = []
        for match in re.finditer(r"\S+", line.group()):
            start = line.start() + match.start()
            end = start + len(match[0])
            chars = [textpage.get_charbox(i) for i in range(start, end)]
            left = min(char[0] for char in chars)
            bottom = min(char[1] for char in chars)
            right = max(char[2] for char in chars)
            top = max(char[3] for char in chars)
            box = (left, height - top, right, height - bottom)
            found.append((match[0], box, layer(textpage, start)))
        if found:
            lines.append(found)
    return lines


def drawings(page: Any) -> Images:
    """The page drawn in grey at SCALE, and what its text draws on it: the difference
    from the page drawn with every text object invisible (the page is changed)."""
    drawn = page.render(scale=SCALE).to_pil().convert("L")
    for text_object in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_TEXT]):
        pdfium_c.FPDFTextObj_SetTextRenderMode(
            text_object, pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE
        )
    bare = page.render(scale=SCALE).to_pil().convert("L")
    return drawn, ImageChops.difference(drawn, bare)


def visible(word: Found, size: tuple[float, float], images: Images) -> bool:
    """Whether a reader sees the word: its centre lies on the page, and its text
    changes the drawn page inside its box, or (a layer) the page has ink there."""
    _, (left, top, right, bottom), is_layer = word
    width, height = size
    if not (0 <= (left + right) / 2 <= width and 0 <= (top + bottom) / 2 <= height):
        return False
    drawn, text = images
    area = (
        max(0, int(left * SCALE)),
        max(0, int(top * SCALE)),
        min(drawn.width, int(right * SCALE) + 1),
        min(drawn.height, int(bottom * SCALE) + 1),
    )
    if area[0] >= area[2] or area[1] >= area[3]:
        return False
    _, change = text.crop(area).getextrema()
    if change > 0:
        return True
    darkest, _ = drawn.crop(area).getextrema()
    return is_layer and darkest < INK


def read_pages(path: Path, first: int, last: int) -> dict[int, Page]:
    """Pages `first` to `last` of a PDF as pdfium reads them: each page's size and
    its lines of words, each word marked with whether a reader sees it."""
    pages = {}
    with pypdfium2_lock:
        document = pypdfium2.PdfDocument(path)
        try:
            for number in range(max(first, 1), min(last, len(document)) + 1):
                page = document[number - 1]
                size = page.get_size()
                textpage = page.get_textpage()
                found = text_lines(textpage, size[1])
                textpage.close()
                images = drawings(page) if found else None
                page.close()
                lines = []
                for line in found:
                    marked = []
                    for word in line:
                        seen = images is not None and visible(word, size, images)
                        marked.append((word[0], word[1], seen))
                    lines.append(marked)
                pages[number] = (size, lines)
        finally:
            document.close()
    return pages


def overlaps(box: Box, bbox: Box) -> bool:
    """Whether a word's box overlaps an item's box (widened by MARGIN)."""
    left, top, right, bottom = bbox
    w_left, w_top, w_right, w_bottom = box
    across = w_left < right + MARGIN and w_right > left - MARGIN
    down = w_top < bottom + MARGIN and w_bottom > top - MARGIN
    return across and down


def shows_inside(lines: list[Line], bbox: Box) -> bool:
    """Whether the page shows any word with a letter or digit over the box."""
    for line in lines:
        for text, box, seen in line:
            if seen and key(text) and overlaps(box, bbox):
                return True
    return False


def hidden_inside(lines: list[Line], bbox: Box) -> set[str]:
    """The spellings (as `key`) of the words inside the box that the page hides
    there and shows nowhere else inside it."""
    left, top, right, bottom = bbox
    shown = set()
    hidden = set()
    for line in lines:
        for text, (w_left, w_top, w_right, w_bottom), seen in line:
            x = (w_left + w_right) / 2
            y = (w_top + w_bottom) / 2
            if not (left - MARGIN <= x <= right + MARGIN):
                continue
            if not (top - MARGIN <= y <= bottom + MARGIN):
                continue
            if seen:
                shown.add(key(text))
            else:
                hidden.add(key(text))
    return hidden - shown - {""}


def keep_visible(
    pages: dict[int, Page],
    removed: Counter[int],
    number: int,
    bbox: Box | None,
    text: str,
) -> str:
    """The text of an item on page `number` without the words the page hides inside
    the item's box, and none of it where the page shows no word over the box (text
    docling-parse reads, or places, where no reader sees it). Where pdfium reads no
    word on the page, nothing is known to be hidden. `removed` counts the words taken
    out on each page."""
    page = pages.get(number)
    if page is None or bbox is None or not page[1]:
        return text
    words = text.split()
    if shows_inside(page[1], bbox):
        hidden = hidden_inside(page[1], bbox)
        kept = [word for word in words if key(word) not in hidden]
    else:
        kept = []
    if len(kept) == len(words):
        return text
    removed[number] += len(words) - len(kept)
    return " ".join(kept)
