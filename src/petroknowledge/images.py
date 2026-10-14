"""Pictures as the index keeps them (Lime Green X43 B, carried over for NNPC).

Every picture is stored once in data/images/ as a PNG whose longest side is at most
MAX_SIDE pixels: the format llama.cpp's image loader reads. Pictures come from two
places:
- a post's own uploads, named in its Nuxt data (`nuxt.page_media`): the notices,
  flyers and newsletters downloaded from a list;
- a PDF's figures, cut from their page by pdfium at their box, as `rendering` draws
  pages.

A picture shown in SHARED places or more is a banner or logo, not content, and is
left out (Lime Green X43). A figure on a scanned page is left out too: a scan's text
failed its gate, so it may not come back as text read from a picture. The text
in each picture is read once by GLM-OCR and saved beside it with what read it.
"""

import hashlib
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import pypdfium2
from PIL import Image, ImageStat, UnidentifiedImageError

from petroknowledge import nuxt, transcribe

MAX_SIDE = 1024  # pixels: the longest side of a stored image
LIGHT = 200  # mean grey level above which drawn pixels count as light
DARK_GROUND = (51, 51, 51)  # laid under light drawings, so white lettering shows
MIN_SIDE = 72  # points: a smaller PDF figure is a mark, not a picture
SHARED = 3  # shown in this many places, a picture is a banner or a logo
SCANNED = "little or no text"  # the flag a reading gives a page with no text layer
FIGURE_SCALE = 200 / 72  # a figure is drawn at 200 DPI, as tables are read (pdf.py)
TAG = re.compile(r"<[^>]+>")
# A line of markup alone: a code fence (with its language) or a horizontal rule.
MARKUP_LINE = re.compile(r"```\w*|[-*_=]{3,}")
Box = tuple[float, float, float, float]  # left, top, right, bottom, from top-left


def normalised(data: bytes) -> bytes:
    """The image as an RGB PNG at most MAX_SIDE pixels on its longest side. Any
    transparency is laid on a ground that contrasts with what is drawn: dark under a
    light drawing (white icons and lettering), white otherwise. Raises
    PIL.UnidentifiedImageError for a format PIL cannot read (SVG)."""
    with Image.open(BytesIO(data)) as opened:
        opened.load()
        image = opened.convert("RGBA")
    drawn = image.getchannel("A").point(lambda alpha: 255 if alpha > 128 else 0)
    level = (
        ImageStat.Stat(image.convert("L"), mask=drawn).mean[0] if drawn.getbbox() else 0
    )
    flat = Image.new("RGB", image.size, DARK_GROUND if level > LIGHT else "white")
    flat.paste(image, mask=image.getchannel("A"))
    flat.thumbnail((MAX_SIDE, MAX_SIDE))
    out = BytesIO()
    flat.save(out, format="PNG")
    return out.getvalue()


def image_id(png: bytes) -> str:
    """A stored image's id: the SHA-256 of its PNG."""
    return hashlib.sha256(png).hexdigest()


def crop(pdf: Path, page: int, box: Box) -> bytes:
    """A figure's picture as PNG bytes: page `page` (from 1) drawn by pdfium at
    FIGURE_SCALE and cut at `box`, in points from the page's top-left corner."""
    document = pypdfium2.PdfDocument(str(pdf))
    try:
        drawn = document[page - 1]
        width, height = drawn.get_size()
        left, top, right, bottom = box
        margins = (left, height - bottom, width - right, top)
        bitmap = drawn.render(scale=FIGURE_SCALE, crop=margins)
        out = BytesIO()
        bitmap.to_pil().save(out, format="PNG")
        bitmap.close()
    finally:
        document.close()
    return normalised(out.getvalue())


def figure_boxes(reading: dict[str, Any]) -> list[dict[str, Any]]:
    """A document reading's figures large enough to be pictures (MIN_SIDE points on
    each side) and not on a scanned page, with their page, box, caption and
    section."""
    pages = reading["validation"]["pages"]
    scanned = {page["page"] for page in pages if SCANNED in page["flags"]}
    found = []
    for element in reading["elements"]:
        box = element.get("bbox")
        if element["kind"] != "figure" or not box or element["page"] in scanned:
            continue
        left, top, right, bottom = box
        if right - left >= MIN_SIDE and bottom - top >= MIN_SIDE:
            found.append({
                "page": element["page"],
                "box": box,
                "alt": element["text"],
                "section": element["section"],
            })  # fmt: skip
    return found


def collect(
    posts: list[tuple[str, str, str]],
    readings: list[dict[str, Any]],
    stored: dict[str, Path],
    out: Path,
) -> list[dict[str, Any]]:
    """Every picture of the posts and documents stored once in `out` as PNG, and
    each place it is shown: the post or document (its URL), the page of a document,
    its alt text or caption and its section. `posts` holds each post's URL, HTML and
    media host (its uploads are named relative to it); `stored` each collected
    file's path by URL. Pictures shown in SHARED places or more are left out."""
    out.mkdir(parents=True, exist_ok=True)
    places = []

    def keep(png: bytes, place: dict[str, Any]) -> None:
        identity = image_id(png)
        target = out / f"{identity}.png"
        if not target.exists():
            target.write_bytes(png)
        places.append(place | {"id": identity})

    for url, raw, media_base in posts:
        for media in nuxt.page_media(raw):
            path = stored.get(media_base + media["url"])
            if path is None:
                continue  # not collected: only the listed files were downloaded
            try:
                png = normalised(path.read_bytes())
            except UnidentifiedImageError:
                continue  # vector images (SVG) hold no pixels to read
            address = media_base + media["url"]
            keep(png, {"source": url, "page": None, "alt": media["alt"],
                       "section": [], "image_url": address})  # fmt: skip
    for reading in readings:
        url = reading["urls"][0]
        for figure in figure_boxes(reading):
            png = crop(stored[url], figure["page"], tuple(figure["box"]))
            keep(png, {"source": url, "page": figure["page"], "alt": figure["alt"],
                       "section": figure["section"], "image_url": ""})  # fmt: skip
    shown: dict[str, set[tuple[str, int | None]]] = {}
    for place in places:
        shown.setdefault(place["id"], set()).add((place["source"], place["page"]))
    return [place for place in places if len(shown[place["id"]]) < SHARED]


def ocr_lines(reading: str) -> list[str]:
    """The text a model read in a picture, as lines: inline LaTeX read as the
    characters it prints, markup tags dropped, empty lines and lines of markup alone
    left out. GLM-OCR answers a picture with no text by an empty code fence (Lime
    Green X44 F1), and a swatch by a horizontal rule."""
    from petroknowledge.tables import unlatex

    plain = TAG.sub(" ", unlatex(reading))
    lines = [" ".join(line.split()) for line in plain.splitlines()]
    return [line for line in lines if line and not MARKUP_LINE.fullmatch(line)]


def read_all(
    places: list[dict[str, Any]], folder: Path, url: str, fingerprint: dict[str, Any]
) -> int:
    """Read the text in each placed picture once with the vision model at `url`,
    saved as folder/<id>.json with what read it; a picture read the same way before
    is kept. A reply the model did not finish (a loop) is no text. The count read."""
    read = 0
    for identity in dict.fromkeys(place["id"] for place in places):
        target = folder / f"{identity}.json"
        if target.exists():
            saved = json.loads(target.read_text(encoding="utf-8"))
            if saved["fingerprint"] == fingerprint:
                continue
        with Image.open(folder / f"{identity}.png") as picture:
            answer = transcribe.ask(picture.convert("RGB"), url, transcribe.PROMPT)
        found = {"id": identity, "fingerprint": fingerprint, "ocr": answer or ""}
        target.write_text(json.dumps(found, ensure_ascii=False), encoding="utf-8")
        read += 1
    return read
