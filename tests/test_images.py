"""Pictures as the index keeps them (Lime Green X43 B). Invented images, posts and a
blank PDF."""

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pypdfium2
import pytest
from PIL import Image, UnidentifiedImageError

from petroknowledge import images, transcribe
from tests.test_nuxt import page


def encoded(image: Image.Image, form: str) -> bytes:
    out = BytesIO()
    image.save(out, format=form)
    return out.getvalue()


def test_an_image_is_stored_as_an_rgb_png_at_most_max_side() -> None:
    wide = encoded(Image.new("RGB", (3000, 1000), "green"), "WEBP")
    clear = encoded(Image.new("RGBA", (10, 10), (0, 0, 0, 0)), "PNG")

    stored = Image.open(BytesIO(images.normalised(wide)))
    flat = Image.open(BytesIO(images.normalised(clear)))

    assert (stored.format, stored.mode, stored.size) == ("PNG", "RGB", (1024, 341))
    assert flat.getpixel((0, 0)) == (255, 255, 255)  # transparency laid on white
    icon = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    icon.putpixel((5, 5), (255, 255, 255, 255))  # a white drawing on transparency
    shown = Image.open(BytesIO(images.normalised(encoded(icon, "PNG"))))
    assert shown.getpixel((0, 0)) == images.DARK_GROUND
    assert shown.getpixel((5, 5)) == (255, 255, 255)
    assert images.image_id(b"png") == images.image_id(b"png") != images.image_id(b"x")
    with pytest.raises(UnidentifiedImageError):
        images.normalised(b"<svg xmlns='http://www.w3.org/2000/svg'/>")


def blank_pdf(path: Path, height: int = 200) -> None:
    document = pypdfium2.PdfDocument.new()
    document.new_page(200, height)
    document.save(str(path))
    document.close()


def test_a_figure_is_cut_from_its_page_at_its_box(tmp_path: Path) -> None:
    path = tmp_path / "sheet.pdf"
    blank_pdf(path, 100)

    figure = Image.open(BytesIO(images.crop(path, 1, (10, 10, 110, 60))))

    # 100 x 50 points drawn at 200 DPI: 277.8 x 138.9 pixels, which pdfium truncates
    assert figure.size == (277, 138) or figure.size == (278, 138)


FIGURE = {"kind": "figure", "page": 1, "text": "Fig 1", "section": ["Gas"],
          "bbox": [10, 10, 110, 110]}  # fmt: skip


def report(figures: list[dict[str, Any]], flags: list[list[str]]) -> dict[str, Any]:
    pages = [{"page": n, "flags": f} for n, f in enumerate(flags, 1)]
    text = {"kind": "paragraph", "page": 1, "text": "Mix", "section": [], "bbox": None}
    return {"urls": ["https://x/report.pdf"], "validation": {"pages": pages},
            "elements": [*figures, text]}  # fmt: skip


def test_figures_are_pictures_when_large_and_not_on_a_scanned_page() -> None:
    small = FIGURE | {"bbox": [0, 0, 20, 20]}
    on_scan = FIGURE | {"page": 2}
    found = images.figure_boxes(report([FIGURE, small, on_scan],
                                       [[], ["little or no text"]]))  # fmt: skip

    assert found == [
        {"page": 1, "box": [10, 10, 110, 110], "alt": "Fig 1", "section": ["Gas"]}
    ]


def post(media: str, alt: str) -> str:
    return page([{"data": 1}, {"image": 2}, {"url": 3, "mime": 4,
                 "alternativeText": 5}, media, "image/png", alt])  # fmt: skip


def test_every_post_picture_and_document_figure_is_stored_once(tmp_path: Path) -> None:
    flyer = encoded(Image.new("RGB", (40, 20), "green"), "JPEG")
    banner = encoded(Image.new("RGB", (40, 20), "red"), "PNG")
    (tmp_path / "flyer.jpg").write_bytes(flyer)
    (tmp_path / "banner.png").write_bytes(banner)
    (tmp_path / "logo.svg").write_bytes(b"<svg xmlns='http://www.w3.org/2000/svg'/>")
    blank_pdf(tmp_path / "report.pdf")
    cms = "https://cms.x"
    stored = {
        cms + "/uploads/flyer.jpg": tmp_path / "flyer.jpg",
        cms + "/uploads/banner.png": tmp_path / "banner.png",
        cms + "/uploads/logo.svg": tmp_path / "logo.svg",
        "https://x/report.pdf": tmp_path / "report.pdf",
    }
    posts = [
        ("https://x/a", post("/uploads/flyer.jpg", "flyer.jpg"), cms),
        ("https://x/a", post("/uploads/logo.svg", ""), cms),  # no pixels to read
        ("https://x/b", post("/uploads/gone.png", ""), cms),  # never collected
    ]
    posts += [(f"https://x/{n}", post("/uploads/banner.png", "banner"), cms)
              for n in "cde"]  # fmt: skip
    reading = report([FIGURE], [[]])
    out = tmp_path / "out"

    places = images.collect(posts, [reading], stored, out)

    assert [(p["source"], p["page"], p["alt"], p["image_url"]) for p in places] == [
        ("https://x/a", None, "flyer.jpg", cms + "/uploads/flyer.jpg"),
        ("https://x/report.pdf", 1, "Fig 1", ""),
    ]  # the banner, shown in three posts, is left out
    assert len(list(out.glob("*.png"))) == 3  # every picture kept once on disk
    assert (
        images.image_id((out / f"{places[0]['id']}.png").read_bytes())
        == (places[0]["id"])
    )


def test_a_reading_becomes_lines_without_markup() -> None:
    reading = "```markdown\n\n```\nSolo 2nd pass\n---\n\n<b>Duro</b> $25\\mathrm{kg}$"

    assert images.ocr_lines(reading) == ["Solo 2nd pass", "Duro 25kg"]


def test_each_picture_is_read_once_per_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for identity in ("p1", "p2"):
        Image.new("RGB", (4, 4)).save(tmp_path / f"{identity}.png")
    places = [{"id": "p1"}, {"id": "p2"}, {"id": "p1"}]
    asked: list[tuple[str, str]] = []
    replies = {"p1": "Notice of AGM", "p2": None}  # p2: a loop, no text

    def ask(picture: Image.Image, url: str, prompt: str) -> str | None:
        identity = "p1" if len(asked) % 2 == 0 else "p2"
        asked.append((url, prompt))
        return replies[identity]

    monkeypatch.setattr(transcribe, "ask", ask)
    reader = {"vision_model": "GLM-OCR"}

    assert images.read_all(places, tmp_path, "http://vlm", reader) == 2
    saved = json.loads((tmp_path / "p1.json").read_text(encoding="utf-8"))
    assert saved == {"id": "p1", "fingerprint": reader, "ocr": "Notice of AGM"}
    assert json.loads((tmp_path / "p2.json").read_text(encoding="utf-8"))["ocr"] == ""
    assert asked == [("http://vlm", "Text Recognition:")] * 2
    assert images.read_all(places, tmp_path, "http://vlm", reader) == 0  # kept
    assert images.read_all(places, tmp_path, "http://vlm", {"vision_model": "x"}) == 2


def test_the_command_line_stores_reads_and_embeds_the_pictures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from petroknowledge import cli, config, documents, fetch, llm, siglip, sources

    files, readings = tmp_path / "files", tmp_path / "elements"
    files.mkdir()
    readings.mkdir()
    (files / ("a" * 64)).write_text("<html>post</html>", encoding="utf-8")
    pdf = {"urls": ["https://x/r.pdf"], "fingerprint": {"packages": {"pypdfium2": "5"}}}
    sheet = {"fingerprint": {"packages": {"openpyxl": "3"}}}
    (readings / f"{'b' * 64}.json").write_text(json.dumps(pdf))
    (readings / f"{'d' * 64}.json").write_text(json.dumps(sheet))
    # The run's report carries the PDF reader's fingerprint but is no reading.
    report = {"fingerprint": pdf["fingerprint"], "read": []}
    (readings / "report.json").write_text(json.dumps(report))
    manifest = [
        {"url": "https://x/post", "kind": "post", "sha256": "a" * 64},
        {"url": "https://x/r.pdf", "kind": "document", "sha256": "c" * 64},
        {"url": "https://x/blocked", "kind": "page"},  # never stored
    ]
    monkeypatch.setattr(fetch, "read_manifest", lambda: manifest)
    monkeypatch.setattr(config, "FILE_STORE", files)
    monkeypatch.setattr(documents, "OUT", readings)
    monkeypatch.setattr(config, "IMAGES", tmp_path / "images")
    monkeypatch.setattr(
        sources, "host_rules", lambda url: {"media_base": "https://cms"}
    )
    monkeypatch.setattr(llm, "healthy", lambda url: url == "http://up")
    seen: dict[str, Any] = {}

    def collect(posts: Any, found: Any, stored: Any, out: Path) -> list[dict[str, Any]]:
        seen.update(posts=posts, readings=found, stored=stored)
        out.mkdir(parents=True, exist_ok=True)
        return [{"id": "p1"}, {"id": "p1"}]

    def read_all(places: Any, folder: Path, url: str, reader: Any) -> int:
        seen["reader"] = reader
        return 1

    fingerprint = {"vision_model": "GLM-OCR", "prompt": "Text Recognition:",
                   "max_tokens": 8192, "sampling": {},
                   "layout_model": "heron"}  # fmt: skip
    monkeypatch.setattr(images, "collect", collect)
    monkeypatch.setattr(images, "read_all", read_all)
    monkeypatch.setattr(transcribe, "fingerprint", lambda url: fingerprint)
    monkeypatch.setattr(siglip, "picture_model", lambda folder: lambda paths: [[1.0]])

    assert cli.main(["read-images"]) == 1
    assert "give --vlm, --vectors or both" in capsys.readouterr().err
    assert cli.main(["read-images", "--vlm", "http://down"]) == 1
    assert "no vision model ready" in capsys.readouterr().err
    assert cli.main(["read-images", "--vlm", "http://up", "--vectors"]) == 0
    assert "2 places of 1 pictures; 1 read and 1 vectors made now" in (
        capsys.readouterr().out
    )
    assert seen["posts"] == [("https://x/post", "<html>post</html>", "https://cms")]
    assert seen["readings"] == [pdf]
    assert set(seen["reader"]) == {"vision_model", "prompt", "max_tokens", "sampling"}
    vectors = json.loads((tmp_path / "images" / "vectors.json").read_text())
    assert vectors == {"p1": [1.0]}
    assert cli.main(["read-images", "--vectors"]) == 0  # kept: no new vector
    assert "0 read and 0 vectors made now" in capsys.readouterr().out
