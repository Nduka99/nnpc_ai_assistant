"""Pages read by docling-parse and drawn by pdfium, on a small PDF made in the test.
No Docling model runs."""

from io import BytesIO
from pathlib import Path

import pypdfium2
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling_core.types.doc.base import BoundingBox, CoordOrigin

from petroknowledge import pdf
from petroknowledge.rendering import PdfiumPage, PdfiumRenderedBackend


def two_pages(path: Path) -> None:
    """A PDF of two blank pages, 200 x 100 points and 300 x 150 points."""
    document = pypdfium2.PdfDocument.new()
    document.new_page(200, 100)
    document.new_page(300, 150)
    document.save(path)
    document.close()


def backend(source: Path | BytesIO) -> PdfiumRenderedBackend:
    document = InputDocument(
        path_or_stream=source,
        format=InputFormat.PDF,
        backend=PdfiumRenderedBackend,
        filename="blank.pdf",
    )
    found = document._backend
    assert isinstance(found, PdfiumRenderedBackend)
    return found


def test_each_page_is_drawn_by_pdfium_at_the_asked_scale(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    two_pages(path)
    reader = backend(path)

    # The threaded parser can return pages in any order, so each is found by number.
    pages = {page.page_no: page for page in reader.iter_pages()}

    assert sorted(pages) == [1, 2]
    assert all(isinstance(page, PdfiumPage) for page in pages.values())
    assert pages[1].get_page_image(scale=1).size == (200, 100)
    whole = pages[2].get_page_image(scale=2)
    assert (whole.size, whole.mode) == ((600, 300), "RGB")
    box = BoundingBox(l=10, t=20, r=110, b=70, coord_origin=CoordOrigin.TOPLEFT)
    part = pages[2].get_page_image(scale=2, cropbox=box)
    assert part.size == (200, 100)
    assert part.getpixel((50, 50)) == (255, 255, 255)  # a blank page is white
    for page in pages.values():
        page.unload()
    reader.unload()


def test_a_document_given_as_bytes_is_read_too(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    two_pages(path)
    stream = BytesIO(path.read_bytes())
    stream.seek(40)  # drawn from the start, wherever the stream was left

    reader = backend(stream)

    sizes = {
        page.page_no: page.get_page_image(scale=1).size for page in reader.iter_pages()
    }
    assert sizes == {1: (200, 100), 2: (300, 150)}
    reader.unload()


def test_the_converter_draws_pages_with_pdfium() -> None:
    found = pdf.converter().format_to_options[InputFormat.PDF]

    assert found.backend is PdfiumRenderedBackend
    assert found.backend_options is not None
    assert found.backend_options.render_pages is False  # type: ignore[union-attr]
