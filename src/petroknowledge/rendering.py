"""Page images from pdfium for Docling's docling-parse backend.

On Windows, docling-parse's threaded page renderer corrupts memory when its worker
threads exit (docling-parse #357). Measured here: one silent access violation in
twenty documents, and a crash on the IWI Installation Guide. So docling-parse still
reads each page's text, its renderer is switched off, and pages are drawn by pdfium,
Docling's own renderer until v2.128, the way Docling drew them. pdfium is not
thread-safe, so every call holds Docling's pdfium lock.

Imported only when a PDF is read, so the API never needs the `ingest` group.
"""

from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import Any

import pypdfium2
from docling.backend.docling_parse_backend import (
    ThreadedDoclingParseDocumentBackend,
    ThreadedDoclingParsePageBackend,
)
from docling.datamodel.backend_options import ThreadedDoclingParseBackendOptions
from docling.utils.locks import pypdfium2_lock
from docling_core.types.doc.base import BoundingBox, CoordOrigin
from PIL.Image import Image

# docling-parse reads the text and draws nothing.
OPTIONS = ThreadedDoclingParseBackendOptions.model_validate({"render_pages": False})


class PdfiumPage(ThreadedDoclingParsePageBackend):
    """A page read by docling-parse and drawn by pdfium."""

    def __init__(self, result: Any, page: pypdfium2.PdfPage) -> None:
        super().__init__(result, rendered=False)
        self._page = page

    def get_page_image(
        self, scale: float = 1, cropbox: BoundingBox | None = None
    ) -> Image:
        """The page (or `cropbox` of it) at `scale`, drawn at 1.5 times the scale and
        reduced, for sharper text, as Docling's pdfium backend draws it."""
        size = self.get_size()
        if cropbox is None:
            cropbox = BoundingBox(
                l=0, t=0, r=size.width, b=size.height, coord_origin=CoordOrigin.TOPLEFT
            )
            margins = (0.0, 0.0, 0.0, 0.0)
        else:
            box = cropbox.to_bottom_left_origin(size.height)
            margins = (box.l, box.b, size.width - box.r, size.height - box.t)
        with pypdfium2_lock:
            bitmap = self._page.render(scale=scale * 1.5, crop=margins)
            image: Image = bitmap.to_pil().copy()
            bitmap.close()
        width = round(cropbox.width * scale)
        height = round(cropbox.height * scale)
        return image.resize((width, height)).convert("RGB")

    def unload(self) -> None:
        with pypdfium2_lock:
            self._page.close()
        super().unload()


class PdfiumRenderedBackend(ThreadedDoclingParseDocumentBackend):
    """docling-parse's backend with its renderer off and pdfium drawing the pages."""

    def __init__(
        self, in_doc: Any, path_or_stream: BytesIO | Path, options: Any = None
    ) -> None:
        super().__init__(in_doc, path_or_stream, OPTIONS)
        if isinstance(path_or_stream, BytesIO):
            path_or_stream.seek(0)
        with pypdfium2_lock:
            self._pdf = pypdfium2.PdfDocument(path_or_stream)

    def iter_pages(self) -> Iterator[PdfiumPage]:
        for page in super().iter_pages():
            with pypdfium2_lock:
                native = self._pdf[page.page_no - 1]
            yield PdfiumPage(page._result, native)

    def unload(self) -> None:
        super().unload()
        with pypdfium2_lock:
            self._pdf.close()
