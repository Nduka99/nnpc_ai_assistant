"""Regions of a page image: where its paragraphs, headings, tables and pictures are.

A scanned page has no text layer, so Docling's own reading cannot place it. The same
layout model Docling runs on every PDF page (docling-layout-heron) finds the regions
on the page's image instead, so the regions the GLM-OCR SDK leaves out (page
headers, footers, pictures) can be left out of the page's transcription
(`transcribe`).

Detections are kept as Docling keeps them: at least its confidence for their label.
The model often marks one area twice (a list item that is also text) or marks a
wrapper and its parts (a table and its cells, a picture and its labels), so regions
are taken largest first and a region lying mostly inside one already kept is part of
it, read with it once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image

MODEL = "docling-project/docling-layout-heron"
REVISION = (
    "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8"  # the cached snapshot Docling uses
)
# Docling's confidence for each label (docling.utils.layout_postprocessor); 0.5 else.
THRESHOLDS = {
    "section_header": 0.45,
    "title": 0.45,
    "code": 0.45,
    "checkbox_selected": 0.45,
    "checkbox_unselected": 0.45,
    "form": 0.45,
    "key_value_region": 0.45,
    "document_index": 0.45,
}
THRESHOLD = 0.5
DETECTED = 0.3  # Docling's lowest score for any detection, before the label's own
INSIDE = 0.8  # a region with this share of its area inside a kept one is part of it

Box = tuple[float, float, float, float]  # left, top, right, bottom; top-left origin


@dataclass(frozen=True)
class Region:
    label: str
    confidence: float
    box: Box  # in the image's pixels


def load() -> tuple[Any, Any]:
    """The layout model and its image processor, from the pinned revision."""
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    processor = AutoImageProcessor.from_pretrained(MODEL, revision=REVISION)  # type: ignore[no-untyped-call]
    model = AutoModelForObjectDetection.from_pretrained(MODEL, revision=REVISION)
    return processor, model.eval()


def detect(image: Image, processor: Any, model: Any) -> list[Region]:
    """Every detection the model makes on the image above Docling's lowest score."""
    import torch

    inputs = processor(images=[image], return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
    size = torch.tensor([[image.height, image.width]])
    found = processor.post_process_object_detection(
        outputs, target_sizes=size, threshold=DETECTED
    )[0]
    names = model.config.id2label
    regions = []
    for label, score, box in zip(
        found["labels"], found["scores"], found["boxes"], strict=True
    ):
        left, top, right, bottom = (float(value) for value in box)
        regions.append(
            Region(names[int(label)], float(score), (left, top, right, bottom))
        )
    return regions


def area(box: Box) -> float:
    left, top, right, bottom = box
    return max(0.0, right - left) * max(0.0, bottom - top)


def shared(first: Box, second: Box) -> float:
    """The area two boxes share."""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    return area((left, top, right, bottom))


def kept(found: list[Region]) -> list[Region]:
    """The confident regions, each area read once: largest first, a region lying
    mostly inside one already kept is dropped; the same box twice keeps the more
    confident label."""
    confident = [
        region
        for region in found
        if region.confidence >= THRESHOLDS.get(region.label, THRESHOLD)
    ]
    confident.sort(key=lambda region: (-area(region.box), -region.confidence))
    chosen: list[Region] = []
    for region in confident:
        inside = any(
            shared(region.box, other.box) >= INSIDE * area(region.box)
            for other in chosen
        )
        if not inside:
            chosen.append(region)
    return chosen


def find(image: Image, processor: Any, model: Any) -> list[Region]:
    """The page image's regions, each area once."""
    return kept(detect(image, processor, model))
