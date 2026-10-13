"""Regions of a page image. The model's detections are invented."""

from typing import Any

import pytest
import torch
from PIL import Image

from petroknowledge import regions
from petroknowledge.regions import Region


def test_confident_regions_are_kept_once_each_largest_first() -> None:
    found = [
        Region("text", 0.9, (10, 10, 100, 20)),
        Region("list_item", 0.95, (10, 10, 100, 20)),  # the same box, more confident
        Region("table", 0.98, (10, 50, 200, 150)),
        Region("text", 0.7, (20, 60, 90, 70)),  # a cell inside the table
        Region("text", 0.49, (10, 200, 100, 210)),  # below Docling's 0.5 for text
        Region("section_header", 0.46, (10, 180, 100, 190)),  # above its 0.45
        Region("text", 0.8, (150, 140, 250, 160)),  # overlaps the table's corner only
    ]

    assert regions.kept(found) == [
        Region("table", 0.98, (10, 50, 200, 150)),
        Region("text", 0.8, (150, 140, 250, 160)),
        Region("list_item", 0.95, (10, 10, 100, 20)),
        Region("section_header", 0.46, (10, 180, 100, 190)),
    ]


class Processor:
    def __call__(self, images: list[Any], return_tensors: str) -> dict[str, Any]:
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros(1)}

    def post_process_object_detection(
        self, outputs: Any, target_sizes: Any, threshold: float
    ) -> list[dict[str, Any]]:
        assert target_sizes.tolist() == [[50, 80]] and threshold == regions.DETECTED
        return [{
            "labels": torch.tensor([8, 9]),
            "scores": torch.tensor([0.75, 0.5]),
            "boxes": torch.tensor([[1.0, 2.0, 30.0, 40.0], [5.0, 5.0, 6.0, 6.0]]),
        }]  # fmt: skip


class Model:
    class config:  # noqa: N801 - the attribute name transformers models use
        id2label = {8: "table", 9: "text"}

    def __call__(self, **inputs: Any) -> str:
        assert "pixel_values" in inputs
        return "outputs"

    def eval(self) -> "Model":
        return self


def test_detections_are_read_with_their_labels_and_boxes() -> None:
    image = Image.new("RGB", (80, 50))

    found = regions.detect(image, Processor(), Model())

    assert found == [
        Region("table", 0.75, (1.0, 2.0, 30.0, 40.0)),
        Region("text", 0.5, (5.0, 5.0, 6.0, 6.0)),
    ]
    # the small text lies inside the table, so the table is read once, with it
    assert regions.find(image, Processor(), Model()) == found[:1]


def test_the_model_is_loaded_from_its_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    asked: list[tuple[str, str]] = []

    def loader(result: Any) -> Any:
        def from_pretrained(name: str, revision: str) -> Any:
            asked.append((name, revision))
            return result

        return from_pretrained

    model = Model()
    monkeypatch.setattr(
        transformers.AutoImageProcessor, "from_pretrained", loader("processor")
    )
    monkeypatch.setattr(
        transformers.AutoModelForObjectDetection, "from_pretrained", loader(model)
    )

    assert regions.load() == ("processor", model)
    assert asked == [(regions.MODEL, regions.REVISION)] * 2
