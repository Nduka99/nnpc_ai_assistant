"""Picture vectors from SigLIP2. The model is replaced by a small invented one."""

import time
from functools import cache
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from petroknowledge import config, siglip


def test_vectors_are_scaled_to_length_one() -> None:
    found = siglip.unit(torch.tensor([[3.0, 4.0], [0.0, 2.0]]))

    assert found[0] == pytest.approx([0.6, 0.8]) and found[1] == [0.0, 1.0]


def test_picture_vectors_are_kept_or_embedded_in_batches(tmp_path: Path) -> None:
    calls: list[list[Path]] = []

    def embed(paths: list[Path]) -> list[list[float]]:
        calls.append(paths)
        return [[float(len(path.stem))] for path in paths]

    ids = [f"p{n}" for n in range(10)]
    found = siglip.picture_vectors(ids, tmp_path, embed, {"p0": [9.0]})

    assert found["p0"] == [9.0] and found["p1"] == [2.0]
    assert [len(batch) for batch in calls] == [8, 1]  # nine left, in batches of 8
    assert calls[0][0] == tmp_path / "p1.png"


def test_the_picture_encoder_runs_the_model_on_each_picture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transformers

    class Output:
        pooler_output = torch.tensor([[0.0, 5.0]])

    class Model:
        def eval(self) -> "Model":
            return self

        def get_image_features(self, **batch: Any) -> Output:
            assert batch == {"pixel_values": "pixels"}
            return Output()

    def processor(images: list[Image.Image], return_tensors: str) -> dict[str, str]:
        assert [image.mode for image in images] == ["RGB"] and return_tensors == "pt"
        return {"pixel_values": "pixels"}

    loaded: list[Path] = []

    def model_from(folder: Path) -> Model:
        loaded.append(folder)
        return Model()

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", model_from)
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", lambda folder: processor
    )
    Image.new("L", (4, 4)).save(tmp_path / "p.png")

    pictures = siglip.picture_model(tmp_path)

    assert pictures([tmp_path / "p.png"]) == [[0.0, 1.0]]
    assert loaded == [tmp_path]


def test_a_question_s_vector_comes_from_the_text_tower_loaded_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    loaded: list[Any] = []

    class TextModel:
        def eval(self) -> "TextModel":
            return self

        def __call__(self, **batch: Any) -> Any:
            rows = len(batch["input_ids"])
            return type(
                "Output", (), {"pooler_output": torch.tensor([[0.0, 3.0]] * rows)}
            )

    def text_model(folder: Path) -> TextModel:
        loaded.append(folder)
        return TextModel()

    def processor(**options: Any) -> dict[str, Any]:
        loaded.append((options["text"], options["max_length"]))
        return {"input_ids": [0] * len(options["text"])}

    monkeypatch.setattr(transformers.SiglipTextModel, "from_pretrained", text_model)
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", lambda folder: processor
    )
    siglip.text_encoder.cache_clear()

    assert siglip.text_vector("Show me Bonga FPSO") == [0.0, 1.0]
    assert siglip.text_vector("Show me a flare") == [0.0, 1.0]
    assert loaded == [config.SIGLIP, (["show me bonga fpso"], 64),
                      (["show me a flare"], 64)]  # fmt: skip
    siglip.text_encoder.cache_clear()


def test_questions_asked_at_once_load_the_text_tower_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    loads: list[int] = []

    def slow_encoder() -> siglip.EmbedTexts:
        loads.append(1)
        time.sleep(0.05)  # a second thread arrives while the first still loads
        return lambda found: [[1.0] for _ in found]

    monkeypatch.setattr(siglip, "text_encoder", cache(slow_encoder))

    with ThreadPoolExecutor(max_workers=4) as pool:
        found = list(pool.map(siglip.text_vector, ["a", "b", "c", "d"]))

    assert found == [[1.0]] * 4 and loads == [1]
