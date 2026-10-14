"""Pictures ranked by what they show: SigLIP2 so400m on the CPU (Lime Green X43 B3,
X44 F2).

Used as its model card shows: pictures through its processor (384 px), text padded to
SIGLIP_TEXT_TOKENS tokens and lowercased (SigLIP2 was trained on lowercased text, and
this checkpoint's Gemma tokenizer does not lowercase). Vectors are normalised, so a
dot product is the cosine. Picture vectors are made once, when pictures are read, and
stored with each picture; a question's vector is made at search time by the text
tower, loaded once per process. Needs torch and transformers.
"""

import threading
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

from petroknowledge import config

BATCH = 8
LOADING = threading.Lock()  # the text tower is loaded by one thread only
Vectors = dict[str, list[float]]
EmbedPictures = Callable[[list[Path]], list[list[float]]]
EmbedTexts = Callable[[list[str]], list[list[float]]]


def picture_model(folder: Path) -> EmbedPictures:
    """The model's picture encoder, returning unit vectors."""
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(folder).eval()
    processor = AutoProcessor.from_pretrained(folder)  # type: ignore[no-untyped-call]

    def pictures(paths: list[Path]) -> list[list[float]]:
        images = [Image.open(path).convert("RGB") for path in paths]
        batch = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            return unit(model.get_image_features(**batch).pooler_output)

    return pictures


def unit(found: Any) -> list[list[float]]:
    """A batch of vectors (a torch tensor) scaled to length 1, as lists."""
    vectors: list[list[float]] = (found / found.norm(dim=-1, keepdim=True)).tolist()
    return vectors


def picture_vectors(
    ids: list[str], folder: Path, embed: EmbedPictures, known: Vectors
) -> Vectors:
    """Every picture's vector, from `known` when it holds the picture (a run resumes),
    else embedded from `folder/<id>.png` in batches."""
    found = {identity: known[identity] for identity in ids if identity in known}
    left = [identity for identity in ids if identity not in found]
    for start in range(0, len(left), BATCH):
        batch = left[start : start + BATCH]
        paths = [folder / f"{identity}.png" for identity in batch]
        found.update(zip(batch, embed(paths), strict=True))
    return found


def text_vectors(
    processor: Any, run: Callable[..., Any], found: list[str]
) -> list[list[float]]:
    """Texts through SigLIP2's text side as its model card shows: lowercased, padded
    to SIGLIP_TEXT_TOKENS tokens; `run` is the model call giving the pooled output."""
    import torch

    batch = processor(
        text=[text.lower() for text in found],
        padding="max_length",
        max_length=config.SIGLIP_TEXT_TOKENS,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        return unit(run(**batch).pooler_output)


@cache
def text_encoder() -> EmbedTexts:
    """The text tower alone, loaded once per process on first use: it gives the
    whole model's text vectors exactly and holds 2.7 GB of RAM, not 4.4 (Lime Green
    X44 G7)."""
    from transformers import AutoProcessor, SiglipTextModel

    model = SiglipTextModel.from_pretrained(config.SIGLIP)
    model.eval()  # type: ignore[no-untyped-call]
    processor = AutoProcessor.from_pretrained(config.SIGLIP)  # type: ignore[no-untyped-call]
    return lambda found: text_vectors(processor, model, found)


def text_vector(question: str) -> list[float]:
    """A question's SigLIP2 vector, for ranking pictures by what they show. A
    question with several parts asks from several threads at once; the lock lets one
    of them load the text tower while the others wait, since two loads at once fail
    in transformers' lazy imports (seen on a two-part question)."""
    with LOADING:
        encode = text_encoder()
    return encode([question])[0]
