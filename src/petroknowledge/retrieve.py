"""The retrieval steps shared by every index: fusion and reranking.

Two independent rankings are merged by reciprocal-rank fusion (RRF): keyword search
(BM25) finds exact terms such as product names, and vector search (cosine similarity
of embeddings) finds paraphrases; `store.search` runs both in Postgres. A
cross-encoder reranker then reads the question with each of the best fused
candidates and orders them, so the passages the model sees first are the most
relevant ones.
"""

import math
from collections.abc import Callable, Sequence

from petroknowledge import config
from petroknowledge.records import Passage, described

Embed = Callable[[list[str]], list[list[float]]]
# One relevance score per document for a query, in the documents' order.
Rerank = Callable[[str, list[str]], list[float]]


def unit_vector(vector: Sequence[float]) -> list[float]:
    """Scale to length 1, so cosine similarity becomes a plain dot product."""
    length = math.sqrt(sum(x * x for x in vector))
    return [x / length for x in vector] if length else list(vector)


def fuse(rankings: Sequence[list[int]], k: int = config.RRF_K) -> list[int]:
    """Reciprocal-rank fusion: each ranking adds 1 / (k + rank) to a passage.

    Ties are broken by passage id, so the result is deterministic.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, passage_id in enumerate(ranking, start=1):
            scores[passage_id] = scores.get(passage_id, 0.0) + 1 / (k + rank)
    return sorted(scores, key=lambda passage_id: (-scores[passage_id], passage_id))


def rerank_top(
    question: str,
    candidates: list[Passage],
    rerank: Rerank,
    top: int | None = None,
) -> list[Passage]:
    """The reranker's best passages among the fused candidates, best first: `top` of
    them, or `config.TOP_K`."""
    if not candidates:
        return []
    # The title (and a PDF passage's context) tells the reranker which product and
    # section a short passage is about.
    texts = [described(p.title, p.context, p.text) for p in candidates]
    scores = rerank(question, texts)
    # A stable sort keeps the fused order among equal scores, so ties are deterministic.
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    return [candidates[i] for i in order[: config.TOP_K if top is None else top]]
