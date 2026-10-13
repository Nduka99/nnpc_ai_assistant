"""Fusion and reranking (Lime Green's retrieve, unchanged). Every passage here is
invented."""

import pytest

from petroknowledge import config
from petroknowledge.records import Passage, as_read, described
from petroknowledge.retrieve import fuse, rerank_top, unit_vector


def candidates(count: int) -> list[Passage]:
    """Passages whose texts grow longer, in fused order."""
    return [
        Passage(i, "https://example.test/p", f"Page {i}", "Section", "x" * i, "")
        for i in range(1, count + 1)
    ]


def test_fusion_rewards_agreement_and_breaks_ties_by_id() -> None:
    # Passages 1 and 3 appear in both rankings with equal fused scores; 2 and 4 once.
    assert fuse([[3, 1, 2], [1, 3, 4]]) == [1, 3, 2, 4]


def test_the_reranker_orders_the_fused_candidates_and_the_top_k_are_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "TOP_K", 3)
    seen: list[str] = []

    def rerank(query: str, documents: list[str]) -> list[float]:
        seen.extend(documents)
        return [float(len(document)) for document in documents]  # longest first

    found = rerank_top("OB3 gas pipeline", candidates(6), rerank)

    assert len(seen) == 6  # every fused candidate is scored, with its page title
    assert all(document.startswith("Page ") for document in seen)
    assert [p.id for p in found] == [6, 5, 4]


def test_equal_scores_keep_the_fused_order() -> None:
    def rerank(query: str, documents: list[str]) -> list[float]:
        return [0.0] * len(documents)

    found = rerank_top("gas pipeline", candidates(10), rerank)

    assert [p.id for p in found] == list(range(1, config.TOP_K + 1))


def test_no_candidates_are_returned_without_calling_the_reranker() -> None:
    def rerank(query: str, documents: list[str]) -> list[float]:
        raise AssertionError("nothing to rerank")

    assert rerank_top("anything", [], rerank) == []


def test_vectors_are_scaled_to_unit_length_and_passages_read_as_sent() -> None:
    assert unit_vector([3.0, 4.0]) == [0.6, 0.8]
    assert unit_vector([0.0, 0.0]) == [0.0, 0.0]
    passage = Passage(1, "https://x/p", "Report", "Results", "Revenue rose.", "",
                      context="Results › Group")  # fmt: skip
    assert described(passage.title, passage.context, passage.text) == (
        "Report\nResults › Group\nRevenue rose."
    )
    assert as_read(passage) == "Report\nResults\nRevenue rose."
