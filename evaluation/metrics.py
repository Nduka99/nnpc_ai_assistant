"""Standard information-retrieval measures, computed from ranked passage ids.

Relevance is binary and comes from a frozen answer key: a passage is relevant to
a question part when it contains one of that part's evidence quotes. Each
function takes one ranked list and the relevant ids, so the same numbers can be
reproduced from the saved run and qrels files with any IR tool.
"""

import math
import random
from collections.abc import Sequence


def first_relevant_rank(ranking: Sequence[int], relevant: set[int]) -> int | None:
    """1-based rank of the first relevant passage, or None if there is none."""
    return next((i for i, pid in enumerate(ranking, 1) if pid in relevant), None)


def success_at_k(ranking: Sequence[int], relevant: set[int], k: int) -> float:
    """1.0 when at least one relevant passage is in the top k (hit rate)."""
    return float(any(pid in relevant for pid in ranking[:k]))


def recall_at_k(ranking: Sequence[int], relevant: set[int], k: int) -> float:
    """Share of the relevant passages that are in the top k."""
    if not relevant:
        return 0.0
    return sum(pid in relevant for pid in ranking[:k]) / len(relevant)


def precision_at_k(ranking: Sequence[int], relevant: set[int], k: int) -> float:
    """Share of the top k that is relevant (low by design: a part has 1-3
    relevant passages among hundreds)."""
    return sum(pid in relevant for pid in ranking[:k]) / k if k else 0.0


def reciprocal_rank(ranking: Sequence[int], relevant: set[int], k: int) -> float:
    """1 / rank of the first relevant passage within the top k, else 0."""
    rank = first_relevant_rank(ranking[:k], relevant)
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(ranking: Sequence[int], relevant: set[int], k: int) -> float:
    """Discounted cumulative gain at k, normalised by the best possible order."""
    gains = sum(
        1 / math.log2(i + 1) for i, pid in enumerate(ranking[:k], 1) if pid in relevant
    )
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return gains / ideal if ideal else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if not ordered:
        return 0.0
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def wilson(found: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval (95% by default) for `found` out of `total`; with no
    data, the whole range. Sound at small counts and at 0, unlike the normal
    approximation (Brown, Cai and DasGupta 2001)."""
    if total == 0:
        return (0.0, 1.0)
    p = found / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohen_kappa(first: Sequence[str], second: Sequence[str]) -> float:
    """Agreement between two graders' labels on the same items beyond what their label
    frequencies give by chance (Cohen 1960); 1.0 when both give one label throughout."""
    total = len(first)
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / total
    labels = set(first) | set(second)
    expected = sum(
        (list(first).count(label) / total) * (list(second).count(label) / total)
        for label in labels
    )
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def auc(pairs: Sequence[tuple[float, bool]]) -> float:
    """ROC AUC of (score, positive) pairs: the chance a positive scores above a
    negative, ties counting half (Mann-Whitney)."""
    positives = [score for score, positive in pairs if positive]
    negatives = [score for score, positive in pairs if not positive]
    if not positives or not negatives:
        return 0.0
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives
    )
    return wins / (len(positives) * len(negatives))


def bootstrap_interval(
    values: Sequence[float], rounds: int = 10000, seed: int = 42
) -> tuple[float, float]:
    """95% interval for the mean, by resampling the questions."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(
        mean([values[rng.randrange(len(values))] for _ in values])
        for _ in range(rounds)
    )
    return (means[int(0.025 * rounds)], means[int(0.975 * rounds)])


def paired_bootstrap(
    better: Sequence[float],
    baseline: Sequence[float],
    rounds: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Difference of means on the same questions, its 95% interval and p value.

    The p value is the share of resamples of the centred differences whose mean
    is at least as far from zero as the observed difference (two-sided).
    """
    differences = [b - a for b, a in zip(better, baseline, strict=True)]
    observed = mean(differences)
    rng = random.Random(seed)
    centred = [d - observed for d in differences]
    resampled = sorted(
        mean([differences[rng.randrange(len(differences))] for _ in differences])
        for _ in range(rounds)
    )
    null = [
        mean([centred[rng.randrange(len(centred))] for _ in centred])
        for _ in range(rounds)
    ]
    extreme = sum(abs(value) >= abs(observed) for value in null)
    return {
        "difference": observed,
        "low": resampled[int(0.025 * rounds)],
        "high": resampled[int(0.975 * rounds)],
        "p_value": (extreme + 1) / (rounds + 1),
    }


def paired_cluster_bootstrap(
    better: Sequence[float],
    baseline: Sequence[float],
    clusters: Sequence[str],
    rounds: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Difference of means on the same questions and its 95% interval, resampling
    whole clusters: questions from one table (or one turn) are not independent."""
    groups: dict[str, list[float]] = {}
    for new, old, cluster in zip(better, baseline, clusters, strict=True):
        groups.setdefault(cluster, []).append(new - old)
    names = sorted(groups)
    observed = mean([difference for name in names for difference in groups[name]])
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        drawn = [groups[names[rng.randrange(len(names))]] for _ in names]
        means.append(mean([difference for group in drawn for difference in group]))
    means.sort()
    return {
        "difference": observed,
        "low": means[int(0.025 * rounds)],
        "high": means[int(0.975 * rounds)],
    }
