"""The harness's safeguards and measures: hashed sets and the standard statistics.
Adopted with the modules from the Lime Green assistant. Every set here is invented."""

import json
import math
from pathlib import Path

import pytest

from evaluation import metrics, sets

RANKING = [5, 9, 3, 7, 1]
RELEVANT = {3, 7}


def make_set(root: Path) -> Path:
    folder = root / "demo"
    (folder / "runs").mkdir(parents=True)
    (folder / "truth.json").write_text('{"pages": []}')
    (folder / "pages.json").write_text('{"pages": []}')
    (folder / "runs" / "first.json").write_text("{}")
    return folder


def test_registering_hashes_every_file_and_verify_finds_no_problem(
    tmp_path: Path,
) -> None:
    make_set(tmp_path)
    registry = tmp_path / "sets.json"

    added = sets.register("demo", "invented", tmp_path, registry)

    assert added == ["pages.json", "runs/first.json", "truth.json"]
    assert sets.problems("demo", tmp_path, registry) == []
    assert sets.require("demo", tmp_path, registry) == tmp_path / "demo"
    assert json.loads(registry.read_text())["sets"]["demo"]["description"] == "invented"


def test_a_changed_missing_or_new_file_is_caught(tmp_path: Path) -> None:
    folder = make_set(tmp_path)
    registry = tmp_path / "sets.json"
    sets.register("demo", "invented", tmp_path, registry)
    (folder / "truth.json").write_text("{}")
    (folder / "pages.json").unlink()
    (folder / "extra.json").write_text("[]")

    assert sets.problems("demo", tmp_path, registry) == [
        "demo/pages.json: missing",
        "demo/truth.json: changed since it was registered",
        "demo/extra.json: not registered",
    ]
    with pytest.raises(ValueError, match="changed since it was registered"):
        sets.require("demo", tmp_path, registry)


def test_a_registered_file_can_never_be_replaced(tmp_path: Path) -> None:
    folder = make_set(tmp_path)
    registry = tmp_path / "sets.json"
    sets.register("demo", "invented", tmp_path, registry)
    (folder / "truth.json").write_text("{}")

    with pytest.raises(ValueError, match="truth.json: changed"):
        sets.register("demo", "invented", tmp_path, registry)


def test_new_files_can_be_added_to_a_registered_set(tmp_path: Path) -> None:
    folder = make_set(tmp_path)
    registry = tmp_path / "sets.json"
    sets.register("demo", "invented", tmp_path, registry)
    (folder / "runs" / "second.json").write_text("{}")

    assert sets.register("demo", "", tmp_path, registry) == ["runs/second.json"]
    assert sets.problems("demo", tmp_path, registry) == []


def test_an_unregistered_set_is_refused(tmp_path: Path) -> None:
    assert sets.read_registry(tmp_path / "none.json") == {"sets": {}}
    assert sets.problems("demo", tmp_path, tmp_path / "none.json") == [
        "demo: not registered"
    ]


def test_rank_hit_recall_and_precision() -> None:
    assert metrics.first_relevant_rank(RANKING, RELEVANT) == 3
    assert metrics.first_relevant_rank(RANKING, {99}) is None
    assert metrics.success_at_k(RANKING, RELEVANT, 2) == 0.0
    assert metrics.success_at_k(RANKING, RELEVANT, 3) == 1.0
    assert metrics.recall_at_k(RANKING, RELEVANT, 4) == 1.0
    assert metrics.recall_at_k(RANKING, RELEVANT, 3) == 0.5
    assert metrics.recall_at_k(RANKING, set(), 3) == 0.0
    assert metrics.precision_at_k(RANKING, RELEVANT, 4) == 0.5
    assert metrics.precision_at_k(RANKING, RELEVANT, 0) == 0.0


def test_reciprocal_rank_and_ndcg_match_their_definitions() -> None:
    assert metrics.reciprocal_rank(RANKING, RELEVANT, 10) == 1 / 3
    assert metrics.reciprocal_rank(RANKING, RELEVANT, 2) == 0.0
    # Relevant at ranks 3 and 4; the best possible order puts them at 1 and 2.
    gains = 1 / math.log2(4) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert metrics.ndcg_at_k(RANKING, RELEVANT, 5) == gains / ideal
    assert metrics.ndcg_at_k([1, 2], {1}, 2) == 1.0
    assert metrics.ndcg_at_k([1, 2], set(), 2) == 0.0


def test_mean_median_and_interval() -> None:
    assert metrics.mean([]) == 0.0
    assert metrics.median([]) == 0.0
    assert metrics.median([3.0, 1.0, 2.0]) == 2.0
    assert metrics.median([4.0, 1.0, 3.0, 2.0]) == 2.5
    assert metrics.bootstrap_interval([]) == (0.0, 0.0)
    assert metrics.bootstrap_interval([1.0] * 20, rounds=200) == (1.0, 1.0)


def test_paired_bootstrap_finds_a_real_difference_and_ignores_none() -> None:
    better = [1.0] * 30 + [0.0] * 5
    baseline = [0.0] * 35
    result = metrics.paired_bootstrap(better, baseline, rounds=2000)
    assert result["difference"] > 0.8
    assert result["low"] > 0.6
    assert result["p_value"] < 0.05
    same = metrics.paired_bootstrap(baseline, baseline, rounds=2000)
    assert same["difference"] == 0.0
    assert same["p_value"] > 0.05
    mixed = metrics.paired_cluster_bootstrap([1, 0, 1], [0, 0, 0], ["a", "a", "b"])
    assert mixed["low"] <= mixed["difference"] <= mixed["high"]


def test_wilson_bounds_hold_at_zero_and_in_the_middle() -> None:
    assert metrics.wilson(0, 0) == (0.0, 1.0)
    low, high = metrics.wilson(0, 73)
    assert low == 0.0 and high == pytest.approx(0.05, abs=0.001)
    low, high = metrics.wilson(5, 10)
    assert low == pytest.approx(1 - high)


def test_kappa_corrects_agreement_for_chance_and_auc_needs_both_labels() -> None:
    assert metrics.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "b", "b"]) == 0.5
    assert metrics.cohen_kappa(["a", "a"], ["a", "a"]) == 1.0
    assert metrics.auc([(1.0, True), (0.0, False)]) == 1.0
    assert metrics.auc([(1.0, True)]) == 0.0
