"""Unit tests for ``backend.core.progress_estimator``.

Pinned by issue 008 of the image-pipeline-and-project-ux-overhaul PRD:
phase boundaries, p50 update, cold-start fallback, monotonicity, and
the "stats doc seeded by new jobs only" contract (no historical
replay path exists in the module surface).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from azure.cosmos import exceptions

from backend.core.progress_estimator import (
    DEFAULT_P50_SECONDS,
    GENERATING_CEILING,
    GENERATING_FLOOR,
    SAMPLE_RING_SIZE,
    ProgressEstimator,
    _stats_doc_id,
)


# ---------------------------------------------------------------------------
# In-memory stand-in for a Cosmos ContainerProxy.
# ---------------------------------------------------------------------------


class _InMemoryContainer:
    """Minimal stub matching the ContainerProxy methods the estimator
    uses. Backed by a plain dict keyed by doc id.
    """

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.read_calls: list[str] = []
        self.upsert_calls: list[dict[str, Any]] = []

    def read_item(self, *, item: str, partition_key: str) -> dict[str, Any]:
        self.read_calls.append(item)
        if item not in self.docs:
            raise exceptions.CosmosResourceNotFoundError(
                status_code=404, message="not found"
            )
        # Return a copy so callers can't mutate our store accidentally.
        return dict(self.docs[item])

    def upsert_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        self.upsert_calls.append(dict(body))
        self.docs[body["id"]] = dict(body)
        return dict(body)


# ---------------------------------------------------------------------------
# estimate(): phase, monotonicity, boundaries
# ---------------------------------------------------------------------------


def test_estimate_at_zero_elapsed_returns_generating_floor():
    est = ProgressEstimator(container=False)

    phase, progress = est.estimate(
        model="m", kind="regenerate_variation", elapsed_seconds=0
    )

    assert phase == "generating"
    assert progress == GENERATING_FLOOR


def test_estimate_at_p50_elapsed_returns_midpoint():
    """Half-life curve hits 50% of the generating window at elapsed=p50."""
    est = ProgressEstimator(container=False, default_p50_seconds=20.0)

    _, progress = est.estimate(
        model="m", kind="k", elapsed_seconds=20.0
    )

    midpoint = GENERATING_FLOOR + (GENERATING_CEILING - GENERATING_FLOOR) // 2
    # Allow ±1 for integer flooring.
    assert abs(progress - midpoint) <= 1


def test_estimate_caps_at_generating_ceiling():
    """A long-overrunning job approaches but never crosses the ceiling.

    The bar must not "freeze" at p50 (so users see progress on slow
    runs) but also must not falsely reach 100% before the dispatcher
    returns.
    """
    est = ProgressEstimator(container=False, default_p50_seconds=10.0)

    _, progress = est.estimate(
        model="m", kind="k", elapsed_seconds=10_000.0
    )

    assert progress == GENERATING_CEILING
    assert progress < FINALIZING_FLOOR_VALUE


# Mirror of FINALIZING_FLOOR for the assertion above without coupling
# the test to the module-private import.
FINALIZING_FLOOR_VALUE = 90


def test_estimate_monotonic_with_prior_progress():
    """When a recomputation would lower progress, the prior wins."""
    est = ProgressEstimator(container=False, default_p50_seconds=60.0)

    # At elapsed=0 the curve says GENERATING_FLOOR; passing a higher
    # prior must NOT drop the bar back.
    _, progress = est.estimate(
        model="m", kind="k", elapsed_seconds=0.0, prior_progress=42
    )

    assert progress == 42


def test_estimate_negative_elapsed_treated_as_zero():
    est = ProgressEstimator(container=False)

    _, progress = est.estimate(
        model="m", kind="k", elapsed_seconds=-5.0
    )

    assert progress == GENERATING_FLOOR


def test_estimate_strictly_non_decreasing_across_increasing_elapsed():
    est = ProgressEstimator(container=False, default_p50_seconds=10.0)

    prior = 0
    last = -1
    for elapsed in [0, 1, 2, 5, 10, 20, 50, 100, 1000]:
        _, p = est.estimate(
            model="m", kind="k", elapsed_seconds=elapsed, prior_progress=prior
        )
        assert p >= last, f"non-monotonic at elapsed={elapsed}: {p} < {last}"
        last = p
        prior = p


# ---------------------------------------------------------------------------
# Cold-start fallback (no doc, no samples, missing/corrupt p50)
# ---------------------------------------------------------------------------


def test_cold_start_uses_default_p50_when_no_container():
    est = ProgressEstimator(container=False, default_p50_seconds=15.0)

    _, progress = est.estimate(
        model="any", kind="any", elapsed_seconds=15.0
    )

    midpoint = GENERATING_FLOOR + (GENERATING_CEILING - GENERATING_FLOOR) // 2
    assert abs(progress - midpoint) <= 1


def test_cold_start_uses_default_when_doc_missing():
    container = _InMemoryContainer()
    est = ProgressEstimator(container=container, default_p50_seconds=DEFAULT_P50_SECONDS)

    _, progress = est.estimate(
        model="m", kind="k", elapsed_seconds=DEFAULT_P50_SECONDS
    )

    midpoint = GENERATING_FLOOR + (GENERATING_CEILING - GENERATING_FLOOR) // 2
    assert abs(progress - midpoint) <= 1


def test_cold_start_uses_default_when_samples_empty():
    container = _InMemoryContainer()
    container.docs[_stats_doc_id("m", "k")] = {
        "id": _stats_doc_id("m", "k"),
        "samples": [],
        "p50": None,
    }
    est = ProgressEstimator(container=container)

    _, progress = est.estimate(
        model="m", kind="k", elapsed_seconds=DEFAULT_P50_SECONDS
    )

    midpoint = GENERATING_FLOOR + (GENERATING_CEILING - GENERATING_FLOOR) // 2
    assert abs(progress - midpoint) <= 1


@pytest.mark.parametrize("bad_p50", [0, -1, "garbage"])
def test_corrupt_p50_falls_back_to_default(bad_p50):
    container = _InMemoryContainer()
    container.docs[_stats_doc_id("m", "k")] = {
        "id": _stats_doc_id("m", "k"),
        "samples": [],
        "p50": bad_p50,
    }
    est = ProgressEstimator(container=container, default_p50_seconds=20.0)

    _, progress = est.estimate(
        model="m", kind="k", elapsed_seconds=20.0
    )

    midpoint = GENERATING_FLOOR + (GENERATING_CEILING - GENERATING_FLOOR) // 2
    assert abs(progress - midpoint) <= 1


def test_estimate_swallows_cosmos_read_failure():
    container = MagicMock()
    container.read_item.side_effect = RuntimeError("cosmos blip")
    est = ProgressEstimator(container=container, default_p50_seconds=10.0)

    _, progress = est.estimate(
        model="m", kind="k", elapsed_seconds=10.0
    )

    midpoint = GENERATING_FLOOR + (GENERATING_CEILING - GENERATING_FLOOR) // 2
    assert abs(progress - midpoint) <= 1


# ---------------------------------------------------------------------------
# record_completion(): seeds doc, updates p50, ring-caps, drops bad input
# ---------------------------------------------------------------------------


def test_record_completion_creates_doc_on_first_call():
    container = _InMemoryContainer()
    est = ProgressEstimator(container=container)

    est.record_completion(model="m", kind="k", elapsed_seconds=12.5)

    doc_id = _stats_doc_id("m", "k")
    assert doc_id in container.docs
    doc = container.docs[doc_id]
    assert doc["samples"] == [12.5]
    assert doc["p50"] == 12.5
    assert doc["count"] == 1
    assert doc["model"] == "m"
    assert doc["kind"] == "k"


def test_record_completion_appends_and_recomputes_median():
    container = _InMemoryContainer()
    est = ProgressEstimator(container=container)

    for s in [10.0, 20.0, 30.0]:
        est.record_completion(model="m", kind="k", elapsed_seconds=s)

    doc = container.docs[_stats_doc_id("m", "k")]
    assert doc["samples"] == [10.0, 20.0, 30.0]
    assert doc["p50"] == 20.0
    assert doc["count"] == 3


def test_record_completion_caps_ring_at_max_samples():
    container = _InMemoryContainer()
    est = ProgressEstimator(container=container)

    for s in range(SAMPLE_RING_SIZE + 5):
        est.record_completion(
            model="m", kind="k", elapsed_seconds=float(s + 1)
        )

    doc = container.docs[_stats_doc_id("m", "k")]
    assert len(doc["samples"]) == SAMPLE_RING_SIZE
    # The earliest 5 entries (1.0..5.0) were dropped.
    assert doc["samples"][0] == 6.0
    assert doc["count"] == SAMPLE_RING_SIZE + 5


@pytest.mark.parametrize("bad", [0, -1.0, None])
def test_record_completion_drops_non_positive_durations(bad):
    container = _InMemoryContainer()
    est = ProgressEstimator(container=container)

    est.record_completion(model="m", kind="k", elapsed_seconds=bad)

    assert container.docs == {}
    assert container.upsert_calls == []


def test_record_completion_swallows_cosmos_failure():
    container = MagicMock()
    container.read_item.side_effect = exceptions.CosmosResourceNotFoundError(
        status_code=404, message="missing"
    )
    container.upsert_item.side_effect = RuntimeError("transient")
    est = ProgressEstimator(container=container)

    # Must NOT raise.
    est.record_completion(model="m", kind="k", elapsed_seconds=10.0)


def test_record_completion_no_op_when_persistence_disabled():
    est = ProgressEstimator(container=False)

    # Must NOT raise; nothing to assert beyond "no exception".
    est.record_completion(model="m", kind="k", elapsed_seconds=10.0)


# ---------------------------------------------------------------------------
# End-to-end: recorded p50 changes the curve.
# ---------------------------------------------------------------------------


def test_recorded_p50_drives_estimate_curve():
    """After recording fast completions, a job at the same elapsed
    advances further than a cold-start estimate would.
    """
    container = _InMemoryContainer()
    est = ProgressEstimator(container=container, default_p50_seconds=30.0)

    # Pre-record fast completions: median = 5s.
    for s in [4.0, 5.0, 6.0]:
        est.record_completion(model="m", kind="k", elapsed_seconds=s)

    # At elapsed=5s the bar should now be near the midpoint of the
    # generating window (because elapsed = recorded p50).
    _, progress = est.estimate(model="m", kind="k", elapsed_seconds=5.0)
    midpoint = GENERATING_FLOOR + (GENERATING_CEILING - GENERATING_FLOOR) // 2
    assert abs(progress - midpoint) <= 1

    # A cold-start (different kind) at the same elapsed should be much
    # earlier on the curve (default p50 is 30s; 5s ≈ 1/6 of p50).
    _, cold = est.estimate(model="m", kind="other", elapsed_seconds=5.0)
    assert cold < progress


# ---------------------------------------------------------------------------
# "Stats doc seeded by new jobs only" — pinned by surface absence.
# ---------------------------------------------------------------------------


def test_estimator_exposes_no_historical_backfill_method():
    """The PRD's Out-of-Scope clause forbids replaying historical jobs
    into the stats doc. Pin that by asserting the public surface has no
    backfill / replay / import method — only ``estimate`` and
    ``record_completion`` mutate state, and the latter is one-sample-
    at-a-time from the live worker.
    """
    forbidden = {
        "backfill",
        "replay",
        "import_history",
        "seed_from_jobs",
        "rebuild",
    }
    public = {name for name in dir(ProgressEstimator) if not name.startswith("_")}
    assert public.isdisjoint(forbidden), (
        f"unexpected backfill-style method exposed: "
        f"{sorted(public & forbidden)}"
    )
