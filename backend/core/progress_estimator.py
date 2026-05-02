"""``ProgressEstimator`` — synthetic 3-phase progress for image-pipeline
jobs, calibrated against a rolling p50 per ``(model, kind)``.

Single responsibility: turn "how long has this job been running, given
how long jobs of its shape usually take" into a ``(phase, progress)``
tuple the SSE clients can render. The actual emission cadence + the
``status=succeeded`` terminal write live in ``JobWorker``; this module
is pure estimation + ``stats`` doc maintenance.

Phase ranges (PRD § ProgressEstimator):

  - ``queued``     — 0–10%  (pre-pickup; worker emits this implicitly
                              by the ``status=pending`` doc state)
  - ``generating`` — 10–90% (this estimator's territory; mapped from
                              elapsed_seconds / p50)
  - ``finalizing`` — 90–100% (worker writes 100 + ``finalizing`` on
                              success per the existing state machine)

Stats persistence (PRD § Out of Scope: "stats doc seeded by new jobs
only"):

  - One Cosmos doc per ``(model, kind)`` in a ``stats`` container,
    partition key ``/id``. Doc shape::

        {
          "id":      "<model>:<kind>",
          "model":   "<model>",
          "kind":    "<kind>",
          "samples": [s0, s1, ... ]   (ring, last SAMPLE_RING_SIZE)
          "p50":     <cached median>,
          "count":   <total ever recorded>,
          "updated_at": "<iso>"
        }

  - ``record_completion`` is the ONLY writer; called by the worker
    after a successful dispatch. No batch backfill — the doc is
    seeded by live traffic, exactly as the PRD specifies.

  - Cold start (no doc, or empty samples ring): falls back to
    ``DEFAULT_P50_SECONDS``. Sensible default for a typical image-gen
    request.

Failure posture:

  - All Cosmos calls are best-effort. A read failure → cold-start
    default. A write failure → log a warning and move on. Progress
    estimation is cosmetic; under no circumstance should a stats blip
    take down a worker mid-job.

  - Output is monotonic non-decreasing within a single job because the
    caller threads ``prior_progress`` through every estimate call. The
    estimator clamps the computed progress to ``max(prior_progress,
    computed)``. The worker keeps the prior in memory between heartbeat
    ticks; nothing depends on Cosmos for monotonicity.
"""
from __future__ import annotations

import logging
import math
import statistics
from datetime import datetime, timezone
from typing import Any, Optional

from azure.cosmos import ContainerProxy, CosmosClient, exceptions
from azure.identity import DefaultAzureCredential

from backend.core.config import settings

logger = logging.getLogger(__name__)


STATS_CONTAINER_ID = "stats"

# Phase boundaries (inclusive lower / exclusive upper for the middle
# range). The worker owns the queued and finalizing transitions; this
# module produces values inside the generating window.
QUEUED_MAX = 10
GENERATING_FLOOR = 10
GENERATING_CEILING = 89  # never reach 90 from estimation; only success
FINALIZING_FLOOR = 90

# How many recent durations to keep per (model, kind). 50 is enough to
# track gradual drift in image-gen latency without a huge per-doc
# payload; the median is robust to outliers in this window.
SAMPLE_RING_SIZE = 50

# Cold-start fallback when no samples exist. Image-gen is typically
# 10–30s; 30 gives a slow-but-not-frozen-looking curve so the bar
# always advances visibly even on the very first job in a fresh
# environment.
DEFAULT_P50_SECONDS = 30.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stats_doc_id(model: str, kind: str) -> str:
    return f"{model}:{kind}"


class ProgressEstimator:
    """Cosmos-backed rolling-p50 estimator for image-job progress.

    Constructor injection: pass a ``container`` for tests (mock or
    emulator). Default construction uses managed identity to reach the
    configured Cosmos account and creates the ``stats`` container if
    missing (partition key ``/id``).

    Pass ``container=False`` to disable persistence entirely; the
    estimator then becomes a pure function (returns the cold-start
    default for every ``(model, kind)`` and silently no-ops on
    ``record_completion``). Useful in environments where no Cosmos
    account is available — e.g. local dev without the emulator, or
    early-boot before Cosmos is reachable.
    """

    def __init__(
        self,
        container: Optional[ContainerProxy] | bool = None,
        *,
        default_p50_seconds: float = DEFAULT_P50_SECONDS,
    ):
        self._default_p50 = float(default_p50_seconds)
        if container is False:
            self.container = None
            return
        if container is not None:
            self.container = container  # type: ignore[assignment]
            return

        # Managed-identity / key fallback parity with JobStore.
        cosmos_key = getattr(settings, "AZURE_COSMOS_DB_KEY", None) or None
        if cosmos_key:
            client = CosmosClient(
                url=settings.AZURE_COSMOS_DB_ENDPOINT, credential=cosmos_key
            )
        else:
            client = CosmosClient(
                url=settings.AZURE_COSMOS_DB_ENDPOINT,
                credential=DefaultAzureCredential(),
            )
        database = client.get_database_client(settings.AZURE_COSMOS_DB_ID)
        self.container = database.create_container_if_not_exists(
            id=STATS_CONTAINER_ID,
            partition_key={"paths": ["/id"], "kind": "Hash"},
        )

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def estimate(
        self,
        *,
        model: str,
        kind: str,
        elapsed_seconds: float,
        prior_progress: int = 0,
    ) -> tuple[str, int]:
        """Return ``(phase, progress)`` for a job that has been running
        ``elapsed_seconds`` since pickup.

        ``progress`` is an int in ``[GENERATING_FLOOR, GENERATING_CEILING]``,
        clamped to be ``>= prior_progress``. Phase is always
        ``"generating"`` from this method — the worker writes
        ``"queued"`` (pre-pickup) and ``"finalizing"`` (post-success)
        itself.

        Curve: exponential approach to the ceiling. At ``elapsed = p50``
        the bar reads exactly the midpoint of the generating window
        (50%); jobs that overrun their p50 keep moving but asymptote to
        ``GENERATING_CEILING`` so the bar never "freezes" but also
        never falsely reaches 100% before the dispatcher returns.
        """
        p50 = self._get_p50(model=model, kind=kind)
        # Defensive: a corrupt stats doc with p50<=0 would otherwise
        # divide-by-zero or return NaN. Treat as cold start.
        if not (p50 and p50 > 0):
            p50 = self._default_p50
        # Negative elapsed is nonsense; treat as 0 rather than raising.
        elapsed = max(0.0, float(elapsed_seconds))

        # Half-life curve: ratio = 1 - 2^(-elapsed/p50). Reaches 0.5 at
        # elapsed=p50, asymptotes to 1.0. Maps onto the generating
        # window via linear interp.
        ratio = 1.0 - math.pow(2.0, -elapsed / p50)
        span = GENERATING_CEILING - GENERATING_FLOOR
        computed = GENERATING_FLOOR + int(math.floor(span * ratio))
        # Clamp to the generating ceiling, then enforce monotonicity.
        computed = min(computed, GENERATING_CEILING)
        progress = max(int(prior_progress), computed)
        return "generating", progress

    # ------------------------------------------------------------------
    # Stats persistence
    # ------------------------------------------------------------------

    def record_completion(
        self, *, model: str, kind: str, elapsed_seconds: float
    ) -> None:
        """Append a completed-job duration to the rolling sample ring
        and refresh the cached median. Best-effort.

        Negative or zero durations are silently dropped — they would
        skew the median and almost certainly indicate a bug at the
        caller (clock skew, missed start time) rather than a real
        sub-second image-gen.
        """
        if self.container is None:
            return
        if elapsed_seconds is None or elapsed_seconds <= 0:
            return

        doc_id = _stats_doc_id(model, kind)
        try:
            doc = self._read_doc(doc_id)
            if doc is None:
                doc = {
                    "id": doc_id,
                    "model": model,
                    "kind": kind,
                    "samples": [],
                    "p50": None,
                    "count": 0,
                }
            samples = list(doc.get("samples") or [])
            samples.append(float(elapsed_seconds))
            # Drop oldest entries when the ring overflows.
            if len(samples) > SAMPLE_RING_SIZE:
                samples = samples[-SAMPLE_RING_SIZE:]
            doc["samples"] = samples
            doc["p50"] = float(statistics.median(samples))
            doc["count"] = int(doc.get("count") or 0) + 1
            doc["model"] = model
            doc["kind"] = kind
            doc["updated_at"] = _now_iso()
            self.container.upsert_item(body=doc)
        except Exception:  # noqa: BLE001 — cosmetic stat, never fatal
            logger.warning(
                "progress_estimator.record_completion failed model=%s kind=%s",
                model,
                kind,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_p50(self, *, model: str, kind: str) -> float:
        """Return cached p50 or the cold-start default. Best-effort."""
        if self.container is None:
            return self._default_p50
        doc_id = _stats_doc_id(model, kind)
        try:
            doc = self._read_doc(doc_id)
        except Exception:  # noqa: BLE001 — fall back to default
            logger.warning(
                "progress_estimator._get_p50 read failed model=%s kind=%s",
                model,
                kind,
                exc_info=True,
            )
            return self._default_p50
        if doc is None:
            return self._default_p50
        cached = doc.get("p50")
        if cached is None:
            samples = doc.get("samples") or []
            if not samples:
                return self._default_p50
            try:
                return float(statistics.median(samples))
            except (TypeError, statistics.StatisticsError):
                return self._default_p50
        try:
            return float(cached)
        except (TypeError, ValueError):
            return self._default_p50

    def _read_doc(self, doc_id: str) -> Optional[dict[str, Any]]:
        try:
            return self.container.read_item(item=doc_id, partition_key=doc_id)
        except exceptions.CosmosResourceNotFoundError:
            return None
