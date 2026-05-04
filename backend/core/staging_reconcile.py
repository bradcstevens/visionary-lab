"""Reconcile stale 'processing' states for staging projects.

When the server stops mid-generation, projects/rooms/variations can be left
in 'processing' status permanently.  This module detects and repairs those
states so users are never stuck in an unrecoverable view.

Issue 003 of the active-and-queued-jobs-ux-redesign PRD split this module
into two responsibilities:

* ``reconcile_project`` — variation/room cleanup ONLY, gated on the existing
  staleness check.  Does NOT mutate ``project_data["status"]`` on any path.
* ``compute_project_status_from_jobs`` — derives the canonical project
  status from the active job in the jobs container, falling back to a pure
  ``_derive_status_from_rooms`` helper when the job is terminal or missing.

The buggy "mixed room statuses ⇒ failed" branch is removed entirely.  No
reconcile path produces ``failed``; that decision belongs exclusively to
the worker (dispatch failure, poison-queue exhaustion), the cancellation
cascade, or a producer-side hard error.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from backend.core.config import settings
from backend.core.project_lease import TERMINAL_JOB_STATUSES

logger = logging.getLogger(__name__)

_STALE_SECONDS = settings.STAGING_STALE_PROCESSING_MINUTES * 60


def _is_stale(updated_at: Optional[str], threshold_seconds: int = _STALE_SECONDS) -> bool:
    """Return True if *updated_at* is older than *threshold_seconds* ago."""
    if not updated_at:
        return True
    try:
        ts = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        return age > threshold_seconds
    except (ValueError, TypeError):
        return True


def _reset_variation(v: dict) -> bool:
    """Reset a single variation to pending.  Returns True if it changed."""
    if v.get("status") not in ("processing", "failed"):
        return False
    v["status"] = "pending"
    v["error"] = None
    v["image_url"] = None
    v["thumbnail_url"] = None
    v["generation_metadata"] = None
    return True


def _derive_status_from_rooms(rooms: Iterable[Dict[str, Any]]) -> str:
    """Pure: derive a project status from per-room statuses.

    Rules (issue 003 of the active-and-queued-jobs-ux-redesign PRD):

    * Empty rooms → ``pending`` (vacuously: nothing to be done yet).
    * All rooms ``completed`` → ``completed``.
    * Mix that contains at least one ``completed`` → ``completed``
      (preserves the legacy "user can regenerate individual rooms that
      were reset" behavior).
    * Otherwise → ``pending``.

    The reconcile path NEVER produces ``failed``; the bug-report scenario
    that flipped queued projects to ``failed`` lived in this branch.
    """
    rooms_list: List[Dict[str, Any]] = list(rooms)
    if not rooms_list:
        return "pending"
    statuses = {r.get("status") for r in rooms_list}
    if statuses == {"completed"}:
        return "completed"
    if "completed" in statuses:
        return "completed"
    return "pending"


def compute_project_status_from_jobs(
    project: Dict[str, Any], store: Any
) -> Optional[str]:
    """Compute the canonical project status by inspecting the active job.

    Short-circuits — returns ``None`` (no change) — when:

    * the project's status is not currently ``processing`` (nothing to do
      from this path, the project has already settled);
    * the project has no ``current_project_job_id`` (no active job to
      consult; legacy projects without a lease pointer are intentionally
      left alone here, since flipping them to ``pending`` based on rooms
      could erase the user's last-known state).

    Otherwise fetches the referenced job from the ``JobStore``:

    * Active non-terminal job → returns ``None`` (the worker is still
      processing; do not change status).
    * Terminal job present (``succeeded``, ``failed``, ``cancelled``) OR
      job document missing → returns the status derived from rooms via
      ``_derive_status_from_rooms``.

    The caller is responsible for applying the returned status to the
    project document and persisting the writeback.
    """
    if project.get("status") != "processing":
        return None

    job_id = project.get("current_project_job_id")
    if not job_id:
        return None

    project_id = project.get("id")
    job = store.get_job(job_id, project_id) if project_id else None

    if job is not None and job.get("status") not in TERMINAL_JOB_STATUSES:
        return None

    return _derive_status_from_rooms(project.get("rooms") or [])


def reconcile_project(
    project_data: Dict[str, Any],
    *,
    force: bool = False,
) -> bool:
    """Reconcile stale variation/room states **in-place**.

    * Rooms that are fully ``completed`` are left alone (any stuck
      ``processing`` variations inside a completed room are reset).
    * Rooms stuck in ``processing`` have their incomplete variations reset
      to ``pending`` (the whole room is reset so the pipeline can re-run).
    * Variations stuck in ``processing`` are reset to ``pending`` with
      orphan data (error, URLs, metadata) cleared.

    Issue 003 of the active-and-queued-jobs-ux-redesign PRD removed the
    project-status mutation from this function.  Status derivation now
    lives in ``compute_project_status_from_jobs``; ``reconcile_project``
    no longer touches ``project_data["status"]`` on any path.

    Parameters
    ----------
    project_data : dict
        Raw Cosmos DB document — mutated in place.
    force : bool
        If ``True``, skip the staleness time-check and always reconcile.

    Returns
    -------
    bool
        ``True`` if any field was changed (caller should persist).
    """
    if project_data.get("status") != "processing":
        return False

    if not force and not _is_stale(project_data.get("updated_at")):
        return False

    changed = False
    rooms = project_data.get("rooms", [])

    for room in rooms:
        room_status = room.get("status")

        if room_status == "completed":
            for v in room.get("variations", []):
                if v.get("status") == "processing":
                    _reset_variation(v)
                    changed = True
            continue

        if room_status == "processing":
            room["status"] = "pending"
            room["error"] = None
            for v in room.get("variations", []):
                if v.get("status") in ("processing", "pending"):
                    if _reset_variation(v) or v.get("status") == "pending":
                        changed = True
                    v["status"] = "pending"
            changed = True

    if changed:
        logger.info(
            "Reconciled stale project %s: variation cleanup applied",
            project_data.get("id"),
        )

    return changed
