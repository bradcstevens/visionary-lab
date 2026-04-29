"""Reconcile stale 'processing' states for staging projects.

When the server stops mid-generation, projects/rooms/variations can be left
in 'processing' status permanently.  This module detects and repairs those
states so users are never stuck in an unrecoverable view.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend.core.config import settings

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


def reconcile_project(
    project_data: Dict[str, Any],
    *,
    force: bool = False,
) -> bool:
    """Reconcile stale processing states **in-place**.

    * Rooms that are fully ``completed`` are left alone.
    * Rooms stuck in ``processing`` have their incomplete variations reset to
      ``pending`` (the whole room is reset so the pipeline can re-run it).
    * Variations stuck in ``processing`` are reset to ``pending`` with orphan
      data (error, URLs, metadata) cleared.

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
            # Preserve fully completed rooms but fix any orphan processing variations
            for v in room.get("variations", []):
                if v.get("status") == "processing":
                    _reset_variation(v)
                    changed = True
            continue

        if room_status == "processing":
            # Reset the whole room — pipeline processes all variations per room
            room["status"] = "pending"
            room["error"] = None
            for v in room.get("variations", []):
                if v.get("status") in ("processing", "pending"):
                    if _reset_variation(v) or v.get("status") == "pending":
                        changed = True
                    v["status"] = "pending"
            changed = True

    # Recompute project-level status from rooms
    statuses = {r.get("status") for r in rooms}
    if not rooms:
        new_status = "pending"
    elif statuses == {"completed"}:
        new_status = "completed"
    elif "completed" in statuses:
        # Mix of completed and pending/failed — mark completed so user can
        # regenerate individual rooms that were reset
        new_status = "completed"
    elif statuses <= {"pending"}:
        new_status = "pending"
    else:
        new_status = "failed"

    if project_data["status"] != new_status:
        project_data["status"] = new_status
        changed = True

    if changed:
        logger.info(
            "Reconciled stale project %s: status %s → %s",
            project_data.get("id"),
            "processing",
            new_status,
        )

    return changed
