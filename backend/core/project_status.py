"""Single source of truth for project-level status from per-room status.

The legacy code base computed project status inline in three places —
``StagingPipeline.generate_project``, ``regenerate_room`` (event_stream
finally), and ``regenerate_variation`` (event_stream finally) — and the
three branches drifted into subtly inconsistent shapes. The result was the
PRD's headlining "Issue 1 bug": a project with one completed room and
several pending rooms could read ``completed`` in the header badge,
because one of the inline branches forgot to check for outstanding
``pending`` work.

This module exposes one pure function that all four call sites delegate
to (the fourth, slice 004's edit-prompt endpoint, picks it up later).
The helper is pure: no I/O, no mutation, no logging.
"""

from typing import Iterable

from backend.models.staging import ProjectStatus, Room


_OUTSTANDING_STATUSES = ("pending", "processing")
_COMPLETED_STATUS = "completed"


class ProjectStatusCalculator:
    """Derives ``ProjectStatus`` from a collection of ``Room`` objects."""

    @staticmethod
    def compute_status(rooms: Iterable[Room]) -> ProjectStatus:
        """Return the project status that matches the current room states.

        Rules per PRD § Solution → 1. Truthful project status:

        - ``PENDING`` if any room is ``pending`` or ``processing``
          (work is outstanding — the project is not done).
        - ``FAILED`` if every room reached a terminal state (``completed``
          or ``failed``) AND none of them are ``completed``.
        - ``COMPLETED`` otherwise — i.e. every room is terminal and at
          least one completed (mixed completed + failed terminal still
          counts as completed since at least one room finished).

        Empty rooms list:
            Returns ``FAILED`` to follow the strict 3-rule reading
            (vacuously every room terminal and none completed). In
            practice this case is unreachable from the regen paths
            because they all reject empty-room projects up front
            (``if not project.rooms: raise HTTPException(400)``); the
            value here is a defensive default for any future caller
            and the literal-spec match avoids future drift back from
            another reader concluding "FAILED is the correct output".

        The function reads only ``Room.status``. It does NOT mutate any
        room or assign back to a project — the caller is responsible for
        persisting the returned status into the ``StagingProject``.
        """
        statuses = [r.status for r in rooms]
        if any(s in _OUTSTANDING_STATUSES for s in statuses):
            return ProjectStatus.PENDING
        if any(s == _COMPLETED_STATUS for s in statuses):
            return ProjectStatus.COMPLETED
        return ProjectStatus.FAILED
