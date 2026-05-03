"use client";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/utils/cn";
import type { Room } from "@/services/stagingApi";
import type { RecoveryState } from "@/utils/recovery-state";

/**
 * Issue 003 of the projects-page-stalled-stream-error-cleanup PRD.
 *
 * Single component owning all five visual room states:
 *   - `pending`     → Badge variant `outline`
 *   - `processing`  → Badge variant `secondary`
 *   - `completed`   → Badge variant `default`
 *   - `failed`      → Badge variant `destructive`
 *   - `stalled`     → Badge variant `outline` + amber tint (NEW)
 *
 * The four pre-existing visuals stay pixel-identical to the prior
 * inline `getStatusVariant(...)` renders in `RoomGroup` and
 * `ProgressTracker`. Only the amber `stalled` treatment is new.
 *
 * The stalled treatment fires iff:
 *   status === 'processing' &&
 *     (projectRecoveryState.kind === 'interrupted' ||
 *      projectRecoveryState.kind === 'stream-lost')
 *
 * Both project-level recovery states leave `processing` rooms stuck
 * without user action; the amber pill encodes that uniformly. The
 * `error` recovery state does NOT trigger the stalled treatment —
 * the destructive banner already communicates the problem at the
 * project level and a per-room amber pill would over-state the room
 * itself as the source of the failure.
 *
 * `data-status` and (when stalled) `data-stalled="true"` are exposed
 * for behavioral testing — assertions key off these attributes rather
 * than CSS class names or snapshots so future visual tweaks don't
 * cascade into test churn.
 */

type RoomStatus = Room["status"];

interface RoomStatusPillProps {
  status: RoomStatus;
  projectRecoveryState: RecoveryState;
  /**
   * Optional override for the visible text inside the pill. Defaults to
   * the literal status word (`pending` / `processing` / `completed` /
   * `failed`) — that's the room-list call-site contract. The
   * ProgressTracker summary call site renders the room *label* instead
   * (e.g. "Living Room"), preserving the pre-migration content there
   * verbatim.
   */
  label?: string;
  className?: string;
}

function isStalled(
  status: RoomStatus,
  recovery: RecoveryState,
): boolean {
  if (status !== "processing") return false;
  return recovery.kind === "interrupted" || recovery.kind === "stream-lost";
}

function variantFor(
  status: RoomStatus,
): "default" | "secondary" | "destructive" | "outline" {
  switch (status) {
    case "completed":
      return "default";
    case "processing":
      return "secondary";
    case "failed":
      return "destructive";
    case "pending":
    default:
      return "outline";
  }
}

export function RoomStatusPill({
  status,
  projectRecoveryState,
  label,
  className,
}: RoomStatusPillProps) {
  const stalled = isStalled(status, projectRecoveryState);
  const variant = stalled ? "outline" : variantFor(status);
  return (
    <Badge
      data-testid="room-status-pill"
      data-status={status}
      data-stalled={stalled ? "true" : undefined}
      variant={variant}
      className={cn(
        "text-xs",
        stalled && "text-amber-600 border-amber-500/30",
        className,
      )}
    >
      {label ?? status}
    </Badge>
  );
}
