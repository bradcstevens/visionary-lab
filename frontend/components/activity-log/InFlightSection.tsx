"use client";

import { useEffect, useState } from "react";
import { Loader2, RefreshCw, Pencil, Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { InFlightOp } from "@/context/activity-log-context";
import { cn } from "@/utils/utils";

/**
 * Issue 008 of the projects-page-improvements PRD: live "In Flight (N)"
 * section pinned to the top of the activity log panel. Renders one row
 * per active operation with its label, a phase indicator
 * ("Starting…" / "Running"), and a live-updating elapsed timer.
 *
 * The elapsed timer ticks from `op.startedAt` IMMEDIATELY (per AC and
 * the rubber-duck blocking finding — the panel is honest about how
 * long the user has been waiting since their click, not when the
 * backend started returning data). The phase label flips from
 * "Starting…" to "Running" when the first SSE event lands so the user
 * has an honest signal for "queued behind a backend semaphore slot"
 * vs "actually running".
 *
 * One shared interval (1s) drives all rows' re-renders so we don't
 * spawn N timers when N ops are in flight. The interval only runs
 * when there is at least one op to time, and is torn down when the
 * In Flight section disappears.
 */
interface InFlightSectionProps {
  inFlight: ReadonlyArray<InFlightOp>;
}

const KIND_ICON: Record<InFlightOp["kind"], typeof Loader2> = {
  project: Sparkles,
  room: Loader2,
  variation: RefreshCw,
  "edit-prompt": Pencil,
};

const KIND_LABEL: Record<InFlightOp["kind"], string> = {
  project: "Project",
  room: "Room",
  variation: "Variation",
  "edit-prompt": "Edit prompt",
};

function formatElapsed(seconds: number): string {
  const safe = Math.max(0, Math.floor(seconds));
  if (safe < 60) return `0:${String(safe).padStart(2, "0")}`;
  const m = Math.floor(safe / 60);
  const s = safe % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function InFlightSection({ inFlight }: InFlightSectionProps) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (inFlight.length === 0) return;
    // Tick every 1s so all rows re-render together. Initial render
    // already computed `now` from the lazy initializer; we don't
    // setNow synchronously here (which would trigger a cascading
    // render). The first interval tick is at most 1s away, which
    // is acceptable for "live elapsed timer" semantics.
    const interval = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(interval);
  }, [inFlight.length]);

  if (inFlight.length === 0) return null;

  return (
    <div
      className="border-b border-border/60 bg-muted/30 shrink-0"
      data-testid="in-flight-section"
    >
      <div className="flex items-center justify-between px-4 py-2 border-b border-border/30">
        <div className="flex items-center gap-2">
          <h3 className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground/70">
            In Flight
          </h3>
          <Badge
            variant="secondary"
            className="text-[10px] px-1.5 py-0 h-[18px] font-mono bg-background text-foreground/70 border border-border/60"
            data-testid="in-flight-count"
          >
            {inFlight.length}
          </Badge>
        </div>
        <span
          className="inline-flex items-center gap-1.5 text-[10px] font-mono text-emerald-600 dark:text-emerald-400/70"
          aria-label="Live"
        >
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500 dark:bg-emerald-400/70 animate-pulse" />
          live
        </span>
      </div>
      <ul className="divide-y divide-border/30">
        {inFlight.map((op) => {
          const Icon = KIND_ICON[op.kind];
          const elapsedSec = Math.max(0, (now - op.startedAt.getTime()) / 1000);
          const phase = op.hasFirstEvent ? "Running" : "Starting…";
          return (
            <li
              key={op.id}
              data-testid={`in-flight-row-${op.id}`}
              data-kind={op.kind}
              data-phase={op.hasFirstEvent ? "running" : "starting"}
              className="flex items-center gap-2 px-4 py-2 text-[12px]"
            >
              <Icon
                className={cn(
                  "h-3.5 w-3.5 shrink-0 text-muted-foreground/70",
                  !op.hasFirstEvent && "animate-pulse",
                  op.hasFirstEvent && op.kind !== "edit-prompt" && "animate-spin",
                )}
                aria-hidden
              />
              <span className="sr-only">{KIND_LABEL[op.kind]}: </span>
              <span
                className="flex-1 truncate text-foreground/80 font-medium"
                data-testid={`in-flight-label-${op.id}`}
              >
                {op.label}
              </span>
              <span
                className={cn(
                  "text-[10px] font-mono tabular-nums",
                  op.hasFirstEvent
                    ? "text-muted-foreground/60"
                    : "text-amber-600 dark:text-amber-400/70",
                )}
                data-testid={`in-flight-phase-${op.id}`}
              >
                {phase}
              </span>
              <span
                className="text-[10px] font-mono tabular-nums text-muted-foreground/70 min-w-[2.5rem] text-right"
                data-testid={`in-flight-elapsed-${op.id}`}
              >
                {formatElapsed(elapsedSec)}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
