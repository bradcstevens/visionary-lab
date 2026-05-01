"use client";

import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from "react";

export interface LogEntry {
  id: string;
  timestamp: Date;
  level: "info" | "success" | "error" | "warn";
  message: string;
  detail?: string;
  icon?: string;
}

/**
 * Issue 008 of the projects-page-improvements PRD: live "In Flight (N)"
 * panel inside the activity log. The activity-log context is extended
 * (rather than introducing a parallel store) so the existing singleton
 * mounted in `app/layout.tsx` becomes the consolidated `ActivityFeed`
 * the PRD describes — chronological log + In Flight surface together.
 *
 * The page-level `useGenerationFleet` hook calls into this context via
 * `startOp` / `markOpStarted` / `endOp` so the panel and the per-op
 * flags driving buttons cannot drift apart. `id` matches the fleet
 * hook's stream id, which is what makes `endOp(opId)` unambiguous.
 *
 * `clear()` clears chronological entries only — NOT inFlight. A queued
 * op is still real even if the user clears history.
 */
export interface InFlightOp {
  id: string;
  label: string;
  kind: "project" | "room" | "variation" | "edit-prompt";
  startedAt: Date;
  /**
   * False until the first SSE event of any type lands for this stream.
   * Drives a "Starting…" / "Running" phase label in the panel. The
   * elapsed timer ticks from `startedAt` regardless of this flag —
   * "live elapsed timer" per AC is honest about how long the user has
   * been waiting, not when the backend started returning data.
   */
  hasFirstEvent: boolean;
}

interface ActivityLogContextValue {
  entries: LogEntry[];
  log: (entry: Omit<LogEntry, "id" | "timestamp">) => void;
  clear: () => void;
  isOpen: boolean;
  setOpen: (open: boolean) => void;
  hasActivity: boolean;
  inFlight: InFlightOp[];
  startOp: (op: Omit<InFlightOp, "startedAt" | "hasFirstEvent">) => void;
  markOpStarted: (opId: string) => void;
  endOp: (opId: string) => void;
}

const MAX_ENTRIES = 500;

const ActivityLogContext = createContext<ActivityLogContextValue | null>(null);

export function ActivityLogProvider({ children }: { children: ReactNode }) {
  const entriesRef = useRef<LogEntry[]>([]);
  const inFlightRef = useRef<InFlightOp[]>([]);
  const [revision, setRevision] = useState(0);
  const [isOpen, setIsOpen] = useState(false);

  const log = useCallback((entry: Omit<LogEntry, "id" | "timestamp">) => {
    const newEntry: LogEntry = {
      ...entry,
      id: crypto.randomUUID(),
      timestamp: new Date(),
    };
    entriesRef.current = [...entriesRef.current, newEntry].slice(-MAX_ENTRIES);

    setRevision((r) => r + 1);
  }, []);

  const clear = useCallback(() => {
    entriesRef.current = [];
    setRevision((r) => r + 1);
  }, []);

  const startOp = useCallback(
    (op: Omit<InFlightOp, "startedAt" | "hasFirstEvent">) => {
      // Idempotent on id — a duplicate startOp for the same stream id is
      // a no-op so the fleet hook's idempotent-on-same-scope behavior
      // does not need to filter.
      if (inFlightRef.current.some((existing) => existing.id === op.id)) return;
      inFlightRef.current = [
        ...inFlightRef.current,
        { ...op, startedAt: new Date(), hasFirstEvent: false },
      ];
      setRevision((r) => r + 1);
    },
    [],
  );

  const markOpStarted = useCallback((opId: string) => {
    let mutated = false;
    inFlightRef.current = inFlightRef.current.map((op) => {
      if (op.id !== opId) return op;
      if (op.hasFirstEvent) return op;
      mutated = true;
      return { ...op, hasFirstEvent: true };
    });
    if (mutated) setRevision((r) => r + 1);
  }, []);

  const endOp = useCallback((opId: string) => {
    const before = inFlightRef.current.length;
    inFlightRef.current = inFlightRef.current.filter((op) => op.id !== opId);
    if (inFlightRef.current.length !== before) {
      setRevision((r) => r + 1);
    }
  }, []);

  const value: ActivityLogContextValue = {
    entries: entriesRef.current,
    log,
    clear,
    isOpen,
    setOpen: setIsOpen,
    hasActivity: entriesRef.current.length > 0,
    inFlight: inFlightRef.current,
    startOp,
    markOpStarted,
    endOp,
  };

  // `revision` participates in the value object lifecycle so React
  // re-renders consumers when entries / inFlight mutate via refs.
  void revision;

  return (
    <ActivityLogContext.Provider value={value}>
      {children}
    </ActivityLogContext.Provider>
  );
}

export function useActivityLog(): ActivityLogContextValue {
  const context = useContext(ActivityLogContext);
  if (!context) {
    return {
      entries: [],
      log: () => {},
      clear: () => {},
      isOpen: false,
      setOpen: () => {},
      hasActivity: false,
      inFlight: [],
      startOp: () => {},
      markOpStarted: () => {},
      endOp: () => {},
    };
  }
  return context;
}
