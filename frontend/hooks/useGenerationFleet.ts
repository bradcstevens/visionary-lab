import { useCallback, useEffect, useRef, useState } from 'react';
import {
  streamGeneration,
  streamRoomRegeneration,
  streamVariationRegeneration,
  streamVariationEditPrompt,
  type StagingStreamEvent,
  type StagingStreamEventCallback,
} from '@/services/stagingApi';

/**
 * Issue 007 of the projects-page-improvements PRD.
 *
 * Per-operation generation state with a per-stream silence watchdog. Replaces
 * the page's prior global ``isGenerating`` flag, which serialized every
 * Generate / Regenerate button on the page and got stuck grayed out when an
 * SSE stream stalled.
 *
 * The hook owns the lifecycle of every staging SSE stream the page opens:
 *   - registers the operation in the appropriate in-flight set on click
 *     (BEFORE the first SSE event lands, per AC #2 — UI ack is immediate),
 *   - wraps the user-supplied event callback with watchdog reset + finalize,
 *   - aborts and removes the operation on terminal events / user abort /
 *     unmount / watchdog fire,
 *   - cascades aborts when a larger op supersedes a smaller one
 *     (``startRoom`` aborts variations in the same room; ``startProject``
 *     aborts everything),
 *   - records lost ops with full descriptors so retry replays the EXACT
 *     operation (a lost variation regen is not replayed as a room regen).
 *
 * The 120-second silence threshold is exposed as ``SSE_SILENCE_TIMEOUT_MS``
 * so it can be tuned in one place per the PRD's "Further Notes". The timer
 * starts ON THE FIRST SSE EVENT, not at stream open — this is the deliberate
 * deviation from the literal AC wording "attaches a 120-second silence
 * watchdog [at start]". The reason: the backend's ``STAGING_CONCURRENT_ROOMS``
 * semaphore can hold a request open with no events for tens of seconds while
 * other rooms are active. Pre-first-event silence is queue latency, not
 * stall. The watchdog measures silence BETWEEN events.
 *
 * See PRD §"Solution → 7. Per-room concurrent generation with a watchdog",
 * §"Implementation Decisions → Frontend modules" (useGenerationFleet bullet),
 * §"Cross-cutting decisions" (per-stream watchdog rationale), §"Further
 * Notes" (120s threshold tuning), and §"Testing Decisions" (concurrent-
 * generation Playwright scenario; watchdog is unit-tested with fake timers).
 */

/** Single named constant per AC #6. PRD §"Further Notes": "may need tuning". */
export const SSE_SILENCE_TIMEOUT_MS = 120_000;

export type LostOp =
  | { id: string; kind: 'project'; projectId: string; lostAt: Date }
  | { id: string; kind: 'room'; projectId: string; roomId: string; lostAt: Date }
  | {
      id: string;
      kind: 'variation';
      projectId: string;
      roomId: string;
      variationId: string;
      strategy: 'retry' | 'fresh';
      lostAt: Date;
    }
  | {
      id: string;
      kind: 'edit-prompt';
      projectId: string;
      roomId: string;
      variationId: string;
      prompt: string;
      lostAt: Date;
    };

export interface UseGenerationFleetParams {
  /**
   * Optional callback invoked when the watchdog fires for a stream. The page
   * uses this to log a "Stream lost" entry to the activity log. The lost op
   * descriptor is also queryable via ``lostOps`` (populated synchronously
   * with this callback).
   */
  onStreamLost?: (lostOp: LostOp) => void;

  /**
   * Issue 008 of the projects-page-improvements PRD: lifecycle callbacks
   * the In Flight panel uses to mirror the fleet's per-stream state in
   * the activity feed. The activity-feed context calls `startOp` /
   * `markOpStarted` / `endOp` from these callbacks, so the panel and the
   * per-op flags driving buttons cannot drift apart.
   *
   * Contract:
   *   - `onOpStart` fires SYNCHRONOUSLY on click for each new stream
   *     (NOT on the idempotent re-call path). Carries the descriptor
   *     fields the page needs to derive a human-readable label.
   *   - `onOpProgress` fires ONCE on the first SSE event of any type
   *     for the stream. Used by the panel to flip the row's status
   *     label from "Starting…" to "Running".
   *   - `onOpEnd` fires EXACTLY ONCE per stream lifetime, for every
   *     termination reason (terminal, watchdog, abort, supersede,
   *     unmount). Late stale events delivered after finalize are
   *     no-ops. Routed through `finalize()` so the one-shot guard
   *     prevents double-firing.
   */
  onOpStart?: (op: {
    id: string;
    kind: StreamDescriptor['kind'];
    projectId: string;
    roomId?: string;
    variationId?: string;
  }) => void;
  onOpProgress?: (opId: string) => void;
  onOpEnd?: (opId: string) => void;
}

export interface UseGenerationFleetResult {
  inFlightProject: boolean;
  inFlightRooms: ReadonlySet<string>;
  inFlightVariations: ReadonlySet<string>;
  isAnyInFlight: boolean;
  lostOps: ReadonlyArray<LostOp>;
  startProject: (projectId: string, eventHandler: StagingStreamEventCallback) => void;
  startRoom: (
    projectId: string,
    roomId: string,
    eventHandler: StagingStreamEventCallback,
  ) => void;
  startVariation: (
    projectId: string,
    roomId: string,
    variationId: string,
    strategy: 'retry' | 'fresh',
    eventHandler: StagingStreamEventCallback,
  ) => void;
  editPrompt: (
    projectId: string,
    roomId: string,
    variationId: string,
    prompt: string,
    eventHandler: StagingStreamEventCallback,
  ) => void;
  retryLostOp: (id: string, eventHandler: StagingStreamEventCallback) => void;
  dismissLostOp: (id: string) => void;
  abortAll: () => void;
}

interface ProjectRecord {
  kind: 'project';
  projectId: string;
}
interface RoomRecord {
  kind: 'room';
  projectId: string;
  roomId: string;
}
interface VariationRecord {
  kind: 'variation';
  projectId: string;
  roomId: string;
  variationId: string;
  strategy: 'retry' | 'fresh';
}
interface EditPromptRecord {
  kind: 'edit-prompt';
  projectId: string;
  roomId: string;
  variationId: string;
  prompt: string;
}

type StreamDescriptor = ProjectRecord | RoomRecord | VariationRecord | EditPromptRecord;

interface StreamRecord {
  id: string;
  descriptor: StreamDescriptor;
  abort: () => void;
  watchdogTimer: ReturnType<typeof setTimeout> | null;
  finalized: boolean;
  startedAt: Date;
  /**
   * Issue 008: flips on the first SSE event of any type. Used to fire
   * ``onOpProgress`` exactly once per stream, even if the very first
   * event is terminal. Late stale events after finalize see the record
   * removed from the map and are no-ops.
   */
  firstEventReceived: boolean;
  /**
   * Captured user event handler so finalize('watchdog') can surface a
   * synthetic terminal event to the caller. Without this, a watchdog-
   * aborted edit-prompt would leave the dialog's submit Promise pending
   * forever (it only resolves/rejects on terminal SSE events).
   */
  userHandler: StagingStreamEventCallback;
}

type FinalizeReason = 'terminal' | 'watchdog' | 'abort' | 'unmount' | 'supersede';

let nextStreamSeq = 0;
function newStreamId(): string {
  // Plain counter is fine — the id is process-local and never persisted.
  // crypto.randomUUID would be better but jsdom's crypto sometimes lacks it.
  nextStreamSeq += 1;
  return `s${nextStreamSeq}_${Date.now()}`;
}

function isTerminalEvent(event: StagingStreamEvent): boolean {
  return (
    event.type === 'project_completed' ||
    event.type === 'stream_ended' ||
    event.type === 'error'
  );
}

export function useGenerationFleet(
  params: UseGenerationFleetParams,
): UseGenerationFleetResult {
  const { onStreamLost, onOpStart, onOpProgress, onOpEnd } = params;

  // Refs for stable identity across renders. The state-shadow pattern (refs
  // for control flow + useState for re-renders) is the same shape useRetryQueue
  // uses; it avoids same-tick state-batching races where a second click would
  // see a stale "is in flight?" check before React re-renders.
  const streamsRef = useRef<Map<string, StreamRecord>>(new Map());
  const onStreamLostRef = useRef(onStreamLost);
  const onOpStartRef = useRef(onOpStart);
  const onOpProgressRef = useRef(onOpProgress);
  const onOpEndRef = useRef(onOpEnd);
  useEffect(() => {
    onStreamLostRef.current = onStreamLost;
  }, [onStreamLost]);
  useEffect(() => {
    onOpStartRef.current = onOpStart;
  }, [onOpStart]);
  useEffect(() => {
    onOpProgressRef.current = onOpProgress;
  }, [onOpProgress]);
  useEffect(() => {
    onOpEndRef.current = onOpEnd;
  }, [onOpEnd]);

  // State shadows for re-rendering consumers. Recomputed by syncShadows()
  // whenever streamsRef changes.
  const [inFlightProject, setInFlightProject] = useState(false);
  const [inFlightRooms, setInFlightRooms] = useState<ReadonlySet<string>>(
    new Set(),
  );
  const [inFlightVariations, setInFlightVariations] = useState<
    ReadonlySet<string>
  >(new Set());
  const [lostOps, setLostOps] = useState<ReadonlyArray<LostOp>>([]);
  // Ref shadow so retryLostOp reads the current lost-op list synchronously,
  // without relying on a setState updater closure setting an outer-scope
  // variable (React schedules updaters; the variable wouldn't be populated
  // before the dispatch line ran). Mirrored by syncLostOps below.
  const lostOpsRef = useRef<LostOp[]>([]);
  const syncLostOps = useCallback((next: LostOp[]) => {
    lostOpsRef.current = next;
    setLostOps(next);
  }, []);

  const syncShadows = useCallback(() => {
    let project = false;
    const rooms = new Set<string>();
    const variations = new Set<string>();
    for (const record of streamsRef.current.values()) {
      const d = record.descriptor;
      if (d.kind === 'project') {
        project = true;
      } else if (d.kind === 'room') {
        rooms.add(d.roomId);
      } else if (d.kind === 'variation' || d.kind === 'edit-prompt') {
        variations.add(d.variationId);
      }
    }
    setInFlightProject(project);
    setInFlightRooms(rooms);
    setInFlightVariations(variations);
  }, []);

  /**
   * One-shot cleanup. Every termination path (terminal SSE event, watchdog
   * fire, explicit abort, unmount, supersede) flows through here and is
   * guarded by the record's ``finalized`` flag so a late timer fire after
   * a terminal event does not double-record a lost op or double-call the
   * user handler. (Rubber-duck blocking finding #4.)
   *
   * Issue 008: also fires ``onOpEnd`` exactly once per stream lifetime via
   * the same one-shot guard. Late stale events delivered after finalize
   * see the record gone from the map and are no-ops.
   */
  const finalize = useCallback(
    (streamId: string, reason: FinalizeReason): void => {
      const record = streamsRef.current.get(streamId);
      if (!record || record.finalized) return;
      record.finalized = true;

      if (record.watchdogTimer !== null) {
        clearTimeout(record.watchdogTimer);
        record.watchdogTimer = null;
      }

      // Watchdog and supersede paths abort the underlying fetch. Terminal,
      // unmount also call abort defensively (AbortController.abort is
      // idempotent if the fetch already settled).
      if (reason !== 'terminal') {
        try {
          record.abort();
        } catch {
          // Ignore — the underlying fetch's abort is fire-and-forget.
        }
      }

      streamsRef.current.delete(streamId);

      if (reason === 'watchdog') {
        // Surface a synthetic 'error' SSE event to the captured user
        // handler so callers waiting on a Promise (e.g. EditPromptDialog's
        // submit) can settle their pending state. Without this, a
        // watchdog fire would silently abort the underlying fetch and
        // leave the dialog stuck in isSubmitting forever — the user
        // would have to dismiss the dialog by hand. Per finalize's
        // one-shot guard above, this runs at most once per stream.
        try {
          record.userHandler({
            type: 'error',
            error: 'Stream lost — no SSE events for 2 minutes',
          });
        } catch {
          // Caller's handler threw — ignore (we still want to record
          // the lost op + abort).
        }

        // Build the lost-op descriptor with full operation metadata so
        // ``retryLostOp`` can replay the EXACT operation (a lost variation
        // regen is replayed as a variation regen, not as a room regen).
        const lostOp: LostOp = (() => {
          const d = record.descriptor;
          const id = `lost_${streamId}`;
          const lostAt = new Date();
          if (d.kind === 'project') {
            return { id, kind: 'project', projectId: d.projectId, lostAt };
          }
          if (d.kind === 'room') {
            return { id, kind: 'room', projectId: d.projectId, roomId: d.roomId, lostAt };
          }
          if (d.kind === 'variation') {
            return {
              id,
              kind: 'variation',
              projectId: d.projectId,
              roomId: d.roomId,
              variationId: d.variationId,
              strategy: d.strategy,
              lostAt,
            };
          }
          return {
            id,
            kind: 'edit-prompt',
            projectId: d.projectId,
            roomId: d.roomId,
            variationId: d.variationId,
            prompt: d.prompt,
            lostAt,
          };
        })();
        setLostOps((prev) => [...prev, lostOp]);
        lostOpsRef.current = [...lostOpsRef.current, lostOp];
        onStreamLostRef.current?.(lostOp);
      }

      // Issue 008: fire onOpEnd for every termination reason. The activity-
      // feed context's endOp is idempotent on a missing id, so a double-
      // call here would be safe — but the record.finalized one-shot guard
      // above already ensures this branch runs at most once per stream.
      try {
        onOpEndRef.current?.(streamId);
      } catch {
        // Subscriber threw — ignore (we still want to sync shadow state).
      }

      syncShadows();
    },
    [syncShadows],
  );

  const wrapEventHandler = useCallback(
    (
      streamId: string,
      userHandler: StagingStreamEventCallback,
    ): StagingStreamEventCallback => {
      return (event) => {
        const record = streamsRef.current.get(streamId);
        if (record && !record.finalized) {
          // Issue 008: fire onOpProgress exactly once on the FIRST SSE
          // event of any type. Used by the activity feed's panel to flip
          // the row's status label from "Starting…" to "Running". The
          // very-first-event-is-terminal case still triggers progress
          // BEFORE finalize (so the panel briefly sees "Running" then
          // the row is removed by onOpEnd — symmetric with normal flow).
          if (!record.firstEventReceived) {
            record.firstEventReceived = true;
            try {
              onOpProgressRef.current?.(streamId);
            } catch {
              // Subscriber threw — ignore.
            }
          }

          // Reset / start the watchdog on EVERY SSE event of any type.
          // The first event implicitly STARTS the timer (timer was null);
          // subsequent events RESET it. This is the rubber-duck-flagged
          // "don't false-fire while queued behind STAGING_CONCURRENT_ROOMS"
          // semantic — pre-first-event silence is queue latency, not stall.
          if (record.watchdogTimer !== null) {
            clearTimeout(record.watchdogTimer);
          }
          if (!isTerminalEvent(event)) {
            record.watchdogTimer = setTimeout(() => {
              finalize(streamId, 'watchdog');
            }, SSE_SILENCE_TIMEOUT_MS);
          } else {
            record.watchdogTimer = null;
          }
        }

        // Forward to the page's event handler BEFORE finalizing on terminal
        // so the page sees the terminal event's payload (e.g., it can pull
        // adapted_prompt off variation_completed before we tear down).
        userHandler(event);

        if (isTerminalEvent(event)) {
          finalize(streamId, 'terminal');
        }
      };
    },
    [finalize],
  );

  /**
   * Idempotent on the same descriptor scope. Returns the streamId of the
   * existing in-flight stream when one already exists for this scope, or
   * the new stream when one is started.
   */
  const startStream = useCallback(
    (
      descriptor: StreamDescriptor,
      userHandler: StagingStreamEventCallback,
    ): string => {
      // Idempotent gates per scope.
      for (const existing of streamsRef.current.values()) {
        const d = existing.descriptor;
        if (descriptor.kind === 'project' && d.kind === 'project') {
          return existing.id;
        }
        if (
          descriptor.kind === 'room' &&
          d.kind === 'room' &&
          d.roomId === descriptor.roomId
        ) {
          return existing.id;
        }
        if (
          (descriptor.kind === 'variation' || descriptor.kind === 'edit-prompt') &&
          (d.kind === 'variation' || d.kind === 'edit-prompt') &&
          d.variationId === descriptor.variationId
        ) {
          return existing.id;
        }
      }

      // Supersede cascades — abort smaller in-flight ops that conflict.
      if (descriptor.kind === 'project') {
        // Project is exclusive — abort everything.
        for (const [id] of streamsRef.current.entries()) {
          finalize(id, 'supersede');
        }
      } else if (descriptor.kind === 'room') {
        // Aborts variations in the same room; preserves other rooms.
        for (const [id, record] of streamsRef.current.entries()) {
          const d = record.descriptor;
          if (
            (d.kind === 'variation' || d.kind === 'edit-prompt') &&
            d.roomId === descriptor.roomId
          ) {
            finalize(id, 'supersede');
          }
        }
      }

      const streamId = newStreamId();
      const record: StreamRecord = {
        id: streamId,
        descriptor,
        abort: () => {},
        watchdogTimer: null,
        finalized: false,
        startedAt: new Date(),
        firstEventReceived: false,
        userHandler,
      };
      streamsRef.current.set(streamId, record);

      // Issue 008: fire onOpStart synchronously so the activity-feed's
      // In Flight section gains the row BEFORE the SSE stream opens.
      // AC #4: "Operations appear in In Flight on click (before the
      // first SSE event), so the panel acknowledges intent immediately."
      // The supersede cascades above already finalized superseded streams
      // (firing their onOpEnd), so the order user sees in the activity
      // feed is: end:<superseded-ids> THEN start:<new-id>.
      try {
        const d = descriptor;
        if (d.kind === 'project') {
          onOpStartRef.current?.({ id: streamId, kind: 'project', projectId: d.projectId });
        } else if (d.kind === 'room') {
          onOpStartRef.current?.({
            id: streamId,
            kind: 'room',
            projectId: d.projectId,
            roomId: d.roomId,
          });
        } else {
          onOpStartRef.current?.({
            id: streamId,
            kind: d.kind,
            projectId: d.projectId,
            roomId: d.roomId,
            variationId: d.variationId,
          });
        }
      } catch {
        // Subscriber threw — ignore.
      }

      const wrappedHandler = wrapEventHandler(streamId, userHandler);

      let abort: () => void;
      if (descriptor.kind === 'project') {
        abort = streamGeneration(descriptor.projectId, wrappedHandler);
      } else if (descriptor.kind === 'room') {
        abort = streamRoomRegeneration(
          descriptor.projectId,
          descriptor.roomId,
          wrappedHandler,
        );
      } else if (descriptor.kind === 'variation') {
        abort = streamVariationRegeneration(
          descriptor.projectId,
          descriptor.roomId,
          descriptor.variationId,
          descriptor.strategy,
          wrappedHandler,
        );
      } else {
        abort = streamVariationEditPrompt(
          descriptor.projectId,
          descriptor.roomId,
          descriptor.variationId,
          descriptor.prompt,
          wrappedHandler,
        );
      }
      record.abort = abort;

      // Sync shadow state synchronously so the calling render sees the
      // updated in-flight set immediately. AC #2 — UI ack on click.
      syncShadows();
      return streamId;
    },
    [finalize, syncShadows, wrapEventHandler],
  );

  const startProject = useCallback(
    (projectId: string, eventHandler: StagingStreamEventCallback) => {
      startStream({ kind: 'project', projectId }, eventHandler);
    },
    [startStream],
  );

  const startRoom = useCallback(
    (
      projectId: string,
      roomId: string,
      eventHandler: StagingStreamEventCallback,
    ) => {
      startStream({ kind: 'room', projectId, roomId }, eventHandler);
    },
    [startStream],
  );

  const startVariation = useCallback(
    (
      projectId: string,
      roomId: string,
      variationId: string,
      strategy: 'retry' | 'fresh',
      eventHandler: StagingStreamEventCallback,
    ) => {
      startStream(
        { kind: 'variation', projectId, roomId, variationId, strategy },
        eventHandler,
      );
    },
    [startStream],
  );

  const editPrompt = useCallback(
    (
      projectId: string,
      roomId: string,
      variationId: string,
      prompt: string,
      eventHandler: StagingStreamEventCallback,
    ) => {
      startStream(
        { kind: 'edit-prompt', projectId, roomId, variationId, prompt },
        eventHandler,
      );
    },
    [startStream],
  );

  const dismissLostOp = useCallback((id: string) => {
    syncLostOps(lostOpsRef.current.filter((op) => op.id !== id));
  }, [syncLostOps]);

  const retryLostOp = useCallback(
    (id: string, eventHandler: StagingStreamEventCallback) => {
      const lostOp = lostOpsRef.current.find((op) => op.id === id);
      if (!lostOp) return;
      // Remove first (idempotent against double-click).
      syncLostOps(lostOpsRef.current.filter((op) => op.id !== id));
      if (lostOp.kind === 'project') {
        startProject(lostOp.projectId, eventHandler);
      } else if (lostOp.kind === 'room') {
        startRoom(lostOp.projectId, lostOp.roomId, eventHandler);
      } else if (lostOp.kind === 'variation') {
        startVariation(
          lostOp.projectId,
          lostOp.roomId,
          lostOp.variationId,
          lostOp.strategy,
          eventHandler,
        );
      } else {
        editPrompt(
          lostOp.projectId,
          lostOp.roomId,
          lostOp.variationId,
          lostOp.prompt,
          eventHandler,
        );
      }
    },
    [syncLostOps, startProject, startRoom, startVariation, editPrompt],
  );

  const abortAll = useCallback(() => {
    for (const [id] of streamsRef.current.entries()) {
      finalize(id, 'abort');
    }
  }, [finalize]);

  // Cleanup all streams on unmount. Per useEffect's setup/teardown contract,
  // the function returned here runs exactly once at unmount.
  //
  // Issue 008: route per-stream cleanup through ``finalize(id, 'unmount')``
  // so onOpEnd fires for each active stream. Pre-issue-008 this loop
  // manually flipped ``record.finalized`` and called abort, bypassing
  // finalize and leaking In Flight rows in the (provider-still-mounted)
  // activity feed when the page navigated away. Capture finalize via a
  // ref so the unmount effect's empty deps array stays valid AND the
  // cleanup uses the latest finalize (which depends on syncShadows, which
  // changes identity once at mount).
  const finalizeRef = useRef(finalize);
  useEffect(() => {
    finalizeRef.current = finalize;
  }, [finalize]);
  useEffect(() => {
    const streams = streamsRef.current;
    return () => {
      // Snapshot ids first — finalize mutates the Map by deleting entries,
      // so iterating the live Map.values() while finalize runs would skip
      // entries.
      const ids = Array.from(streams.keys());
      for (const id of ids) {
        finalizeRef.current(id, 'unmount');
      }
    };
  }, []);

  const isAnyInFlight =
    inFlightProject || inFlightRooms.size > 0 || inFlightVariations.size > 0;

  return {
    inFlightProject,
    inFlightRooms,
    inFlightVariations,
    isAnyInFlight,
    lostOps,
    startProject,
    startRoom,
    startVariation,
    editPrompt,
    retryLostOp,
    dismissLostOp,
    abortAll,
  };
}
