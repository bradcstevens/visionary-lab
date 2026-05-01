import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

// MUST be defined before the hook import is evaluated so the hook's module-
// level `import { streamGeneration, ... } from '@/services/stagingApi'` picks
// up the mocks. Vitest's `vi.mock` is hoisted; the factory must reference
// only top-level identifiers.
vi.mock('@/services/stagingApi', () => ({
  streamGeneration: vi.fn(),
  streamRoomRegeneration: vi.fn(),
  streamVariationRegeneration: vi.fn(),
  streamVariationEditPrompt: vi.fn(),
}));

import {
  useGenerationFleet,
  SSE_SILENCE_TIMEOUT_MS,
} from '../useGenerationFleet';
import type { StagingStreamEvent, StagingStreamEventCallback } from '@/services/stagingApi';
import {
  streamGeneration,
  streamRoomRegeneration,
  streamVariationRegeneration,
  streamVariationEditPrompt,
} from '@/services/stagingApi';

const mockedStreamGeneration = streamGeneration as ReturnType<typeof vi.fn>;
const mockedStreamRoomRegeneration = streamRoomRegeneration as ReturnType<typeof vi.fn>;
const mockedStreamVariationRegeneration = streamVariationRegeneration as ReturnType<typeof vi.fn>;
const mockedStreamVariationEditPrompt = streamVariationEditPrompt as ReturnType<typeof vi.fn>;

interface CapturedStream {
  callback: StagingStreamEventCallback;
  abort: ReturnType<typeof vi.fn>;
}

/**
 * Captures the (callback, abort) pair from a streamGeneration-style mock so
 * tests can synthesize SSE events into the hook on demand and assert on the
 * abort fn being called by finalize / watchdog / supersede paths.
 */
function captureStreamMock(mock: ReturnType<typeof vi.fn>): CapturedStream[] {
  const captures: CapturedStream[] = [];
  mock.mockImplementation((...args: unknown[]) => {
    const callback = args[args.length - 1] as StagingStreamEventCallback;
    const abort = vi.fn();
    captures.push({ callback, abort });
    return abort;
  });
  return captures;
}

describe('useGenerationFleet — module exports', () => {
  it('exports SSE_SILENCE_TIMEOUT_MS as 120 seconds', () => {
    // The PRD's "Further Notes" calls out 120s as the conservative threshold.
    // The constant being a single named export from the hook module is AC #6.
    expect(SSE_SILENCE_TIMEOUT_MS).toBe(120_000);
  });
});

describe('useGenerationFleet — initial state', () => {
  beforeEach(() => {
    mockedStreamGeneration.mockReset();
    mockedStreamRoomRegeneration.mockReset();
    mockedStreamVariationRegeneration.mockReset();
    mockedStreamVariationEditPrompt.mockReset();
  });

  it('starts with all in-flight sets empty and no lost ops', () => {
    const { result } = renderHook(() => useGenerationFleet({}));
    expect(result.current.inFlightProject).toBe(false);
    expect(result.current.inFlightRooms.size).toBe(0);
    expect(result.current.inFlightVariations.size).toBe(0);
    expect(result.current.isAnyInFlight).toBe(false);
    expect(result.current.lostOps).toEqual([]);
  });
});

describe('useGenerationFleet — startProject', () => {
  beforeEach(() => {
    mockedStreamGeneration.mockReset();
    mockedStreamRoomRegeneration.mockReset();
  });

  it('opens the project SSE stream synchronously and flips inFlightProject + isAnyInFlight on click', () => {
    // AC #2: the in-flight set is populated on click, not on first SSE event,
    // so the UI acknowledges intent immediately.
    const captures = captureStreamMock(mockedStreamGeneration);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startProject('p1', () => {});
    });

    expect(captures).toHaveLength(1);
    expect(mockedStreamGeneration).toHaveBeenCalledWith('p1', expect.any(Function));
    expect(result.current.inFlightProject).toBe(true);
    expect(result.current.isAnyInFlight).toBe(true);
  });

  it('forwards SSE events to the user-supplied event handler', () => {
    const captures = captureStreamMock(mockedStreamGeneration);
    const userHandler = vi.fn();
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startProject('p1', userHandler);
    });

    const event: StagingStreamEvent = { type: 'room_started', room_id: 'r1' };
    act(() => {
      captures[0].callback(event);
    });

    expect(userHandler).toHaveBeenCalledTimes(1);
    expect(userHandler).toHaveBeenCalledWith(event);
  });

  it('clears inFlightProject on terminal project_completed event', () => {
    const captures = captureStreamMock(mockedStreamGeneration);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startProject('p1', () => {});
    });
    expect(result.current.inFlightProject).toBe(true);

    act(() => {
      captures[0].callback({ type: 'project_completed' });
    });

    expect(result.current.inFlightProject).toBe(false);
    expect(result.current.isAnyInFlight).toBe(false);
  });

  it('clears inFlightProject on terminal stream_ended event', () => {
    const captures = captureStreamMock(mockedStreamGeneration);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startProject('p1', () => {});
    });

    act(() => {
      captures[0].callback({ type: 'stream_ended' });
    });

    expect(result.current.inFlightProject).toBe(false);
  });

  it('clears inFlightProject on terminal error event', () => {
    const captures = captureStreamMock(mockedStreamGeneration);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startProject('p1', () => {});
    });

    act(() => {
      captures[0].callback({ type: 'error', error: 'boom' });
    });

    expect(result.current.inFlightProject).toBe(false);
  });

  it('is idempotent: calling startProject twice while already in flight does not open a second stream', () => {
    const captures = captureStreamMock(mockedStreamGeneration);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startProject('p1', () => {});
    });
    act(() => {
      result.current.startProject('p1', () => {});
    });

    expect(captures).toHaveLength(1);
    expect(mockedStreamGeneration).toHaveBeenCalledTimes(1);
  });
});

describe('useGenerationFleet — startRoom (concurrent rooms)', () => {
  beforeEach(() => {
    mockedStreamRoomRegeneration.mockReset();
  });

  it('flips inFlightRooms.has(roomId) for the targeted room on click', () => {
    const captures = captureStreamMock(mockedStreamRoomRegeneration);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });

    expect(captures).toHaveLength(1);
    expect(result.current.inFlightRooms.has('roomA')).toBe(true);
    expect(result.current.inFlightRooms.has('roomB')).toBe(false);
    expect(result.current.isAnyInFlight).toBe(true);
  });

  it('AC #4: opens TWO concurrent streams when two different rooms are started', () => {
    const captures = captureStreamMock(mockedStreamRoomRegeneration);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
      result.current.startRoom('p1', 'roomB', () => {});
    });

    expect(captures).toHaveLength(2);
    expect(result.current.inFlightRooms.has('roomA')).toBe(true);
    expect(result.current.inFlightRooms.has('roomB')).toBe(true);
    expect(result.current.inFlightRooms.size).toBe(2);
  });

  it('opens THREE concurrent streams when three rooms are started in rapid succession', () => {
    const captures = captureStreamMock(mockedStreamRoomRegeneration);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startRoom('p1', 'r1', () => {});
      result.current.startRoom('p1', 'r2', () => {});
      result.current.startRoom('p1', 'r3', () => {});
    });

    expect(captures).toHaveLength(3);
    expect(result.current.inFlightRooms.size).toBe(3);
  });

  it('removes only the terminating roomId from inFlightRooms (other rooms remain)', () => {
    const captures = captureStreamMock(mockedStreamRoomRegeneration);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
      result.current.startRoom('p1', 'roomB', () => {});
    });

    act(() => {
      // roomA's stream completes
      captures[0].callback({ type: 'project_completed' });
    });

    expect(result.current.inFlightRooms.has('roomA')).toBe(false);
    expect(result.current.inFlightRooms.has('roomB')).toBe(true);
    expect(result.current.isAnyInFlight).toBe(true);
  });

  it('is idempotent on the same roomId: second startRoom for the same room is a no-op', () => {
    const captures = captureStreamMock(mockedStreamRoomRegeneration);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });
    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });

    expect(captures).toHaveLength(1);
  });
});

describe('useGenerationFleet — startVariation', () => {
  beforeEach(() => {
    mockedStreamVariationRegeneration.mockReset();
  });

  it('flips inFlightVariations.has(variationId) and forwards to streamVariationRegeneration with the strategy', () => {
    const captures = captureStreamMock(mockedStreamVariationRegeneration);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startVariation('p1', 'roomA', 'v0', 'fresh', () => {});
    });

    expect(captures).toHaveLength(1);
    expect(mockedStreamVariationRegeneration).toHaveBeenCalledWith(
      'p1',
      'roomA',
      'v0',
      'fresh',
      expect.any(Function),
    );
    expect(result.current.inFlightVariations.has('v0')).toBe(true);
    expect(result.current.isAnyInFlight).toBe(true);
  });

  it('removes the variationId on terminal stream_ended event', () => {
    const captures = captureStreamMock(mockedStreamVariationRegeneration);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startVariation('p1', 'roomA', 'v0', 'retry', () => {});
    });

    act(() => {
      captures[0].callback({ type: 'stream_ended' });
    });

    expect(result.current.inFlightVariations.has('v0')).toBe(false);
  });

  it('is idempotent on the same variationId: second call is a no-op', () => {
    const captures = captureStreamMock(mockedStreamVariationRegeneration);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startVariation('p1', 'roomA', 'v0', 'fresh', () => {});
    });
    act(() => {
      result.current.startVariation('p1', 'roomA', 'v0', 'retry', () => {});
    });

    expect(captures).toHaveLength(1);
  });
});

describe('useGenerationFleet — editPrompt', () => {
  beforeEach(() => {
    mockedStreamVariationEditPrompt.mockReset();
  });

  it('marks the source variation as in flight and forwards to streamVariationEditPrompt with the prompt body', () => {
    const captures = captureStreamMock(mockedStreamVariationEditPrompt);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.editPrompt('p1', 'roomA', 'v0', 'a fresh prompt', () => {});
    });

    expect(captures).toHaveLength(1);
    expect(mockedStreamVariationEditPrompt).toHaveBeenCalledWith(
      'p1',
      'roomA',
      'v0',
      'a fresh prompt',
      expect.any(Function),
    );
    expect(result.current.inFlightVariations.has('v0')).toBe(true);
  });
});

describe('useGenerationFleet — supersede semantics on startRoom', () => {
  beforeEach(() => {
    mockedStreamRoomRegeneration.mockReset();
    mockedStreamVariationRegeneration.mockReset();
  });

  it('aborts in-flight variations within the same room when startRoom fires (preserves retry-queue scenario 3 supersede behavior)', () => {
    const variationCaptures = captureStreamMock(mockedStreamVariationRegeneration);
    const roomCaptures = captureStreamMock(mockedStreamRoomRegeneration);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startVariation('p1', 'roomA', 'v0', 'fresh', () => {});
    });
    expect(result.current.inFlightVariations.has('v0')).toBe(true);

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });

    // The variation stream's abort fn is called (supersede).
    expect(variationCaptures[0].abort).toHaveBeenCalledTimes(1);
    // The variation is no longer in flight.
    expect(result.current.inFlightVariations.has('v0')).toBe(false);
    // The room IS now in flight.
    expect(result.current.inFlightRooms.has('roomA')).toBe(true);
    expect(roomCaptures).toHaveLength(1);
  });

  it('does NOT abort variations in OTHER rooms when startRoom fires (concurrent rooms preserved)', () => {
    const variationCaptures = captureStreamMock(mockedStreamVariationRegeneration);
    const roomCaptures = captureStreamMock(mockedStreamRoomRegeneration);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startVariation('p1', 'roomB', 'v0', 'fresh', () => {});
    });

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });

    // The variation in roomB is NOT aborted.
    expect(variationCaptures[0].abort).not.toHaveBeenCalled();
    expect(result.current.inFlightVariations.has('v0')).toBe(true);
    expect(roomCaptures).toHaveLength(1);
  });
});

describe('useGenerationFleet — supersede semantics on startProject', () => {
  beforeEach(() => {
    mockedStreamGeneration.mockReset();
    mockedStreamRoomRegeneration.mockReset();
    mockedStreamVariationRegeneration.mockReset();
  });

  it('aborts ALL in-flight rooms and variations when startProject fires (project is exclusive)', () => {
    const projectCaptures = captureStreamMock(mockedStreamGeneration);
    const roomCaptures = captureStreamMock(mockedStreamRoomRegeneration);
    const variationCaptures = captureStreamMock(mockedStreamVariationRegeneration);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
      result.current.startVariation('p1', 'roomB', 'vX', 'fresh', () => {});
    });

    act(() => {
      result.current.startProject('p1', () => {});
    });

    expect(roomCaptures[0].abort).toHaveBeenCalledTimes(1);
    expect(variationCaptures[0].abort).toHaveBeenCalledTimes(1);
    expect(result.current.inFlightRooms.size).toBe(0);
    expect(result.current.inFlightVariations.size).toBe(0);
    expect(result.current.inFlightProject).toBe(true);
    expect(projectCaptures).toHaveLength(1);
  });
});

describe('useGenerationFleet — watchdog', () => {
  beforeEach(() => {
    mockedStreamRoomRegeneration.mockReset();
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('does NOT fire the watchdog before the first SSE event arrives (queued behind STAGING_CONCURRENT_ROOMS semaphore)', () => {
    // Rationale: the backend's STAGING_CONCURRENT_ROOMS semaphore can hold a
    // request open with no events for tens of seconds. Pre-first-event
    // silence is queue latency, not stall. Without this guard, a burst of
    // concurrent room generates would false-fire watchdogs on rooms still
    // waiting for a slot.
    const captures = captureStreamMock(mockedStreamRoomRegeneration);
    const onStreamLost = vi.fn();
    const { result } = renderHook(() => useGenerationFleet({ onStreamLost }));

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });
    expect(result.current.inFlightRooms.has('roomA')).toBe(true);

    // Simulate 10 minutes passing with NO events at all.
    act(() => {
      vi.advanceTimersByTime(10 * 60 * 1000);
    });

    // Still in flight, watchdog has not fired, no lost op recorded.
    expect(result.current.inFlightRooms.has('roomA')).toBe(true);
    expect(result.current.lostOps).toEqual([]);
    expect(onStreamLost).not.toHaveBeenCalled();
    expect(captures[0].abort).not.toHaveBeenCalled();
  });

  it('fires after SSE_SILENCE_TIMEOUT_MS of silence following the first event: clears flag, aborts, and records a lost op', () => {
    const captures = captureStreamMock(mockedStreamRoomRegeneration);
    const onStreamLost = vi.fn();
    const { result } = renderHook(() => useGenerationFleet({ onStreamLost }));

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });

    // First event arrives — watchdog timer attaches.
    act(() => {
      captures[0].callback({ type: 'room_started', room_id: 'roomA' });
    });

    // Just under the threshold: still alive.
    act(() => {
      vi.advanceTimersByTime(SSE_SILENCE_TIMEOUT_MS - 1);
    });
    expect(result.current.inFlightRooms.has('roomA')).toBe(true);
    expect(onStreamLost).not.toHaveBeenCalled();

    // Cross the threshold.
    act(() => {
      vi.advanceTimersByTime(2);
    });

    expect(result.current.inFlightRooms.has('roomA')).toBe(false);
    expect(captures[0].abort).toHaveBeenCalledTimes(1);
    expect(result.current.lostOps).toHaveLength(1);
    expect(result.current.lostOps[0]).toMatchObject({ kind: 'room', roomId: 'roomA' });
    expect(onStreamLost).toHaveBeenCalledTimes(1);
    expect(onStreamLost).toHaveBeenCalledWith(expect.objectContaining({ kind: 'room', roomId: 'roomA' }));
  });

  it('resets the watchdog timer on EVERY subsequent SSE event of any type (healthy long-running streams stay alive)', () => {
    const captures = captureStreamMock(mockedStreamRoomRegeneration);
    const onStreamLost = vi.fn();
    const { result } = renderHook(() => useGenerationFleet({ onStreamLost }));

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });

    // First event starts the timer.
    act(() => {
      captures[0].callback({ type: 'room_started', room_id: 'roomA' });
    });

    // Tick almost to the threshold.
    act(() => {
      vi.advanceTimersByTime(SSE_SILENCE_TIMEOUT_MS - 1000);
    });

    // A non-terminal event of ANY type resets the timer.
    act(() => {
      captures[0].callback({ type: 'variation_started', variation_id: 'vX' });
    });

    // Tick almost to the threshold AGAIN — no fire because the previous tick
    // was zeroed by the variation_started event.
    act(() => {
      vi.advanceTimersByTime(SSE_SILENCE_TIMEOUT_MS - 1000);
    });

    expect(result.current.inFlightRooms.has('roomA')).toBe(true);
    expect(onStreamLost).not.toHaveBeenCalled();
  });

  it('terminal event clears the watchdog (no spurious fire after natural completion)', () => {
    const captures = captureStreamMock(mockedStreamRoomRegeneration);
    const onStreamLost = vi.fn();
    const { result } = renderHook(() => useGenerationFleet({ onStreamLost }));

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });

    act(() => {
      captures[0].callback({ type: 'room_started' });
    });
    act(() => {
      captures[0].callback({ type: 'project_completed' });
    });

    expect(result.current.inFlightRooms.has('roomA')).toBe(false);

    // Long after the terminal event, no late watchdog fire.
    act(() => {
      vi.advanceTimersByTime(10 * 60 * 1000);
    });
    expect(onStreamLost).not.toHaveBeenCalled();
    expect(result.current.lostOps).toEqual([]);
  });

  it('surfaces a synthetic error event to the user handler when the watchdog fires (so callers awaiting a Promise can settle)', () => {
    // Pinned regression: pre-fix, EditPromptDialog's submit Promise
    // resolves only on `project_completed` / `stream_ended` and rejects
    // only on `error`. A watchdog fire produced none of those, so the
    // dialog stuck in `isSubmitting` indefinitely. The fix surfaces a
    // synthetic 'error' event with a stream-lost message so the page's
    // existing 'error' handler runs and the Promise rejects.
    const captures = captureStreamMock(mockedStreamVariationEditPrompt);
    const userHandler = vi.fn();
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.editPrompt('p1', 'roomA', 'v0', 'a prompt', userHandler);
    });

    // First event starts the watchdog.
    act(() => {
      captures[0].callback({ type: 'variation_started' });
    });

    // Watchdog fires — the user handler must receive a synthetic
    // 'error' event so its existing error case can settle the dialog
    // Promise.
    act(() => {
      vi.advanceTimersByTime(SSE_SILENCE_TIMEOUT_MS + 100);
    });

    // The user handler was called for variation_started AND the
    // synthetic error event.
    expect(userHandler).toHaveBeenCalledTimes(2);
    const lastCall = userHandler.mock.calls[1]?.[0];
    expect(lastCall).toMatchObject({
      type: 'error',
      error: expect.stringContaining('Stream lost'),
    });
  });

  it('records the strategy and prompt fields on lost ops so retry can replay the exact operation', () => {
    // Watchdog fire on a variation regen → lost op carries strategy.
    const variationCaptures = captureStreamMock(mockedStreamVariationRegeneration);
    const editPromptCaptures = captureStreamMock(mockedStreamVariationEditPrompt);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startVariation('p1', 'roomA', 'v0', 'retry', () => {});
      result.current.editPrompt('p1', 'roomA', 'v1', 'a fresh edit', () => {});
    });

    act(() => {
      variationCaptures[0].callback({ type: 'variation_started' });
      editPromptCaptures[0].callback({ type: 'variation_started' });
    });

    act(() => {
      vi.advanceTimersByTime(SSE_SILENCE_TIMEOUT_MS + 100);
    });

    expect(result.current.lostOps).toHaveLength(2);
    const variationLost = result.current.lostOps.find((op) => op.kind === 'variation');
    const editPromptLost = result.current.lostOps.find((op) => op.kind === 'edit-prompt');
    expect(variationLost).toMatchObject({
      kind: 'variation',
      roomId: 'roomA',
      variationId: 'v0',
      strategy: 'retry',
    });
    expect(editPromptLost).toMatchObject({
      kind: 'edit-prompt',
      roomId: 'roomA',
      variationId: 'v1',
      prompt: 'a fresh edit',
    });
  });
});

describe('useGenerationFleet — retryLostOp', () => {
  beforeEach(() => {
    mockedStreamRoomRegeneration.mockReset();
    mockedStreamVariationRegeneration.mockReset();
    mockedStreamVariationEditPrompt.mockReset();
    mockedStreamGeneration.mockReset();
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('replays a lost room op as a fresh startRoom call', () => {
    const captures = captureStreamMock(mockedStreamRoomRegeneration);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });
    act(() => {
      captures[0].callback({ type: 'room_started' });
    });
    act(() => {
      vi.advanceTimersByTime(SSE_SILENCE_TIMEOUT_MS + 100);
    });

    const lostId = result.current.lostOps[0].id;
    act(() => {
      result.current.retryLostOp(lostId, () => {});
    });

    // A second room stream was opened.
    expect(captures).toHaveLength(2);
    expect(result.current.inFlightRooms.has('roomA')).toBe(true);
    // The lost op was removed.
    expect(result.current.lostOps).toEqual([]);
  });

  it('replays a lost variation op with the original strategy', () => {
    const captures = captureStreamMock(mockedStreamVariationRegeneration);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startVariation('p1', 'roomA', 'v0', 'retry', () => {});
    });
    act(() => {
      captures[0].callback({ type: 'variation_started' });
    });
    act(() => {
      vi.advanceTimersByTime(SSE_SILENCE_TIMEOUT_MS + 100);
    });

    const lostId = result.current.lostOps[0].id;
    act(() => {
      result.current.retryLostOp(lostId, () => {});
    });

    expect(captures).toHaveLength(2);
    expect(mockedStreamVariationRegeneration).toHaveBeenLastCalledWith(
      'p1',
      'roomA',
      'v0',
      'retry',
      expect.any(Function),
    );
    expect(result.current.inFlightVariations.has('v0')).toBe(true);
  });

  it('replays a lost edit-prompt op with the original prompt body', () => {
    const captures = captureStreamMock(mockedStreamVariationEditPrompt);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.editPrompt('p1', 'roomA', 'v0', 'original prompt text', () => {});
    });
    act(() => {
      captures[0].callback({ type: 'variation_started' });
    });
    act(() => {
      vi.advanceTimersByTime(SSE_SILENCE_TIMEOUT_MS + 100);
    });

    const lostId = result.current.lostOps[0].id;
    act(() => {
      result.current.retryLostOp(lostId, () => {});
    });

    expect(captures).toHaveLength(2);
    expect(mockedStreamVariationEditPrompt).toHaveBeenLastCalledWith(
      'p1',
      'roomA',
      'v0',
      'original prompt text',
      expect.any(Function),
    );
  });
});

describe('useGenerationFleet — dismissLostOp', () => {
  beforeEach(() => {
    mockedStreamRoomRegeneration.mockReset();
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('removes the lost op without replaying it', () => {
    const captures = captureStreamMock(mockedStreamRoomRegeneration);
    const { result } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });
    act(() => {
      captures[0].callback({ type: 'room_started' });
    });
    act(() => {
      vi.advanceTimersByTime(SSE_SILENCE_TIMEOUT_MS + 100);
    });

    const lostId = result.current.lostOps[0].id;
    act(() => {
      result.current.dismissLostOp(lostId);
    });

    expect(result.current.lostOps).toEqual([]);
    // No second stream was opened.
    expect(captures).toHaveLength(1);
  });
});

describe('useGenerationFleet — finalize idempotency', () => {
  beforeEach(() => {
    mockedStreamRoomRegeneration.mockReset();
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('does not double-record a lost op when terminal event arrives just before the watchdog could fire', () => {
    // Pre-fix: a race where finalize('terminal') and finalize('watchdog') both
    // fire could record duplicate lost ops or call the user handler twice.
    // The internal `finalized` guard makes the cleanup one-shot.
    const captures = captureStreamMock(mockedStreamRoomRegeneration);
    const onStreamLost = vi.fn();
    const userHandler = vi.fn();
    const { result } = renderHook(() => useGenerationFleet({ onStreamLost }));

    act(() => {
      result.current.startRoom('p1', 'roomA', userHandler);
    });

    act(() => {
      captures[0].callback({ type: 'room_started' });
    });

    // Terminal event arrives.
    act(() => {
      captures[0].callback({ type: 'project_completed' });
    });

    // Simulate the watchdog timer firing AFTER the terminal — must be no-op.
    act(() => {
      vi.advanceTimersByTime(SSE_SILENCE_TIMEOUT_MS + 100);
    });

    expect(result.current.lostOps).toEqual([]);
    expect(onStreamLost).not.toHaveBeenCalled();
  });

  it('aborts and removes all in-flight streams on unmount', () => {
    const captures = captureStreamMock(mockedStreamRoomRegeneration);
    const { result, unmount } = renderHook(() => useGenerationFleet({}));

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
      result.current.startRoom('p1', 'roomB', () => {});
    });
    expect(result.current.inFlightRooms.size).toBe(2);

    unmount();

    expect(captures[0].abort).toHaveBeenCalledTimes(1);
    expect(captures[1].abort).toHaveBeenCalledTimes(1);
  });
});

/**
 * Issue 008 of the projects-page-improvements PRD: the In Flight panel
 * inside the activity log derives its rows from the fleet hook via three
 * lifecycle callbacks — onOpStart (synchronous on click), onOpProgress
 * (first SSE event of any type), onOpEnd (every termination reason).
 * The activity-feed context maintains the In Flight list using these
 * callbacks; the page wires them to startOp/markOpStarted/endOp.
 *
 * The contract is one-shot per stream:
 *   - onOpStart fires once per new stream (NOT on idempotent re-call).
 *   - onOpProgress fires once on the first event (NOT on subsequent).
 *   - onOpEnd fires exactly once per stream lifetime, for every reason
 *     (terminal, watchdog, abort, supersede, unmount).
 *
 * Late stale events delivered after finalize must be no-ops on all
 * three callbacks (the underlying stream record is gone from the map).
 */
describe('useGenerationFleet — onOpStart / onOpProgress / onOpEnd lifecycle (issue 008)', () => {
  beforeEach(() => {
    mockedStreamGeneration.mockReset();
    mockedStreamRoomRegeneration.mockReset();
    mockedStreamVariationRegeneration.mockReset();
    mockedStreamVariationEditPrompt.mockReset();
  });

  it('fires onOpStart once per new stream with kind + descriptor fields', () => {
    captureStreamMock(mockedStreamRoomRegeneration);
    const onOpStart = vi.fn();
    const { result } = renderHook(() =>
      useGenerationFleet({ onOpStart }),
    );

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });

    expect(onOpStart).toHaveBeenCalledTimes(1);
    expect(onOpStart).toHaveBeenCalledWith(
      expect.objectContaining({
        id: expect.any(String),
        kind: 'room',
        projectId: 'p1',
        roomId: 'roomA',
      }),
    );
  });

  it('does NOT fire onOpStart on the idempotent re-call path (same scope)', () => {
    captureStreamMock(mockedStreamRoomRegeneration);
    const onOpStart = vi.fn();
    const { result } = renderHook(() =>
      useGenerationFleet({ onOpStart }),
    );

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });
    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });

    expect(onOpStart).toHaveBeenCalledTimes(1);
  });

  it('fires onOpStart for kind=project with the projectId field', () => {
    captureStreamMock(mockedStreamGeneration);
    const onOpStart = vi.fn();
    const { result } = renderHook(() =>
      useGenerationFleet({ onOpStart }),
    );

    act(() => {
      result.current.startProject('p9', () => {});
    });

    expect(onOpStart).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'project', projectId: 'p9' }),
    );
    const arg = onOpStart.mock.calls[0][0];
    expect(arg.roomId).toBeUndefined();
    expect(arg.variationId).toBeUndefined();
  });

  it('fires onOpStart for kind=variation with roomId, variationId, and strategy fields', () => {
    captureStreamMock(mockedStreamVariationRegeneration);
    const onOpStart = vi.fn();
    const { result } = renderHook(() =>
      useGenerationFleet({ onOpStart }),
    );

    act(() => {
      result.current.startVariation('p1', 'roomA', 'v3', 'fresh', () => {});
    });

    expect(onOpStart).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'variation',
        projectId: 'p1',
        roomId: 'roomA',
        variationId: 'v3',
      }),
    );
  });

  it('fires onOpStart for kind=edit-prompt with the prompt body field', () => {
    captureStreamMock(mockedStreamVariationEditPrompt);
    const onOpStart = vi.fn();
    const { result } = renderHook(() =>
      useGenerationFleet({ onOpStart }),
    );

    act(() => {
      result.current.editPrompt('p1', 'roomA', 'v3', 'a brand new sofa', () => {});
    });

    expect(onOpStart).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'edit-prompt',
        projectId: 'p1',
        roomId: 'roomA',
        variationId: 'v3',
      }),
    );
  });

  it('fires onOpProgress exactly once on the FIRST SSE event of any type', () => {
    const captures = captureStreamMock(mockedStreamRoomRegeneration);
    const onOpProgress = vi.fn();
    const { result } = renderHook(() =>
      useGenerationFleet({ onOpProgress }),
    );

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });
    expect(onOpProgress).not.toHaveBeenCalled();

    act(() => {
      captures[0].callback({ type: 'room_started' });
    });
    expect(onOpProgress).toHaveBeenCalledTimes(1);

    // Subsequent events do not re-fire progress.
    act(() => {
      captures[0].callback({ type: 'variation_completed', room_id: 'roomA' });
    });
    expect(onOpProgress).toHaveBeenCalledTimes(1);
  });

  it('fires onOpProgress AND onOpEnd cleanly when the very first SSE event is terminal', () => {
    // E.g., the backend errors out before any progress event lands.
    const captures = captureStreamMock(mockedStreamRoomRegeneration);
    const onOpProgress = vi.fn();
    const onOpEnd = vi.fn();
    const { result } = renderHook(() =>
      useGenerationFleet({ onOpProgress, onOpEnd }),
    );

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });

    act(() => {
      captures[0].callback({ type: 'error', error: 'backend died' });
    });

    expect(onOpProgress).toHaveBeenCalledTimes(1);
    expect(onOpEnd).toHaveBeenCalledTimes(1);
  });

  it('fires onOpEnd on terminal SSE event', () => {
    const captures = captureStreamMock(mockedStreamRoomRegeneration);
    let opId: string | undefined;
    const onOpStart = vi.fn((op: { id: string }) => {
      opId = op.id;
    });
    const onOpEnd = vi.fn();
    const { result } = renderHook(() =>
      useGenerationFleet({ onOpStart, onOpEnd }),
    );

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });

    act(() => {
      captures[0].callback({ type: 'project_completed' });
    });

    expect(onOpEnd).toHaveBeenCalledTimes(1);
    expect(onOpEnd).toHaveBeenCalledWith(opId);
    expect(result.current.inFlightRooms.size).toBe(0);
  });

  it('fires onOpEnd for every superseded variation when startRoom aborts them', () => {
    captureStreamMock(mockedStreamVariationRegeneration);
    captureStreamMock(mockedStreamRoomRegeneration);
    const onOpStart = vi.fn();
    const onOpEnd = vi.fn();
    const { result } = renderHook(() =>
      useGenerationFleet({ onOpStart, onOpEnd }),
    );

    act(() => {
      result.current.startVariation('p1', 'roomA', 'v1', 'retry', () => {});
      result.current.startVariation('p1', 'roomA', 'v2', 'fresh', () => {});
    });
    expect(onOpStart).toHaveBeenCalledTimes(2);
    const v1Id = onOpStart.mock.calls[0][0].id;
    const v2Id = onOpStart.mock.calls[1][0].id;

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });

    // Both variations were superseded — onOpEnd fires for each.
    expect(onOpEnd).toHaveBeenCalledTimes(2);
    expect(onOpEnd).toHaveBeenCalledWith(v1Id);
    expect(onOpEnd).toHaveBeenCalledWith(v2Id);
  });

  it('fires onOpEnd on watchdog fire (after SSE_SILENCE_TIMEOUT_MS of silence following first event)', () => {
    vi.useFakeTimers();
    const captures = captureStreamMock(mockedStreamRoomRegeneration);
    const onOpEnd = vi.fn();
    const { result } = renderHook(() =>
      useGenerationFleet({ onOpEnd }),
    );

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });

    // First event STARTS the watchdog timer (per the start-on-first-event
    // semantic that prevents queue-latency false-fires).
    act(() => {
      captures[0].callback({ type: 'room_started' });
    });
    expect(onOpEnd).not.toHaveBeenCalled();

    act(() => {
      vi.advanceTimersByTime(SSE_SILENCE_TIMEOUT_MS + 100);
    });

    expect(onOpEnd).toHaveBeenCalledTimes(1);
    vi.useRealTimers();
  });

  it('fires onOpEnd for every active stream on unmount (cleanup routes through finalize)', () => {
    captureStreamMock(mockedStreamRoomRegeneration);
    const onOpEnd = vi.fn();
    const { result, unmount } = renderHook(() =>
      useGenerationFleet({ onOpEnd }),
    );

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
      result.current.startRoom('p1', 'roomB', () => {});
    });
    expect(onOpEnd).not.toHaveBeenCalled();

    unmount();

    expect(onOpEnd).toHaveBeenCalledTimes(2);
  });

  it('LATE STALE event after finalize does not re-fire onOpProgress or onOpEnd', () => {
    // Real-world race: the SSE library could deliver one more callback
    // invocation after we finalize on a terminal event. The wrapper
    // must look up the record + finalized flag and no-op.
    const captures = captureStreamMock(mockedStreamRoomRegeneration);
    const onOpProgress = vi.fn();
    const onOpEnd = vi.fn();
    const { result } = renderHook(() =>
      useGenerationFleet({ onOpProgress, onOpEnd }),
    );

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });

    act(() => {
      captures[0].callback({ type: 'room_started' });
    });
    expect(onOpProgress).toHaveBeenCalledTimes(1);

    // Terminal — finalize runs.
    act(() => {
      captures[0].callback({ type: 'project_completed' });
    });
    expect(onOpEnd).toHaveBeenCalledTimes(1);

    // Late stale event — must be a no-op for both callbacks.
    act(() => {
      captures[0].callback({ type: 'variation_completed', room_id: 'roomA' });
    });
    expect(onOpProgress).toHaveBeenCalledTimes(1);
    expect(onOpEnd).toHaveBeenCalledTimes(1);
  });

  it('preserves the order: superseded onOpEnd calls fire BEFORE the new onOpStart', () => {
    // The supersede cascade must happen before the new record is inserted
    // so the activity feed sees the Set's count drop, then bump up — not
    // the other way around. (Otherwise In Flight rows could briefly
    // overlap in odd visual ways.)
    captureStreamMock(mockedStreamVariationRegeneration);
    captureStreamMock(mockedStreamRoomRegeneration);
    const order: string[] = [];
    const onOpStart = vi.fn((op: { id: string; kind: string }) => {
      order.push(`start:${op.kind}:${op.id}`);
    });
    const onOpEnd = vi.fn((opId: string) => {
      order.push(`end:${opId}`);
    });
    const { result } = renderHook(() =>
      useGenerationFleet({ onOpStart, onOpEnd }),
    );

    act(() => {
      result.current.startVariation('p1', 'roomA', 'v1', 'retry', () => {});
    });
    const variationStartIdx = order.length;

    act(() => {
      result.current.startRoom('p1', 'roomA', () => {});
    });

    // Last 2 entries must be: end:<variation-id>, then start:room:<room-id>.
    expect(order[variationStartIdx]).toMatch(/^end:/);
    expect(order[variationStartIdx + 1]).toMatch(/^start:room:/);
  });
});
