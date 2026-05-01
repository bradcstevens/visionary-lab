import { describe, it, expect, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useRetryQueue } from '../useRetryQueue';
import type { Room } from '@/services/stagingApi';

const ISO = '2026-01-01T00:00:00Z';

function makeFailedRoom(roomId = 'r1', failedId = 'v0'): Room {
  return {
    id: roomId,
    label: 'Test Room',
    original_image_url: 'https://x.test/orig.png',
    status: 'processing',
    variations: [
      {
        id: failedId,
        status: 'failed',
        error: 'simulated',
        created_at: ISO,
        updated_at: ISO,
      },
    ],
  };
}

function makeProject(rooms: Room[]) {
  return {
    id: 'proj-1',
    name: 'P',
    prompt: '',
    status: 'processing' as const,
    settings: {} as never,
    rooms,
    total_variations: rooms.reduce((n, r) => n + r.variations.length, 0),
    completed_variations: 0,
    created_at: ISO,
    updated_at: ISO,
  };
}

describe('useRetryQueue — enqueue when idle', () => {
  it('returns "dispatched" and synchronously calls onDispatch with the right (room, index, "fresh")', () => {
    const room = makeFailedRoom();
    const project = makeProject([room]);
    const onDispatch = vi.fn();

    const { result } = renderHook(() =>
      useRetryQueue({
        project,
        isGenerating: false,
        regeneratingVariationId: null,
        onDispatch,
      }),
    );

    let outcome: ReturnType<typeof result.current.enqueue> | undefined;
    act(() => {
      outcome = result.current.enqueue('v0');
    });

    expect(outcome).toBe('dispatched');
    expect(onDispatch).toHaveBeenCalledTimes(1);
    expect(onDispatch).toHaveBeenCalledWith(room, 0, 'fresh');
    expect(result.current.queuedIds.size).toBe(0);
  });
});

describe('useRetryQueue — enqueue while busy', () => {
  it('returns "queued", does NOT call onDispatch, and exposes the id in queuedIds', () => {
    const room = makeFailedRoom();
    const project = makeProject([room]);
    const onDispatch = vi.fn();

    const { result } = renderHook(() =>
      useRetryQueue({
        project,
        isGenerating: true,
        regeneratingVariationId: null,
        onDispatch,
      }),
    );

    let outcome: ReturnType<typeof result.current.enqueue> | undefined;
    act(() => {
      outcome = result.current.enqueue('v0');
    });

    expect(outcome).toBe('queued');
    expect(onDispatch).not.toHaveBeenCalled();
    expect(result.current.queuedIds.has('v0')).toBe(true);
    expect(result.current.queuedIds.size).toBe(1);
  });

  it('also queues when regeneratingVariationId is set (single-variation regen in flight)', () => {
    const room = makeFailedRoom();
    const project = makeProject([room]);
    const onDispatch = vi.fn();

    const { result } = renderHook(() =>
      useRetryQueue({
        project,
        isGenerating: false,
        regeneratingVariationId: 'someone-else',
        onDispatch,
      }),
    );

    let outcome: ReturnType<typeof result.current.enqueue> | undefined;
    act(() => {
      outcome = result.current.enqueue('v0');
    });

    expect(outcome).toBe('queued');
    expect(onDispatch).not.toHaveBeenCalled();
  });
});

describe('useRetryQueue — drain on idle', () => {
  it('dispatches the next queued entry when (isGenerating, regeneratingVariationId) both go idle', () => {
    const room = makeFailedRoom();
    const project = makeProject([room]);
    const onDispatch = vi.fn();

    type Props = {
      project: ReturnType<typeof makeProject>;
      isGenerating: boolean;
      regeneratingVariationId: string | null;
    };
    const initialProps: Props = {
      project,
      isGenerating: true,
      regeneratingVariationId: null,
    };

    const { result, rerender } = renderHook(
      (props: Props) =>
        useRetryQueue({
          ...props,
          onDispatch,
        }),
      { initialProps },
    );

    act(() => {
      result.current.enqueue('v0');
    });
    expect(result.current.queuedIds.has('v0')).toBe(true);
    expect(onDispatch).not.toHaveBeenCalled();

    act(() => {
      rerender({
        project,
        isGenerating: false,
        regeneratingVariationId: null,
      });
    });

    expect(onDispatch).toHaveBeenCalledTimes(1);
    expect(onDispatch).toHaveBeenCalledWith(room, 0, 'fresh');
    expect(result.current.queuedIds.size).toBe(0);
  });

  it('drains serially: queue [A, B] dispatches A, waits for completion, then dispatches B', () => {
    const room: Room = {
      id: 'r1',
      label: 'Room',
      original_image_url: 'https://x.test/o.png',
      status: 'processing',
      variations: [
        { id: 'A', status: 'failed', error: 'e', created_at: ISO, updated_at: ISO },
        { id: 'B', status: 'failed', error: 'e', created_at: ISO, updated_at: ISO },
      ],
    };
    const project = makeProject([room]);
    const onDispatch = vi.fn();

    type Props = {
      project: ReturnType<typeof makeProject>;
      isGenerating: boolean;
      regeneratingVariationId: string | null;
    };
    const initialProps: Props = {
      project,
      isGenerating: true,
      regeneratingVariationId: null,
    };

    const { result, rerender } = renderHook(
      (props: Props) =>
        useRetryQueue({
          ...props,
          onDispatch,
        }),
      { initialProps },
    );

    act(() => {
      result.current.enqueue('A');
      result.current.enqueue('B');
    });
    expect(result.current.queuedIds.size).toBe(2);

    // Global stream completes → A fires.
    act(() => {
      rerender({ project, isGenerating: false, regeneratingVariationId: null });
    });
    expect(onDispatch).toHaveBeenCalledTimes(1);
    expect(onDispatch).toHaveBeenLastCalledWith(room, 0, 'fresh');
    expect(result.current.queuedIds.has('A')).toBe(false);
    expect(result.current.queuedIds.has('B')).toBe(true);

    // Consumer marks A in flight: regeneratingVariationId moves to 'A'.
    act(() => {
      rerender({ project, isGenerating: false, regeneratingVariationId: 'A' });
    });
    expect(onDispatch).toHaveBeenCalledTimes(1); // no extra dispatch while A is in flight

    // A completes: regeneratingVariationId clears → B fires.
    act(() => {
      rerender({ project, isGenerating: false, regeneratingVariationId: null });
    });
    expect(onDispatch).toHaveBeenCalledTimes(2);
    expect(onDispatch).toHaveBeenLastCalledWith(room, 1, 'fresh');
    expect(result.current.queuedIds.size).toBe(0);
  });
});

describe('useRetryQueue — dedup', () => {
  it('returns "deduped" on repeated enqueue of the same id while queued; queue length unchanged', () => {
    const room = makeFailedRoom();
    const project = makeProject([room]);
    const onDispatch = vi.fn();

    const { result } = renderHook(() =>
      useRetryQueue({
        project,
        isGenerating: true,
        regeneratingVariationId: null,
        onDispatch,
      }),
    );

    let first: ReturnType<typeof result.current.enqueue> | undefined;
    let second: ReturnType<typeof result.current.enqueue> | undefined;
    let third: ReturnType<typeof result.current.enqueue> | undefined;
    act(() => {
      first = result.current.enqueue('v0');
      second = result.current.enqueue('v0');
      third = result.current.enqueue('v0');
    });

    expect(first).toBe('queued');
    expect(second).toBe('deduped');
    expect(third).toBe('deduped');
    expect(result.current.queuedIds.size).toBe(1);
    expect(onDispatch).not.toHaveBeenCalled();
  });

  it('returns "deduped" on rapid-fire same-id while idle (inFlight protects after sync dispatch)', () => {
    const room = makeFailedRoom();
    const project = makeProject([room]);
    const onDispatch = vi.fn();

    const { result } = renderHook(() =>
      useRetryQueue({
        project,
        isGenerating: false,
        regeneratingVariationId: null,
        onDispatch,
      }),
    );

    let first: ReturnType<typeof result.current.enqueue> | undefined;
    let second: ReturnType<typeof result.current.enqueue> | undefined;
    act(() => {
      first = result.current.enqueue('v0');
      second = result.current.enqueue('v0');
    });

    expect(first).toBe('dispatched');
    expect(second).toBe('deduped');
    expect(onDispatch).toHaveBeenCalledTimes(1);
  });
});

describe('useRetryQueue — clear()', () => {
  it('empties the queue and prevents pending dispatches', () => {
    const room = makeFailedRoom();
    const project = makeProject([room]);
    const onDispatch = vi.fn();

    type Props = {
      project: ReturnType<typeof makeProject>;
      isGenerating: boolean;
      regeneratingVariationId: string | null;
    };
    const initialProps: Props = {
      project,
      isGenerating: true,
      regeneratingVariationId: null,
    };

    const { result, rerender } = renderHook(
      (props: Props) => useRetryQueue({ ...props, onDispatch }),
      { initialProps },
    );

    act(() => {
      result.current.enqueue('v0');
    });
    expect(result.current.queuedIds.has('v0')).toBe(true);

    act(() => {
      result.current.clear();
    });
    expect(result.current.queuedIds.size).toBe(0);

    // Idle transition after clear: nothing should dispatch.
    act(() => {
      rerender({ project, isGenerating: false, regeneratingVariationId: null });
    });
    expect(onDispatch).not.toHaveBeenCalled();
  });

  it('returns the number of cleared entries (issue 004 of failed-variation-retry-queue PRD)', () => {
    // The drop-on-error path needs a truthful count for its
    // activity-log entry without racing against the rendered
    // ``queuedIds`` Set (which is one render behind ``queueRef``).
    const room: Room = {
      id: 'r1',
      label: 'Room',
      original_image_url: 'https://x.test/o.png',
      status: 'processing',
      variations: [
        { id: 'v0', status: 'failed', error: 'e', created_at: ISO, updated_at: ISO },
        { id: 'v1', status: 'failed', error: 'e', created_at: ISO, updated_at: ISO },
        { id: 'v2', status: 'failed', error: 'e', created_at: ISO, updated_at: ISO },
      ],
    };
    const project = makeProject([room]);
    const onDispatch = vi.fn();

    const { result } = renderHook(() =>
      useRetryQueue({
        project,
        isGenerating: true,
        regeneratingVariationId: null,
        onDispatch,
      }),
    );

    // Empty queue → clear() returns 0, no log entry on the call site.
    let count = 0;
    act(() => {
      count = result.current.clear();
    });
    expect(count).toBe(0);

    // Three queued → clear() returns 3.
    act(() => {
      result.current.enqueue('v0');
      result.current.enqueue('v1');
      result.current.enqueue('v2');
    });
    expect(result.current.queuedIds.size).toBe(3);
    act(() => {
      count = result.current.clear();
    });
    expect(count).toBe(3);
    expect(result.current.queuedIds.size).toBe(0);

    // Subsequent clear() on an already-empty queue → 0 again.
    act(() => {
      count = result.current.clear();
    });
    expect(count).toBe(0);
  });
});

describe('useRetryQueue — drop rule', () => {
  it('drops a queued id whose variation is no longer "failed" at drain time, considers next entry', () => {
    const room: Room = {
      id: 'r1',
      label: 'Room',
      original_image_url: 'https://x.test/o.png',
      status: 'processing',
      variations: [
        // A was failed when queued, but is "completed" at drain time
        // (e.g., a sibling regen filled it).
        { id: 'A', status: 'failed', error: 'e', created_at: ISO, updated_at: ISO },
        { id: 'B', status: 'failed', error: 'e', created_at: ISO, updated_at: ISO },
      ],
    };
    const project = makeProject([room]);
    const onDispatch = vi.fn();
    const onDrop = vi.fn();

    type Props = {
      project: ReturnType<typeof makeProject>;
      isGenerating: boolean;
      regeneratingVariationId: string | null;
    };
    const initialProps: Props = {
      project,
      isGenerating: true,
      regeneratingVariationId: null,
    };

    const { result, rerender } = renderHook(
      (props: Props) => useRetryQueue({ ...props, onDispatch, onDrop }),
      { initialProps },
    );

    act(() => {
      result.current.enqueue('A');
      result.current.enqueue('B');
    });
    expect(result.current.queuedIds.size).toBe(2);

    // Before draining, A's status flips to 'completed' (race).
    const updatedProject = {
      ...project,
      rooms: [
        {
          ...room,
          variations: [
            { ...room.variations[0], status: 'completed' as const, image_url: 'https://x.test/a.png' },
            room.variations[1],
          ],
        },
      ],
    };

    act(() => {
      rerender({
        project: updatedProject,
        isGenerating: false,
        regeneratingVariationId: null,
      });
    });

    expect(onDrop).toHaveBeenCalledTimes(1);
    expect(onDrop).toHaveBeenCalledWith('A');
    // B is the next valid entry → dispatched
    expect(onDispatch).toHaveBeenCalledTimes(1);
    expect(onDispatch).toHaveBeenCalledWith(updatedProject.rooms[0], 1, 'fresh');
    expect(result.current.queuedIds.size).toBe(0);
  });

  it('drops a queued id whose variation is missing from the project entirely', () => {
    const room = makeFailedRoom('r1', 'A');
    const project = makeProject([room]);
    const onDispatch = vi.fn();
    const onDrop = vi.fn();

    type Props = {
      project: ReturnType<typeof makeProject>;
      isGenerating: boolean;
      regeneratingVariationId: string | null;
    };
    const initialProps: Props = {
      project,
      isGenerating: true,
      regeneratingVariationId: null,
    };

    const { result, rerender } = renderHook(
      (props: Props) => useRetryQueue({ ...props, onDispatch, onDrop }),
      { initialProps },
    );

    act(() => {
      result.current.enqueue('A');
    });

    // Project replaced; the variation is gone (e.g., room deleted server-side).
    const replacedProject = makeProject([
      { ...room, variations: [] },
    ]);
    act(() => {
      rerender({
        project: replacedProject,
        isGenerating: false,
        regeneratingVariationId: null,
      });
    });

    expect(onDrop).toHaveBeenCalledWith('A');
    expect(onDispatch).not.toHaveBeenCalled();
    expect(result.current.queuedIds.size).toBe(0);
  });
});

describe('useRetryQueue — page unmount discards queue', () => {
  // PRD: "Page unmount discards the queue (this is implicit from React state
  // but should be verified by test)." Verifies that an unmounted hook's
  // pending queue cannot fire onDispatch after unmount, even when a fresh
  // hook instance is mounted with isGenerating=false.
  it('does not dispatch from an unmounted hook after the busy state goes idle elsewhere', () => {
    const room = makeFailedRoom();
    const project = makeProject([room]);
    const onDispatch = vi.fn();

    const { result, unmount } = renderHook(() =>
      useRetryQueue({
        project,
        isGenerating: true,
        regeneratingVariationId: null,
        onDispatch,
      }),
    );

    act(() => {
      result.current.enqueue('v0');
    });
    expect(result.current.queuedIds.has('v0')).toBe(true);

    act(() => {
      unmount();
    });

    // A new hook instance with idle state — should NOT see the previous queue.
    const onDispatch2 = vi.fn();
    const { result: result2 } = renderHook(() =>
      useRetryQueue({
        project,
        isGenerating: false,
        regeneratingVariationId: null,
        onDispatch: onDispatch2,
      }),
    );
    expect(result2.current.queuedIds.size).toBe(0);
    expect(onDispatch).not.toHaveBeenCalled();
    expect(onDispatch2).not.toHaveBeenCalled();
  });
});
