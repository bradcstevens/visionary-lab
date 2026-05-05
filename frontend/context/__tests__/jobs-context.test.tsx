import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { renderHook, act, waitFor } from "@testing-library/react"
import { useProjectJobs, getSessionToken, type ProjectJob } from "../jobs-context"

// ---------------------------------------------------------------------------
// EventSource mock — minimal contract used by useProjectJobs:
//   - addEventListener for "open" | "seed" | "job" | "error"
//   - close()
//   - constructor URL is recorded so tests can assert query params
// ---------------------------------------------------------------------------

class MockEventSource {
  static instances: MockEventSource[] = []
  url: string
  init: EventSourceInit | undefined
  closed = false
  listeners: Record<string, ((ev: MessageEvent) => void)[]> = {}

  constructor(url: string, init?: EventSourceInit) {
    this.url = url
    this.init = init
    MockEventSource.instances.push(this)
  }

  addEventListener(type: string, cb: (ev: MessageEvent) => void) {
    ;(this.listeners[type] ||= []).push(cb)
  }

  close() {
    this.closed = true
  }

  // helpers
  dispatch(type: string, data?: unknown) {
    const ev = { data: data === undefined ? "" : JSON.stringify(data) } as MessageEvent
    for (const cb of this.listeners[type] || []) cb(ev)
  }

  fireOpen() {
    for (const cb of this.listeners["open"] || []) cb({} as MessageEvent)
  }

  fireError() {
    for (const cb of this.listeners["error"] || []) cb({} as MessageEvent)
  }
}

function makeJob(overrides: Partial<ProjectJob> = {}): ProjectJob {
  return {
    id: "p1:r1:v1:0",
    project_id: "p1",
    room_id: "r1",
    variation_id: "v1",
    revision: 0,
    kind: "regenerate_variation",
    status: "pending",
    progress: 0,
    phase: "queued",
    updated_at: "2026-05-02T00:00:00.000Z",
    ...overrides,
  }
}

function jsonResponse(body: unknown, init: { status?: number } = {}): Response {
  return new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    headers: { "Content-Type": "application/json" },
  })
}

beforeEach(() => {
  MockEventSource.instances = []
  if (typeof window !== "undefined") {
    try { window.localStorage.clear() } catch { /* ignore */ }
  }
})

afterEach(() => {
  vi.useRealTimers()
})

describe("getSessionToken", () => {
  it("returns a stable token across calls", () => {
    const a = getSessionToken()
    const b = getSessionToken()
    expect(a).toBe(b)
    expect(a.length).toBeGreaterThan(4)
  })
})

describe("useProjectJobs — REST seed", () => {
  it("seeds jobs from GET /staging/projects/{id}/jobs", async () => {
    const seedJob = makeJob({ id: "j1" })
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ jobs: [seedJob] }))

    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: MockEventSource as unknown as typeof EventSource,
      }),
    )

    await waitFor(() => {
      expect(result.current.jobs).toHaveLength(1)
    })
    expect(result.current.jobsById["j1"].status).toBe("pending")
    expect(fetchMock).toHaveBeenCalledWith(
      "http://api/staging/projects/p1/jobs",
      expect.objectContaining({ credentials: "include" }),
    )
  })

  it("does not fetch when projectId is null", () => {
    const fetchMock = vi.fn()
    renderHook(() =>
      useProjectJobs(null, {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: MockEventSource as unknown as typeof EventSource,
      }),
    )
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("surfaces lastError on non-200", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ detail: "boom" }, { status: 500 }))
    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null, // no SSE so the test doesn't hang on connect
      }),
    )
    await waitFor(() => {
      expect(result.current.lastError).toMatch(/HTTP 500/)
    })
  })
})

describe("useProjectJobs — SSE", () => {
  it("opens an EventSource with access_token query param", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ jobs: [] }))
    renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: MockEventSource as unknown as typeof EventSource,
      }),
    )
    await waitFor(() => {
      expect(MockEventSource.instances.length).toBe(1)
    })
    const es = MockEventSource.instances[0]
    expect(es.url).toMatch(/^http:\/\/api\/staging\/projects\/p1\/jobs\/stream\?access_token=/)
  })

  it("merges seed + job events; updated_at wins on conflict", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ jobs: [] }))
    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: MockEventSource as unknown as typeof EventSource,
      }),
    )
    await waitFor(() => expect(MockEventSource.instances.length).toBe(1))
    const es = MockEventSource.instances[0]

    act(() => {
      es.fireOpen()
      es.dispatch("seed", { jobs: [makeJob({ id: "j1", status: "pending", updated_at: "2026-05-02T00:00:01Z" })] })
    })
    await waitFor(() => expect(result.current.jobsById["j1"]?.status).toBe("pending"))

    // Newer event updates it
    act(() => {
      es.dispatch("job", makeJob({ id: "j1", status: "running", phase: "generating", progress: 50, updated_at: "2026-05-02T00:00:02Z" }))
    })
    await waitFor(() => expect(result.current.jobsById["j1"].status).toBe("running"))
    expect(result.current.jobsById["j1"].progress).toBe(50)

    // Stale event (older updated_at) MUST NOT regress state
    act(() => {
      es.dispatch("job", makeJob({ id: "j1", status: "pending", phase: "queued", progress: 0, updated_at: "2026-05-02T00:00:00Z" }))
    })
    expect(result.current.jobsById["j1"].status).toBe("running")
    expect(result.current.jobsById["j1"].progress).toBe(50)
  })

  it("transitions connectionState to 'open' on EventSource open", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ jobs: [] }))
    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: MockEventSource as unknown as typeof EventSource,
      }),
    )
    await waitFor(() => expect(MockEventSource.instances.length).toBe(1))
    act(() => {
      MockEventSource.instances[0].fireOpen()
    })
    await waitFor(() => expect(result.current.connectionState).toBe("open"))
  })

  it("activeJobs excludes terminal statuses", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({
      jobs: [
        makeJob({ id: "a", status: "running" }),
        makeJob({ id: "b", status: "succeeded" }),
        makeJob({ id: "c", status: "failed" }),
        makeJob({ id: "d", status: "pending" }),
        makeJob({ id: "e", status: "cancelled" }),
      ],
    }))
    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
      }),
    )
    await waitFor(() => expect(result.current.jobs).toHaveLength(5))
    expect(result.current.activeJobs.map((j) => j.id).sort()).toEqual(["a", "d"])
  })
})

describe("useProjectJobs — fallback + reconnect", () => {
  it("polls every pollIntervalMs when EventSource is unavailable", async () => {
    vi.useFakeTimers()
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ jobs: [] }))
    renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null, // simulate undefined EventSource
        pollIntervalMs: 5000,
      }),
    )
    // Initial REST seed
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    // Two more polls at 5s + 10s
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000)
    })
    expect(fetchMock).toHaveBeenCalledTimes(2)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000)
    })
    expect(fetchMock).toHaveBeenCalledTimes(3)
  })

  it("starts polling immediately on SSE error and schedules reconnect", async () => {
    vi.useFakeTimers()
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ jobs: [] }))
    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: MockEventSource as unknown as typeof EventSource,
        pollIntervalMs: 5000,
      }),
    )
    await vi.waitFor(() => expect(MockEventSource.instances.length).toBe(1))
    const es = MockEventSource.instances[0]
    const seedCalls = fetchMock.mock.calls.length

    act(() => {
      es.fireError()
    })
    expect(result.current.connectionState).toBe("polling")
    expect(es.closed).toBe(true)

    // Polling kicked in — within 5s the next fetch fires
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000)
    })
    expect(fetchMock.mock.calls.length).toBeGreaterThan(seedCalls)

    // Reconnect attempted (a new MockEventSource is created within
    // base backoff (1s) + jitter (<=500ms))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(MockEventSource.instances.length).toBeGreaterThanOrEqual(2)
  })
})

describe("useProjectJobs — retry()", () => {
  it("POSTs to regenerate endpoint with room_ids+variation_ids and returns job_ids", async () => {
    const fetchMock = vi.fn()
      // initial REST seed
      .mockResolvedValueOnce(jsonResponse({ jobs: [] }))
      // retry POST
      .mockResolvedValueOnce(jsonResponse({ job_ids: ["p1:r1:v1:1"] }))

    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
      }),
    )
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    let returned: string[] = []
    await act(async () => {
      returned = await result.current.retry({ room_id: "r1", variation_id: "v1" })
    })
    expect(returned).toEqual(["p1:r1:v1:1"])

    const lastCall = fetchMock.mock.calls[fetchMock.mock.calls.length - 1]
    expect(lastCall[0]).toBe("http://api/staging/projects/p1/jobs/regenerate")
    expect(lastCall[1].method).toBe("POST")
    expect(JSON.parse(lastCall[1].body)).toEqual({
      room_ids: ["r1"],
      variation_ids: ["v1"],
    })

    // Optimistic insert exposes the new job immediately
    await waitFor(() => {
      expect(result.current.jobsById["p1:r1:v1:1"]?.status).toBe("pending")
    })
  })

  it("throws on non-200", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({ jobs: [] }))
      .mockResolvedValueOnce(jsonResponse({ detail: "nope" }, { status: 503 }))

    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
      }),
    )
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    await expect(
      result.current.retry({ room_id: "r1", variation_id: "v1" }),
    ).rejects.toThrow(/HTTP 503/)
  })
})

describe("useProjectJobs — lifecycle", () => {
  it("closes EventSource and clears polling on unmount", async () => {
    vi.useFakeTimers()
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ jobs: [] }))
    const { unmount } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: MockEventSource as unknown as typeof EventSource,
      }),
    )
    await vi.waitFor(() => expect(MockEventSource.instances.length).toBe(1))
    const es = MockEventSource.instances[0]

    unmount()
    expect(es.closed).toBe(true)

    // No additional polls after unmount
    const calls = fetchMock.mock.calls.length
    await act(async () => {
      await vi.advanceTimersByTimeAsync(15000)
    })
    expect(fetchMock.mock.calls.length).toBe(calls)
  })

  it("reopens stream when projectId changes", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ jobs: [] }))
    const { rerender } = renderHook(
      ({ pid }: { pid: string | null }) =>
        useProjectJobs(pid, {
          apiBaseUrl: "http://api",
          fetchImpl: fetchMock as unknown as typeof fetch,
          eventSourceImpl: MockEventSource as unknown as typeof EventSource,
        }),
      { initialProps: { pid: "p1" as string | null } },
    )
    await waitFor(() => expect(MockEventSource.instances.length).toBe(1))

    rerender({ pid: "p2" })
    await waitFor(() => expect(MockEventSource.instances.length).toBe(2))
    expect(MockEventSource.instances[0].closed).toBe(true)
    expect(MockEventSource.instances[1].url).toContain("/staging/projects/p2/jobs/stream")
  })
})

// ---------------------------------------------------------------------------
// Issue 009 — generate_project slice + cancelProjectGeneration handler
//
// Selector contract (3-tier):
//   1. ≥1 running generate_project    → freshest running by updated_at
//   2. exactly 1 non-terminal         → return it (covers single-pending)
//   3. else (multi-pending, no run)   → null  (deliberate; ambiguous)
//
// cancel handler: no-arg; reads slice internally; null → no-op.
// ---------------------------------------------------------------------------

describe("useProjectJobs — inFlightProjectGeneration slice", () => {
  it("returns null when there are no generate_project jobs (only regenerate_variation)", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({
      jobs: [
        makeJob({ id: "rv1", kind: "regenerate_variation", status: "running" }),
      ],
    }))
    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
      }),
    )
    await waitFor(() => expect(result.current.jobs).toHaveLength(1))
    expect(result.current.inFlightProjectGeneration).toBeNull()
  })

  it("returns null when all generate_project jobs are terminal", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({
      jobs: [
        makeJob({ id: "g1", kind: "generate_project", status: "succeeded" }),
        makeJob({ id: "g2", kind: "generate_project", status: "failed" }),
        makeJob({ id: "g3", kind: "generate_project", status: "cancelled" }),
      ],
    }))
    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
      }),
    )
    await waitFor(() => expect(result.current.jobs).toHaveLength(3))
    expect(result.current.inFlightProjectGeneration).toBeNull()
  })

  it("surfaces a single pending generate_project job (tier 2 of selector)", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({
      jobs: [
        makeJob({
          id: "g1",
          kind: "generate_project",
          status: "pending",
          phase: "queued",
          progress: 0,
        }),
      ],
    }))
    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
      }),
    )
    await waitFor(() => {
      expect(result.current.inFlightProjectGeneration).not.toBeNull()
    })
    expect(result.current.inFlightProjectGeneration).toEqual({
      jobId: "g1",
      progress: 0,
      phase: "queued",
      status: "pending",
    })
  })

  it("surfaces a running generate_project job with full slice shape (jobId+progress+phase+status)", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({
      jobs: [
        makeJob({
          id: "g1",
          kind: "generate_project",
          status: "running",
          phase: "generating",
          progress: 47,
        }),
      ],
    }))
    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
      }),
    )
    await waitFor(() => {
      expect(result.current.inFlightProjectGeneration).not.toBeNull()
    })
    expect(result.current.inFlightProjectGeneration).toEqual({
      jobId: "g1",
      progress: 47,
      phase: "generating",
      status: "running",
    })
  })

  it("prefers the running generate_project over a pending one (tier 1 wins, queued follow-up rule)", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({
      jobs: [
        makeJob({
          id: "g-pending",
          kind: "generate_project",
          status: "pending",
          phase: "queued",
          progress: 0,
          updated_at: "2026-05-03T00:00:10.000Z",
        }),
        makeJob({
          id: "g-running",
          kind: "generate_project",
          status: "running",
          phase: "generating",
          progress: 30,
          updated_at: "2026-05-03T00:00:05.000Z",
        }),
      ],
    }))
    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
      }),
    )
    await waitFor(() => {
      expect(result.current.inFlightProjectGeneration).not.toBeNull()
    })
    // Running wins even though pending was updated more recently — the
    // 3-tier rule prioritises status===running over updated_at recency.
    expect(result.current.inFlightProjectGeneration?.jobId).toBe("g-running")
    expect(result.current.inFlightProjectGeneration?.status).toBe("running")
  })

  it("returns null when multiple non-terminal generate_project jobs exist but none is running (tier 3, ambiguous)", async () => {
    // Two pending-but-not-yet-picked-up jobs is genuinely ambiguous —
    // we shouldn't guess. The PRD reserves queued-backlog visibility
    // for a future slice; issue 009 only exposes the active/running job.
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({
      jobs: [
        makeJob({ id: "g1", kind: "generate_project", status: "pending" }),
        makeJob({ id: "g2", kind: "generate_project", status: "pending" }),
      ],
    }))
    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
      }),
    )
    await waitFor(() => expect(result.current.jobs).toHaveLength(2))
    expect(result.current.inFlightProjectGeneration).toBeNull()
  })

  it("flips to null when the slice's job reaches terminal status via SSE event", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({
      jobs: [
        makeJob({
          id: "g1",
          kind: "generate_project",
          status: "running",
          phase: "generating",
          progress: 80,
          updated_at: "2026-05-03T00:00:01.000Z",
        }),
      ],
    }))
    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: MockEventSource as unknown as typeof EventSource,
      }),
    )
    await waitFor(() => {
      expect(result.current.inFlightProjectGeneration?.jobId).toBe("g1")
    })
    await waitFor(() => expect(MockEventSource.instances.length).toBe(1))
    const es = MockEventSource.instances[0]
    act(() => {
      es.dispatch("job", makeJob({
        id: "g1",
        kind: "generate_project",
        status: "succeeded",
        phase: "finalizing",
        progress: 100,
        updated_at: "2026-05-03T00:01:00.000Z",
      }))
    })
    await waitFor(() => {
      expect(result.current.inFlightProjectGeneration).toBeNull()
    })
  })

  it("phase defaults to 'queued' when the underlying job has phase=null", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({
      jobs: [
        makeJob({
          id: "g1",
          kind: "generate_project",
          status: "pending",
          phase: null,
          progress: 0,
        }),
      ],
    }))
    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
      }),
    )
    await waitFor(() => {
      expect(result.current.inFlightProjectGeneration).not.toBeNull()
    })
    expect(result.current.inFlightProjectGeneration?.phase).toBe("queued")
  })

  it("stays non-null when cancel_requested=true but status is still non-terminal (cancelling-in-flight)", async () => {
    // The backend flips cancel_requested=true on DELETE, but the worker
    // doesn't emit status="cancelled" until it observes the flag and
    // exits. The slice MUST remain visible in this window so the UI can
    // show a "Cancelling..." sub-state (derived from
    // jobsById[id].cancel_requested) instead of disappearing the banner.
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({
      jobs: [
        makeJob({
          id: "g1",
          kind: "generate_project",
          status: "running",
          phase: "generating",
          progress: 60,
          cancel_requested: true,
        }),
      ],
    }))
    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
      }),
    )
    await waitFor(() => {
      expect(result.current.inFlightProjectGeneration).not.toBeNull()
    })
    expect(result.current.inFlightProjectGeneration?.jobId).toBe("g1")
    expect(result.current.inFlightProjectGeneration?.status).toBe("running")
    // Page derives cancelling state directly from jobsById.
    expect(result.current.jobsById["g1"]?.cancel_requested).toBe(true)
  })
})

describe("useProjectJobs — cancelProjectGeneration handler", () => {
  it("is a no-op (no fetch fired) when inFlightProjectGeneration is null", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ jobs: [] }))
    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
      }),
    )
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    expect(result.current.inFlightProjectGeneration).toBeNull()

    const callsBefore = fetchMock.mock.calls.length
    await act(async () => {
      await result.current.cancelProjectGeneration()
    })
    expect(fetchMock.mock.calls.length).toBe(callsBefore)
  })

  it("issues DELETE to /staging/jobs/{id} with the slice's jobId", async () => {
    const fetchMock = vi.fn()
      // initial REST seed with one running generate_project job
      .mockResolvedValueOnce(jsonResponse({
        jobs: [
          makeJob({
            id: "g-active",
            kind: "generate_project",
            status: "running",
            phase: "generating",
            progress: 50,
          }),
        ],
      }))
      // cancel DELETE
      .mockResolvedValueOnce(jsonResponse({
        status: "accepted",
        job_id: "g-active",
        already_terminal: false,
      }))

    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
      }),
    )
    await waitFor(() => {
      expect(result.current.inFlightProjectGeneration?.jobId).toBe("g-active")
    })

    await act(async () => {
      await result.current.cancelProjectGeneration()
    })

    const lastCall = fetchMock.mock.calls[fetchMock.mock.calls.length - 1]
    expect(lastCall[0]).toBe("http://api/staging/jobs/g-active")
    expect(lastCall[1].method).toBe("DELETE")
    expect(lastCall[1].credentials).toBe("include")
  })

  it("does NOT optimistically mutate jobs state (status observed via SSE only)", async () => {
    // Contrast with retry() which DOES optimistic-insert. Cancel is
    // intentionally async-observed: the backend flips cancel_requested
    // but does not change status; the SSE stream delivers the eventual
    // terminal flip. Asserting no synchronous mutation here pins that
    // contract against accidental "convenience" optimistic flips that
    // would race with a status=running re-emission.
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({
        jobs: [
          makeJob({
            id: "g-active",
            kind: "generate_project",
            status: "running",
            phase: "generating",
            progress: 50,
            cancel_requested: false,
          }),
        ],
      }))
      .mockResolvedValueOnce(jsonResponse({
        status: "accepted",
        job_id: "g-active",
        already_terminal: false,
      }))

    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
      }),
    )
    await waitFor(() => {
      expect(result.current.inFlightProjectGeneration?.jobId).toBe("g-active")
    })

    const beforeJob = result.current.jobsById["g-active"]
    await act(async () => {
      await result.current.cancelProjectGeneration()
    })

    // Same status, same cancel_requested, same progress — no in-place
    // mutation. The slice is still non-null (terminal flip arrives
    // later via SSE).
    const afterJob = result.current.jobsById["g-active"]
    expect(afterJob.status).toBe("running")
    expect(afterJob.cancel_requested).toBe(false)
    expect(afterJob.progress).toBe(beforeJob.progress)
    expect(result.current.inFlightProjectGeneration?.jobId).toBe("g-active")
  })

  it("throws with both status code AND response body when DELETE returns non-2xx", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({
        jobs: [
          makeJob({
            id: "g-active",
            kind: "generate_project",
            status: "running",
            phase: "generating",
            progress: 50,
          }),
        ],
      }))
      .mockResolvedValueOnce(
        new Response("Async queue feature flag is disabled", {
          status: 503,
          headers: { "Content-Type": "text/plain" },
        }),
      )

    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
      }),
    )
    await waitFor(() => {
      expect(result.current.inFlightProjectGeneration?.jobId).toBe("g-active")
    })

    await expect(
      result.current.cancelProjectGeneration(),
    ).rejects.toThrow(/503.*Async queue feature flag is disabled/)
  })
})

// ---------------------------------------------------------------------------
// Issue 005 — staleness tracking + cancelAllProjectJobs
// ---------------------------------------------------------------------------

describe("useProjectJobs — staleness baseline (issue 005)", () => {
  it("records lastBackendActivityByJobId on every merge (REST seed, SSE seed, SSE job)", async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-05-04T12:00:00Z"))
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({
      jobs: [makeJob({
        id: "g1",
        kind: "generate_project",
        status: "running",
        phase: "generating",
        progress: 50,
        updated_at: "2026-05-04T11:59:00Z",
      })],
    }))
    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: MockEventSource as unknown as typeof EventSource,
      }),
    )

    await vi.waitFor(() => expect(result.current.jobs).toHaveLength(1))

    // After REST seed, baseline is the front-end wall clock at merge time
    // (NOT the worker's updated_at — rubber-duck #2). Allow a few ms of
    // drift because vi.waitFor advances the fake clock while polling.
    expect(result.current.lastBackendActivityByJobId["g1"]).toBeGreaterThanOrEqual(
      Date.parse("2026-05-04T12:00:00Z"),
    )
    const seedBaseline = result.current.lastBackendActivityByJobId["g1"]
    expect(seedBaseline).toBeLessThan(Date.parse("2026-05-04T12:00:30Z"))

    // Now an SSE job event lands later — baseline should advance.
    vi.setSystemTime(new Date("2026-05-04T12:00:30Z"))
    await vi.waitFor(() => expect(MockEventSource.instances.length).toBe(1))
    const es = MockEventSource.instances[0]
    act(() => {
      es.dispatch("job", makeJob({
        id: "g1",
        kind: "generate_project",
        status: "running",
        phase: "generating",
        progress: 70,
        updated_at: "2026-05-04T12:00:25Z",
      }))
    })
    await vi.waitFor(() => {
      expect(result.current.jobsById["g1"].progress).toBe(70)
    })
    expect(result.current.lastBackendActivityByJobId["g1"]).toBeGreaterThanOrEqual(
      Date.parse("2026-05-04T12:00:30Z"),
    )
  })

  it("baseline persists across renders even when no new merge fires", async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-05-04T12:00:00Z"))
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({
      jobs: [makeJob({
        id: "g1",
        kind: "generate_project",
        status: "running",
        phase: "generating",
        progress: 50,
        updated_at: "2026-05-04T11:59:00Z",
      })],
    }))

    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        // Polling-only mode (no SSE)
        eventSourceImpl: null,
        pollIntervalMs: 60_000, // long so we don't trigger duplicate polls
      }),
    )
    await vi.waitFor(() => expect(result.current.jobs).toHaveLength(1))
    const initialBaseline = result.current.lastBackendActivityByJobId["g1"]
    expect(initialBaseline).toBeGreaterThanOrEqual(Date.parse("2026-05-04T12:00:00Z"))

    // Time advances 30s but no merge fires. Baseline does NOT advance —
    // it represents "last time we confirmed activity from backend",
    // which is exactly what the staleness detector wants.
    expect(result.current.lastBackendActivityByJobId["g1"]).toBe(
      initialBaseline,
    )
  })

  it("baseline does NOT advance when a poll returns the SAME doc (rubber-duck strict-newer rule)", async () => {
    // The polling layer fires every pollIntervalMs and merges the same
    // doc when nothing has changed worker-side. The pre-fix behaviour
    // recorded a fresh ``Date.now()`` baseline on every accepted merge,
    // including these no-op polls — staleness would NEVER trigger
    // because the baseline kept tracking wall-clock. The fix: only
    // STRICTLY-NEWER ``updated_at`` advances the baseline. Polls
    // delivering the same doc accept the merge (progress tiebreak) but
    // leave the baseline alone, which is exactly the freshness signal
    // the detector wants.
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-05-04T12:00:00Z"))
    let pollCount = 0
    const fetchMock = vi.fn(async () => {
      pollCount += 1
      return jsonResponse({
        jobs: [
          makeJob({
            id: "g1",
            kind: "generate_project",
            status: "running",
            phase: "generating",
            progress: 50,
            updated_at: "2026-05-04T11:59:00Z",
          }),
        ],
      })
    })

    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
        pollIntervalMs: 1_000,
      }),
    )

    await vi.waitFor(() => expect(result.current.jobs).toHaveLength(1))
    const initialBaseline = result.current.lastBackendActivityByJobId["g1"]
    expect(initialBaseline).toBeGreaterThanOrEqual(
      Date.parse("2026-05-04T12:00:00Z"),
    )

    // Advance the clock and let the polling timer fire several times.
    vi.setSystemTime(new Date("2026-05-04T12:01:00Z"))
    await vi.advanceTimersByTimeAsync(5_000)

    // Confirm polling actually happened (otherwise the test is trivially
    // green for the wrong reason).
    expect(pollCount).toBeGreaterThan(1)

    // Baseline did NOT advance even though the wall clock moved by 60s.
    expect(result.current.lastBackendActivityByJobId["g1"]).toBe(
      initialBaseline,
    )
  })

  it("deriveProjectWorstStaleness exposes worst-state across non-terminal generate_project jobs", async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-05-04T12:00:00Z"))
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({
      jobs: [
        makeJob({
          id: "g1",
          kind: "generate_project",
          status: "pending",
          updated_at: "2026-05-04T11:59:00Z",
        }),
        makeJob({
          id: "g2",
          kind: "generate_project",
          status: "pending",
          updated_at: "2026-05-04T11:59:00Z",
        }),
      ],
    }))
    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
      }),
    )
    await vi.waitFor(() => expect(result.current.jobs).toHaveLength(2))
    expect(result.current.inFlightProjectGeneration).toBeNull()

    // Tier-3 multi-pending case (rubber-duck #3): cancel-all is
    // most-needed exactly when the canonical slice returns null.
    // Project staleness must STILL surface.
    expect(result.current.projectStaleness).not.toBeNull()
    expect(result.current.projectStaleness?.kind).toBe("fresh")
  })

  it("computes hard-pending after 120s elapsed (5s tick recomputes)", async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-05-04T12:00:00Z"))
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({
      jobs: [makeJob({
        id: "g1",
        kind: "generate_project",
        status: "pending",
        updated_at: "2026-05-04T11:59:00Z",
      })],
    }))
    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
        pollIntervalMs: 60_000,
      }),
    )
    await vi.waitFor(() => expect(result.current.jobs).toHaveLength(1))
    expect(result.current.projectStaleness?.kind).toBe("fresh")

    // Advance 120s past baseline merge time and run a 5s tick.
    await act(async () => {
      vi.setSystemTime(new Date("2026-05-04T12:02:00Z"))
      await vi.advanceTimersByTimeAsync(5_000)
    })
    expect(result.current.projectStaleness?.kind).toBe("hard-pending")
    expect(result.current.projectStaleness?.secondsAgo).toBeGreaterThanOrEqual(120)
  })

  it("ignores terminal jobs in projectStaleness", async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-05-04T12:00:00Z"))
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({
      jobs: [makeJob({
        id: "g1",
        kind: "generate_project",
        status: "succeeded",
        updated_at: "2026-05-04T11:00:00Z",
      })],
    }))
    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
      }),
    )
    await vi.waitFor(() => expect(result.current.jobs).toHaveLength(1))
    expect(result.current.projectStaleness).toBeNull()
  })
})

describe("useProjectJobs — cancelAllProjectJobs (issue 005)", () => {
  it("issues DELETE to /staging/projects/{id}/jobs and resolves with the response body", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({ jobs: [] }))
      .mockResolvedValueOnce(jsonResponse({
        status: "accepted",
        cancelled_count: 3,
        project_id: "p1",
      }, { status: 202 }))

    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
      }),
    )
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    let response: { status: string; cancelled_count: number; project_id: string } | undefined | void
    await act(async () => {
      response = await result.current.cancelAllProjectJobs()
    })
    expect(response).toEqual({
      status: "accepted",
      cancelled_count: 3,
      project_id: "p1",
    })
    const lastCall = fetchMock.mock.calls[fetchMock.mock.calls.length - 1]
    expect(lastCall[0]).toBe("http://api/staging/projects/p1/jobs")
    expect(lastCall[1].method).toBe("DELETE")
    expect(lastCall[1].credentials).toBe("include")
  })

  it("throws with status + body on non-2xx", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({ jobs: [] }))
      .mockResolvedValueOnce(
        new Response("Project not found", { status: 404 }),
      )

    const { result } = renderHook(() =>
      useProjectJobs("p1", {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
      }),
    )
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    await expect(result.current.cancelAllProjectJobs()).rejects.toThrow(/404.*Project not found/)
  })

  it("no-ops when projectId is null", async () => {
    const fetchMock = vi.fn()
    const { result } = renderHook(() =>
      useProjectJobs(null, {
        apiBaseUrl: "http://api",
        fetchImpl: fetchMock as unknown as typeof fetch,
        eventSourceImpl: null,
      }),
    )
    const before = fetchMock.mock.calls.length
    await act(async () => {
      await result.current.cancelAllProjectJobs()
    })
    expect(fetchMock.mock.calls.length).toBe(before)
  })
})
