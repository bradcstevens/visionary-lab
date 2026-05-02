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
