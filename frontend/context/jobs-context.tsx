"use client"

import React, { createContext, useContext, useState, useCallback, useEffect, useRef, useMemo } from "react"
import { API_BASE_URL } from "@/services/stagingApi"

// ---------------------------------------------------------------------------
// Project-scoped job state (issue 006 — image-pipeline-and-project-ux-overhaul)
// ---------------------------------------------------------------------------
//
// Provides a REST-seeded, SSE-subscribed live view of staging-pipeline jobs
// for a single project. Used by the per-image / per-project progress bars
// (issue 009), the StorageImage skeleton (011), and any UI surface that needs
// to reflect job status without manual refresh.
//
// Transport order, by design:
//   1. REST seed via GET /staging/projects/{id}/jobs (always)
//   2. EventSource subscription to /staging/projects/{id}/jobs/stream
//      (when typeof EventSource !== "undefined")
//   3. Polling fallback (5s) when EventSource is unavailable OR while we are
//      between SSE reconnection attempts (so the UI never goes stale).
//
// Merge rule: jobs are keyed by id; the doc with the larger updated_at wins
// (NOT arrival order). This defends against a delayed REST poll overwriting
// fresher SSE-delivered state during a reconnect window.

export type JobStatus =
  | "pending"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled"
  | string

export type JobPhase = "queued" | "generating" | "finalizing" | string | null

export interface ProjectJob {
  id: string
  project_id: string
  room_id: string
  variation_id: string
  revision: number
  kind: string
  status: JobStatus
  progress?: number | null
  phase?: JobPhase
  attempts?: number
  error?: string | null
  result?: unknown
  cancel_requested?: boolean
  created_at?: string
  updated_at?: string
}

export type ConnectionState =
  | "idle"
  | "connecting"
  | "open"
  | "polling"
  | "closed"

export const TERMINAL_JOB_STATUSES = new Set<JobStatus>([
  "succeeded",
  "failed",
  "cancelled",
])

const SESSION_TOKEN_STORAGE_KEY = "vlab.session_token"
const POLL_INTERVAL_MS = 5_000
const RECONNECT_BASE_MS = 1_000
const RECONNECT_MAX_MS = 30_000
const RECONNECT_JITTER_MS = 500

/** Mint or read a stable opaque session token used as the SSE auth key. */
export function getSessionToken(): string {
  if (typeof window === "undefined") return "ssr"
  try {
    let t = window.localStorage.getItem(SESSION_TOKEN_STORAGE_KEY)
    if (!t) {
      const cryptoObj = (window as unknown as { crypto?: Crypto }).crypto
      t = cryptoObj?.randomUUID
        ? cryptoObj.randomUUID()
        : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
      window.localStorage.setItem(SESSION_TOKEN_STORAGE_KEY, t)
    }
    return t
  } catch {
    // Storage disabled (Safari private mode, etc.) — degrade to a per-tab token.
    return `ephemeral-${Math.random().toString(36).slice(2, 10)}`
  }
}

function _isNewer(incoming: ProjectJob, existing: ProjectJob | undefined): boolean {
  if (!existing) return true
  const a = incoming.updated_at ? Date.parse(incoming.updated_at) : 0
  const b = existing.updated_at ? Date.parse(existing.updated_at) : 0
  if (a !== b) return a >= b
  // Same updated_at (clock granularity collision) — prefer the doc with
  // higher progress so a status flip from running→succeeded that arrives
  // simultaneously with a heartbeat doesn't get stomped.
  return (incoming.progress ?? 0) >= (existing.progress ?? 0)
}

/** Public type returned by useProjectJobs. */
export interface ProjectJobsState {
  jobs: ProjectJob[]
  jobsById: Record<string, ProjectJob>
  activeJobs: ProjectJob[]
  connectionState: ConnectionState
  lastError: string | null
  retry: (job: { room_id: string; variation_id: string }) => Promise<string[]>
  refresh: () => Promise<void>
  // Issue 009 of project-generation-async-queue-cutover PRD: live view of
  // the in-flight ``kind="generate_project"`` job for this project, derived
  // from ``activeJobs`` (single source of truth for "non-terminal").
  //
  // Selector contract (3-tier — see useMemo body):
  //   1. ≥1 running generate_project    → freshest running by updated_at
  //   2. exactly 1 non-terminal         → return it (covers single pending)
  //   3. else (multiple pending, none running) → null  (deliberate)
  //
  // The tier-3 ambiguous case deliberately surfaces no banner / no cancel
  // affordance — picking one would wire the UI to an arbitrary job, and the
  // PRD reserves "queued backlog visibility" for a future slice. This
  // contract is pinned by the "two pending → null" test.
  inFlightProjectGeneration: {
    jobId: string
    progress: number
    phase: string
    status: string
  } | null
  // Issue 009 of project-generation-async-queue-cutover PRD: cancels the
  // currently exposed ``inFlightProjectGeneration`` job. No args by design
  // — the slice is the canonical "active job"; passing an explicit jobId
  // would let callers cancel queued/wrong jobs out of sync with the read
  // side. Null slice → no-op (no DELETE fired). Status flip arrives via
  // the SSE/REST stream after the worker observes ``cancel_requested=true``;
  // we intentionally do NOT optimistically mutate (contrast with retry()).
  cancelProjectGeneration: () => Promise<void>
}

interface UseProjectJobsOptions {
  /** Override the default base URL (tests). */
  apiBaseUrl?: string
  /** Override the default fetch (tests). */
  fetchImpl?: typeof fetch
  /** Override the default EventSource constructor (tests). */
  eventSourceImpl?: typeof EventSource | null
  /** Override the polling interval (tests). */
  pollIntervalMs?: number
  /** Skip mounting (e.g. when projectId is not yet known). */
  enabled?: boolean
}

/**
 * Subscribe to live job state for a single project.
 *
 * Lifecycle:
 *   - mount → REST seed → open EventSource → merge events
 *   - SSE error → close, schedule reconnect with backoff+jitter, AND start
 *     polling immediately so the UI keeps updating during the gap
 *   - EventSource unavailable (typeof undefined) → polling-only mode
 *   - unmount or projectId change → close ES, clear poll/reconnect timers
 */
export function useProjectJobs(
  projectId: string | null | undefined,
  opts: UseProjectJobsOptions = {},
): ProjectJobsState {
  const {
    apiBaseUrl = API_BASE_URL,
    fetchImpl,
    eventSourceImpl,
    pollIntervalMs = POLL_INTERVAL_MS,
    enabled = true,
  } = opts

  const [jobsById, setJobsById] = useState<Record<string, ProjectJob>>({})
  const [connectionState, setConnectionState] = useState<ConnectionState>("idle")
  const [lastError, setLastError] = useState<string | null>(null)

  const esRef = useRef<EventSource | null>(null)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const aliveRef = useRef(true)
  const projectIdRef = useRef(projectId)
  // Forward-ref so scheduleReconnect can call openStream defined later
  // without an init-order cycle (lint rule disallows reading a const
  // before its declaration site).
  const openStreamRef = useRef<() => void>(() => {})

  const fetchFn: typeof fetch = useMemo(
    () => fetchImpl ?? (typeof fetch !== "undefined" ? fetch.bind(globalThis) : (() => {
      throw new Error("fetch is unavailable in this environment")
    }) as typeof fetch),
    [fetchImpl],
  )

  const ESCtor = useMemo<typeof EventSource | null>(() => {
    if (eventSourceImpl !== undefined) return eventSourceImpl
    return typeof EventSource !== "undefined" ? EventSource : null
  }, [eventSourceImpl])

  const mergeJobs = useCallback((incoming: ProjectJob[]) => {
    if (!incoming || incoming.length === 0) return
    setJobsById((prev) => {
      let changed = false
      const next: Record<string, ProjectJob> = { ...prev }
      for (const job of incoming) {
        if (!job || !job.id) continue
        if (_isNewer(job, prev[job.id])) {
          next[job.id] = job
          changed = true
        }
      }
      return changed ? next : prev
    })
  }, [])

  const seedFromRest = useCallback(async (): Promise<void> => {
    const pid = projectIdRef.current
    if (!pid) return
    try {
      const resp = await fetchFn(`${apiBaseUrl}/staging/projects/${pid}/jobs`, {
        credentials: "include",
      })
      if (!resp.ok) {
        // 404 means the project has been deleted — let the caller's project
        // load handle the redirect; we just stop trying to fetch jobs.
        if (resp.status === 404) {
          setLastError("Project not found")
          return
        }
        setLastError(`REST seed failed: HTTP ${resp.status}`)
        return
      }
      const body = (await resp.json()) as { jobs: ProjectJob[] }
      if (!aliveRef.current || projectIdRef.current !== pid) return
      mergeJobs(body.jobs ?? [])
      setLastError(null)
    } catch (err) {
      if (!aliveRef.current) return
      setLastError(`REST seed failed: ${(err as Error).message}`)
    }
  }, [apiBaseUrl, fetchFn, mergeJobs])

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
  }, [])

  const startPolling = useCallback(() => {
    if (pollTimerRef.current) return
    pollTimerRef.current = setInterval(() => {
      void seedFromRest()
    }, pollIntervalMs)
  }, [pollIntervalMs, seedFromRest])

  const closeStream = useCallback(() => {
    if (esRef.current) {
      try { esRef.current.close() } catch { /* swallow */ }
      esRef.current = null
    }
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
  }, [])

  const scheduleReconnect = useCallback(() => {
    if (!aliveRef.current) return
    const attempt = reconnectAttemptsRef.current
    const backoff = Math.min(RECONNECT_BASE_MS * 2 ** attempt, RECONNECT_MAX_MS)
    const jitter = Math.floor(Math.random() * RECONNECT_JITTER_MS)
    reconnectAttemptsRef.current = attempt + 1
    reconnectTimerRef.current = setTimeout(() => {
      reconnectTimerRef.current = null
      openStreamRef.current()
    }, backoff + jitter)
  }, [])

  const openStream = useCallback(() => {
    const pid = projectIdRef.current
    if (!pid || !aliveRef.current) return
    if (!ESCtor) {
      // Polling-only mode.
      setConnectionState("polling")
      startPolling()
      return
    }
    closeStream()
    setConnectionState("connecting")
    const token = getSessionToken()
    const url = `${apiBaseUrl}/staging/projects/${pid}/jobs/stream?access_token=${encodeURIComponent(token)}`
    let es: EventSource
    try {
      es = new ESCtor(url, { withCredentials: true })
    } catch (err) {
      setLastError(`SSE open failed: ${(err as Error).message}`)
      setConnectionState("polling")
      startPolling()
      scheduleReconnect()
      return
    }
    esRef.current = es

    es.addEventListener("open", () => {
      if (!aliveRef.current) return
      reconnectAttemptsRef.current = 0
      setConnectionState("open")
      setLastError(null)
      // SSE is healthy — stop the fallback poll to avoid duplicate work.
      stopPolling()
    })

    es.addEventListener("seed", (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data) as { jobs: ProjectJob[] }
        mergeJobs(payload.jobs ?? [])
      } catch (err) {
        setLastError(`Bad seed payload: ${(err as Error).message}`)
      }
    })

    es.addEventListener("job", (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data) as ProjectJob
        mergeJobs([payload])
      } catch (err) {
        setLastError(`Bad job payload: ${(err as Error).message}`)
      }
    })

    es.addEventListener("error", () => {
      if (!aliveRef.current) return
      setLastError("SSE connection error")
      setConnectionState("polling")
      // Start polling immediately so the UI keeps updating during backoff.
      startPolling()
      // Close the broken stream and schedule reconnect.
      try { es.close() } catch { /* swallow */ }
      if (esRef.current === es) esRef.current = null
      scheduleReconnect()
    })
  }, [ESCtor, apiBaseUrl, closeStream, mergeJobs, scheduleReconnect, startPolling, stopPolling])

  const retry = useCallback(
    async (job: { room_id: string; variation_id: string }): Promise<string[]> => {
      const pid = projectIdRef.current
      if (!pid) return []
      const resp = await fetchFn(
        `${apiBaseUrl}/staging/projects/${pid}/jobs/regenerate`,
        {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            room_ids: [job.room_id],
            variation_ids: [job.variation_id],
          }),
        },
      )
      if (!resp.ok) {
        throw new Error(`Retry failed: HTTP ${resp.status}`)
      }
      const body = (await resp.json()) as { job_ids: string[] }
      const ids = body.job_ids ?? []
      // Optimistic insert so the UI reflects the new active job(s)
      // immediately even if SSE is degraded.
      if (ids.length) {
        const now = new Date().toISOString()
        const optimistic: ProjectJob[] = ids.map((id) => ({
          id,
          project_id: pid,
          room_id: job.room_id,
          variation_id: job.variation_id,
          revision: 0,
          kind: "regenerate_variation",
          status: "pending",
          phase: "queued",
          progress: 0,
          updated_at: now,
        }))
        mergeJobs(optimistic)
      }
      return ids
    },
    [apiBaseUrl, fetchFn, mergeJobs],
  )

  // Mount effect: seed via REST, then open the stream. Reset on projectId change.
  useEffect(() => {
    projectIdRef.current = projectId
    if (!enabled || !projectId) {
      // Defer the state writes so we don't call setState synchronously in
      // the effect body (lint: react-hooks/set-state-in-effect).
      const t = setTimeout(() => setConnectionState("idle"), 0)
      return () => clearTimeout(t)
    }
    aliveRef.current = true
    openStreamRef.current = openStream
    // Reset state when projectId changes — these setState calls are
    // intentional (the previous project's jobs must not bleed through).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setJobsById({})
    setConnectionState("connecting")
    void seedFromRest().then(() => {
      if (!aliveRef.current || projectIdRef.current !== projectId) return
      openStream()
    })
    return () => {
      aliveRef.current = false
      closeStream()
      stopPolling()
      setConnectionState("closed")
    }
    // openStream / seedFromRest deps are intentionally excluded — they're
    // recreated on every render but we only want to (re)open the stream
    // when the projectId itself changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, enabled])

  const jobs = useMemo(() => Object.values(jobsById), [jobsById])
  const activeJobs = useMemo(
    () => jobs.filter((j) => !TERMINAL_JOB_STATUSES.has(j.status)),
    [jobs],
  )

  // Issue 009 of project-generation-async-queue-cutover PRD: derive the
  // single "active project-generation" slice from activeJobs (NOT jobs)
  // so we share the same canonical "non-terminal" definition. If a future
  // status (e.g. ``timed_out``) is added to TERMINAL_JOB_STATUSES it
  // automatically removes those jobs from this slice without a separate
  // edit here.
  const inFlightProjectGeneration = useMemo(() => {
    const candidates = activeJobs.filter((j) => j.kind === "generate_project")
    if (candidates.length === 0) return null
    const running = candidates.filter((j) => j.status === "running")
    let chosen: ProjectJob | null = null
    if (running.length > 0) {
      // Tier 1: at least one running — pick the freshest (highest updated_at).
      // Defensive sort even though there should normally be exactly one
      // running per project (concurrent jobs are queued, not co-running).
      chosen = [...running].sort((a, b) => {
        const aT = a.updated_at ? Date.parse(a.updated_at) : 0
        const bT = b.updated_at ? Date.parse(b.updated_at) : 0
        return bT - aT
      })[0]
    } else if (candidates.length === 1) {
      // Tier 2: exactly one non-terminal (likely a pending job not yet
      // picked up by the worker) — surface it.
      chosen = candidates[0]
    } else {
      // Tier 3: multiple pending, none running — ambiguous. Returning null
      // is deliberate; queued-backlog visibility is a future slice.
      return null
    }
    return {
      jobId: chosen.id,
      progress: chosen.progress ?? 0,
      phase: (chosen.phase ?? "queued") as string,
      status: chosen.status,
    }
  }, [activeJobs])

  const cancelProjectGeneration = useCallback(async (): Promise<void> => {
    // Read the slice's jobId at call time — the closure captures the
    // memoised slice, which rebinds when activeJobs changes.
    const jobId = inFlightProjectGeneration?.jobId
    if (!jobId) return
    const resp = await fetchFn(`${apiBaseUrl}/staging/jobs/${jobId}`, {
      method: "DELETE",
      credentials: "include",
    })
    if (!resp.ok) {
      // Preserve the response body (matches enqueueProjectGeneration's
      // convention) — the cancel endpoint returns 404 for unknown ids
      // and 503 when the async-queue feature flag is off; both have
      // useful body text the UI can surface.
      let detail = ""
      try { detail = await resp.text() } catch { /* ignore */ }
      throw new Error(
        `Cancel failed: HTTP ${resp.status}${detail ? ` - ${detail}` : ""}`,
      )
    }
    // Intentionally NO optimistic mutation — the SSE stream will deliver
    // the cancel_requested flip and the eventual terminal status.
  }, [apiBaseUrl, fetchFn, inFlightProjectGeneration])

  return {
    jobs,
    jobsById,
    activeJobs,
    connectionState,
    lastError,
    retry,
    refresh: seedFromRest,
    inFlightProjectGeneration,
    cancelProjectGeneration,
  }
}

// ---------------------------------------------------------------------------
// Legacy refresh-handler API (preserved for app/jobs/page.tsx — videos page)
// ---------------------------------------------------------------------------

type JobsContextType = {
  refreshJobs: () => Promise<void>
  setRefreshHandler: (handler: (() => Promise<void>) | null) => void
  hasRefreshHandler: boolean
}

const JobsContext = createContext<JobsContextType>({
  refreshJobs: async () => {},
  setRefreshHandler: () => {},
  hasRefreshHandler: false
})

export const useJobs = () => useContext(JobsContext)

export function JobsProvider({ children }: { children: React.ReactNode }) {
  const [refreshHandler, setRefreshHandlerState] = useState<(() => Promise<void>) | null>(null)
  const [hasHandler, setHasHandler] = useState(false)
  const refreshOperationInProgress = useRef(false)
  const isMounted = useRef(false)

  // This effect safely updates the mount status
  useEffect(() => {
    isMounted.current = true;
    
    return () => {
      isMounted.current = false;
    };
  }, []);
  
  // This effect safely updates the hasHandler state based on the refreshHandler
  useEffect(() => {
    if (isMounted.current) {
      // Use setTimeout to ensure state updates don't happen during render
      setTimeout(() => {
        if (isMounted.current) {
          const handlerExists = !!refreshHandler && typeof refreshHandler === 'function';
          setHasHandler(handlerExists);
          console.log("JobsProvider: refresh handler updated, exists:", handlerExists);
        }
      }, 0);
    }
  }, [refreshHandler]);

  const refreshJobs = useCallback(async () => {
    // Prevent concurrent refresh operations
    if (refreshOperationInProgress.current) {
      console.log("Refresh operation already in progress, skipping");
      return;
    }

    console.log("RefreshJobs called, handler exists:", !!refreshHandler);
    if (refreshHandler && typeof refreshHandler === 'function') {
      try {
        refreshOperationInProgress.current = true;
        await refreshHandler();
        console.log("Refresh handler executed successfully");
      } catch (error) {
        console.error("Error in refresh handler:", error);
        throw error;
      } finally {
        refreshOperationInProgress.current = false;
      }
    } else {
      console.warn("No refresh handler registered or it's not a function");
    }
  }, [refreshHandler]);
  
  const setRefreshHandlerWrapper = useCallback((handler: (() => Promise<void>) | null) => {
    console.log("Setting refresh handler:", !!handler, typeof handler);
    if (isMounted.current) {
      // Use setTimeout to ensure state updates don't happen during render
      setTimeout(() => {
        if (isMounted.current) {
          setRefreshHandlerState(handler);
        }
      }, 0);
    }
  }, []);

  return (
    <JobsContext.Provider value={{ 
      refreshJobs, 
      setRefreshHandler: setRefreshHandlerWrapper,
      hasRefreshHandler: hasHandler
    }}>
      {children}
    </JobsContext.Provider>
  );
} 