"use client"

import React, { createContext, useContext, useState, useCallback, useEffect, useRef, useMemo } from "react"
import { API_BASE_URL } from "@/services/stagingApi"
import {
  computeStaleness,
  deriveProjectWorstStaleness,
  type StalenessState,
} from "@/lib/job-staleness"

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
  // Issue 004 of active-and-queued-jobs-ux-redesign PRD: classified
  // failure kind written by the worker on terminal-failure transitions
  // and by the producer on enqueue failures (issue 002). Mirrors the
  // backend's ``ErrorKind`` enum (.value form). Optional because most
  // jobs don't carry one — only failed ones, and even then only after
  // the issue 002 backend ship.
  error_kind?: string | null
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
  // Issue 004 of active-and-queued-jobs-ux-redesign PRD: inject a synthetic
  // optimistic job into the merged jobsById so the in-flight banner / room
  // tile renders immediately on a 202 response, before the SSE seed catches
  // up (~1-3s window). The job's ``updated_at`` is forced to epoch zero
  // ("1970-01-01T00:00:00Z") so any real SSE doc with a non-empty
  // ``updated_at`` strictly supersedes it via ``_isNewer``. Page-level
  // callers populate the synthetic doc with the REAL ``job_id`` returned
  // from the producer 202; if SSE never delivers (network down), the
  // optimistic doc lingers — same failure mode as before, just with a
  // visible banner instead of a silent gap.
  injectOptimisticJob: (job: ProjectJob) => void
  // Issue 005 of active-and-queued-jobs-ux-redesign PRD: per-job baseline
  // recording the front-end wall-clock timestamp of the most recent merge
  // that ACCEPTED a doc (i.e., the doc passed ``_isNewer`` and won the
  // comparison). Consumed by the staleness detector — exposed for tests
  // and for any UI that wants raw freshness data.
  lastBackendActivityByJobId: Record<string, number>
  // Issue 005: pre-computed worst-case staleness across all non-terminal
  // ``generate_project`` jobs. NULL when there are no in-flight jobs OR
  // every in-flight job is fresh and there's nothing actionable. Recomputed
  // every 5s and on every merge so the page header doesn't have to call
  // ``computeStaleness`` itself.
  projectStaleness: StalenessState | null
  // Issue 005: bulk cancel — the user's "give up" escape hatch when
  // staleness has crossed the 120s hard threshold. Hits
  // DELETE /staging/projects/{id}/jobs which is a thin wrapper around
  // ``_cascade_cancel_project_jobs``. Returns the response body so the
  // page can surface ``cancelled_count`` in the success toast.
  cancelAllProjectJobs: () => Promise<
    { status: string; cancelled_count: number; project_id: string } | void
  >
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
  // Issue 005 of active-and-queued-jobs-ux-redesign PRD: per-job baseline
  // map. Set by ``mergeJobs`` for every doc that wins the ``_isNewer``
  // comparison; consumed by ``computeStaleness``. Tracking
  // front-end wall clock (NOT job.updated_at) defends against worker /
  // client NTP drift poisoning the detector — see job-staleness.ts.
  const [lastBackendActivityByJobId, setLastBackendActivityByJobId] = useState<
    Record<string, number>
  >({})
  // Issue 005: tick state forces a recomputation of projectStaleness even
  // when no merge has fired (a job that's been sitting at 119s elapsed
  // becomes hard-stale at 120s without any new doc arriving — the 5s
  // interval bumps this counter so the useMemo re-evaluates against
  // ``Date.now()``).
  const [stalenessTick, setStalenessTick] = useState(() => Date.now())

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
      // Issue 005: only docs with a STRICTLY-NEWER ``updated_at`` than the
      // existing one (or first-seen) advance the staleness baseline. A poll
      // that returns the same doc is NOT evidence the worker is alive — it's
      // evidence the list-jobs endpoint works. Without the strict check,
      // staleness would never trigger because every 5s poll would reset the
      // baseline to ``Date.now()``. Equal-timestamp merges still accept (so
      // the existing ``_isNewer`` progress-tiebreaker keeps working), they
      // just don't reset the baseline.
      const strictlyNewerHere: string[] = []
      for (const job of incoming) {
        if (!job || !job.id) continue
        if (_isNewer(job, prev[job.id])) {
          next[job.id] = job
          changed = true
          const prevTs = prev[job.id]?.updated_at
            ? Date.parse(prev[job.id].updated_at as string)
            : 0
          const incomingTs = job.updated_at ? Date.parse(job.updated_at) : 0
          if (!prev[job.id] || incomingTs > prevTs) {
            strictlyNewerHere.push(job.id)
          }
        }
      }
      if (strictlyNewerHere.length > 0) {
        // Issue 005: every strictly-newer accepted merge bumps the per-job
        // baseline to *now* (front-end wall clock, NOT job.updated_at, so
        // NTP drift between worker and client doesn't poison the detector).
        // Performed synchronously inside the jobsById update so the baseline
        // lands in the same React commit as the merge.
        const now = Date.now()
        setLastBackendActivityByJobId((prevBaseline) => {
          const nextBaseline = { ...prevBaseline }
          for (const id of strictlyNewerHere) nextBaseline[id] = now
          return nextBaseline
        })
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

  // Issue 004 of active-and-queued-jobs-ux-redesign PRD: inject a synthetic
  // optimistic job. Forces ``updated_at`` to epoch zero so the merge rule
  // (``_isNewer``) lets any real SSE doc supersede us regardless of when
  // the optimistic injection happened.
  const injectOptimisticJob = useCallback((job: ProjectJob): void => {
    mergeJobs([{ ...job, updated_at: "1970-01-01T00:00:00Z" }])
  }, [mergeJobs])

  // Issue 005 of active-and-queued-jobs-ux-redesign PRD: bulk cancel —
  // hits ``DELETE /staging/projects/{project_id}/jobs`` (server-side
  // cascade). Returns the response body so the page can render
  // "Cancelled N queued jobs" in the success toast.
  const cancelAllProjectJobs = useCallback(
    async (): Promise<
      { status: string; cancelled_count: number; project_id: string } | void
    > => {
      const pid = projectIdRef.current
      if (!pid) return
      const resp = await fetchFn(`${apiBaseUrl}/staging/projects/${pid}/jobs`, {
        method: "DELETE",
        credentials: "include",
      })
      if (!resp.ok) {
        let detail = ""
        try { detail = await resp.text() } catch { /* ignore */ }
        throw new Error(
          `Cancel-all failed: HTTP ${resp.status}${detail ? ` - ${detail}` : ""}`,
        )
      }
      try {
        return (await resp.json()) as {
          status: string
          cancelled_count: number
          project_id: string
        }
      } catch {
        // Server may return 202 with empty body in degenerate cases.
        return
      }
    },
    [apiBaseUrl, fetchFn],
  )

  // Issue 005: 5s wall-clock tick to refresh projectStaleness even when
  // no merges fire. A job sitting idle at 119s elapsed must transition to
  // hard-stale at 120s without waiting for a backend event. Plain
  // ``setInterval`` (no ``visibilitychange`` listener): hidden tabs
  // throttle to ≥1000ms but don't pause, which is acceptable here. We
  // store the wall-clock value (not a monotonic counter) so the memo
  // below stays a pure function of state — calling ``Date.now()``
  // during render trips the React 19 ``react-hooks/purity`` rule.
  useEffect(() => {
    if (!enabled || !projectId) return
    const t = setInterval(() => {
      setStalenessTick(Date.now())
    }, 5_000)
    return () => clearInterval(t)
  }, [enabled, projectId])

  // Issue 005: pre-compute project-level worst-staleness so the page
  // header doesn't have to invoke the detector itself. Keyed on jobsById
  // (every merge), lastBackendActivityByJobId (every accepted merge),
  // and stalenessTick (every 5s wall-clock tick — also seeds the
  // initial ``now`` on first render via lazy ``useState`` init).
  const projectStaleness = useMemo<StalenessState | null>(() => {
    const states = computeStaleness(
      jobs,
      lastBackendActivityByJobId,
      stalenessTick,
    )
    return deriveProjectWorstStaleness(states)
  }, [jobs, lastBackendActivityByJobId, stalenessTick])

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
    injectOptimisticJob,
    lastBackendActivityByJobId,
    projectStaleness,
    cancelAllProjectJobs,
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