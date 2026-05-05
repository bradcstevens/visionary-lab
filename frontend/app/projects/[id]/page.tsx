"use client"

import { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, Plus, RefreshCw, Loader2, Play, AlertTriangle, Trash2, MoreHorizontal, ChevronDown, Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { RoomGroup } from "@/components/staging/RoomGroup";
import { ProgressTracker } from "@/components/staging/ProgressTracker";
import { ImageLightbox, LightboxImage } from "@/components/staging/ImageLightbox";
import { ProjectSettingsSheet } from "@/components/staging/ProjectSettingsSheet";
import { EditPromptDialog } from "@/components/staging/EditPromptDialog";
import { CollapsiblePrompt } from "@/components/staging/CollapsiblePrompt";
import { getProject, deleteProject, resetProject, updateProject, updateRoomAddendum, enqueueProjectGeneration, EnqueueGenerationTimeoutError, EnqueueGenerationFailedError, StagingProject, Room, StagingStreamEvent, StagingStreamEventCallback, UpdateProjectBody } from "@/services/stagingApi";
import { sasTokenService } from "@/services/sas-token";
import { toast } from "sonner";
import { parseApiError } from "@/utils/error-utils";
import { getHeaderAction } from "@/utils/staging-header";
import { getRecoveryState, type RecoveryState } from "@/utils/recovery-state";
import { useActivityLog } from "@/context/activity-log-context";
import { useRetryQueue } from "@/hooks/useRetryQueue";
import { useGenerationFleet, type LostOp } from "@/hooks/useGenerationFleet";
import { useProjectJobs } from "@/context/jobs-context";
import type { ProjectJob } from "@/context/jobs-context";
import { ProjectGenerationBanner } from "@/components/staging/ProjectGenerationBanner";
import { EnqueueingBanner } from "@/components/staging/EnqueueingBanner";
import { StalenessSubline } from "@/components/staging/StalenessSubline";
import { getErrorKindCopy } from "@/lib/error-kind-copy";
import { deriveLogEntries } from "@/lib/activity-log-derivation";

export default function ProjectDetailPage() {
  const params = useParams();
  const router = useRouter();
  const projectId = params.id as string;

  const [project, setProject] = useState<StagingProject | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  // Issue 007 of projects-page-improvements PRD: the prior global
  // `isGenerating` flag is replaced by per-operation state tracked in the
  // `useGenerationFleet` hook (see below). Reads of `isAnyInFlight` /
  // `inFlightRooms` / `inFlightVariations` substitute for the prior reads
  // of `isGenerating` / `regeneratingVariationId`.
  const [generationError, setGenerationError] = useState<string | null>(null);
  // Issue 004 of active-and-queued-jobs-ux-redesign PRD: structured
  // enqueue-error state set when the producer returns a classified
  // ``EnqueueGenerationFailedError`` (issue 002 contract). Drives a
  // dedicated recovery banner arm (the friendly-copy one) — kept
  // separate from ``generationError`` so the legacy ``parseApiError``
  // pipeline isn't fed structured-payload strings it would mis-classify.
  const [enqueueErrorState, setEnqueueErrorState] = useState<{
    errorKind: string;
    userMessage: string;
    httpStatus: number;
    detail?: { type?: string; message?: string } | null;
  } | null>(null);
  // Issue 011 of project-generation-async-queue-cutover PRD: tracks the
  // 30-90s window between the user clicking Generate and the producer
  // endpoint returning 202 with a job_id (inline brief composition runs
  // here). The header CTA hides during this window so the user can't
  // double-enqueue and the legacy "Generating" banner doesn't compete
  // with the in-flight banner that's about to appear.
  const [isEnqueueing, setIsEnqueueing] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [showSettingsSheet, setShowSettingsSheet] = useState(false);
  // Issue 004 of projects-page-improvements PRD: per-variation Edit
  // Prompt opens an inline Dialog with the source variation's prior
  // adapted_prompt prefilled. The state captures BOTH the room and
  // variation index so the dialog can read the source variation's
  // metadata at render time (and falls back to project.prompt with a
  // notice when generation_metadata.adapted_prompt is missing).
  const [editPromptTarget, setEditPromptTarget] = useState<
    { roomId: string; variationIndex: number } | null
  >(null);
  const [lightboxImage, setLightboxImage] = useState<LightboxImage | null>(null);
  const reloadTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestLoadIdRef = useRef(0);

  // Issue 005 of active-and-queued-jobs-ux-redesign PRD: cancel-all wiring.
  //
  // ``cancellingState`` flips true synchronously on Cancel-button click and
  // stays true until ONE of:
  //  - the cancelled job flips terminal (SSE confirmation: success path), OR
  //  - the 10s ``cancelTimeoutRef`` fires (fallback path: server didn't confirm).
  //
  // ``dismissedAfterTimeoutFor`` is the jobId whose staleness UI is suppressed
  // AFTER a fallback-timeout fire. Without this, the staleness subline would
  // immediately re-mount because the worker hasn't terminated the job yet.
  // Cleared by an effect when the live non-terminal set no longer contains
  // that jobId (terminal arrival OR replaced by a new job). This implements
  // rubber-duck blocking finding #4.
  //
  // ``cancelInFlightRef`` doubles as the re-entrancy latch (per the issue 004
  // pattern): a non-null value means a cancel is in flight; the click handler
  // early-returns. Stores the captured jobId so the fallback-timeout knows
  // which job to suppress.
  const [cancellingState, setCancellingState] = useState(false);
  const [dismissedAfterTimeoutFor, setDismissedAfterTimeoutFor] = useState<
    string | null
  >(null);
  const cancelInFlightRef = useRef<{
    jobId: string;
    timer: ReturnType<typeof setTimeout>;
  } | null>(null);

  const activityLog = useActivityLog();

  useEffect(() => {
    activityLog.clear();
    return () => activityLog.clear();
  }, []);

  // Abort all active streams and pending reloads on unmount. The
  // useGenerationFleet hook's own unmount effect aborts all active streams
  // it is tracking; the reload timer is page-local so we clear it here.
  useEffect(() => {
    return () => {
      if (reloadTimerRef.current) clearTimeout(reloadTimerRef.current);
    };
  }, []);

  useEffect(() => {
    if (projectId) {
      loadProject();
    }
  }, [projectId]);

  // Sync the open lightbox with the freshly-loaded project after each reload.
  // Drives off `lightboxImage` (the variation actually on screen), not the
  // variation we originally opened on, so arrow-key navigation during a regen
  // doesn't desync the spinner / image. See issue 005 of the
  // single-variation-regeneration PRD.
  useEffect(() => {
    if (!project) return;
    setLightboxImage((prev) => {
      if (!prev) return prev;
      const room = project.rooms.find((r) => r.id === prev.roomId);
      if (!room) return prev;
      const variation = room.variations[prev.variationIndex];
      const refreshedUrl =
        variation?.status === 'completed' && variation.image_url
          ? variation.image_url
          : prev.url;
      return {
        ...prev,
        variations: room.variations,
        url: refreshedUrl,
      };
    });
  }, [project]);

  const resolveImageUrls = async (data: StagingProject) => {
    try {
      const tokens = await sasTokenService.getTokens();
      for (const room of data.rooms) {
        if (room.original_image_url && !room.original_image_url.includes('?')) {
          room.original_image_url = `${room.original_image_url}?${tokens.imageSasToken}`;
        }
        for (const variation of room.variations) {
          if (variation.image_url && !variation.image_url.includes('?')) {
            variation.image_url = `${variation.image_url}?${tokens.imageSasToken}`;
          }
        }
      }
    } catch (sasError) {
      console.warn('Failed to get SAS tokens, images may not load:', sasError);
      toast.warning('Image previews may not load — storage access token unavailable', {
        id: 'sas-token-warning',
        duration: 8000,
      });
    }
  };

  const loadProject = useCallback(async () => {
    const loadId = ++latestLoadIdRef.current;
    try {
      setIsLoading(true);
      const data = await getProject(projectId);
      // Discard stale responses from overlapping fetches
      if (loadId !== latestLoadIdRef.current) return;
      await resolveImageUrls(data);
      if (loadId !== latestLoadIdRef.current) return;
      setProject(data);
    } catch (error) {
      if (loadId !== latestLoadIdRef.current) return;
      console.error('Failed to load project:', error);
      toast.error('Failed to load project');
      router.push('/projects');
    } finally {
      if (loadId === latestLoadIdRef.current) {
        setIsLoading(false);
      }
    }
  }, [projectId]);

  /** Debounced reload — coalesces rapid SSE events into a single fetch. */
  const debouncedReload = useCallback(() => {
    if (reloadTimerRef.current) clearTimeout(reloadTimerRef.current);
    reloadTimerRef.current = setTimeout(() => {
      reloadTimerRef.current = null;
      loadProject();
    }, 500);
  }, [loadProject]);

  // Computed state (must be before callbacks that reference these values)
  const allPending = project?.rooms.every(r => r.status === 'pending') ?? false;
  const hasFailed = project?.rooms.some(r => r.status === 'failed' || r.variations.some(v => v.status === 'failed')) ?? false;
  const totalVariations = project?.rooms.reduce((sum, r) => sum + r.variations.length, 0) ?? 0;
  const completedVariations = project?.rooms.reduce((sum, r) => sum + r.variations.filter(v => v.status === 'completed').length, 0) ?? 0;

  // ``handleStreamEvent`` is defined AFTER ``useRetryQueue`` so its
  // ``'error'`` case can call ``retryQueue.clear()`` directly. The
  // prior placement (above ``handleVariationClick``) was historical.
  // Issue 004 of failed-variation-retry-queue PRD: when the global
  // generation SSE stream itself terminates with an ``'error'`` event,
  // the queued retries are dropped so we don't immediately fire N
  // requests against the same broken upstream after the banner
  // appears. The Retry button on the failed variation is restored so
  // the user can re-trigger manually after acknowledging the error.
  // See PRD §"Page integration" (last sentence about the ``'error'``
  // case) and §"Testing Decisions" → scenario 4.

  // Issue 007 of projects-page-improvements PRD: per-operation generation
  // state replaces the prior global ``isGenerating`` flag. Each
  // ``start*`` method on the hook opens its own SSE stream, registers the
  // operation in the appropriate in-flight set on click (UI ack is
  // immediate per AC #2), attaches a 120-second silence watchdog per
  // stream (resets on every event of any type), and removes the
  // operation on terminal events / abort / unmount / watchdog fire.
  // ``onStreamLost`` is called when the watchdog fires; the hook also
  // exposes ``lostOps`` so the page can render a per-room (or per-project)
  // banner with a Retry action that replays the EXACT lost operation.
  //
  // Issue 008: ``onOpStart`` / ``onOpProgress`` / ``onOpEnd`` mirror the
  // fleet's per-stream lifecycle into the activity-feed context's In Flight
  // surface so the panel and the buttons cannot drift apart. Labels are
  // derived from the latest ``project`` via a ref (the callback identity
  // stays stable; the ref read picks up the freshest project state at
  // call time, which is the click moment for ``onOpStart``).
  const projectRef = useRef<StagingProject | null>(null);
  useEffect(() => {
    projectRef.current = project;
  }, [project]);

  const deriveOpLabel = useCallback(
    (op: {
      kind: 'project' | 'room' | 'variation' | 'edit-prompt';
      projectId: string;
      roomId?: string;
      variationId?: string;
    }): string => {
      const proj = projectRef.current;
      if (op.kind === 'project') {
        return proj?.name ? `Project: ${proj.name}` : 'Project generation';
      }
      const room = op.roomId
        ? proj?.rooms.find((r) => r.id === op.roomId) ?? null
        : null;
      const roomLabel = room?.label ?? (op.roomId ? `Room ${op.roomId.slice(0, 6)}` : 'Room');
      if (op.kind === 'room') {
        return roomLabel;
      }
      const variationIndex = room && op.variationId
        ? room.variations.findIndex((v) => v.id === op.variationId)
        : -1;
      const variationNumber = variationIndex >= 0 ? variationIndex + 1 : '?';
      if (op.kind === 'variation') {
        return `Variation ${variationNumber} in ${roomLabel}`;
      }
      return `Edit prompt: variation ${variationNumber} in ${roomLabel}`;
    },
    [],
  );

  const fleet = useGenerationFleet({
    onStreamLost: useCallback(
      (lostOp: LostOp) => {
        const scopeLabel =
          lostOp.kind === 'project'
            ? 'project generation'
            : lostOp.kind === 'room'
              ? `room ${lostOp.roomId.slice(0, 8)}`
              : `variation in room ${lostOp.roomId.slice(0, 8)}`;
        const toastBody =
          lostOp.kind === 'project'
            ? 'Stream lost — click Retry on the project banner above the rooms list.'
            : 'Stream lost — click Retry on the affected room banner.';
        activityLog.log({
          level: 'warn',
          icon: '⚠',
          message: `Stream lost for ${scopeLabel}`,
          detail: 'No SSE events arrived for 2 minutes — click Retry on the banner to try again.',
        });
        toast.warning(toastBody);
      },
      [activityLog],
    ),
    onOpStart: useCallback(
      (op: {
        id: string;
        kind: 'project' | 'room' | 'variation' | 'edit-prompt';
        projectId: string;
        roomId?: string;
        variationId?: string;
      }) => {
        activityLog.startOp({
          id: op.id,
          kind: op.kind,
          label: deriveOpLabel(op),
        });
      },
      [activityLog, deriveOpLabel],
    ),
    onOpProgress: useCallback(
      (opId: string) => activityLog.markOpStarted(opId),
      [activityLog],
    ),
    onOpEnd: useCallback(
      (opId: string) => activityLog.endOp(opId),
      [activityLog],
    ),
  });
  const { isAnyInFlight, inFlightProject, inFlightRooms, inFlightVariations } = fleet;

  // Issue 009: live job state for the per-image overlay bars and the
  // per-project header aggregate bar. Disabled until projectId is known.
  const projectJobs = useProjectJobs(projectId, { enabled: !!projectId });
  // Issue 011: in-flight project-generation slice (single-tenant: at most
  // one generate_project job in non-terminal status at a time per project).
  // The page mounts the ProjectGenerationBanner whenever this is non-null.
  const inFlightProjectGeneration = projectJobs.inFlightProjectGeneration;
  const cancelProjectGeneration = projectJobs.cancelProjectGeneration;
  // Issue 004 of active-and-queued-jobs-ux-redesign PRD: optimistic-job
  // injection helper (jobs-context API). Called on a 202 response with
  // the real job_id so the in-flight banner / room tile renders before
  // the SSE seed catches up; the SSE-delivered doc supersedes the
  // optimistic placeholder via the merge rule (epoch-zero updated_at).
  const injectOptimisticJob = projectJobs.injectOptimisticJob;
  // Composite busy flag: true when ANY generation flow is mutating the
  // project. The fleet half (isAnyInFlight) covers the legacy variation /
  // room SSE streams; isProjectGenerationBusy covers the new async path
  // (the 30-90s enqueue window AND the worker-running window). Per the
  // single-banner / single-action contract, every site that previously
  // gated on isAnyInFlight must use this composite so concurrent writes
  // from the worker and a stray click can't race.
  const isProjectGenerationBusy = isEnqueueing || inFlightProjectGeneration !== null;
  const isAnyGenerationBusy = isAnyInFlight || isProjectGenerationBusy;
  // Map each variation to its most-recent (highest-revision) job so the
  // overlay reflects the active attempt rather than a stale prior one.
  const jobsByVariationId = (() => {
    const map = new Map<string, import("@/context/jobs-context").ProjectJob>();
    for (const j of projectJobs.jobs) {
      const prev = map.get(j.variation_id);
      if (!prev || (j.revision ?? 0) >= (prev.revision ?? 0)) {
        map.set(j.variation_id, j);
      }
    }
    return map;
  })();

  // Issue 004 of active-and-queued-jobs-ux-redesign PRD: activity-log
  // derivation effect. Compares the previous jobs-by-id snapshot to the
  // current one and emits structured log entries via deriveLogEntries
  // (pure module). The bootstrap suppression guard (``bootstrappedRef``)
  // ensures the first non-empty snapshot — which arrives via the SSE
  // seed event mid-page load — does NOT backfill log entries for jobs
  // that have been running since before the user opened the page.
  // Without this, a refresh during an active run would emit "queued"
  // entries for every existing job. The ref flips to false once the
  // first non-empty snapshot is processed; subsequent updates emit
  // log entries normally.
  const prevJobsByIdRef = useRef<Record<string, ProjectJob>>({});
  const bootstrappedRef = useRef(true);
  useEffect(() => {
    const current = projectJobs.jobsById;
    const isBootstrap = bootstrappedRef.current;
    const entries = deriveLogEntries(prevJobsByIdRef.current, current, {
      bootstrap: isBootstrap,
    });
    for (const entry of entries) {
      activityLog.log(entry);
    }
    prevJobsByIdRef.current = current;
    if (isBootstrap && Object.keys(current).length > 0) {
      bootstrappedRef.current = false;
    }
  }, [projectJobs.jobsById, activityLog]);

  // Issue 005 of active-and-queued-jobs-ux-redesign PRD: cancel-all
  // wiring effect.
  //
  // Watches ``projectJobs.activeJobs`` (the canonical "non-terminal" set)
  // for two state transitions:
  //
  //   1. Cancellation confirmed: while a cancel is in flight
  //      (``cancelInFlightRef.current`` non-null), if the targeted jobId
  //      is no longer present in the live non-terminal generate_project
  //      set, the worker has flipped it terminal. Clear the timer, the
  //      ref, the cancellingState, and any stale ``dismissedAfterTimeout``
  //      suppression. This is the success-path reconciliation.
  //
  //   2. Suppression auto-clear: if ``dismissedAfterTimeoutFor`` is set
  //      AND that jobId is no longer in the non-terminal set, clear the
  //      suppression. This handles the post-timeout case where the
  //      worker eventually catches up (the staleness UI is now safe to
  //      surface again should a NEW job become stale).
  useEffect(() => {
    const liveGenerateProjectJobIds = new Set(
      (projectJobs.activeJobs ?? [])
        .filter((j) => j.kind === "generate_project")
        .map((j) => j.id),
    );

    if (
      cancelInFlightRef.current &&
      !liveGenerateProjectJobIds.has(cancelInFlightRef.current.jobId)
    ) {
      clearTimeout(cancelInFlightRef.current.timer);
      cancelInFlightRef.current = null;
      setCancellingState(false);
      setDismissedAfterTimeoutFor(null);
    }

    if (
      dismissedAfterTimeoutFor &&
      !liveGenerateProjectJobIds.has(dismissedAfterTimeoutFor)
    ) {
      // Guarded reconciliation against the refreshed activeJobs list —
      // only fires when the previously-suppressed job has actually left
      // the non-terminal set, which is the external state-change this
      // effect synchronizes to.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setDismissedAfterTimeoutFor(null);
    }
  }, [projectJobs.activeJobs, dismissedAfterTimeoutFor]);

  // Issue 005: cleanup the cancel-fallback timer on unmount so a
  // setTimeout callback firing post-unmount doesn't try to setState
  // on a torn-down tree (act-warning correctness).
  useEffect(() => {
    return () => {
      if (cancelInFlightRef.current) {
        clearTimeout(cancelInFlightRef.current.timer);
        cancelInFlightRef.current = null;
      }
    };
  }, []);

  // Issue 005 of active-and-queued-jobs-ux-redesign PRD: cancel-all
  // click handler.
  //
  // Imperative re-entrancy guard: ``cancelInFlightRef`` non-null means a
  // cancel is in progress; double-click is a no-op (see issue 004
  // for the same pattern on Retry).
  //
  // Captures the worst-staleness jobId (NOT inFlightProjectGeneration's
  // jobId; the tier-3 multi-pending case has inFlightProjectGeneration
  // === null but projectStaleness still surfaces a worst-state job
  // — rubber-duck blocking finding #3).
  //
  // Schedules a 10s fallback timer: if the worker doesn't confirm
  // (active set still contains the jobId), set ``dismissedAfterTimeoutFor``
  // so the staleness UI hides AND fire a fallback toast so the user
  // gets feedback. The cancel is still in flight server-side; the
  // suppression effect above clears ``dismissedAfterTimeoutFor`` if the
  // job eventually does flip terminal.
  const handleCancelAllProjectJobs = useCallback(() => {
    if (cancelInFlightRef.current) {
      return;
    }
    const jobId = projectJobs.projectStaleness?.jobId;
    if (!jobId) {
      return;
    }

    const fallbackTimer = setTimeout(() => {
      cancelInFlightRef.current = null;
      setCancellingState(false);
      setDismissedAfterTimeoutFor(jobId);
      toast.error(
        "Cancel request didn't confirm in time. Try refreshing the page if jobs remain stuck.",
      );
    }, 10_000);

    cancelInFlightRef.current = { jobId, timer: fallbackTimer };
    setCancellingState(true);

    projectJobs
      .cancelAllProjectJobs()
      .then((res) => {
        const cancelledCount =
          res && typeof res === "object" && "cancelled_count" in res
            ? (res as { cancelled_count: number }).cancelled_count
            : 0;
        toast.success(
          cancelledCount === 0
            ? "No queued jobs to cancel."
            : cancelledCount === 1
              ? "Cancelled 1 queued job."
              : `Cancelled ${cancelledCount} queued jobs.`,
        );
      })
      .catch((err: unknown) => {
        if (cancelInFlightRef.current) {
          clearTimeout(cancelInFlightRef.current.timer);
          cancelInFlightRef.current = null;
        }
        setCancellingState(false);
        const message = err instanceof Error ? err.message : String(err);
        toast.error(`Couldn't cancel jobs: ${message}`);
      });
  }, [projectJobs]);

  const handleVariationClick = (room: Room, variationIndex: number) => {
    const variation = room.variations[variationIndex];
    if (variation.status === 'completed' && variation.image_url) {
      setLightboxImage({
        url: variation.image_url,
        mdUrl: variation.md_url,
        roomId: room.id,
        roomLabel: room.label,
        variationIndex,
        variations: room.variations,
      });
    }
  };

  const handleLightboxNavigate = (variationIndex: number) => {
    if (!lightboxImage) return;
    const variation = lightboxImage.variations[variationIndex];
    if (variation?.status === 'completed' && variation.image_url) {
      setLightboxImage((prev) =>
        prev ? { ...prev, variationIndex, url: variation.image_url!, mdUrl: variation.md_url } : null
      );
    }
  };

  // Issue 007: per-variation regen now flows through the fleet hook so
  // the in-flight Set populates synchronously, the watchdog attaches per
  // stream, and unmount/abort cleanup is centralized. The hook's startVariation
  // is idempotent on the same variationId, so we don't need the prior
  // `if (regeneratingVariationId)` guard at the call site (the hook silently
  // dedupes). The hook also aborts other variations in the same room when
  // the user supersedes via Regenerate Room (preserves retry-queue scenario 3).
  const handleRegenerateVariation = useCallback((room: Room, variationIndex: number, strategy: 'retry' | 'fresh') => {
    const variation = room.variations[variationIndex];
    if (!variation) return;

    // Issue 006 of single-variation-regeneration PRD: closure-scoped flag —
    // each ``handleRegenerateVariation`` invocation gets its own. Set on
    // ``variation_fallback``; read on ``variation_completed`` to compute
    // the effective strategy label "(fresh — no prior prompt)".
    let fellBackToFresh = false;

    fleet.startVariation(projectId, room.id, variation.id, strategy, (event) => {
      switch (event.type) {
        case 'variation_completed': {
          // Issue 006 success activity-log copy: include the strategy
          // label and (when present) a 60-char snippet of the
          // ``adapted_prompt`` that produced the result.
          const completed = event as {
            type: 'variation_completed';
            model?: string;
            tokens_used?: number;
            elapsed_ms?: number;
            adapted_prompt?: string;
          };
          const label = fellBackToFresh
            ? '(fresh — no prior prompt)'
            : strategy === 'retry'
              ? '(retry)'
              : '(fresh)';
          const adaptedPrompt = completed.adapted_prompt;
          const snippet =
            typeof adaptedPrompt === 'string' && adaptedPrompt.length > 0
              ? adaptedPrompt.slice(0, 60) + (adaptedPrompt.length > 60 ? '…' : '')
              : null;
          activityLog.log({
            level: 'success',
            icon: '✓',
            message: `Variation ${variationIndex + 1} regenerated ${label}`,
            detail: [
              completed.model,
              completed.tokens_used ? `${Number(completed.tokens_used).toLocaleString()} tokens` : null,
              completed.elapsed_ms ? `${(completed.elapsed_ms / 1000).toFixed(1)}s` : null,
              snippet,
            ].filter(Boolean).join(' · ') || undefined,
          });
          break;
        }
        case 'variation_fallback':
          // Issue 004 of single-variation-regeneration PRD: backend
          // emits this when ``strategy=retry`` is requested but the
          // variation has no prior ``adapted_prompt`` recorded. The
          // regen continues normally; this is a one-line user
          // notification that the retry silently became a fresh
          // generation. Single info toast + activity-log entry. The
          // hook keeps the variation in the in-flight set until a
          // terminal event lands.
          fellBackToFresh = true;
          activityLog.log({
            level: 'info',
            icon: 'ℹ',
            message: `Variation ${variationIndex + 1}: no previous prompt found`,
            detail: 'Generating a fresh take instead.',
          });
          toast.info('No previous prompt found — generating a fresh take instead.');
          break;
        case 'variation_failed':
          activityLog.log({
            level: 'error',
            icon: '✕',
            message: `Variation ${variationIndex + 1} regeneration failed`,
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            detail: (event as any).error || 'Unknown error',
          });
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          toast.error(`Regeneration failed: ${(event as any).error || 'Unknown error'}`);
          break;
        case 'project_completed':
        case 'stream_ended':
          // Issue 006: drop the spurious ``toast.success('Variation
          // regenerated!')``. The success activity-log entry already
          // landed on ``variation_completed``; an additional toast
          // here would duplicate that signal on the happy path. Both
          // ``project_completed`` and ``stream_ended`` resolve to the
          // same cleanup — reload the project to surface the new
          // variation. The hook clears the variation from inFlightVariations
          // automatically.
          loadProject();
          break;
        case 'error':
          toast.error(event.error || 'Regeneration failed');
          loadProject();
          break;
      }
    });
  }, [fleet, projectId, activityLog, loadProject]);

  const handleLightboxRegenerate = useCallback((strategy: 'retry' | 'fresh') => {
    if (!lightboxImage || !project) return;
    const room = project.rooms.find((r) => r.id === lightboxImage.roomId);
    if (!room) return;
    handleRegenerateVariation(room, lightboxImage.variationIndex, strategy);
  }, [lightboxImage, project, handleRegenerateVariation]);

  // Issue 004 of projects-page-improvements PRD: open the Edit Prompt
  // dialog. Just records the target — submission goes through
  // ``handleEditPromptSubmit`` below. Opening is allowed mid-
  // generation (the dialog itself disables Generate via the dialog's
  // ``isBlocked`` prop) so the user can preview / draft prompts
  // without waiting for the current run to finish.
  const handleEditPromptVariation = useCallback((room: Room, variationIndex: number) => {
    setEditPromptTarget({ roomId: room.id, variationIndex });
  }, []);

  const handleEditPromptSubmit = useCallback(async (adaptedPrompt: string): Promise<void> => {
    if (!editPromptTarget || !project) return;
    const room = project.rooms.find((r) => r.id === editPromptTarget.roomId);
    if (!room) {
      throw new Error('Room not found');
    }
    const variation = room.variations[editPromptTarget.variationIndex];
    if (!variation) {
      throw new Error('Variation not found');
    }
    // Issue 007 per-variation gating (rubber-duck-flagged): the prior
    // global isAnyInFlight gate also blocked Edit Prompt for room B
    // while room A streamed. The new fleet model lets variation-level
    // ops proceed concurrently with unrelated room/variation ops, so
    // the gate now narrows to "this variation, room, OR project is in
    // flight". Same predicate as the per-variation menu-item gate in
    // RoomGroup so the check is consistent end-to-end.
    if (
      inFlightProject ||
      inFlightRooms.has(room.id) ||
      inFlightVariations.has(variation.id)
    ) {
      toast.error('Wait for this room to finish before editing prompts.');
      throw new Error('Generation in flight');
    }

    return new Promise<void>((resolve, reject) => {
      const variationLabel = `Variation ${editPromptTarget.variationIndex + 1}`;
      // The hook marks the source variation as in flight via inFlightVariations
      // (the backend appends a NEW variation but until that surfaces in the
      // UI on loadProject(), the source variation is the user-visible "thing
      // being edited" — same proxy as before issue 007).
      fleet.editPrompt(projectId, room.id, variation.id, adaptedPrompt, (event) => {
        switch (event.type) {
          case 'variation_completed': {
            const completed = event as {
              type: 'variation_completed';
              model?: string;
              tokens_used?: number;
              elapsed_ms?: number;
              adapted_prompt?: string;
            };
            const adapted = completed.adapted_prompt;
            const snippet =
              typeof adapted === 'string' && adapted.length > 0
                ? adapted.slice(0, 60) + (adapted.length > 60 ? '…' : '')
                : null;
            activityLog.log({
              level: 'success',
              icon: '✓',
              message: `${variationLabel}: new variation appended from edited prompt`,
              detail: [
                completed.model,
                completed.tokens_used ? `${Number(completed.tokens_used).toLocaleString()} tokens` : null,
                completed.elapsed_ms ? `${(completed.elapsed_ms / 1000).toFixed(1)}s` : null,
                snippet,
              ].filter(Boolean).join(' · ') || undefined,
            });
            break;
          }
          case 'variation_failed':
            activityLog.log({
              level: 'error',
              icon: '✕',
              message: `${variationLabel}: edit-prompt generation failed`,
              detail: (event as { error?: string }).error || 'Unknown error',
            });
            toast.error(`Edit Prompt failed: ${(event as { error?: string }).error || 'Unknown error'}`);
            break;
          case 'project_completed':
          case 'stream_ended':
            setEditPromptTarget(null);
            loadProject();
            resolve();
            break;
          case 'error':
            toast.error(event.error || 'Edit Prompt failed');
            loadProject();
            reject(new Error(event.error || 'Edit Prompt failed'));
            break;
        }
      });
    });
  }, [editPromptTarget, project, projectId, inFlightProject, inFlightRooms, inFlightVariations, fleet, activityLog, loadProject]);

  // Issue 002 of failed-variation-retry-queue PRD: per-page in-memory
  // retry queue. Failed-variation Retry clicks during in-flight
  // generation are routed through this hook's `enqueue`. Queued retries
  // drain serially via `handleRegenerateVariation` once both
  // `isGenerating` and `regeneratingVariationId` go idle.
  const onDropQueuedRetry = useCallback(
    (variationId: string) => {
      // Drop rule: the variation no longer exists in the current
      // project OR its status is no longer 'failed' at drain time.
      // No toast — silent UX per PRD; activity log entry only.
      activityLog.log({
        level: 'info',
        icon: 'ℹ',
        message: 'Queued retry skipped',
        detail: `Variation ${variationId.slice(0, 8)} was no longer in a failed state when the queue drained.`,
      });
    },
    [activityLog],
  );
  const retryQueue = useRetryQueue({
    project,
    // Issue 007: the retry queue's "is anything in flight?" check is now
    // satisfied by the fleet's derived `isAnyInFlight` boolean, replacing
    // the prior (isGenerating, regeneratingVariationId) pair. The retry
    // queue's interface intentionally stays as-is — the second field is
    // wired to `null` so its drain effect's `regeneratingVariationId !==
    // null` check is benign. The first field carries the unified busy
    // signal. This preserves the four existing retry-queue scenarios
    // (queue, dedup, supersede on Regenerate Room, drop on global error)
    // verbatim against the new fleet-driven state.
    //
    // Issue 011: extended to ``isAnyGenerationBusy`` so a project-scope
    // job (enqueueing OR worker-running) ALSO blocks retry-queue drain.
    // Without this, a queued failed-variation retry could drain mid-
    // project-job and race the worker for the same variation.
    isGenerating: isAnyGenerationBusy,
    regeneratingVariationId: null,
    onDispatch: handleRegenerateVariation,
    onDrop: onDropQueuedRetry,
  });

  const handleStreamEvent = useCallback((event: StagingStreamEvent) => {
    switch (event.type) {
      case 'room_started':
        activityLog.log({
          level: 'info',
          icon: '▶',
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          message: `Starting generation for "${(event as any).label ?? 'room'}"`,
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          detail: `Room ${(event as any).room_id?.slice(0, 8) ?? ''}`,
        });
        debouncedReload();
        break;
      case 'variation_completed':
        activityLog.log({
          level: 'success',
          icon: '✓',
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          message: `Variation ${((event as any).variation_index ?? 0) + 1} saved`,
          detail: [
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (event as any).model,
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (event as any).tokens_used ? `${Number((event as any).tokens_used).toLocaleString()} tokens` : null,
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (event as any).elapsed_ms ? `${((event as any).elapsed_ms / 1000).toFixed(1)}s` : null,
          ].filter(Boolean).join(' · ') || undefined,
        });
        debouncedReload();
        break;
      case 'variation_failed':
        activityLog.log({
          level: 'error',
          icon: '✕',
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          message: `Variation ${((event as any).variation_index ?? 0) + 1} failed`,
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          detail: (event as any).error || 'Unknown error',
        });
        debouncedReload();
        break;
      case 'room_uploaded':
        debouncedReload();
        break;
      case 'room_completed':
        activityLog.log({
          level: 'success',
          icon: '✓',
          message: 'Room complete',
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          detail: `Room ${(event as any).room_id?.slice(0, 8) ?? ''}`,
        });
        debouncedReload();
        break;
      case 'room_failed':
        activityLog.log({
          level: 'error',
          icon: '✕',
          message: 'Room failed',
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          detail: (event as any).error || 'Unknown error',
        });
        debouncedReload();
        break;
      case 'project_completed':
        // Cancel any pending debounced reload
        if (reloadTimerRef.current) clearTimeout(reloadTimerRef.current);
        activityLog.log({
          level: 'success',
          icon: '🎉',
          message: 'Generation complete!',
        });
        // The fleet hook clears the in-flight flag on terminal events; the
        // page just resets its own page-level error state.
        setGenerationError(null);
        toast.success('Generation completed!');
        loadProject();
        break;
      case 'stream_ended':
        // Fallback: stream closed without a terminal event — reconcile state
        if (reloadTimerRef.current) clearTimeout(reloadTimerRef.current);
        loadProject();
        break;
      case 'error': {
        if (reloadTimerRef.current) clearTimeout(reloadTimerRef.current);
        activityLog.log({
          level: 'error',
          icon: '✕',
          message: 'Generation error',
          detail: event.error || 'Unknown error',
        });
        // Issue 001 of projects-page-stalled-stream-error-cleanup PRD:
        // only real server-sent error events flip the destructive
        // generationError banner. Watchdog-synthesized errors carry
        // ``synthetic: true`` and are surfaced via the amber lost-op
        // banner + activity log + toast above instead, so we don't
        // double-paint a red banner for a transient stream-lost
        // condition that the user can recover from with one click.
        if (!event.synthetic) {
          setGenerationError(event.error || 'Generation failed');
        }
        toast.error(event.error || 'Generation failed');
        loadProject();
        // Issue 004 of failed-variation-retry-queue PRD: drop queued
        // retries on global stream error so we don't immediately fire N
        // more requests against the same broken upstream after the banner
        // appears. ``clear()`` returns the count of dropped entries so
        // the activity-log message is truthful without racing against the
        // rendered ``queuedIds`` Set (which is populated via setState
        // and may be one render behind ``queueRef``). When the queue was
        // already empty the call is a no-op and no log entry fires.
        const droppedCount = retryQueue.clear();
        if (droppedCount > 0) {
          activityLog.log({
            level: 'info',
            icon: 'ℹ',
            message: `Queued retries cleared (${droppedCount})`,
            detail: 'Generation stream errored; retry manually after acknowledging the error.',
          });
        }
        break;
      }
    }
  }, [retryQueue, activityLog, debouncedReload, loadProject]);

  // Issue 003 of failed-variation-retry-queue PRD + issue 007 of projects-
  // page-improvements PRD + issue 011 of project-generation-async-queue-
  // cutover PRD: the three "larger regen action" entry points each call
  // ``retryQueue.clear()`` BEFORE delegating. The supersede semantic is
  // preserved: when the user triggers a regen action that subsumes
  // individual failed-variation retries, the queued per-variation retries
  // are silently cleared.
  //
  // Post-cutover (issue 011): startGeneration enqueues a project-scope
  // job via the async producer endpoint instead of opening a long-lived
  // SSE stream. Inline brief composition runs on the producer side
  // (30-90s blocking) - the user sees `isEnqueueing=true` during this
  // window, then the worker begins processing and the change feed
  // delivers a job event that the inFlightProjectGeneration slice
  // consumes. The legacy fleet.startProject path is no longer wired here
  // (variation/room regen still use the fleet).
  //
  // ``regenerateAll: false`` is intentional and matches the established
  // "Generate Remaining (N)" semantic - process pending + failed only,
  // leave completed rooms alone. The destructive `regenerate_all=true`
  // backend path is reserved for a distinct future affordance (per PRD
  // issue 006: it pre-cancels in-flight variation jobs and reruns
  // everything from scratch).
  //
  // Issue 004 of active-and-queued-jobs-ux-redesign PRD rewrite:
  //
  //   1. Click → ``isEnqueueing=true`` immediately mounts the
  //      ``EnqueueingBanner`` synchronously. NO upfront activity-log
  //      entry — the branch decides what to log.
  //   2. Helper resolves with ``{ job_id, already_in_flight }``:
  //      - 202 (already_in_flight=false) → ``injectOptimisticJob`` so
  //        the inFlightProjectGeneration slice has a doc to surface
  //        BEFORE the SSE seed catches up; activity log "Submitted to
  //        queue".
  //      - 200 (already_in_flight=true) → toast "Generation already
  //        in progress" + scroll into view; activity log info entry.
  //        No optimistic injection (the existing job is already in
  //        the SSE stream).
  //   3. Helper rejects with ``EnqueueGenerationFailedError`` →
  //      structured ``enqueueErrorState`` so the dedicated recovery
  //      banner arm renders friendly copy + technical-details
  //      collapsible.
  //   4. Helper rejects with ``EnqueueGenerationTimeoutError`` →
  //      toast (preserves the issue 011 contract).
  //   5. Other errors → legacy ``generationError`` path (parsed via
  //      parseApiError) so anything we can't classify still produces
  //      a user-visible recovery banner.
  //
  // Re-entrancy guard (rubber-duck issue 004): two fast clicks on
  // Retry can call startGeneration twice in the same React tick,
  // before isAnyGenerationBusy has flushed via re-render. The
  // closure-captured boolean is stale on the second call, so the
  // ``if (isAnyGenerationBusy) return`` guard would let both calls
  // race to the producer. The imperative ref below flips
  // synchronously and is read at call time so duplicate calls
  // short-circuit even when no re-render has occurred.
  const startGenerationInFlightRef = useRef(false);
  const startGeneration = useCallback(async () => {
    retryQueue.clear();
    if (isAnyGenerationBusy) return;
    if (startGenerationInFlightRef.current) return;
    startGenerationInFlightRef.current = true;
    setGenerationError(null);
    setEnqueueErrorState(null);
    setIsEnqueueing(true);
    try {
      const result = await enqueueProjectGeneration(projectId, { regenerateAll: false });
      if (result.already_in_flight) {
        // 200 dedupe path — a generate_project job is already in
        // flight server-side. Surface a small info toast + scroll
        // into the room grid. The SSE seed will deliver the
        // existing job's doc within a moment.
        toast.info("Generation already in progress");
        activityLog.log({
          level: 'info',
          icon: 'ℹ',
          message: 'Generation already in progress',
          detail: `Server reused existing job ${result.job_id.slice(0, 8)}.`,
        });
        // Best-effort scroll into the rooms grid so the user sees
        // their work. The selector matches the project rooms region
        // mounted lower on the page.
        if (typeof document !== 'undefined') {
          const target = document.querySelector('[data-testid="project-rooms"]');
          if (target && typeof target.scrollIntoView === 'function') {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        }
      } else {
        // 202 happy path — fresh enqueue. Inject an optimistic job
        // with the real job_id so the in-flight banner / room tile
        // surfaces immediately. Real SSE doc supersedes via merge.
        //
        // Issue 004: NO manual activity-log entry here. The
        // ``deriveLogEntries`` effect (line ~360) sees the optimistic
        // job appear in jobsById with phase=queued and emits a
        // single "Queued" entry. Emitting both "Submitted to queue"
        // here AND letting derivation emit "Queued" produces two
        // entries for the same logical event. Derivation is the
        // single source of truth for job-driven log entries.
        const optimistic: ProjectJob = {
          id: result.job_id,
          project_id: projectId,
          room_id: '__project__',
          variation_id: '__project__',
          revision: 0,
          kind: 'generate_project',
          status: 'pending',
          progress: 0,
          phase: 'queued',
          updated_at: '1970-01-01T00:00:00Z',
        };
        injectOptimisticJob(optimistic);
      }
    } catch (err: unknown) {
      // 180s client-side abort surfaces as a user-visible toast rather
      // than a frozen spinner (PRD AC: "user-visible error message
      // 'Couldn't start generation, please try again'").
      if (err instanceof EnqueueGenerationTimeoutError) {
        toast.error("Couldn't start generation, please try again.");
      } else if (err instanceof EnqueueGenerationFailedError) {
        // Issue 004: structured backend failure. Set the dedicated
        // ``enqueueErrorState`` so the friendly recovery banner arm
        // renders kind-specific copy via getErrorKindCopy. NO toast —
        // the banner is the recovery surface, not a fire-and-forget
        // notification.
        setEnqueueErrorState({
          errorKind: err.errorKind,
          userMessage: err.userMessage,
          httpStatus: err.httpStatus,
          detail: err.detail ?? null,
        });
        const copy = getErrorKindCopy(err.errorKind);
        activityLog.log({
          level: 'error',
          icon: '⚠',
          message: copy.friendlyTitle,
          detail: err.detail?.message
            ? `${err.detail.type ?? 'Error'}: ${err.detail.message}`
            : err.userMessage,
        });
      } else {
        // Non-classified errors: surface as a destructive recovery
        // banner via setGenerationError. The recovery classifier maps
        // this to the existing 'error' arm and renders the prior
        // collapsible "Full error" detail block.
        const message = err instanceof Error ? err.message : String(err);
        setGenerationError(message);
        toast.error("Couldn't start generation, please try again.");
      }
    } finally {
      setIsEnqueueing(false);
      startGenerationInFlightRef.current = false;
    }
  }, [retryQueue, activityLog, isAnyGenerationBusy, projectId, injectOptimisticJob]);

  const handleRegenerateRoom = useCallback((room: Room) => {
    retryQueue.clear();
    // Per the new disabling rules (issue 007), the room-level Regenerate
    // is disabled iff `inFlightProject || inFlightRooms.has(room.id)`.
    // Issue 011 adds `isProjectGenerationBusy` to the gate so a click
    // can't race a worker-running project-scope job.
    if (isProjectGenerationBusy || inFlightProject || inFlightRooms.has(room.id)) return;
    setGenerationError(null);
    toast.info(`Regenerating ${room.label}...`);
    fleet.startRoom(projectId, room.id, handleStreamEvent);
  }, [retryQueue, isProjectGenerationBusy, inFlightProject, inFlightRooms, fleet, projectId, handleStreamEvent]);

  const handleRegenerateAll = () => {
    retryQueue.clear();
    if (isAnyGenerationBusy) return;
    toast.info('Regenerating all rooms...');
    // Delegates to startGeneration (which enqueues with regenerateAll=
    // false). The toast difference is the only user-visible delta vs
    // the header "Generate" path.
    startGeneration();
  };

  const handleRetryVariation = (room: Room, variationIndex: number) => {
    // Failed-variation Retry regenerates ONLY that variation, leaving sibling
    // completed variations untouched. The room-header "Regenerate" button
    // remains available for users who explicitly want a full-room redo.
    // See PRD: prds/2026-04-29-single-variation-regeneration-prd.md (Frontend → RoomGroup).
    //
    // Issue 002 of failed-variation-retry-queue PRD: route through the
    // retry queue so clicks during in-flight generation are visibly
    // queued (toast + thumbnail indicator + activity log) instead of
    // silently no-op'ing.
    const variation = room.variations[variationIndex];
    if (!variation) return;
    const outcome = retryQueue.enqueue(variation.id);
    if (outcome === 'queued') {
      toast.info('Retry queued — will run when generation completes');
      activityLog.log({
        level: 'info',
        icon: 'ℹ',
        message: `Variation ${variationIndex + 1} retry queued`,
        detail: 'Will run when generation completes.',
      });
    }
    // 'dispatched' and 'deduped' outcomes are silent — the existing flow
    // (handleRegenerateVariation) takes over for 'dispatched', and the
    // user is intentionally given no feedback on duplicate clicks.
  };

  const handleAddRooms = () => {
    toast.info('Add rooms feature coming soon');
  };

  const handleUpdateRoomAddendum = useCallback(async (room: Room, promptAddendum: string | null) => {
    // Issue 003 of projects-page-improvements PRD: persist a per-room
    // addendum without triggering any regeneration. The backend
    // normalizes empty / whitespace-only to null and returns the freshly-
    // updated project. We MUST resolveImageUrls() before swapping state
    // in — otherwise the bare blob URLs from the PATCH response would
    // replace the SAS-suffixed URLs already in local state, breaking
    // image previews and the lightbox until the next full reload.
    try {
      const updated = await updateRoomAddendum(projectId, room.id, promptAddendum);
      await resolveImageUrls(updated);
      setProject(updated);
      toast.success(
        promptAddendum === null
          ? `Cleared addendum for ${room.label}`
          : `Saved addendum for ${room.label}`
      );
      activityLog.log({
        level: 'info',
        icon: '✏',
        message:
          promptAddendum === null
            ? `Cleared per-image addendum for ${room.label}`
            : `Updated per-image addendum for ${room.label}`,
        detail: 'Applies to future generations only.',
      });
    } catch (error) {
      toast.error(parseApiError(error, 'Failed to save addendum'));
      throw error;
    }
  }, [projectId, activityLog]);

  const handleProjectSettingsSave = useCallback(async (updates: UpdateProjectBody) => {
    // Issue 002 of projects-page-improvements PRD: persist edits to
    // ``name``, ``prompt``, and/or ``settings`` without triggering any
    // regeneration. Mirrors ``handleUpdateRoomAddendum``'s pattern:
    // call ``resolveImageUrls(updated)`` BEFORE ``setProject(updated)``
    // so the SAS-suffixed URLs already in local state aren't replaced
    // by the bare blob URLs in the PATCH response — same regression
    // the per-room-prompt-addendum spec pinned.
    try {
      const updated = await updateProject(projectId, updates);
      await resolveImageUrls(updated);
      setProject(updated);
      toast.success('Project settings saved — applies to future generations');
      const changedKeys = Object.keys(updates).filter((k) => k !== 'design_brief' || updates.design_brief !== undefined);
      activityLog.log({
        level: 'info',
        icon: '⚙',
        message: `Project settings updated`,
        detail: `Changed: ${changedKeys.join(', ')}. Applies to future generations only.`,
      });
    } catch (error) {
      toast.error(parseApiError(error, 'Failed to save project settings'));
      throw error;
    }
  }, [projectId, activityLog]);

  // Issue 004 of project-settings-completeness PRD: room operations
  // (rename today; delete/add in subsequent slices) persist
  // immediately per action via the room-scoped API endpoints. The
  // ProjectRoomsManager calls this after each successful mutation so
  // local state stays in sync with the server. Mirrors
  // handleProjectSettingsSave's post-PATCH steps: resolveImageUrls
  // FIRST so SAS-suffixed URLs already in local state aren't replaced
  // by the bare blob URLs in the response — same regression the
  // per-room-prompt-addendum spec pinned. No success toast: room
  // operations are persist-immediately UX where success is silent
  // (the new label appearing in the row IS the feedback); the
  // ProjectRoomsManager toasts errors itself.
  const handleProjectUpdate = useCallback(async (updated: StagingProject) => {
    await resolveImageUrls(updated);
    setProject(updated);
  }, []);

  const handleDeleteProject = async () => {
    setIsDeleting(true);
    try {
      await deleteProject(projectId);
      toast.success('Project and all artifacts deleted');
      router.push('/projects');
    } catch (error) {
      console.error('Failed to delete project:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to delete project');
      setIsDeleting(false);
      setShowDeleteConfirm(false);
    }
  };

  const handleResetProject = async () => {
    setIsResetting(true);
    try {
      const updated = await resetProject(projectId);
      await resolveImageUrls(updated);
      setProject(updated);
      toast.success('Project reset — ready to generate');
    } catch (error) {
      console.error('Failed to reset project:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to reset project');
    } finally {
      setIsResetting(false);
    }
  };

  // Issue 002 of projects-page-stalled-stream-error-cleanup PRD:
  // single recovery-state classifier replaces the prior trio of in-line
  // banner blocks (`generationError`, `isStaleProcessing`, project-scope
  // `fleet.lostOps`) and the now-deleted `isStaleProcessing` derived
  // boolean. The banner block, the header `<Badge>`, and the header CTA
  // visibility all derive from the single classifier so they cannot
  // drift out of sync. Computed unconditionally (project may be null on
  // first paint) using the project's current status (or 'pending' as a
  // safe pre-load default that resolves to `kind: 'none'`).
  const projectLostOps = fleet.lostOps.filter(
    (op): op is Extract<LostOp, { kind: 'project' }> => op.kind === 'project',
  );
  const parsedGenerationError = generationError
    ? parseApiError(generationError)
    : null;
  const recoveryState: RecoveryState = getRecoveryState({
    projectStatus: project?.status ?? 'pending',
    // Issue 011: feed the composite busy flag so the 'interrupted' arm
    // (status='processing' && !isAnyInFlight) cannot fire while a
    // project-scope async job is enqueueing or running. Without this,
    // refreshing the page during an in-flight async generation would
    // briefly flash the destructive recovery banner before the change-
    // feed seeded the inFlightProjectGeneration slice.
    isAnyInFlight: isAnyGenerationBusy,
    projectLostOps,
    generationError: parsedGenerationError
      ? {
          statusCode: parsedGenerationError.statusCode ?? undefined,
          detail: parsedGenerationError.detail ?? undefined,
          raw: generationError ?? '',
        }
      : null,
  });

  if (isLoading || !project) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-center h-64">
          <div className="flex items-center gap-2 text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin" />
            Loading project...
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8 space-y-8">
      {/* Header */}
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" asChild>
            <Link href="/projects">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Projects
            </Link>
          </Button>
        </div>

        <div className="flex items-start justify-between">
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <h1 className="text-3xl font-bold">{project.name}</h1>
              {(() => {
                // Issue 002: badge derives from recoveryState.kind first
                // so the header agrees with the unified banner below.
                // `stream-lost` and `interrupted` share the `interrupted`
                // label intentionally — they describe the same
                // user-visible truth ("a previous run didn't finish").
                if (recoveryState.kind === 'error') {
                  return (
                    <Badge variant="destructive" className="text-xs">
                      error
                    </Badge>
                  );
                }
                if (
                  recoveryState.kind === 'stream-lost' ||
                  recoveryState.kind === 'interrupted'
                ) {
                  return (
                    <Badge variant="secondary" className="text-xs">
                      interrupted
                    </Badge>
                  );
                }
                return (
                  <Badge variant={project.status === 'completed' ? 'default' : project.status === 'failed' ? 'destructive' : project.status === 'processing' ? 'secondary' : 'outline'} className="text-xs">
                    {project.status === 'pending' ? 'ready' : project.status}
                  </Badge>
                );
              })()}
            </div>
            {/* Issue 014 of image-pipeline-and-project-ux-overhaul PRD:
                collapsed prompt header. Renders ``prompt_summary`` by
                default; the in-place editor saves via the existing
                PATCH endpoint and the backend regenerates the summary
                (PromptSummarizer; issue 013). Editing the prompt does
                NOT enqueue any regeneration job — the call only mutates
                the project document. */}
            <CollapsiblePrompt
              prompt={project.prompt}
              promptSummary={project.prompt_summary ?? null}
              onSave={async (newPrompt) => {
                await handleProjectSettingsSave({ prompt: newPrompt });
              }}
              disabled={isAnyInFlight}
            />
            {/* Issue 005 of active-and-queued-jobs-ux-redesign PRD:
                staleness subline lives INSIDE the header (rubber-duck
                blocking finding #5: NOT a separate banner). Renders the
                4 staleness-tier copies + the 120s "Cancel queued jobs"
                button when the worker has gone hard-stale; falls back
                to the existing ``{N}/{M} variations complete`` counter
                line when staleness is fresh. The component itself owns
                the cancelling-state copy override so a stale-running
                flip during the in-flight cancel doesn't flash. */}
            {(() => {
              const counterFallback =
                totalVariations > 0 ? (
                  <p className="text-xs text-muted-foreground">
                    {completedVariations}/{totalVariations} variations complete across {project.rooms.length} images
                  </p>
                ) : null;

              const suppressed =
                projectJobs.projectStaleness?.jobId !== undefined &&
                projectJobs.projectStaleness?.jobId === dismissedAfterTimeoutFor;

              return (
                <StalenessSubline
                  staleness={
                    cancellingState || suppressed
                      ? null
                      : projectJobs.projectStaleness
                  }
                  cancelling={cancellingState}
                  onCancelAllClick={handleCancelAllProjectJobs}
                  fallback={counterFallback}
                />
              );
            })()}
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {/* Primary action — derived from a pure 3-state helper so the
                label tells the truth (issue 002 of per-room-generation-control).
                See `frontend/utils/staging-header.ts` for the contract.
                Issue 002 of projects-page-stalled-stream-error-cleanup
                PRD: hidden (not disabled) while a recovery banner is
                showing, so the page presents a single recovery path
                instead of competing CTAs. Issue 011: ALSO hidden while
                a project-scope async job is enqueueing or running -
                the ProjectGenerationBanner owns the in-progress UI
                surface (single-action contract). */}
            {recoveryState.kind === 'none' && !isProjectGenerationBusy && (() => {
              const action = getHeaderAction(project.rooms);
              if (action.kind === 'hidden') return null;
              if (action.kind === 'generate') {
                return (
                  <Button
                    data-testid="project-header-action"
                    onClick={startGeneration}
                    disabled={isAnyGenerationBusy}
                  >
                    {isAnyGenerationBusy ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Play className="h-4 w-4 mr-2" />}
                    Generate
                  </Button>
                );
              }
              return (
                <Button
                  data-testid="project-header-action"
                  variant="outline"
                  onClick={handleRegenerateAll}
                  disabled={isAnyGenerationBusy}
                >
                  {isAnyGenerationBusy ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <RefreshCw className="h-4 w-4 mr-2" />}
                  Generate Remaining ({action.count})
                </Button>
              );
            })()}

            {/* Overflow menu — secondary and destructive actions. The
                duplicate `Regenerate all` item that used to appear here
                in mixed-state was removed in issue 002 (the header CTA
                already exposes the same action). */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="icon" disabled={isAnyInFlight || isDeleting}>
                  <MoreHorizontal className="h-4 w-4" />
                  <span className="sr-only">More actions</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={handleAddRooms} disabled={isAnyInFlight}>
                  <Plus className="h-4 w-4 mr-2" />
                  Add more images
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => setShowSettingsSheet(true)}
                  disabled={isAnyInFlight}
                  data-testid="overflow-menu-project-settings"
                >
                  <Settings className="h-4 w-4 mr-2" />
                  Project settings
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  className="text-destructive focus:text-destructive"
                  onClick={() => setShowDeleteConfirm(true)}
                  disabled={isAnyInFlight || isDeleting}
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Delete project
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </div>

      {/* Issue 004 of active-and-queued-jobs-ux-redesign PRD: synchronous
          preflight banner. Mounts immediately on click and stays up until
          either (a) the producer returns (202 → SSE seed populates the
          inFlightProjectGeneration slice → its banner takes over; 200 →
          dedupe toast clears `isEnqueueing`) or (b) the producer rejects
          with a structured EnqueueGenerationFailedError, in which case
          we mount the dedicated enqueue-error banner below instead. The
          two visibility predicates are mutually exclusive so exactly one
          banner is rendered at a time (single-banner contract). */}
      {isEnqueueing && !inFlightProjectGeneration && !enqueueErrorState && (
        <EnqueueingBanner />
      )}

      {/* Issue 011 of project-generation-async-queue-cutover PRD:
          in-flight project-generation banner. The slice is non-null
          for any non-terminal generate_project job; mounting this
          banner takes precedence over the recovery banner blocks
          below (single-banner contract). The Cancel button signals
          the worker via cancelProjectGeneration; the slice stays
          non-null until the worker observes the cancel_requested
          flag and flips status to 'cancelled' (during which window
          the banner shows the disabled "Cancelling…" state via the
          cancelling prop). */}

      {/* in-flight project-generation banner. The slice is non-null
          for any non-terminal generate_project job; mounting this
          banner takes precedence over the recovery banner blocks
          below (single-banner contract). The Cancel button signals
          the worker via cancelProjectGeneration; the slice stays
          non-null until the worker observes the cancel_requested
          flag and flips status to 'cancelled' (during which window
          the banner shows the disabled "Cancelling…" state via the
          cancelling prop). */}
      {inFlightProjectGeneration && (
        <ProjectGenerationBanner
          progress={inFlightProjectGeneration.progress}
          phase={inFlightProjectGeneration.phase}
          status={inFlightProjectGeneration.status}
          onCancel={() => {
            cancelProjectGeneration().catch((err: unknown) => {
              const message = err instanceof Error ? err.message : String(err);
              toast.error(`Couldn't cancel generation: ${message}`);
            });
          }}
          cancelling={
            projectJobs.jobsById[inFlightProjectGeneration.jobId]
              ?.cancel_requested === true
          }
        />
      )}

      {/* Issue 004 of active-and-queued-jobs-ux-redesign PRD: structured
          enqueue-error banner. Replaces the legacy generationError
          recovery banner specifically for classified producer failures
          (issue 002 ErrorKind contract). Friendly title + user-message
          come from getErrorKindCopy(errorKind) so a single source of
          truth maps the 5 enum values to UX copy. The collapsible
          "Show technical details" surfaces detail.type / detail.message
          for support / debugging. Retry clears the structured state and
          re-runs startGeneration; Dismiss clears the state without
          retrying. This block takes precedence over the legacy
          recovery banner (the suppression below).
          The ``!inFlightProjectGeneration`` gate is belt-and-suspenders:
          ``startGeneration`` early-returns when a project job is in
          flight, so we cannot normally reach this state. Add the gate
          anyway in case a future race (e.g. SSE delivers a stale doc
          AFTER setEnqueueErrorState) would otherwise mount two banners. */}
      {enqueueErrorState && !inFlightProjectGeneration && (() => {
        const copy = getErrorKindCopy(enqueueErrorState.errorKind);
        return (
          <div
            data-testid="enqueue-error-banner"
            data-error-kind={enqueueErrorState.errorKind}
            className="overflow-hidden rounded-lg border border-destructive/20 bg-destructive/[0.04]"
          >
            <div className="flex items-start gap-3 p-4">
              <AlertTriangle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
              <div className="flex-1 min-w-0 space-y-1">
                <p className="text-sm font-medium text-destructive">
                  {copy.friendlyTitle}
                </p>
                <p className="text-xs text-destructive/80 break-words">
                  {copy.userMessage}
                </p>
                {(enqueueErrorState.detail?.type || enqueueErrorState.detail?.message) && (
                  <Collapsible>
                    <CollapsibleTrigger className="group inline-flex items-center gap-1 text-[11px] text-destructive/60 hover:text-destructive transition-colors mt-1 cursor-pointer">
                      <ChevronDown className="h-3 w-3 transition-transform group-data-[state=open]:rotate-180" />
                      Show technical details
                    </CollapsibleTrigger>
                    <CollapsibleContent forceMount>
                      <pre
                        data-testid="enqueue-error-detail"
                        className="mt-2 rounded-md bg-destructive/[0.06] border border-destructive/10 px-3 py-2 text-[11px] text-destructive/70 font-mono whitespace-pre-wrap break-all max-h-32 overflow-y-auto"
                      >
                        {enqueueErrorState.detail?.type ?? 'Error'}: {enqueueErrorState.detail?.message ?? '(no message)'}
                        {enqueueErrorState.httpStatus ? ` [HTTP ${enqueueErrorState.httpStatus}]` : ''}
                      </pre>
                    </CollapsibleContent>
                  </Collapsible>
                )}
              </div>
              <div className="flex flex-col items-end gap-1 shrink-0">
                <div className="flex items-center gap-2">
                  <Button
                    size="sm"
                    variant="ghost"
                    data-testid="enqueue-error-dismiss"
                    onClick={() => setEnqueueErrorState(null)}
                  >
                    Dismiss
                  </Button>
                  {copy.retryable && (
                    <Button
                      size="sm"
                      variant="outline"
                      data-testid="enqueue-error-retry"
                      onClick={() => {
                        setEnqueueErrorState(null);
                        if (!isAnyInFlight) {
                          startGeneration();
                        }
                      }}
                    >
                      <RefreshCw className="h-3.5 w-3.5 mr-1" />
                      Retry
                    </Button>
                  )}
                </div>
                {copy.showAdminContact && (
                  <p className="text-[11px] text-muted-foreground">
                    Contact your administrator if this persists.
                  </p>
                )}
              </div>
            </div>
          </div>
        );
      })()}

      {/* Issue 002 of projects-page-stalled-stream-error-cleanup PRD:
          unified recovery banner. Replaces the three prior in-line
          banner blocks (`generationError`, `isStaleProcessing`,
          project-scope `fleet.lostOps`). The classifier resolves
          precedence so exactly one arm renders. `data-testid` and
          `data-recovery-kind` decouple test assertions from copy.
          Issue 011: SUPPRESSED while a project-scope async job is
          enqueueing or running (the in-flight banner above takes
          precedence — single-banner contract). */}
      {!isProjectGenerationBusy && !enqueueErrorState && recoveryState.kind === 'error' && (
          <div
            data-testid="recovery-banner"
            data-recovery-kind="error"
            className="overflow-hidden rounded-lg border border-destructive/20 bg-destructive/[0.04]"
          >
              <div className="flex items-start gap-3 p-4">
                <AlertTriangle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0 space-y-1">
                  <p className="text-sm font-medium text-destructive">
                    Generation encountered an error
                    {parsedGenerationError?.statusCode ? ` (${parsedGenerationError.statusCode})` : ""}
                  </p>
                  {parsedGenerationError?.detail && (
                    <Collapsible>
                      <p className="text-xs text-destructive/80 line-clamp-2 break-words">
                        {parsedGenerationError.detail}
                      </p>
                      {(parsedGenerationError.isTruncated || (parsedGenerationError.detail?.length ?? 0) > 120) && (
                        <CollapsibleTrigger className="group inline-flex items-center gap-1 text-[11px] text-destructive/60 hover:text-destructive transition-colors mt-1 cursor-pointer">
                          <ChevronDown className="h-3 w-3 transition-transform group-data-[state=open]:rotate-180" />
                          Full error
                        </CollapsibleTrigger>
                      )}
                      <CollapsibleContent>
                        <pre
                          data-testid="recovery-banner-detail"
                          className="mt-2 rounded-md bg-destructive/[0.06] border border-destructive/10 px-3 py-2 text-[11px] text-destructive/70 font-mono whitespace-pre-wrap break-all max-h-32 overflow-y-auto"
                        >
                          {parsedGenerationError.detail}{parsedGenerationError.isTruncated && "…"}
                        </pre>
                      </CollapsibleContent>
                    </Collapsible>
                  )}
                </div>
                <div className="flex flex-col items-end gap-1 shrink-0">
                  <Button
                    size="sm"
                    variant="outline"
                    data-testid="recovery-banner-primary"
                    onClick={() => {
                      // PRD error-arm chain: clear `generationError`,
                      // dismiss every project-scope lost op (in case
                      // one coexists with the server error), then
                      // regenerate. Single click leaves the page in
                      // a fully cleared state.
                      setGenerationError(null);
                      fleet.lostOps
                        .filter((op) => op.kind === 'project')
                        .forEach((op) => fleet.dismissLostOp(op.id));
                      retryQueue.clear();
                      if (!isAnyInFlight) {
                        startGeneration();
                        toast.info('Regenerating all rooms...');
                      }
                    }}
                  >
                    <RefreshCw className="h-3.5 w-3.5 mr-1" />
                    Retry
                  </Button>
                  <p className="text-[11px] text-muted-foreground">
                    Retries the failed run.
                  </p>
                </div>
              </div>
            </div>
          )}

      {!isProjectGenerationBusy && !enqueueErrorState && recoveryState.kind === 'stream-lost' && (
        <div
          data-testid="recovery-banner"
          data-recovery-kind="stream-lost"
          className="flex items-start gap-3 p-4 bg-amber-500/10 border border-amber-500/30 rounded-lg"
        >
          <AlertTriangle className="h-5 w-5 text-amber-500 flex-shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium">Generation stalled — no SSE events for 2 minutes</p>
            <p className="text-xs text-muted-foreground mt-0.5">
              The project-level stream watchdog fired. You can resume the existing run or dismiss this notice to re-sync with the server.
            </p>
          </div>
          <div className="flex flex-col items-end gap-1 shrink-0">
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                variant="ghost"
                data-testid="recovery-banner-secondary"
                onClick={() => {
                  // PRD: dismiss then re-sync with server.
                  fleet.dismissLostOp(recoveryState.lostOpId);
                  loadProject();
                }}
              >
                Dismiss
              </Button>
              <Button
                size="sm"
                variant="outline"
                data-testid="recovery-banner-primary"
                disabled={isAnyInFlight}
                onClick={() => {
                  // PRD: dismiss then startGeneration. NO
                  // loadProject interleave — the POST response
                  // reconciles state and inserting a fetch round-
                  // trip risks the user clicking Retry again
                  // before the start fires.
                  fleet.dismissLostOp(recoveryState.lostOpId);
                  startGeneration();
                }}
              >
                <RefreshCw className="h-3.5 w-3.5 mr-1" />
                Retry generation
              </Button>
            </div>
            <p className="text-[11px] text-muted-foreground">
              Resumes the existing run.
            </p>
          </div>
        </div>
      )}

      {!isProjectGenerationBusy && !enqueueErrorState && recoveryState.kind === 'interrupted' && (
        <div
          data-testid="recovery-banner"
          data-recovery-kind="interrupted"
          className="flex items-start gap-3 p-4 bg-amber-500/10 border border-amber-500/20 rounded-lg"
        >
          <AlertTriangle className="h-5 w-5 text-amber-500 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-sm font-medium">Generation was interrupted</p>
            <p className="text-xs text-muted-foreground mt-0.5">
              A previous run didn&apos;t finish. Reset to start over, or refresh to re-sync with the server.
            </p>
          </div>
          <div className="flex flex-col items-end gap-1 shrink-0">
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                variant="outline"
                data-testid="recovery-banner-secondary"
                onClick={loadProject}
                disabled={isResetting}
              >
                <RefreshCw className="h-3.5 w-3.5 mr-1" />
                Refresh
              </Button>
              <Button
                size="sm"
                data-testid="recovery-banner-primary"
                onClick={handleResetProject}
                disabled={isResetting}
              >
                {isResetting ? <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" /> : <Play className="h-3.5 w-3.5 mr-1" />}
                Reset &amp; Retry
              </Button>
            </div>
            <p className="text-[11px] text-muted-foreground">
              Discards prior progress and starts over.
            </p>
          </div>
        </div>
      )}

      {/* Call-to-action for pending projects */}
      {allPending && project.rooms.length > 0 && !isAnyInFlight && (
        <div className="flex flex-col items-center gap-4 py-8 px-6 bg-muted/30 border border-dashed border-muted-foreground/25 rounded-xl">
          <div className="text-center space-y-2">
            <h3 className="text-lg font-semibold">Ready to generate</h3>
            <p className="text-sm text-muted-foreground max-w-md">
              {project.rooms.length} image{project.rooms.length !== 1 ? 's' : ''} uploaded with {totalVariations} variations queued. 
              Click generate to start the AI image generation pipeline.
            </p>
          </div>
          <Button size="lg" onClick={startGeneration}>
            <Play className="h-4 w-4 mr-2" />
            Generate {totalVariations} Variations
          </Button>
        </div>
      )}

      {/* Generating progress */}
      {isAnyInFlight && (
        <div className="flex items-center gap-3 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
          <Loader2 className="h-5 w-5 animate-spin text-blue-500 flex-shrink-0" />
          <div>
            <p className="text-sm font-medium">Generating variations...</p>
            <p className="text-xs text-muted-foreground mt-0.5">
              {completedVariations}/{totalVariations} complete — this may take a few minutes per image
            </p>
          </div>
        </div>
      )}

      {/* Progress Tracker */}
      <ProgressTracker
        project={project}
        isGenerating={isAnyInFlight}
        projectRecoveryState={recoveryState}
      />

      {/* Issue 009: per-project aggregate progress bar fed by useProjectJobs.
          Hidden when no active jobs (component returns null). Coexists with
          the legacy summary tracker above — one drives off project.rooms
          variation status, the other off live job docs from SSE. */}
      <ProgressTracker kind="per-project" jobs={projectJobs.jobs} />

      {/* Issue 002 of projects-page-stalled-stream-error-cleanup PRD:
          the project-level stream-lost banner has been collapsed into
          the unified `recovery-banner` block above. Per-room
          (room-scope) lost-op banners remain in the room-list section
          below. */}

      {/* Room Groups */}
      <div className="space-y-12">
        {project.rooms.length === 0 ? (
          <div className="text-center py-12">
            <div className="space-y-3">
              <h3 className="text-xl font-semibold">No rooms uploaded</h3>
              <p className="text-muted-foreground">
                Add room images to start generating staged variations
              </p>
              <Button onClick={handleAddRooms}>
                <Plus className="h-4 w-4 mr-2" />
                Add Rooms
              </Button>
            </div>
          </div>
        ) : (
          project.rooms.map((room, index) => {
            // Issue 007: per-room derived state. The hook tracks each
            // operation's scope; the page projects it down to the props
            // RoomGroup (and its children) need.
            const isRoomBusy = inFlightProject || inFlightRooms.has(room.id);
            // At most one variation per room can be in flight (per the
            // disabling rules — variation actions are gated on this).
            const regeneratingVariationIdForRoom =
              room.variations.find((v) => inFlightVariations.has(v.id))?.id ?? null;
            // Lost ops scoped to this room (any kind that carries a roomId
            // matching this room).
            const roomLostOps = fleet.lostOps.filter(
              (op): op is Extract<LostOp, { roomId: string }> =>
                op.kind !== 'project' && op.roomId === room.id,
            );
            return (
              <div key={room.id} className="space-y-3">
                {roomLostOps.map((op) => {
                  const scopeLabel =
                    op.kind === 'room'
                      ? 'room generation'
                      : op.kind === 'variation'
                        ? `variation regeneration (${op.strategy})`
                        : 'edit-prompt generation';
                  // Per-op busy gate (rubber-duck-flagged): a lost-op
                  // Retry must obey the same disabling rules as the
                  // original click. Project in flight blocks all kinds;
                  // this room in flight blocks all room-scoped kinds;
                  // this variation in flight blocks variation/edit-
                  // prompt scoped kinds.
                  const variationInFlight =
                    op.kind === 'variation' || op.kind === 'edit-prompt'
                      ? inFlightVariations.has(op.variationId)
                      : false;
                  const retryDisabled =
                    inFlightProject || inFlightRooms.has(room.id) || variationInFlight;
                  // Route through the same page-level handlers so
                  // retryQueue.clear / toasts / activity-log entries
                  // fire as if the user clicked Retry / Regenerate
                  // through the normal UI affordance.
                  const handleRetryClick = () => {
                    fleet.dismissLostOp(op.id);
                    if (op.kind === 'room') {
                      handleRegenerateRoom(room);
                    } else if (op.kind === 'variation') {
                      const idx = room.variations.findIndex((v) => v.id === op.variationId);
                      if (idx >= 0) {
                        handleRegenerateVariation(room, idx, op.strategy);
                      }
                    } else {
                      // edit-prompt: open the dialog with the original
                      // prompt prefilled. The user must click Generate
                      // again — replaying silently with the same prompt
                      // would skip the user's chance to revise it given
                      // the upstream just stalled.
                      const idx = room.variations.findIndex((v) => v.id === op.variationId);
                      if (idx >= 0) {
                        setEditPromptTarget({ roomId: room.id, variationIndex: idx });
                      }
                    }
                  };
                  return (
                    <div
                      key={op.id}
                      data-testid={`stream-lost-banner-${room.id}`}
                      data-stream-lost-kind={op.kind}
                      className="flex items-center gap-3 p-4 bg-amber-500/10 border border-amber-500/30 rounded-lg"
                    >
                      <AlertTriangle className="h-5 w-5 text-amber-500 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium">
                          Stream lost — {scopeLabel} stalled for {room.label}
                        </p>
                        <p className="text-xs text-muted-foreground mt-0.5">
                          No SSE events arrived for 2 minutes. Click Retry to replay this exact operation.
                        </p>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => fleet.dismissLostOp(op.id)}
                        >
                          Dismiss
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          data-testid={`stream-lost-retry-${room.id}`}
                          disabled={retryDisabled}
                          onClick={handleRetryClick}
                        >
                          <RefreshCw className="h-3.5 w-3.5 mr-1" />
                          Retry
                        </Button>
                      </div>
                    </div>
                  );
                })}
                <RoomGroup
                  room={room}
                  onVariationClick={handleVariationClick}
                  onRetryVariation={handleRetryVariation}
                  onRegenerateRoom={handleRegenerateRoom}
                  onRegenerateVariation={handleRegenerateVariation}
                  onEditPromptVariation={handleEditPromptVariation}
                  onUpdateAddendum={handleUpdateRoomAddendum}
                  regeneratingVariationId={regeneratingVariationIdForRoom}
                  isRoomBusy={isRoomBusy}
                  inFlightVariationIds={inFlightVariations}
                  queuedVariationIds={retryQueue.queuedIds}
                  jobsByVariationId={jobsByVariationId}
                  roomIndex={index + 1}
                  totalRooms={project.rooms.length}
                  projectRecoveryState={recoveryState}
                />
              </div>
            );
          })
        )}
      </div>

      {/* Delete confirmation dialog */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-background border rounded-xl shadow-lg p-6 max-w-md mx-4 space-y-4">
            <div className="space-y-2">
              <h3 className="text-lg font-semibold">Delete project?</h3>
              <p className="text-sm text-muted-foreground">
                This will permanently delete <strong>{project.name}</strong>, including all {project.rooms.length} uploaded images and {totalVariations} generated variations from Azure storage. This cannot be undone.
              </p>
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setShowDeleteConfirm(false)} disabled={isDeleting}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={handleDeleteProject} disabled={isDeleting}>
                {isDeleting ? (
                  <><Loader2 className="h-4 w-4 mr-2 animate-spin" />Deleting...</>
                ) : (
                  <><Trash2 className="h-4 w-4 mr-2" />Delete Project</>
                )}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Image lightbox */}
      <ImageLightbox
        image={lightboxImage}
        onClose={() => setLightboxImage(null)}
        onNavigate={handleLightboxNavigate}
        onRegenerate={lightboxImage ? handleLightboxRegenerate : undefined}
        isRegenerating={
          lightboxImage
            ? inFlightVariations.has(
                lightboxImage.variations[lightboxImage.variationIndex]?.id ?? '',
              )
            : false
        }
        // Issue 001 of failed-variation-retry-queue PRD + issue 007 of
        // projects-page-improvements PRD: any in-flight generation
        // (project, room, OR variation) blocks the lightbox's
        // discretionary regen action. ``isAnyInFlight`` collapses the
        // prior (isGenerating || regeneratingVariationId !== null)
        // expression into a single fleet-derived boolean. The
        // ImageLightbox suppresses the tooltip when isRegenerating
        // (this variation) is true so the existing spinner UI wins.
        isBlocked={isAnyInFlight}
      />

      {/* Project Settings side sheet — issue 002 of projects-page-
          improvements PRD. Lets the user edit name/prompt/settings
          mid-project. Saves apply to FUTURE generations only. */}
      <ProjectSettingsSheet
        open={showSettingsSheet}
        onOpenChange={setShowSettingsSheet}
        project={project}
        onSave={handleProjectSettingsSave}
        onProjectUpdate={handleProjectUpdate}
      />

      {/* Edit Prompt dialog — issue 004 of projects-page-improvements
          PRD; mount-pattern standardized in issue 003 of
          radix-dialog-body-lock-fix PRD. ALWAYS MOUNTED (controlled
          via the `open` prop) so the close path does not race
          Radix's body-lock cleanup. The dialog's own open-edge
          effect re-snaps the draft on each open. */}
      {(() => {
        const room = editPromptTarget
          ? project.rooms.find((r) => r.id === editPromptTarget.roomId)
          : null;
        const variation = editPromptTarget && room
          ? room.variations[editPromptTarget.variationIndex]
          : null;
        const initialPrompt = variation?.generation_metadata?.adapted_prompt;
        // Issue 007: per-variation gate, not global isAnyInFlight. Allows
        // the user to draft Edit Prompt for room B's variation while room
        // A streams concurrently.
        const editPromptBlocked = editPromptTarget
          ? (
              inFlightProject ||
              (room ? inFlightRooms.has(room.id) : false) ||
              (variation ? inFlightVariations.has(variation.id) : false)
            )
          : false;
        const variationLabel = editPromptTarget
          ? `Variation ${editPromptTarget.variationIndex + 1}`
          : "variation";
        return (
          <EditPromptDialog
            open={!!editPromptTarget}
            onOpenChange={(open) => {
              if (!open) setEditPromptTarget(null);
            }}
            initialPrompt={initialPrompt}
            fallbackPrompt={project.prompt}
            variationLabel={variationLabel}
            isBlocked={editPromptBlocked}
            onSubmit={handleEditPromptSubmit}
          />
        );
      })()}
    </div>
  );
}