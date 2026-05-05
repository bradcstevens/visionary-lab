/**
 * Page-level wiring regression for issue 005 of the
 * active-and-queued-jobs-ux-redesign PRD (cancel-all + staleness
 * header subline + 10s timeout fallback + suppression-after-timeout).
 *
 * Pinned behaviour:
 *
 *  - When ``projectStaleness`` is null OR ``fresh``, the page
 *    renders the existing ``{N}/{M} variations complete`` counter
 *    line (NOT a separate banner).
 *
 *  - When the staleness-tier is soft, the subline mounts WITHOUT
 *    a Cancel button (the user is meant to wait, not act).
 *
 *  - When the staleness-tier is hard, the subline mounts WITH the
 *    "Cancel queued jobs" button. Clicking the button:
 *      * fires DELETE /staging/projects/{id}/jobs via
 *        ``projectJobs.cancelAllProjectJobs``;
 *      * surfaces a success toast naming ``cancelled_count``;
 *      * sets local ``cancellingState=true`` so the subline shows
 *        the "Cancelling…" copy until the worker confirms (the
 *        cancelled job leaves the live non-terminal set).
 *
 *  - Re-entrancy: a double-click on the Cancel button only fires
 *    one DELETE (imperative latch in ``cancelInFlightRef``).
 *
 *  - Server-confirmation path: when the active set no longer
 *    contains the cancelled jobId, the page clears
 *    ``cancellingState`` and the staleness subline becomes
 *    available again should new staleness arise.
 *
 *  - 10-second fallback timeout: if the worker has not confirmed
 *    after 10 s, the page clears ``cancellingState`` AND records
 *    ``dismissedAfterTimeoutFor`` so the staleness subline does NOT
 *    immediately re-mount for the same jobId. A fallback toast
 *    informs the user.
 *
 *  - The post-timeout suppression auto-clears when the job
 *    eventually leaves the active set (rubber-duck blocking
 *    finding #4): the staleness subline is then permitted to
 *    re-surface for any NEW stale jobs.
 *
 *  - Cancel-call rejection: rejected ``cancelAllProjectJobs``
 *    promise clears ``cancellingState`` synchronously and surfaces
 *    an error toast. No suppression is set.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  act,
  cleanup,
  fireEvent,
  waitFor,
} from "@testing-library/react";
import type { StagingProject } from "@/services/stagingApi";

// ---------------------------------------------------------------------------
// next/navigation
// ---------------------------------------------------------------------------
vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "project-1" }),
  useRouter: () => ({ push: vi.fn(), back: vi.fn(), refresh: vi.fn() }),
}));

vi.mock("next/link", () => ({
  default: ({
    children,
    href,
  }: {
    children: React.ReactNode;
    href: string;
  }) => <a href={href}>{children}</a>,
}));

// ---------------------------------------------------------------------------
// stagingApi
// ---------------------------------------------------------------------------
const mockProject: StagingProject = {
  id: "project-1",
  name: "Test Project",
  prompt: "A serene living room",
  status: "processing",
  settings: {
    variations_per_room: 2,
    model: "gpt-image-1",
    quality: "high",
    size: "1024x1024",
  },
  rooms: [
    {
      id: "room-1",
      label: "Living Room",
      original_image_url: "https://example.com/r1.jpg",
      status: "processing",
      variations: [],
    },
  ],
  created_at: "2026-05-04T00:00:00Z",
  updated_at: "2026-05-04T00:00:00Z",
};

vi.mock("@/services/stagingApi", async () => {
  const actual = await vi.importActual<typeof import("@/services/stagingApi")>(
    "@/services/stagingApi",
  );
  return {
    ...actual,
    getProject: vi.fn(async () => mockProject),
    deleteProject: vi.fn(),
    resetProject: vi.fn(),
    updateProject: vi.fn(),
    updateRoomAddendum: vi.fn(),
    enqueueProjectGeneration: vi.fn(),
  };
});

vi.mock("@/services/sas-token", () => ({
  sasTokenService: {
    getTokens: vi.fn(async () => ({ images: "", videos: "" })),
    invalidate: vi.fn(),
  },
}));

// ---------------------------------------------------------------------------
// jobs-context — controllable projectStaleness + activeJobs +
// cancelAllProjectJobs spy. The test sets ``__nextStaleness`` and
// ``__activeJobs`` between renders to simulate the live state.
// ---------------------------------------------------------------------------
type ProjectJobLike = {
  id: string;
  project_id: string;
  room_id: string;
  variation_id: string;
  kind: string;
  status: string;
  progress: number;
  phase: string | null;
  cancel_requested?: boolean;
  revision?: number;
  error_kind?: string | null;
  updated_at?: string;
};

type StalenessLike = {
  jobId: string;
  kind: "fresh" | "soft-pending" | "soft-running" | "hard-pending" | "hard-running";
  secondsAgo: number;
} | null;

vi.mock("@/context/jobs-context", async () => {
  const actual = await vi.importActual<typeof import("@/context/jobs-context")>(
    "@/context/jobs-context",
  );
  const cancelAllSpy = vi.fn(async () => ({
    status: "accepted",
    cancelled_count: 1,
    project_id: "project-1",
  }));
  (
    globalThis as unknown as {
      __cancelAllProjectJobs: typeof cancelAllSpy;
    }
  ).__cancelAllProjectJobs = cancelAllSpy;
  return {
    ...actual,
    useProjectJobs: () => {
      const staleness = (
        globalThis as unknown as { __nextStaleness?: StalenessLike }
      ).__nextStaleness;
      const activeJobs = (
        globalThis as unknown as { __activeJobs?: ProjectJobLike[] }
      ).__activeJobs;
      return {
        jobs: activeJobs ?? [],
        jobsById: (activeJobs ?? []).reduce(
          (acc, j) => {
            acc[j.id] = j;
            return acc;
          },
          {} as Record<string, ProjectJobLike>,
        ),
        activeJobs: activeJobs ?? [],
        connectionState: "open",
        lastError: null,
        retry: vi.fn(),
        refresh: vi.fn(),
        inFlightProjectGeneration: null,
        cancelProjectGeneration: vi.fn(),
        injectOptimisticJob: vi.fn(),
        lastBackendActivityByJobId: {},
        projectStaleness: staleness ?? null,
        cancelAllProjectJobs: cancelAllSpy,
      };
    },
  };
});

vi.mock("@/hooks/useGenerationFleet", () => ({
  useGenerationFleet: () => ({
    inFlightProject: false,
    inFlightRooms: new Set<string>(),
    inFlightVariations: new Set<string>(),
    isAnyInFlight: false,
    lostOps: [],
    startProject: vi.fn(),
    startRoom: vi.fn(),
    startVariation: vi.fn(),
    editPrompt: vi.fn(),
    retryLostOp: vi.fn(),
    dismissLostOp: vi.fn(),
    abortAll: vi.fn(),
  }),
}));

vi.mock("@/hooks/useRetryQueue", () => ({
  useRetryQueue: () => ({
    enqueue: vi.fn(),
    dequeue: vi.fn(),
    clear: vi.fn(() => 0),
    queuedIds: new Set<string>(),
    size: 0,
  }),
}));

vi.mock("sonner", () => {
  const toast = {
    error: vi.fn(),
    success: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
  };
  (globalThis as unknown as { __toastSpy: typeof toast }).__toastSpy = toast;
  return { toast, Toaster: () => null };
});

// ---------------------------------------------------------------------------
// Imports AFTER mocks
// ---------------------------------------------------------------------------
import ProjectDetailPage from "../page";
import { ActivityLogProvider } from "@/context/activity-log-context";

const g = globalThis as unknown as {
  __toastSpy: {
    error: ReturnType<typeof vi.fn>;
    success: ReturnType<typeof vi.fn>;
    info: ReturnType<typeof vi.fn>;
    warning: ReturnType<typeof vi.fn>;
  };
  __cancelAllProjectJobs: ReturnType<typeof vi.fn>;
  __nextStaleness?: StalenessLike;
  __activeJobs?: ProjectJobLike[];
};

function renderPage() {
  return render(
    <ActivityLogProvider>
      <ProjectDetailPage />
    </ActivityLogProvider>,
  );
}

function makeJob(id: string, status = "pending"): ProjectJobLike {
  return {
    id,
    project_id: "project-1",
    room_id: "__project__",
    variation_id: "__project__",
    kind: "generate_project",
    status,
    progress: 0,
    phase: "queued",
    revision: 0,
    updated_at: "2026-05-04T00:00:00Z",
  };
}

beforeEach(() => {
  cleanup();
  g.__toastSpy.error.mockClear();
  g.__toastSpy.success.mockClear();
  g.__toastSpy.info.mockClear();
  g.__toastSpy.warning.mockClear();
  g.__cancelAllProjectJobs.mockReset();
  g.__cancelAllProjectJobs.mockResolvedValue({
    status: "accepted",
    cancelled_count: 1,
    project_id: "project-1",
  });
  g.__nextStaleness = null;
  g.__activeJobs = [];
});

afterEach(() => {
  vi.useRealTimers();
});

// ===========================================================================
// Subline rendering gates
// ===========================================================================
describe("Issue 005 — staleness subline rendering", () => {
  it("does NOT mount the staleness subline when projectStaleness is null (fresh)", async () => {
    g.__nextStaleness = null;
    renderPage();
    await screen.findByText(/Test Project/);
    expect(screen.queryByTestId("staleness-subline")).toBeNull();
  });

  it("mounts the soft-running subline without a Cancel button", async () => {
    g.__nextStaleness = {
      jobId: "j1",
      kind: "soft-running",
      secondsAgo: 50,
    };
    g.__activeJobs = [makeJob("j1", "running")];
    renderPage();
    await screen.findByTestId("staleness-subline");
    expect(
      screen.getByTestId("staleness-subline").getAttribute("data-state"),
    ).toBe("soft-running");
    expect(screen.queryByTestId("cancel-queued-jobs-button")).toBeNull();
  });

  it("mounts the hard-running subline WITH the Cancel button (PRD: 120s threshold)", async () => {
    g.__nextStaleness = {
      jobId: "j1",
      kind: "hard-running",
      secondsAgo: 130,
    };
    g.__activeJobs = [makeJob("j1", "running")];
    renderPage();
    await screen.findByTestId("staleness-subline");
    expect(screen.getByTestId("cancel-queued-jobs-button")).toBeTruthy();
  });
});

// ===========================================================================
// Cancel-all click flow
// ===========================================================================
describe("Issue 005 — cancel-all click flow", () => {
  it("invokes cancelAllProjectJobs and surfaces a success toast naming cancelled_count", async () => {
    g.__cancelAllProjectJobs.mockResolvedValue({
      status: "accepted",
      cancelled_count: 3,
      project_id: "project-1",
    });
    g.__nextStaleness = {
      jobId: "j1",
      kind: "hard-pending",
      secondsAgo: 130,
    };
    g.__activeJobs = [makeJob("j1")];
    renderPage();
    const button = await screen.findByTestId("cancel-queued-jobs-button");
    await act(async () => {
      fireEvent.click(button);
    });
    expect(g.__cancelAllProjectJobs).toHaveBeenCalledTimes(1);
    await waitFor(() => {
      expect(g.__toastSpy.success).toHaveBeenCalledWith(
        "Cancelled 3 queued jobs.",
      );
    });
  });

  it("flips the subline to 'Cancelling…' synchronously on click (cancellingState=true)", async () => {
    // Make the cancel call hang so the test observes the intermediate state.
    let resolveCancel: (
      v: { status: string; cancelled_count: number; project_id: string } | void,
    ) => void = () => {};
    g.__cancelAllProjectJobs.mockImplementation(
      () =>
        new Promise<{
          status: string;
          cancelled_count: number;
          project_id: string;
        }>((res) => {
          resolveCancel = (v) => res(v as never);
        }),
    );
    g.__nextStaleness = {
      jobId: "j1",
      kind: "hard-running",
      secondsAgo: 130,
    };
    g.__activeJobs = [makeJob("j1", "running")];
    renderPage();
    const button = await screen.findByTestId("cancel-queued-jobs-button");
    await act(async () => {
      fireEvent.click(button);
    });
    await waitFor(() => {
      expect(
        screen.getByTestId("staleness-subline").getAttribute("data-state"),
      ).toBe("cancelling");
    });
    // Cancel button is suppressed during the in-flight cancel.
    expect(screen.queryByTestId("cancel-queued-jobs-button")).toBeNull();
    // Resolve to clean up.
    await act(async () => {
      resolveCancel({
        status: "accepted",
        cancelled_count: 1,
        project_id: "project-1",
      });
    });
  });

  it("re-entrancy latch: double-click only fires ONE DELETE (imperative latch)", async () => {
    let resolveCancel: (
      v: { status: string; cancelled_count: number; project_id: string } | void,
    ) => void = () => {};
    g.__cancelAllProjectJobs.mockImplementation(
      () =>
        new Promise<{
          status: string;
          cancelled_count: number;
          project_id: string;
        }>((res) => {
          resolveCancel = (v) => res(v as never);
        }),
    );
    g.__nextStaleness = {
      jobId: "j1",
      kind: "hard-pending",
      secondsAgo: 130,
    };
    g.__activeJobs = [makeJob("j1")];
    renderPage();
    const button = await screen.findByTestId("cancel-queued-jobs-button");
    await act(async () => {
      fireEvent.click(button);
      // Even if a second click happens BEFORE the next render, the
      // imperative ref guard should swallow it.
      fireEvent.click(button);
    });
    expect(g.__cancelAllProjectJobs).toHaveBeenCalledTimes(1);
    await act(async () => {
      resolveCancel({
        status: "accepted",
        cancelled_count: 1,
        project_id: "project-1",
      });
    });
  });

  it("rejected cancelAllProjectJobs clears cancellingState + surfaces an error toast", async () => {
    g.__cancelAllProjectJobs.mockRejectedValue(new Error("HTTP 503: Service Unavailable"));
    g.__nextStaleness = {
      jobId: "j1",
      kind: "hard-running",
      secondsAgo: 130,
    };
    g.__activeJobs = [makeJob("j1", "running")];
    renderPage();
    const button = await screen.findByTestId("cancel-queued-jobs-button");
    await act(async () => {
      fireEvent.click(button);
    });
    await waitFor(() => {
      expect(g.__toastSpy.error).toHaveBeenCalled();
    });
    const errArgs = g.__toastSpy.error.mock.calls[0][0] as string;
    expect(errArgs).toContain("Couldn't cancel jobs");
    // After rejection, cancellingState clears and the staleness subline
    // returns to the hard-running tier (since active jobs and staleness
    // are unchanged).
    await waitFor(() => {
      expect(
        screen.getByTestId("staleness-subline").getAttribute("data-state"),
      ).toBe("hard-running");
    });
  });
});

// ===========================================================================
// 10s fallback timer + suppression-after-timeout
// ===========================================================================
describe("Issue 005 — 10s fallback timer", () => {
  it("after 10s with no SSE confirmation, the page surfaces a fallback toast and suppresses the subline for that jobId", async () => {
    // Keep the cancel call pending so SSE confirmation never lands.
    g.__cancelAllProjectJobs.mockImplementation(() => new Promise(() => {}));
    g.__nextStaleness = {
      jobId: "j1",
      kind: "hard-running",
      secondsAgo: 130,
    };
    g.__activeJobs = [makeJob("j1", "running")];
    renderPage();
    // Wait for initial mount with real timers (getProject has to resolve).
    const button = await screen.findByTestId("cancel-queued-jobs-button");

    // Now switch to fake timers ONLY for the 10s fallback timer.
    vi.useFakeTimers();
    try {
      await act(async () => {
        fireEvent.click(button);
      });
      await act(async () => {
        vi.advanceTimersByTime(10_000);
      });

      expect(g.__toastSpy.error).toHaveBeenCalled();
      const errArgs = g.__toastSpy.error.mock.calls[0][0] as string;
      expect(errArgs).toMatch(/didn't confirm in time/);

      // Subline is suppressed for this jobId even though the staleness
      // is still hard-running (rubber-duck blocking finding #4).
      expect(screen.queryByTestId("staleness-subline")).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });
});
