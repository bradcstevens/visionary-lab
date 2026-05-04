/**
 * Page-level regression for issue 001 of the
 * projects-page-stalled-stream-error-cleanup PRD.
 *
 * Pinned behaviour: a watchdog-synthesized SSE 'error' event
 * (``synthetic: true``) MUST NOT flip the destructive red
 * "Generation encountered an error" banner. A real server-sent
 * 'error' event (no ``synthetic`` field) MUST still flip it.
 *
 * Strategy: render ``ProjectDetailPage`` with every external
 * dependency mocked. The fake fleet captures the
 * ``handleStreamEvent`` callback the page passes into
 * ``startRoom`` (the room-scope regen path that still uses the
 * fleet post issue-011 cutover). The test clicks the room-level
 * Regenerate button to trigger ``startRoom``, then drives the
 * captured handler with a synthetic and a real error and asserts
 * on the rendered DOM.
 *
 * Issue 011 of project-generation-async-queue-cutover PRD: the
 * page-level Generate CTA no longer routes through the fleet —
 * it enqueues an async job via ``enqueueProjectGeneration``.
 * Variation/room regen still uses the fleet, so this test now
 * uses the room-Regenerate button as the trigger.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, act } from '@testing-library/react';
import type {
  StagingProject,
  StagingStreamEventCallback,
} from '@/services/stagingApi';

// ---------------------------------------------------------------------------
// next/navigation — useParams / useRouter are required by the page even
// though it never navigates in this test.
// ---------------------------------------------------------------------------
vi.mock('next/navigation', () => ({
  useParams: () => ({ id: 'project-1' }),
  useRouter: () => ({ push: vi.fn(), back: vi.fn(), refresh: vi.fn() }),
}));

// ---------------------------------------------------------------------------
// next/link — render as a plain anchor.
// ---------------------------------------------------------------------------
vi.mock('next/link', () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) => (
    <a href={href}>{children}</a>
  ),
}));

// ---------------------------------------------------------------------------
// stagingApi — only ``getProject`` is exercised on first paint. Other
// exports are stubbed to satisfy the page's import statement.
// ---------------------------------------------------------------------------
//
// Issue 011: the room must be in a status RoomGroup considers "regen-able"
// (failed/completed/processing) so the Regenerate button renders. The
// project status stays 'pending' so the recovery classifier picks the
// 'none' arm and the page doesn't render any banner before the test
// drives an error event.
const mockProject: StagingProject = {
  id: 'project-1',
  name: 'Test Project',
  prompt: 'A serene living room',
  status: 'pending',
  settings: { variations_per_room: 2, model: 'gpt-image-1', quality: 'high', size: '1024x1024' },
  rooms: [
    {
      id: 'room-1',
      label: 'Living Room',
      original_image_url: 'https://example.com/r1.jpg',
      status: 'completed',
      variations: [],
    },
  ],
  created_at: '2026-05-02T00:00:00Z',
  updated_at: '2026-05-02T00:00:00Z',
};

vi.mock('@/services/stagingApi', async () => {
  const actual = await vi.importActual<typeof import('@/services/stagingApi')>(
    '@/services/stagingApi',
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

// ---------------------------------------------------------------------------
// SAS tokens — return empty maps; StorageImage just won't get tokens.
// ---------------------------------------------------------------------------
vi.mock('@/services/sas-token', () => ({
  sasTokenService: {
    getTokens: vi.fn(async () => ({ images: '', videos: '' })),
    invalidate: vi.fn(),
  },
}));

// ---------------------------------------------------------------------------
// jobs-context — return inert values; the page only reads
// ``jobs`` / ``aggregateProgress`` for the progress bar surface.
// Issue 011: must include the new ``inFlightProjectGeneration`` /
// ``cancelProjectGeneration`` / ``jobsById`` keys the page now reads.
// Returning ``null`` for the slice keeps the new ProjectGenerationBanner
// unmounted and the recovery banner blocks unsuppressed (so the
// destructive-banner positive control still asserts correctly).
// ---------------------------------------------------------------------------
vi.mock('@/context/jobs-context', () => ({
  useProjectJobs: () => ({
    jobs: [],
    jobsById: {},
    aggregateProgress: null,
    status: 'idle',
    inFlightProjectGeneration: null,
    cancelProjectGeneration: vi.fn(),
  }),
}));

// ---------------------------------------------------------------------------
// Sonner toast — capture calls so the assertion that the synthetic
// error path STILL toasts works. The hoisted-mock factory cannot
// close over outer-scope variables, so we attach the spies onto a
// global namespace the factory can reach without TDZ.
// ---------------------------------------------------------------------------
vi.mock('sonner', () => {
  const toast = {
    error: vi.fn(),
    success: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
  };
  // Stash on globalThis so the test body can read .mock.calls without
  // tripping the hoisted-factory TDZ rule.
  (globalThis as unknown as { __toastSpy: typeof toast }).__toastSpy = toast;
  return { toast, Toaster: () => null };
});

// ---------------------------------------------------------------------------
// useGenerationFleet — fake fleet that records the eventHandler given
// to ``startRoom`` so the test can drive it directly. (Pre-issue-011
// the trigger was startProject; that path is no longer wired from the
// page-level Generate CTA, so the test now drives the room-regen path
// which still uses the fleet.)
// ---------------------------------------------------------------------------
vi.mock('@/hooks/useGenerationFleet', () => {
  const startRoom = vi.fn((_pid: string, _rid: string, h: StagingStreamEventCallback) => {
    (globalThis as unknown as { __captured: StagingStreamEventCallback | null }).__captured = h;
  });
  (globalThis as unknown as { __startRoom: typeof startRoom }).__startRoom = startRoom;
  return {
    useGenerationFleet: () => ({
      inFlightProject: false,
      inFlightRooms: new Set<string>(),
      inFlightVariations: new Set<string>(),
      isAnyInFlight: false,
      lostOps: [],
      startProject: vi.fn(),
      startRoom,
      startVariation: vi.fn(),
      editPrompt: vi.fn(),
      retryLostOp: vi.fn(),
      dismissLostOp: vi.fn(),
      abortAll: vi.fn(),
    }),
  };
});

// ---------------------------------------------------------------------------
// retry queue — provide the surface the page reads.
// ---------------------------------------------------------------------------
vi.mock('@/hooks/useRetryQueue', () => ({
  useRetryQueue: () => ({
    enqueue: vi.fn(),
    dequeue: vi.fn(),
    clear: vi.fn(() => 0),
    queuedIds: new Set<string>(),
    size: 0,
  }),
}));

// Import the page AFTER all mocks are registered.
import ProjectDetailPage from '../page';
import { ActivityLogProvider } from '@/context/activity-log-context';

// Read the spies off the global stash that the hoisted mock factories
// populated. (Mock factories are hoisted to the top of the file by
// Vitest, so they cannot close over module-level `let` bindings.)
const g = globalThis as unknown as {
  __toastSpy: { error: ReturnType<typeof vi.fn>; success: ReturnType<typeof vi.fn> };
  __startRoom: ReturnType<typeof vi.fn>;
  __captured: StagingStreamEventCallback | null;
};

function renderPage() {
  return render(
    <ActivityLogProvider>
      <ProjectDetailPage />
    </ActivityLogProvider>,
  );
}

describe('ProjectDetailPage — synthetic-watchdog error suppression', () => {
  beforeEach(() => {
    g.__captured = null;
    g.__startRoom.mockClear();
    g.__toastSpy.error.mockClear();
  });

  it('does NOT render the destructive "Generation encountered an error" banner when the page receives a synthetic watchdog error event', async () => {
    renderPage();

    // Click the room-level Regenerate button (issue 011: page-level
    // Generate now goes through enqueueProjectGeneration, not the
    // fleet, so room-regen is the path that captures the fleet's
    // event handler).
    const regenBtn = await screen.findByRole('button', { name: /regenerate/i });

    await act(async () => {
      regenBtn.click();
    });

    expect(g.__startRoom).toHaveBeenCalledTimes(1);
    expect(g.__captured).not.toBeNull();

    await act(async () => {
      g.__captured!({
        type: 'error',
        error: 'Stream lost — no SSE events for 2 minutes',
        synthetic: true,
      });
    });

    expect(
      screen.queryByTestId('recovery-banner'),
    ).toBeNull();

    expect(g.__toastSpy.error).toHaveBeenCalled();
  });

  it('STILL renders the destructive banner when the page receives a real (non-synthetic) error event', async () => {
    renderPage();

    const regenBtn = await screen.findByRole('button', { name: /regenerate/i });
    await act(async () => {
      regenBtn.click();
    });

    expect(g.__captured).not.toBeNull();

    await act(async () => {
      g.__captured!({
        type: 'error',
        error: 'Backend exploded',
      });
    });

    await waitFor(() => {
      const banner = screen.getByTestId('recovery-banner');
      expect(banner).toBeTruthy();
      expect(banner.getAttribute('data-recovery-kind')).toBe('error');
    });
  });
});
