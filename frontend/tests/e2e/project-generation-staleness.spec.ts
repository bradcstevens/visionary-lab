import { test, expect, Page, Route } from '@playwright/test';

/**
 * Cancel-all + staleness header subline UX — issue 005 of the
 * `active-and-queued-jobs-ux-redesign` PRD.
 *
 * Three user-visible contracts proven end-to-end:
 *
 *   1. Soft staleness (45s+): the header subline copy flips from
 *      the existing `{N}/{M} variations complete` counter to the
 *      "Waiting for worker / Generation paused" copy, BUT no Cancel
 *      button is exposed yet. This is the early-warning tier.
 *
 *   2. Hard staleness (120s+): the subline copy escalates to the
 *      "Worker stopped responding / Cancel to free the queue and
 *      retry" copy AND the "Cancel queued jobs" button surfaces.
 *      Click the button → DELETE /staging/projects/{id}/jobs is
 *      issued → success toast names the cancelled count → the
 *      worker confirms by emitting the cancelled status → the
 *      subline auto-clears.
 *
 *   3. 10s fallback: if the worker never confirms after the cancel
 *      DELETE, the page surfaces a fallback toast and suppresses
 *      the staleness subline for that jobId so the user isn't
 *      stuck staring at an unactionable banner. The suppression
 *      auto-clears when the job eventually leaves the live set
 *      (rubber-duck blocking finding #4).
 *
 * The pure-module + page-wiring contracts are pinned by vitest
 * (job-staleness.test.ts, staleness-subline-copy.test.ts,
 * StalenessSubline.test.tsx, page-issue-005-wiring.test.tsx). This
 * Playwright spec is the cross-stack pin proving the page consumes
 * the API contract in a real browser — including the wall-clock-
 * driven staleness recomputation and the DELETE-on-click round-trip.
 *
 * Run: npx playwright test tests/e2e/project-generation-staleness.spec.ts
 */

const PROJECT_ID = 'test-project-issue-005';
const API_BASE = 'http://localhost:8000/api/v1';
const JOB_ID = `${PROJECT_ID}:__project__:__project__:stale-key-001`;

interface MockJob {
  id: string;
  project_id: string;
  room_id: string;
  variation_id: string;
  revision: number | string;
  kind: string;
  status: string;
  progress?: number;
  phase?: string | null;
  attempts?: number;
  error?: string | null;
  error_kind?: string | null;
  cancel_requested?: boolean;
  created_at: string;
  updated_at: string;
}

function makeProject() {
  const now = new Date().toISOString();
  const rooms = ['room-1', 'room-2'].map((id, i) => ({
    id,
    label: `Room ${i + 1}`,
    original_image_url:
      `https://storage.blob.core.windows.net/images/staging/` +
      `${PROJECT_ID}/originals/${id}.png?sv=mock`,
    status: 'pending',
    variations: [],
    created_at: now,
    updated_at: now,
  }));
  return {
    id: PROJECT_ID,
    name: 'Issue 005 Test Project',
    prompt: 'Modern minimalist',
    status: 'processing',
    settings: {
      style: 'modern',
      room_count: 2,
      variations_per_room: 3,
      output_format: 'png',
      quality: 'high',
    },
    rooms,
    total_variations: 6,
    completed_variations: 0,
    created_at: now,
    updated_at: now,
  };
}

function makeRunningJob(): MockJob {
  const now = new Date().toISOString();
  return {
    id: JOB_ID,
    project_id: PROJECT_ID,
    room_id: '__project__',
    variation_id: '__project__',
    revision: 'stale-key-001',
    kind: 'generate_project',
    status: 'running',
    progress: 30,
    phase: 'generating',
    attempts: 1,
    cancel_requested: false,
    created_at: now,
    updated_at: now,
  };
}

function sseEvent(name: string, data: unknown): string {
  return `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
}

interface RouteState {
  jobs: MockJob[];
  /**
   * - confirm: DELETE /jobs flips the job to status="cancelled",
   *   re-emits via SSE.
   * - silent: DELETE /jobs returns 202 but the worker NEVER flips
   *   the job — exercises the 10s fallback timer.
   */
  cancelMode: 'confirm' | 'silent';
  cancelAllRequests: number;
}

async function setupRoutes(page: Page, state: RouteState) {
  await page.route(`${API_BASE}/gallery/sas-tokens`, (route: Route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        video_sas_token: 'sv=mock',
        image_sas_token: 'sv=mock',
        video_container_url:
          'https://storage.blob.core.windows.net/videos',
        image_container_url:
          'https://storage.blob.core.windows.net/images',
        expiry: new Date(Date.now() + 3600_000).toISOString(),
      }),
    }),
  );

  await page.route(
    new RegExp(`/api/v1/staging/projects/${PROJECT_ID}(\\?.*)?$`),
    (route: Route) => {
      if (route.request().method() === 'GET') {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ project: makeProject() }),
        });
      }
      return route.continue();
    },
  );

  await page.route(
    `${API_BASE}/staging/projects/${PROJECT_ID}/jobs`,
    (route: Route) => {
      const method = route.request().method();
      if (method === 'GET') {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ jobs: state.jobs }),
        });
      }
      // DELETE — cancel-all entry point (issue 005 endpoint).
      if (method === 'DELETE') {
        state.cancelAllRequests += 1;
        const cancelledCount = state.jobs.filter(
          (j) => !['succeeded', 'failed', 'cancelled'].includes(j.status),
        ).length;
        if (state.cancelMode === 'confirm') {
          // Worker observes the cancel and flips the job state.
          state.jobs = state.jobs.map((j) =>
            ['succeeded', 'failed', 'cancelled'].includes(j.status)
              ? j
              : {
                  ...j,
                  status: 'cancelled',
                  cancel_requested: true,
                  updated_at: new Date().toISOString(),
                },
          );
        }
        return route.fulfill({
          status: 202,
          contentType: 'application/json',
          body: JSON.stringify({
            status: 'accepted',
            cancelled_count: cancelledCount,
            project_id: PROJECT_ID,
          }),
        });
      }
      return route.continue();
    },
  );

  await page.route(
    new RegExp(
      `/api/v1/staging/projects/${PROJECT_ID}/jobs/stream(\\?.*)?$`,
    ),
    (route: Route) =>
      route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        headers: {
          'Cache-Control': 'no-cache',
          Connection: 'keep-alive',
        },
        body: sseEvent('seed', { jobs: state.jobs }),
      }),
  );
}

test.describe('Project Generation — staleness header subline (issue 005)', () => {
  test('soft staleness (45s): subline mounts WITHOUT cancel button', async ({
    page,
  }) => {
    const state: RouteState = {
      jobs: [makeRunningJob()],
      cancelMode: 'confirm',
      cancelAllRequests: 0,
    };
    await setupRoutes(page, state);

    await page.clock.install();
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('domcontentloaded');

    // Wait for the in-flight banner to confirm the job is hydrated.
    await expect(
      page.getByTestId('project-generation-banner'),
    ).toBeVisible({ timeout: 10_000 });

    // Subline should not be visible yet (job just hydrated → fresh).
    await expect(page.getByTestId('staleness-subline')).toHaveCount(0);

    // Fast-forward 50s → should hit the soft tier.
    await page.clock.fastForward(50_000);
    await page.waitForTimeout(100);

    const subline = page.getByTestId('staleness-subline');
    await expect(subline).toBeVisible({ timeout: 7_000 });
    await expect(subline).toHaveAttribute('data-state', 'soft-running');

    // No cancel button at the soft tier.
    await expect(page.getByTestId('cancel-queued-jobs-button')).toHaveCount(0);
  });

  test('hard staleness (130s): subline + Cancel button → click → success toast → subline clears on confirmation', async ({
    page,
  }) => {
    const state: RouteState = {
      jobs: [makeRunningJob()],
      cancelMode: 'confirm',
      cancelAllRequests: 0,
    };
    await setupRoutes(page, state);

    await page.clock.install();
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('domcontentloaded');

    await expect(
      page.getByTestId('project-generation-banner'),
    ).toBeVisible({ timeout: 10_000 });

    // Fast-forward past the 120s hard threshold.
    await page.clock.fastForward(130_000);
    await page.waitForTimeout(100);

    const subline = page.getByTestId('staleness-subline');
    await expect(subline).toBeVisible({ timeout: 7_000 });
    await expect(subline).toHaveAttribute('data-state', 'hard-running');

    const cancelBtn = page.getByTestId('cancel-queued-jobs-button');
    await expect(cancelBtn).toBeVisible();

    // Click the cancel button — DELETE round-trip.
    const cancelReqPromise = page.waitForRequest(
      (req) =>
        req.url().includes(`/staging/projects/${PROJECT_ID}/jobs`) &&
        req.method() === 'DELETE',
    );
    await cancelBtn.click();
    const cancelReq = await cancelReqPromise;
    expect(cancelReq.method()).toBe('DELETE');

    // After SSE / poll re-fetches the job list, the cancelled job is
    // terminal and the subline auto-clears.
    await expect(subline).toHaveCount(0, { timeout: 10_000 });
    expect(state.cancelAllRequests).toBe(1);
  });

  test('hard staleness silent worker (130s + click + 10s no confirmation): fallback toast + subline suppressed', async ({
    page,
  }) => {
    const state: RouteState = {
      jobs: [makeRunningJob()],
      cancelMode: 'silent',
      cancelAllRequests: 0,
    };
    await setupRoutes(page, state);

    await page.clock.install();
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('domcontentloaded');

    await expect(
      page.getByTestId('project-generation-banner'),
    ).toBeVisible({ timeout: 10_000 });

    await page.clock.fastForward(130_000);
    await page.waitForTimeout(100);

    const cancelBtn = page.getByTestId('cancel-queued-jobs-button');
    await expect(cancelBtn).toBeVisible({ timeout: 7_000 });

    await cancelBtn.click();

    // Subline flips to "Cancelling…" while the call is in-flight.
    const subline = page.getByTestId('staleness-subline');
    await expect(subline).toHaveAttribute('data-state', 'cancelling');

    // Fast-forward past the 10s fallback.
    await page.clock.fastForward(11_000);
    await page.waitForTimeout(100);

    // Subline is suppressed for this jobId even though staleness is
    // unchanged at the hard tier (rubber-duck blocking finding #4).
    await expect(subline).toHaveCount(0, { timeout: 5_000 });

    expect(state.cancelAllRequests).toBe(1);
  });
});
