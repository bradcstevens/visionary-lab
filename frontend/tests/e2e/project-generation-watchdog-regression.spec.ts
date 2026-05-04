import { test, expect, Page, Route } from '@playwright/test';

/**
 * Project-generation watchdog regression E2E (issue 013 of project-
 * generation-async-queue-cutover PRD).
 *
 * Pins three properties at once — per the rubber-duck blocking finding
 * in the issue spec, ALL THREE assertions are required to claim the
 * legacy 120s-silence "stalled stream" symptom is unreachable in the
 * new architecture.
 *
 *   - **Assertion 1**: Generate click triggers a POST to
 *     `/jobs/generate` and does NOT trigger a POST to the legacy
 *     `/staging/projects/{id}/generate`.
 *   - **Assertion 2**: The page does NOT call `streamGeneration` for
 *     the initial-generation path. Asserted at network level via the
 *     absence of the legacy stream POST AND the absence of any
 *     `/regenerate` POST after the Generate click (the fleet's only
 *     other entry points).
 *   - **Assertion 3**: A silent `/jobs/stream` (no SSE events for
 *     130s — comfortably above the legacy 120s watchdog threshold)
 *     does NOT surface any recovery banner over the in-flight
 *     project. The 130s window is mocked via `page.clock` so the
 *     spec runs in deterministic CI time, not wall-clock.
 *
 * Run with: npx playwright test
 *   tests/e2e/project-generation-watchdog-regression.spec.ts
 */

const SCREENSHOT_DIR = 'test-results/screenshots/project-generation-watchdog';
const PROJECT_ID = 'test-project-watchdog';
const API_BASE = 'http://localhost:8000/api/v1';
const JOB_ID = 'job-watchdog-001';

// ---------------------------------------------------------------------------
// Mock data — minimal project + job shapes (the watchdog regression doesn't
// depend on variation count, so the project is intentionally tiny).
// ---------------------------------------------------------------------------

interface MockVariation {
  id: string;
  status: string;
  created_at: string;
  updated_at: string;
}

interface MockRoom {
  id: string;
  label: string;
  original_image_url: string;
  status: string;
  variations: MockVariation[];
  created_at: string;
  updated_at: string;
}

interface MockProject {
  id: string;
  name: string;
  prompt: string;
  status: string;
  settings: Record<string, unknown>;
  rooms: MockRoom[];
  total_variations: number;
  completed_variations: number;
  created_at: string;
  updated_at: string;
}

interface MockJob {
  id: string;
  project_id: string;
  room_id: string;
  variation_id: string;
  revision: number;
  kind: string;
  status: string;
  progress?: number;
  phase?: string | null;
  attempts?: number;
  error?: string | null;
  cancel_requested?: boolean;
  created_at: string;
  updated_at: string;
}

function makeMockProject(): MockProject {
  return {
    id: PROJECT_ID,
    name: 'Watchdog Regression Project',
    prompt: 'Mock prompt — content irrelevant for this spec',
    status: 'pending',
    settings: { style: 'modern', room_count: 1, variations_per_room: 2 },
    rooms: [
      {
        id: 'room-1',
        label: 'Front Yard',
        original_image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/originals/room-1.png?sv=mock`,
        status: 'pending',
        variations: [
          { id: 'room-1-v0', status: 'pending', created_at: '', updated_at: '' },
          { id: 'room-1-v1', status: 'pending', created_at: '', updated_at: '' },
        ],
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
    ],
    total_variations: 2,
    completed_variations: 0,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
}

function makeRunningJob(): MockJob {
  return {
    id: JOB_ID,
    project_id: PROJECT_ID,
    room_id: '',
    variation_id: '',
    revision: 0,
    kind: 'generate_project',
    status: 'running',
    progress: 5,
    phase: 'generating',
    attempts: 1,
    cancel_requested: false,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// SSE helpers
// ---------------------------------------------------------------------------

function sseEvent(name: string, data: unknown): string {
  return `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
}

// ---------------------------------------------------------------------------
// Stateful router
//
// The watchdog regression cares about TWO things from the routes:
//   - the producer endpoint succeeds (so the slice hydrates and the
//     banner mounts);
//   - the SSE stream stays "silent" (no `job` events after the seed)
//     so the test can fast-forward time to prove no fleet watchdog
//     fires against this stream.
// ---------------------------------------------------------------------------

interface WatchdogRouteState {
  jobs: MockJob[];
  project: MockProject;
  enqueuePostCount: number;
}

async function setupWatchdogRoutes(page: Page, state: WatchdogRouteState) {
  await page.route(`${API_BASE}/gallery/sas-tokens`, (route: Route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        video_sas_token: 'sv=mock',
        image_sas_token: 'sv=mock',
        video_container_url: 'https://storage.blob.core.windows.net/videos',
        image_container_url: 'https://storage.blob.core.windows.net/images',
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
          body: JSON.stringify({ project: state.project }),
        });
      }
      return route.continue();
    },
  );

  await page.route(
    `${API_BASE}/staging/projects/${PROJECT_ID}/jobs`,
    (route: Route) => {
      if (route.request().method() === 'GET') {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ jobs: state.jobs }),
        });
      }
      return route.continue();
    },
  );

  // SILENT SSE stream: only the seed event, no progress events. Each
  // EventSource reconnect re-sends the seed so the slice stays at the
  // same `running` state — by construction, no fleet watchdog could
  // have any chance of being reset by this body.
  await page.route(
    new RegExp(`/api/v1/staging/projects/${PROJECT_ID}/jobs/stream(\\?.*)?$`),
    (route: Route) =>
      route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
        body: sseEvent('seed', { jobs: state.jobs }),
      }),
  );

  await page.route(
    `${API_BASE}/staging/projects/${PROJECT_ID}/jobs/generate`,
    (route: Route) => {
      if (route.request().method() === 'POST') {
        state.enqueuePostCount += 1;
        // Bump state to a running job so the next REST/SSE seed
        // mounts the in-flight banner.
        state.jobs = [makeRunningJob()];
        return route.fulfill({
          status: 202,
          contentType: 'application/json',
          body: JSON.stringify({ job_id: JOB_ID }),
        });
      }
      return route.continue();
    },
  );
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

test.describe('Project Generation — watchdog regression (issue 013)', () => {
  test('cutover to /jobs/generate + 130s silent stream produces NO recovery banner', async ({
    page,
  }) => {
    const project = makeMockProject();
    const state: WatchdogRouteState = {
      project,
      jobs: [],
      enqueuePostCount: 0,
    };
    await setupWatchdogRoutes(page, state);

    // -----------------------------------------------------------------
    // Network observer: record EVERY POST to a `/generate` or
    // `/regenerate` URL. Used by Assertions 1 + 2 below.
    // -----------------------------------------------------------------
    const seenGenerateLikePosts: string[] = [];
    page.on('request', (req) => {
      if (req.method() !== 'POST') return;
      const url = req.url();
      if (url.includes('/generate') || url.includes('/regenerate')) {
        seenGenerateLikePosts.push(url);
      }
    });

    // Install a fake clock BEFORE navigation so any setTimeout the page
    // registers (including the fleet's per-stream watchdog) operates on
    // controllable time.
    await page.clock.install();

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('domcontentloaded');

    const generateBtn = page.getByTestId('project-header-action');
    await expect(generateBtn).toBeVisible();

    const [enqueueReq] = await Promise.all([
      page.waitForRequest(
        (req) => req.url().includes('/jobs/generate') && req.method() === 'POST',
      ),
      generateBtn.click(),
    ]);

    // ===================================================================
    // Assertion 1 — Producer endpoint hit (NOT the legacy stream POST).
    // ===================================================================
    expect(enqueueReq.url()).toContain('/jobs/generate');
    expect(enqueueReq.url()).not.toMatch(
      new RegExp(`/staging/projects/${PROJECT_ID}/generate$`),
    );

    // ===================================================================
    // Assertion 2 — `streamGeneration` was NOT called for the initial-
    // generation path.
    //
    // The fleet's `streamGeneration` posts to one of two endpoints:
    //   - `/staging/projects/{id}/generate` (legacy initial generation)
    //   - `/staging/projects/{id}/rooms/{roomId}/regenerate*`
    //     (variation/room regen, kept post-cutover)
    //
    // The variation/room regen entry points are not exercised by this
    // spec (the user only clicks "Generate"), so the absence of any
    // `/generate` or `/regenerate` POST OTHER than `/jobs/generate`
    // proves no fleet stream was opened for the project-generation
    // run. Without an open stream, the fleet's per-stream watchdog
    // is by construction not registered for this run.
    // ===================================================================
    const fleetEntryPosts = seenGenerateLikePosts.filter(
      (u) => !u.includes('/jobs/generate'),
    );
    expect(fleetEntryPosts).toEqual([]);

    // -----------------------------------------------------------------
    // Banner mount sanity check — proves the slice hydrated (so any
    // hypothetical watchdog would have a stream context to monitor).
    // Without this, Assertion 3 below could pass trivially because
    // the page is simply blank.
    // -----------------------------------------------------------------
    const banner = page.getByTestId('project-generation-banner');
    await expect(banner).toBeVisible({ timeout: 10000 });
    await expect(banner).toHaveAttribute('data-status', 'running');

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-banner-mounted-pre-fastforward.png`,
      fullPage: true,
    });

    // ===================================================================
    // Assertion 3 — 130s of SSE silence produces NO recovery banner.
    //
    // Fast-forward the page's clock by 130 seconds (10s past the
    // legacy 120s watchdog threshold). If a fleet watchdog HAD been
    // registered against the silent stream, this would fire it,
    // surface a synthetic 'Stream interrupted' / 'Generation stalled'
    // signal, and the page would render a recovery banner.
    //
    // The post-cutover architecture has no per-stream watchdog for
    // this run; the assertion `recovery-banner toHaveCount(0)` after
    // the fast-forward pins that property closed.
    // ===================================================================
    await page.clock.fastForward(130_000);
    // Yield once so any setTimeout callbacks the fast-forward triggered
    // can run and re-render before we assert.
    await page.waitForTimeout(50);

    // The in-flight banner must STILL be visible (the run is still
    // running per the silent stream); recovery banners must NOT have
    // appeared.
    await expect(banner).toBeVisible();
    await expect(banner).toHaveAttribute('data-status', 'running');
    await expect(page.getByTestId('recovery-banner')).toHaveCount(0);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/02-no-recovery-banner-after-130s.png`,
      fullPage: true,
    });

    // Belt-and-suspenders: the fleet's `lostOps` array would be
    // populated if a watchdog had fired (the watchdog's primary
    // side-effect is `lostOps` accrual via `useGenerationFleet`).
    // That state would surface as a recovery banner with
    // `data-recovery-kind="stream-lost"` per recovery-state.ts.
    // We've already asserted recovery-banner count is 0, so there's
    // no stream-lost banner; this is here as the explicit named
    // assertion the issue spec requires.
    await expect(page.locator('[data-recovery-kind="stream-lost"]')).toHaveCount(0);
    await expect(page.locator('[data-recovery-kind="interrupted"]')).toHaveCount(0);
  });
});
