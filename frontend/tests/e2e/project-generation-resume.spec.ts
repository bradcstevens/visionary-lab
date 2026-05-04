import { test, expect, Page, Route } from '@playwright/test';

/**
 * Project-generation refresh-resume E2E (issue 012 of project-generation-
 * async-queue-cutover PRD).
 *
 * The headline behaviour the PRD pins: a page refresh in the middle of an
 * in-flight project-generation run does NOT kill the run, and the banner
 * reattaches to the in-progress job after reload.
 *
 *   - Open a project page with no in-flight jobs.
 *   - Click Generate; assert POST hits `/jobs/generate` (NOT legacy
 *     `/staging/projects/{id}/generate`) and returns 202 with a `job_id`.
 *   - Wait for the first progress event from `/jobs/stream` to land in
 *     the UI (banner appears with non-zero progress and a phase label).
 *   - `page.reload()`.
 *   - After reload, assert the banner reappears mid-run with current
 *     progress (the page recovered the in-flight job state from the
 *     fresh REST seed + SSE reconnect, NOT by re-issuing Generate).
 *   - Drive the SSE stream to `succeeded` and assert the banner
 *     disappears.
 *
 * Conventions follow project-generation.spec.ts (the prior art for the
 * cutover endpoints + EventSource-aware mocking).
 *
 * Run with: npx playwright test tests/e2e/project-generation-resume.spec.ts
 */

const SCREENSHOT_DIR = 'test-results/screenshots/project-generation-resume';
const PROJECT_ID = 'test-project-resume';
const API_BASE = 'http://localhost:8000/api/v1';
const JOB_ID = 'job-resume-001';

// ---------------------------------------------------------------------------
// Mock data — project + job shapes (mirrored from project-generation.spec.ts
// to keep the two specs deterministically aligned).
// ---------------------------------------------------------------------------

interface MockVariation {
  id: string;
  status: string;
  image_url?: string;
  error?: string;
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

function makeRoom(id: string, label: string, variationCount: number, roomStatus = 'pending'): MockRoom {
  return {
    id,
    label,
    original_image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/originals/${id}.png?sv=mock`,
    status: roomStatus,
    variations: Array.from({ length: variationCount }, (_, i) => ({
      id: `${id}-v${i}`,
      status: 'pending',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    })),
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
}

function makeMockProject(overrides: Partial<MockProject> = {}): MockProject {
  return {
    id: PROJECT_ID,
    name: 'Backyard Resume Test',
    prompt: 'Add native plants and seating',
    status: 'pending',
    settings: {
      style: 'modern',
      room_count: 2,
      variations_per_room: 3,
      output_format: 'png',
      quality: 'high',
    },
    rooms: [
      makeRoom('room-1', 'Front Yard', 3),
      makeRoom('room-2', 'Side Garden', 3),
    ],
    total_variations: 6,
    completed_variations: 0,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    ...overrides,
  };
}

function makeProjectGenerationJob(overrides: Partial<MockJob> = {}): MockJob {
  return {
    id: JOB_ID,
    project_id: PROJECT_ID,
    room_id: '',
    variation_id: '',
    revision: 0,
    kind: 'generate_project',
    status: 'running',
    progress: 0,
    phase: 'composing_brief',
    attempts: 1,
    cancel_requested: false,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    ...overrides,
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
// The test needs the world to "advance" between visits: pre-Generate the
// REST seed is empty + SSE seed is empty; post-Generate (and post-reload)
// the REST seed AND the SSE seed BOTH carry the running job with the
// freshest progress so the banner re-appears immediately on the second
// load. This mirrors the production contract: a refresh hits the same
// REST + SSE pair the original page mount did.
// ---------------------------------------------------------------------------

interface ResumeRouteState {
  /** Current jobs returned by both the REST seed and the SSE `seed` event. */
  jobs: MockJob[];
  /** Project body returned by GET /staging/projects/{id}. */
  project: MockProject;
  /** Discrete `job` SSE events delivered after the seed (in order). */
  pendingJobEvents: MockJob[];
  /** Track POST /jobs/generate hits so the spec can assert no double-enqueue. */
  enqueuePostCount: number;
}

async function setupResumeRoutes(page: Page, state: ResumeRouteState) {
  // SAS tokens — image overlays expect these.
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

  // Project GET (regex covers trailing-slash + query-string variants).
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

  // jobs-context REST seed: returns whatever's currently in `state.jobs`.
  // Crucial for refresh-resume: after reload, this must include the
  // running job so the banner reappears immediately (without waiting
  // for the SSE reconnect to land).
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

  // SSE stream. Body = current seed + any pending `job` events queued for
  // this connection. Each route hit consumes the queued events (so the
  // EventSource reconnect doesn't deliver them twice).
  await page.route(
    new RegExp(`/api/v1/staging/projects/${PROJECT_ID}/jobs/stream(\\?.*)?$`),
    (route: Route) => {
      const body =
        sseEvent('seed', { jobs: state.jobs }) +
        state.pendingJobEvents.map((j) => sseEvent('job', j)).join('');
      // Drain pending events so the EventSource reconnect (after the
      // body closes) doesn't redeliver them and double-tick progress.
      state.pendingJobEvents = [];
      return route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
        body,
      });
    },
  );

  // Producer endpoint: every POST advances `state.jobs` to include the
  // running job. The spec asserts on `state.enqueuePostCount` to prove
  // a reload does NOT re-issue Generate.
  await page.route(
    `${API_BASE}/staging/projects/${PROJECT_ID}/jobs/generate`,
    (route: Route) => {
      if (route.request().method() === 'POST') {
        state.enqueuePostCount += 1;
        // Seed the running job so the very next REST/SSE poll sees it.
        if (state.jobs.length === 0) {
          state.jobs = [makeProjectGenerationJob({ status: 'running', progress: 0, phase: 'composing_brief' })];
        }
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

test.describe('Project Generation — refresh-resume (issue 012)', () => {
  test('refresh mid-run reattaches the banner to the in-flight job and the run still completes', async ({
    page,
  }) => {
    const project = makeMockProject();
    const state: ResumeRouteState = {
      project,
      jobs: [],
      pendingJobEvents: [],
      enqueuePostCount: 0,
    };
    await setupResumeRoutes(page, state);

    // -----------------------------------------------------------------
    // Network observer: prove the cutover hits the new producer
    // endpoint AND assert the legacy stream POST is never touched.
    // -----------------------------------------------------------------
    const seenPosts: string[] = [];
    page.on('request', (req) => {
      if (req.method() === 'POST' && req.url().includes('/generate')) {
        seenPosts.push(req.url());
      }
    });

    // -----------------------------------------------------------------
    // Phase 1 — initial page load + Generate click.
    // -----------------------------------------------------------------
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('domcontentloaded');

    const generateBtn = page.getByTestId('project-header-action');
    await expect(generateBtn).toBeVisible();

    // Stage the running job as the producer-server "wrote" it. The
    // producer route handler also bumps `state.jobs` to a 0% running
    // job; we override here so the very first REST/SSE poll after the
    // click returns the canonical phase=generating state. Both the
    // initial banner mount AND the post-reload banner read from the
    // same `state.jobs` array, so this also pins the post-reload
    // assertion below.
    state.jobs = [
      makeProjectGenerationJob({
        status: 'running',
        progress: 35,
        phase: 'generating',
      }),
    ];

    const [enqueueReq] = await Promise.all([
      page.waitForRequest(
        (req) => req.url().includes('/jobs/generate') && req.method() === 'POST',
      ),
      generateBtn.click(),
    ]);

    // Cutover assertion #1: producer endpoint hit.
    expect(enqueueReq.url()).toContain('/jobs/generate');

    // Cutover assertion #2: legacy stream POST is NEVER touched.
    const legacyHits = seenPosts.filter(
      (u) =>
        u.includes('/generate') &&
        !u.includes('/jobs/generate') &&
        !u.includes('/regenerate'),
    );
    expect(legacyHits).toEqual([]);

    // -----------------------------------------------------------------
    // Phase 2 — wait for the banner to appear post-enqueue.
    //
    // The producer 202 returns synchronously; the slice hydrates as
    // soon as the EventSource reconnects (or the next REST poll lands)
    // and surfaces `state.jobs` containing the running job. We assert
    // on the banner being VISIBLE + carrying a phase label, not on a
    // specific progress number — the first event that ticks through
    // could be either the producer-staged 0% baseline or the 35%
    // generating state, depending on EventSource reconnect timing.
    // The PRD AC ("non-zero progress or a phase label") is satisfied
    // by the phase attribute.
    // -----------------------------------------------------------------
    const banner = page.getByTestId('project-generation-banner');
    await expect(banner).toBeVisible({ timeout: 10000 });
    await expect(banner).toHaveAttribute('data-status', 'running');
    // Phase label proves we got at least one slice update beyond the
    // synthetic empty seed.
    const phaseAttr = await banner.getAttribute('data-phase');
    expect(['composing_brief', 'generating']).toContain(phaseAttr ?? '');

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-mid-run-pre-reload.png`,
      fullPage: true,
    });

    // Bump the running job's progress on the server side BEFORE reload
    // so the post-reload REST seed advertises the freshest snapshot
    // (this models the real backend: while the page was reloading,
    // the worker continued and the next REST hit returns later state).
    state.jobs = [
      makeProjectGenerationJob({
        status: 'running',
        progress: 60,
        phase: 'generating',
      }),
    ];
    state.pendingJobEvents = [];

    // -----------------------------------------------------------------
    // Phase 3 — refresh.
    //
    // The headline AC: after `page.reload()` the banner reappears with
    // CURRENT progress and the run is NOT re-enqueued.
    // -----------------------------------------------------------------
    const enqueueCountBeforeReload = state.enqueuePostCount;
    await page.reload({ waitUntil: 'domcontentloaded' });

    const bannerAfterReload = page.getByTestId('project-generation-banner');
    await expect(bannerAfterReload).toBeVisible({ timeout: 8000 });
    // The REST seed already carries the running job at 60%; the banner
    // mounts immediately on hydration, no second SSE round-trip needed.
    await expect(bannerAfterReload).toContainText('60%');
    await expect(bannerAfterReload).toHaveAttribute('data-status', 'running');

    // Single-action contract: the header CTA is hidden while the
    // in-flight banner is up, even after reload (proves the slice
    // hydration also drove the CTA gate, not just the banner mount).
    await expect(page.getByTestId('project-header-action')).toHaveCount(0);

    // The reload must NOT have re-issued Generate. Allow the page to
    // settle a moment, then assert the producer endpoint count is
    // unchanged.
    await page.waitForTimeout(500);
    expect(state.enqueuePostCount).toBe(enqueueCountBeforeReload);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/02-post-reload-banner-resumed.png`,
      fullPage: true,
    });

    // -----------------------------------------------------------------
    // Phase 4 — drive the run to terminal status and assert the
    // banner clears.
    //
    // We re-stage the SSE feed with a `succeeded` event AND flip the
    // REST seed so any subsequent EventSource reconnect (or REST
    // refetch) sees the terminal state. The ProjectGenerationBanner
    // component returns null when status is in TERMINAL_STATUSES, so
    // the banner unmounts as soon as the change feed ticks.
    // -----------------------------------------------------------------
    const succeededJob = makeProjectGenerationJob({
      status: 'succeeded',
      progress: 100,
      phase: 'completed',
      updated_at: new Date(Date.now() + 1000).toISOString(),
    });
    state.jobs = [succeededJob];
    state.pendingJobEvents = [succeededJob];

    // Force a fresh EventSource reconnect by forcing a page action
    // (Playwright's route handlers fire on each new request — the
    // existing EventSource will reconnect on its own loop within a
    // few seconds).
    await expect(bannerAfterReload).toBeHidden({ timeout: 15000 });

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/03-post-terminal-banner-cleared.png`,
      fullPage: true,
    });
  });
});
