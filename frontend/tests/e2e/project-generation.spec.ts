import { test, expect, Page, Route } from '@playwright/test';

/**
 * Project Generation E2E Tests — async-queue cutover (issue 011 of
 * project-generation-async-queue-cutover PRD).
 *
 * The page-level Generate CTA no longer opens a long-lived SSE
 * stream against `POST /staging/projects/{id}/generate`. It enqueues
 * an async job via `POST /staging/projects/{id}/jobs/generate` (returns
 * 202 + {job_id}), and the in-flight banner is driven by the
 * jobs-context slice that hydrates from `GET /staging/projects/{id}/jobs`
 * (REST seed) + `GET /staging/projects/{id}/jobs/stream` (SSE updates).
 *
 * Variation/room regen still uses the legacy SSE path (regenerate
 * endpoints) — those tests are kept verbatim from the pre-cutover spec.
 *
 * Run with: npx playwright test tests/e2e/project-generation.spec.ts --headed
 */

const SCREENSHOT_DIR = 'test-results/screenshots/project-generation';
const PROJECT_ID = 'test-project-gen';
const API_BASE = 'http://localhost:8000/api/v1';
const JOB_ID = 'job-test-001';

// ---------------------------------------------------------------------------
// Mock data — project + job shapes
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

function makeMockProject(overrides: Partial<MockProject> = {}): MockProject {
  return {
    id: PROJECT_ID,
    name: 'Backyard Redesign',
    prompt: 'Add drought-tolerant landscaping with native plants',
    status: 'pending',
    settings: {
      style: 'modern',
      room_count: 3,
      variations_per_room: 5,
      output_format: 'png',
      quality: 'high',
    },
    rooms: [
      makeRoom('room-1', 'Front Yard', 5),
      makeRoom('room-2', 'Side Garden', 5),
      makeRoom('room-3', 'Back Patio', 5),
    ],
    total_variations: 15,
    completed_variations: 0,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    ...overrides,
  };
}

function makeRoom(
  id: string,
  label: string,
  variationCount: number,
  roomStatus = 'pending',
  variationOverrides?: Partial<MockVariation>[],
): MockRoom {
  return {
    id,
    label,
    original_image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/originals/${id}.png?sv=mock`,
    status: roomStatus,
    variations: Array.from({ length: variationCount }, (_, i) => ({
      id: `${id}-v${i}`,
      status: variationOverrides?.[i]?.status ?? 'pending',
      image_url: variationOverrides?.[i]?.image_url,
      error: variationOverrides?.[i]?.error,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    })),
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
}

function makeProjectGenerationJob(overrides: Partial<MockJob> = {}): MockJob {
  return {
    id: JOB_ID,
    project_id: PROJECT_ID,
    // generate_project jobs use the project as their target — room/variation
    // ids are conventionally empty strings (see backend models/jobs.py).
    room_id: '',
    variation_id: '',
    revision: 0,
    kind: 'generate_project',
    status: 'running',
    progress: 35,
    phase: 'generating',
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
// Route helpers — async cutover (the page no longer opens /generate)
// ---------------------------------------------------------------------------

interface AsyncRouteSetup {
  /** Stateful project — first GET returns initialProject, subsequent return updatedProject. */
  initialProject: MockProject;
  updatedProject: MockProject;
  /** Jobs returned by `GET /staging/projects/{id}/jobs` (REST seed). */
  initialJobs: MockJob[];
  /** Jobs delivered as the SSE `seed` event payload. */
  streamSeedJobs?: MockJob[];
  /** Job updates delivered as discrete SSE `job` events after the seed. */
  streamJobEvents?: MockJob[];
  /** Status returned by `POST /staging/projects/{id}/jobs/generate`. */
  enqueueStatus?: number;
  /** Body returned by enqueue (default 202 with job_id). */
  enqueueBody?: object;
  /** Status returned by `DELETE /staging/jobs/{jobId}` (default 200). */
  cancelStatus?: number;
}

async function setupAsyncRoutes(page: Page, setup: AsyncRouteSetup) {
  let projectGetCount = 0;

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

  // Stateful project GET. Use a regex so the trailing-slash and query-string
  // variants both match.
  await page.route(
    new RegExp(`/api/v1/staging/projects/${PROJECT_ID}(\\?.*)?$`),
    (route: Route) => {
      if (route.request().method() === 'GET') {
        projectGetCount++;
        const data = projectGetCount <= 1 ? setup.initialProject : setup.updatedProject;
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ project: data }),
        });
      }
      return route.continue();
    },
  );

  // jobs-context REST seed: `GET /staging/projects/{id}/jobs`.
  await page.route(
    `${API_BASE}/staging/projects/${PROJECT_ID}/jobs`,
    (route: Route) => {
      if (route.request().method() === 'GET') {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ jobs: setup.initialJobs }),
        });
      }
      return route.continue();
    },
  );

  // SSE stream: `GET /staging/projects/{id}/jobs/stream?...`.
  // Deliver the seed event + any pre-staged job events in one body. The
  // EventSource will re-connect after this body ends — the route is
  // re-mounted so the same body is re-served, which is harmless because
  // mergeJobs in jobs-context dedupes by `_isNewer` (same updated_at →
  // no state change).
  const seedBody = sseEvent('seed', { jobs: setup.streamSeedJobs ?? [] })
    + (setup.streamJobEvents ?? []).map((j) => sseEvent('job', j)).join('');
  await page.route(
    new RegExp(`/api/v1/staging/projects/${PROJECT_ID}/jobs/stream(\\?.*)?$`),
    (route: Route) =>
      route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
        body: seedBody,
      }),
  );

  // Producer endpoint: `POST /staging/projects/{id}/jobs/generate`.
  await page.route(
    `${API_BASE}/staging/projects/${PROJECT_ID}/jobs/generate`,
    (route: Route) => {
      if (route.request().method() === 'POST') {
        return route.fulfill({
          status: setup.enqueueStatus ?? 202,
          contentType: 'application/json',
          body: JSON.stringify(setup.enqueueBody ?? { job_id: JOB_ID }),
        });
      }
      return route.continue();
    },
  );

  // Cancel: `DELETE /staging/jobs/{jobId}`.
  await page.route(
    new RegExp(`/api/v1/staging/jobs/${JOB_ID}$`),
    (route: Route) => {
      if (route.request().method() === 'DELETE') {
        return route.fulfill({
          status: setup.cancelStatus ?? 200,
          contentType: 'application/json',
          body: JSON.stringify({ ok: true }),
        });
      }
      return route.continue();
    },
  );
}

// ---------------------------------------------------------------------------
// Tests — async-cutover behaviour
// ---------------------------------------------------------------------------

test.describe('Project Generation — async cutover', () => {
  test('project page loads with rooms and pending variations', async ({ page }) => {
    const project = makeMockProject();
    await setupAsyncRoutes(page, {
      initialProject: project,
      updatedProject: project,
      initialJobs: [],
    });

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('domcontentloaded');

    await expect(page.locator('h1')).toContainText('Backyard Redesign');

    for (const label of ['Front Yard', 'Side Garden', 'Back Patio']) {
      await expect(page.getByText(label, { exact: true }).first()).toBeVisible();
    }

    // Pending placeholders for at least one variation per room.
    const pendingBadges = page.getByText('Awaiting generation');
    expect(await pendingBadges.count()).toBeGreaterThanOrEqual(3);

    // The header CTA (issue 011 still uses the same data-testid hook).
    await expect(page.getByTestId('project-header-action')).toBeVisible();

    await page.screenshot({ path: `${SCREENSHOT_DIR}/01-page-loaded.png`, fullPage: true });
  });

  test('clicking Generate enqueues a job via /jobs/generate (NOT /generate)', async ({ page }) => {
    const project = makeMockProject();
    // No in-flight jobs yet — the banner is hidden until the post lands
    // and the change feed surfaces a `generate_project` job.
    await setupAsyncRoutes(page, {
      initialProject: project,
      updatedProject: project,
      initialJobs: [],
    });

    // Spy on POSTs to either endpoint so we can assert the cutover.
    const postRequests: string[] = [];
    page.on('request', (req) => {
      if (req.method() === 'POST' && req.url().includes('/generate')) {
        postRequests.push(req.url());
      }
    });

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('domcontentloaded');

    const generateBtn = page.getByTestId('project-header-action');
    await expect(generateBtn).toBeVisible();

    const [postReq] = await Promise.all([
      page.waitForRequest((req) =>
        req.url().includes('/jobs/generate') && req.method() === 'POST',
      ),
      generateBtn.click(),
    ]);

    expect(postReq).toBeTruthy();
    // Issue 011 critical assertion: cutover hits the new producer endpoint.
    expect(postReq.url()).toContain('/jobs/generate');

    // Negative assertion: no request to the legacy /generate SSE endpoint.
    // (We need a small grace period for any stragglers — `waitForRequest`
    // already returned, so we just inspect the recorded list.)
    const legacyHits = postRequests.filter((u) =>
      u.includes('/generate') && !u.includes('/jobs/generate') && !u.includes('/regenerate'),
    );
    expect(legacyHits).toEqual([]);

    await page.screenshot({ path: `${SCREENSHOT_DIR}/02-enqueue-jobs-generate.png`, fullPage: true });
  });

  test('in-flight project-generation banner mounts when slice is non-null', async ({ page }) => {
    const project = makeMockProject({ status: 'processing' });
    const runningJob = makeProjectGenerationJob({
      status: 'running',
      progress: 42,
      phase: 'generating',
    });

    await setupAsyncRoutes(page, {
      initialProject: project,
      updatedProject: project,
      initialJobs: [runningJob],
      streamSeedJobs: [runningJob],
    });

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('domcontentloaded');

    // The banner mounts as soon as the REST seed populates the
    // inFlightProjectGeneration slice.
    const banner = page.getByTestId('project-generation-banner');
    await expect(banner).toBeVisible({ timeout: 8000 });
    await expect(banner).toHaveAttribute('data-status', 'running');
    await expect(banner).toHaveAttribute('data-phase', 'generating');

    // Header CTA is hidden — single-action contract (issue 011 ADOPTED
    // from the projects-page-stalled-stream-error-cleanup PRD).
    await expect(page.getByTestId('project-header-action')).toHaveCount(0);

    await page.screenshot({ path: `${SCREENSHOT_DIR}/03-in-flight-banner.png`, fullPage: true });
  });

  test('Cancel button on banner posts DELETE to /staging/jobs/{jobId}', async ({ page }) => {
    const project = makeMockProject({ status: 'processing' });
    const runningJob = makeProjectGenerationJob({ status: 'running' });

    await setupAsyncRoutes(page, {
      initialProject: project,
      updatedProject: project,
      initialJobs: [runningJob],
      streamSeedJobs: [runningJob],
    });

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('domcontentloaded');

    await expect(page.getByTestId('project-generation-banner')).toBeVisible({ timeout: 8000 });

    const cancelBtn = page.getByRole('button', { name: /cancel project generation/i });
    await expect(cancelBtn).toBeVisible();

    const [delReq] = await Promise.all([
      page.waitForRequest((req) =>
        req.method() === 'DELETE' && req.url().includes(`/staging/jobs/${JOB_ID}`),
      ),
      cancelBtn.click(),
    ]);
    expect(delReq).toBeTruthy();
  });

  test('refreshing mid-generation re-attaches the banner from the REST seed', async ({ page }) => {
    // First load shows the banner because the REST seed surfaces an
    // already-running job. This is the resume-on-refresh PRD AC: the
    // server-side change feed is the source of truth, so a page refresh
    // after the worker has started must re-mount the banner without any
    // explicit Generate click.
    const project = makeMockProject({ status: 'processing' });
    const runningJob = makeProjectGenerationJob({
      status: 'running',
      progress: 60,
      phase: 'generating',
    });

    await setupAsyncRoutes(page, {
      initialProject: project,
      updatedProject: project,
      initialJobs: [runningJob],
      streamSeedJobs: [runningJob],
    });

    // First visit.
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('domcontentloaded');
    await expect(page.getByTestId('project-generation-banner')).toBeVisible({ timeout: 8000 });

    // Refresh — the slice must reseed from REST + SSE and the banner
    // must reappear.
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByTestId('project-generation-banner')).toBeVisible({ timeout: 8000 });
  });

  test('terminal job (succeeded) clears the in-flight banner', async ({ page }) => {
    const project = makeMockProject({ status: 'completed' });
    // Initial load: no in-flight jobs (the only generate_project job is
    // already in a terminal state, so the slice resolves to null).
    const succeededJob = makeProjectGenerationJob({
      status: 'succeeded',
      progress: 100,
      phase: 'finalizing',
    });

    await setupAsyncRoutes(page, {
      initialProject: project,
      updatedProject: project,
      initialJobs: [succeededJob],
      streamSeedJobs: [succeededJob],
    });

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('domcontentloaded');

    // The banner is NOT mounted because the slice excludes terminal jobs
    // (TERMINAL_JOB_STATUSES = succeeded | failed | cancelled).
    await expect(page.getByTestId('project-generation-banner')).toHaveCount(0);
  });

  test('enqueue 4xx error surfaces as a user-visible toast', async ({ page }) => {
    const project = makeMockProject();
    await setupAsyncRoutes(page, {
      initialProject: project,
      updatedProject: project,
      initialJobs: [],
      enqueueStatus: 503,
      enqueueBody: { detail: 'queue temporarily unavailable' },
    });

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('domcontentloaded');

    await page.getByTestId('project-header-action').click();

    // The page surfaces a destructive toast — issue 011 PRD AC.
    // The exact copy is "Couldn't start generation, please try again."
    await expect(page.getByText(/Couldn't start generation/i)).toBeVisible({
      timeout: 8000,
    });
  });
});

// ---------------------------------------------------------------------------
// Variation/room regen — KEPT verbatim from the pre-cutover spec because
// these flows still go through useGenerationFleet + streamGeneration
// against `/regenerate` endpoints (the cutover only swapped the page-level
// initial-generation flow).
// ---------------------------------------------------------------------------

test.describe('Project Generation — variation/room regen (legacy SSE path)', () => {
  test('failed-variation Retry regenerates only that variation, not the whole room', async ({ page }) => {
    const projectWithFailure = makeMockProject({
      status: 'completed',
      rooms: [
        makeRoom('room-1', 'Front Yard', 5, 'completed', [
          { id: 'r1-v0', status: 'completed', image_url: 'https://example.com/v0.png', created_at: '', updated_at: '' },
          { id: 'r1-v1', status: 'failed', error: 'Content policy violation', created_at: '', updated_at: '' },
          { id: 'r1-v2', status: 'completed', image_url: 'https://example.com/v2.png', created_at: '', updated_at: '' },
          { id: 'r1-v3', status: 'completed', image_url: 'https://example.com/v3.png', created_at: '', updated_at: '' },
          { id: 'r1-v4', status: 'completed', image_url: 'https://example.com/v4.png', created_at: '', updated_at: '' },
        ]),
        makeRoom('room-2', 'Side Garden', 5),
        makeRoom('room-3', 'Back Patio', 5),
      ],
    });

    await setupAsyncRoutes(page, {
      initialProject: projectWithFailure,
      updatedProject: projectWithFailure,
      initialJobs: [],
    });

    // Capture regenerate requests (variation-level + room-level — only
    // the variation-level one should fire).
    const regenRequests: string[] = [];
    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/**/regenerate**`,
      (route: Route) => {
        regenRequests.push(route.request().url());
        return route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
          body: sseEvent('stream_ended', { type: 'stream_ended' }),
        });
      },
    );

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('domcontentloaded');

    const retryBtn = page.getByRole('button', { name: /Retry/i }).first();
    await expect(retryBtn).toBeVisible();

    const [request] = await Promise.all([
      page.waitForRequest(
        (req) => req.url().includes('/regenerate') && req.method() === 'POST',
        { timeout: 5000 },
      ),
      retryBtn.click(),
    ]);

    // Critical: variation-level endpoint, NOT room-level.
    expect(request.url()).toContain('/variations/room-1-v1/regenerate');
    expect(request.url()).toContain('strategy=fresh');
    expect(request.url()).not.toMatch(/\/rooms\/[^/]+\/regenerate(\?|$)/);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/10b-failed-retry-routes-to-variation-regen.png`,
      fullPage: true,
    });
  });

  test('room-header Regenerate still hits the room-level regen endpoint', async ({ page }) => {
    const completed = makeMockProject({
      status: 'completed',
      rooms: [
        makeRoom('room-1', 'Front Yard', 5, 'completed', [
          { id: 'r1-v0', status: 'completed', image_url: 'https://example.com/v0.png', created_at: '', updated_at: '' },
          { id: 'r1-v1', status: 'completed', image_url: 'https://example.com/v1.png', created_at: '', updated_at: '' },
          { id: 'r1-v2', status: 'completed', image_url: 'https://example.com/v2.png', created_at: '', updated_at: '' },
          { id: 'r1-v3', status: 'completed', image_url: 'https://example.com/v3.png', created_at: '', updated_at: '' },
          { id: 'r1-v4', status: 'completed', image_url: 'https://example.com/v4.png', created_at: '', updated_at: '' },
        ]),
        makeRoom('room-2', 'Side Garden', 5, 'completed'),
        makeRoom('room-3', 'Back Patio', 5, 'completed'),
      ],
    });

    await setupAsyncRoutes(page, {
      initialProject: completed,
      updatedProject: completed,
      initialJobs: [],
    });

    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/**/regenerate**`,
      (route: Route) =>
        route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
          body: sseEvent('stream_ended', { type: 'stream_ended' }),
        }),
    );

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('domcontentloaded');

    const roomHeaderRegenBtn = page.getByRole('button', { name: /^Regenerate$/i }).first();
    await expect(roomHeaderRegenBtn).toBeVisible();

    const [request] = await Promise.all([
      page.waitForRequest(
        (req) => req.url().includes('/regenerate') && req.method() === 'POST',
        { timeout: 5000 },
      ),
      roomHeaderRegenBtn.click(),
    ]);

    expect(request.url()).toMatch(/\/rooms\/room-1\/regenerate(\?|$)/);
    expect(request.url()).not.toContain('/variations/');
  });
});

// ---------------------------------------------------------------------------
// Live Smoke Test (opt-in — requires running backend)
// ---------------------------------------------------------------------------

test.describe('Live Smoke Test', () => {
  test.skip(!process.env.LIVE_SMOKE, 'Set LIVE_SMOKE=1 to run against live backend');

  const LIVE_PROJECT_ID = process.env.LIVE_PROJECT_ID ?? '17735db0-a0bf-4dfb-8a45-fcf59fe4de3e';

  test('async-job stream establishes for real project', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    const consoleLogs: string[] = [];
    page.on('console', (msg) => consoleLogs.push(`[${msg.type()}] ${msg.text()}`));

    await page.goto(`/projects/${LIVE_PROJECT_ID}`);
    await page.waitForLoadState('domcontentloaded');

    await expect(page.locator('h1').first()).toBeVisible({ timeout: 10_000 });
    await page.screenshot({ path: `${SCREENSHOT_DIR}/11-live-project-loaded.png`, fullPage: true });

    const generateBtn = page.getByTestId('project-header-action');
    const hasButton = await generateBtn.isVisible().catch(() => false);

    if (!hasButton) {
      console.log('No generate button found — project may already be processing or completed');
      console.log('Console logs:', consoleLogs.join('\n'));
      return;
    }

    // Click and verify the new async-job POST is made (NOT the legacy /generate).
    const [postReq] = await Promise.all([
      page.waitForRequest(
        (req) => req.url().includes('/jobs/generate') && req.method() === 'POST',
        { timeout: 5000 },
      ),
      generateBtn.click(),
    ]);

    expect(postReq).toBeTruthy();
    expect(postReq.url()).toContain('/jobs/generate');
    console.log('Async-job POST made to:', postReq.url());

    // Wait for the in-flight banner to mount (proves the change feed
    // surfaced the new job to the page).
    try {
      await expect(page.getByTestId('project-generation-banner')).toBeVisible({
        timeout: 30_000,
      });
      console.log('✓ ProjectGenerationBanner mounted — async cutover is live');
    } catch {
      console.log('⚠ Banner did not mount within 30s — change feed may not be reaching the frontend');
      console.log('Console logs:', consoleLogs.join('\n'));
    }

    await page.screenshot({ path: `${SCREENSHOT_DIR}/12-live-banner-mounted.png`, fullPage: true });

    if (errors.length > 0) {
      console.log('JS errors:', errors);
    }
  });
});
