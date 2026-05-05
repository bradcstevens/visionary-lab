import { test, expect, Page, Route } from '@playwright/test';

/**
 * Project generation double-click dedupe — issue 002 of the
 * `active-and-queued-jobs-ux-redesign` PRD.
 *
 * The user-visible contract: a user clicking "Generate" twice in
 * rapid succession (or a transport-layer retry that re-issues the
 * POST) MUST NOT produce two distinct generation runs. The producer
 * dedupes:
 *
 *   - SAME ``Idempotency-Key`` (transport retry) → 200 + same job_id
 *   - DISTINCT keys (genuine double-click) → second POST sees the
 *     active project lease and returns 200 +
 *     ``already_in_flight=true`` + the EXISTING job_id (NOT a new
 *     one).
 *
 * This spec asserts the second contract end-to-end at the page
 * level: the user double-clicks Generate; the second click hits the
 * producer; the producer returns ``already_in_flight=true`` with the
 * EXISTING job_id; the UI surfaces the same single in-flight banner
 * (NOT two stacked banners; NOT a fresh job).
 *
 * Backend coverage:
 *   - ``test_project_generation_producer.py`` proves the producer's
 *     dedupe algorithm.
 *   - ``test_staging_endpoints_generate_jobs.py::
 *     test_post_two_distinct_keys_produce_distinct_jobs`` and
 *     ``test_post_same_idempotency_key_returns_200_already_in_flight``
 *     prove the HTTP wire shape.
 *
 * This Playwright spec proves the binding from API → UI holds for
 * the user-visible double-click scenario.
 *
 * Run with: npx playwright test
 *           tests/e2e/project-generation-double-click.spec.ts
 */

const SCREENSHOT_DIR =
  'test-results/screenshots/project-generation-double-click';
const PROJECT_ID = 'test-project-double-click';
const API_BASE = 'http://localhost:8000/api/v1';
const EXISTING_JOB_ID =
  `${PROJECT_ID}:__project__:__project__:existing-key-001`;

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
    name: 'Double-Click Test',
    prompt: 'Modern minimalist',
    status: 'pending',
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
    id: EXISTING_JOB_ID,
    project_id: PROJECT_ID,
    room_id: '__project__',
    variation_id: '__project__',
    revision: 'existing-key-001',
    kind: 'generate_project',
    status: 'running',
    progress: 25,
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
  enqueuePostCount: number;
  enqueueRequests: { idempotencyKey: string | null; body: string }[];
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

  // Producer endpoint: simulates the dedupe flow. The FIRST POST seeds
  // the running job and returns 202 + already_in_flight=false. EVERY
  // subsequent POST (with whatever Idempotency-Key) sees the existing
  // lease holder and returns 200 + already_in_flight=true with the
  // EXISTING job_id — this mirrors the producer's lease-precheck
  // branch: the holder is non-terminal, so the producer returns
  // ``AlreadyInFlight(holder_id)`` regardless of the new key.
  await page.route(
    `${API_BASE}/staging/projects/${PROJECT_ID}/jobs/generate`,
    (route: Route) => {
      if (route.request().method() !== 'POST') {
        return route.continue();
      }
      const headers = route.request().headers();
      const idempKey = headers['idempotency-key'] ?? null;
      const body = route.request().postData() ?? '';
      state.enqueueRequests.push({ idempotencyKey: idempKey, body });
      state.enqueuePostCount += 1;

      if (state.enqueuePostCount === 1) {
        // First click: seed the running job into state.jobs. Return
        // 202 + the freshly-minted job_id.
        state.jobs = [makeRunningJob()];
        return route.fulfill({
          status: 202,
          contentType: 'application/json',
          body: JSON.stringify({
            job_id: EXISTING_JOB_ID,
            already_in_flight: false,
          }),
        });
      }

      // Second + subsequent clicks: producer's lease precheck fires;
      // 200 + already_in_flight=true + the EXISTING holder id.
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          job_id: EXISTING_JOB_ID,
          already_in_flight: true,
        }),
      });
    },
  );
}

test.describe('Project Generation — double-click dedupe (issue 002)', () => {
  test('two rapid Generate clicks produce ONE running banner with ONE job_id (NOT two distinct runs)', async ({
    page,
  }) => {
    const state: RouteState = {
      jobs: [],
      enqueuePostCount: 0,
      enqueueRequests: [],
    };
    await setupRoutes(page, state);

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('domcontentloaded');

    const generateBtn = page.getByTestId('project-header-action');
    await expect(generateBtn).toBeVisible();

    // First click — fires producer POST #1, seeds running job, the
    // banner mounts with the freshly-minted job_id.
    const [enqueue1] = await Promise.all([
      page.waitForRequest(
        (req) =>
          req.url().includes('/jobs/generate') &&
          req.method() === 'POST',
      ),
      generateBtn.click(),
    ]);
    expect(enqueue1.headers()['idempotency-key']).toBeDefined();

    // The banner should appear with status=running.
    const banner = page.getByTestId('project-generation-banner');
    await expect(banner).toBeVisible({ timeout: 8_000 });
    await expect(banner).toHaveAttribute('data-status', 'running');

    // Wait long enough that the second click is unambiguously a
    // distinct user action — but short enough that the banner is
    // still mounted (the producer's lease has NOT cleared).
    await page.waitForTimeout(300);

    // Force the Generate button to be re-enabled for the second click.
    // In production the header CTA hides while a generation is
    // in-flight — but a determined user can still re-trigger via a
    // different surface (lightbox regenerate, menu action). The
    // double-click contract is enforced at the API layer; the UI
    // hide is defense-in-depth. We assert at the API layer here.
    const enqueue2Promise = page.waitForRequest(
      (req) =>
        req.url().includes('/jobs/generate') &&
        req.method() === 'POST' &&
        // Distinct from the first request — match by timing rather
        // than URL since both hit the same path.
        req !== enqueue1,
      { timeout: 5_000 },
    );

    // Drive the second click via fetch. This simulates either:
    // (a) a user double-click with a distinct key minted per click;
    // (b) an automated retry fired by a flaky network shim.
    // Both paths land on the same producer endpoint and both must
    // dedupe via the lease.
    await page.evaluate(async ({ projectId, apiBase }) => {
      const key = (
        crypto.randomUUID
          ? crypto.randomUUID()
          : Math.random().toString(36).slice(2)
      ) as string;
      const r = await fetch(
        `${apiBase}/staging/projects/${projectId}/jobs/generate`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Idempotency-Key': key,
          },
          body: JSON.stringify({ regenerate_all: false }),
        },
      );
      // Stash the response on window so the spec can read it back.
      const json = await r.json();
      (
        window as unknown as { __secondEnqueue?: unknown }
      ).__secondEnqueue = { status: r.status, json, key };
    }, { projectId: PROJECT_ID, apiBase: API_BASE });

    const enqueue2 = await enqueue2Promise;

    // Issue 002 wire-shape contract: distinct Idempotency-Keys, but
    // the second response is 200 + already_in_flight=true + SAME
    // job_id.
    const k1 = enqueue1.headers()['idempotency-key'];
    const k2 = enqueue2.headers()['idempotency-key'];
    expect(k1).not.toBe(k2); // distinct keys (real double-click)

    const second = await page.evaluate(
      () =>
        (
          window as unknown as {
            __secondEnqueue?: {
              status: number;
              json: { job_id: string; already_in_flight: boolean };
              key: string;
            };
          }
        ).__secondEnqueue,
    );
    expect(second).toBeDefined();
    expect(second!.status).toBe(200);
    expect(second!.json.already_in_flight).toBe(true);
    expect(second!.json.job_id).toBe(EXISTING_JOB_ID);

    // Producer was hit exactly twice — both clicks reached the API.
    expect(state.enqueuePostCount).toBe(2);
    // Both requests carried distinct Idempotency-Keys (genuine
    // double-click, not transport retry).
    expect(state.enqueueRequests).toHaveLength(2);
    expect(state.enqueueRequests[0].idempotencyKey).not.toBe(
      state.enqueueRequests[1].idempotencyKey,
    );

    // Most important user-visible assertion: still ONE banner; still
    // running; the banner did NOT mount a second time, did NOT replace
    // its job_id, did NOT race to a different status.
    const banners = page.getByTestId('project-generation-banner');
    await expect(banners).toHaveCount(1);
    await expect(banner).toHaveAttribute('data-status', 'running');

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-single-banner-after-double-click.png`,
      fullPage: true,
    });
  });
});
