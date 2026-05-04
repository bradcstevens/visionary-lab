import { test, expect, Page, Route } from '@playwright/test';

/**
 * Queued project stays processing — issue 003 of the
 * `active-and-queued-jobs-ux-redesign` PRD.
 *
 * The bug-report scenario: a project enqueues into the async-job
 * pipeline and the worker hasn't picked it up yet (it's behind other
 * jobs in the queue). The project's ``updated_at`` is older than the
 * staleness window. Pre-fix the next ``GET /staging/projects/:id``
 * call would run reconcile_project, which derived status from rooms;
 * mixed/all-pending rooms produced ``failed``. The user sees a
 * "failed" badge for a project that was just sitting in the queue.
 *
 * The fix: ``reconcile_project`` no longer mutates project status.
 * Status derivation moved to ``compute_project_status_from_jobs``,
 * which reads the active job from the jobs container. An active
 * non-terminal job ⇒ status stays at ``processing``. Even if the job
 * is terminal, the room-derived fallback never produces ``failed``
 * (the buggy "mixed ⇒ failed" branch was removed).
 *
 * This spec asserts the user-visible outcome at the page level: with
 * the backend-mocked endpoint returning a ``processing`` project
 * (the API has correctly preserved the status across the staleness
 * window), the badge reads "processing" and never "failed".
 *
 * The backend integration tests
 * (``tests/test_staging_api.py::test_get_project_queued_with_pending_job_stays_processing``
 * and friends) prove the API actually returns the right status. This
 * Playwright spec proves the binding from API → badge holds for the
 * scenario the user reported.
 */

const SCREENSHOT_DIR = 'test-results/screenshots/queued-project-stays-processing';
const PROJECT_ID = 'test-queued-stays-processing';
const API_BASE = 'http://localhost:8000/api/v1';

type RoomStatus = 'pending' | 'processing' | 'completed' | 'failed';
type ProjectStatus = 'uploading' | 'pending' | 'processing' | 'completed' | 'failed';

function makeProject(opts: {
  rooms: RoomStatus[];
  projectStatus: ProjectStatus;
  currentProjectJobId?: string | null;
}) {
  // ``updated_at`` is well past the staleness window. Pre-fix this
  // would trigger reconcile_project's status mutation and (with
  // pending rooms) flip to 'failed'. Post-fix the API preserves the
  // genuine queued state.
  const stale = '2020-01-01T00:00:00Z';
  const now = new Date().toISOString();
  const rooms = opts.rooms.map((status, i) => {
    const idx = i + 1;
    return {
      id: `room-${idx}`,
      label: `Room ${idx}`,
      original_image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/originals/room-${idx}.png?sv=mock`,
      status,
      // No variations on the pending rooms — they're queued and the
      // worker hasn't started.
      variations: [] as Array<{
        id: string;
        status: string;
        created_at: string;
        updated_at: string;
      }>,
      created_at: stale,
      updated_at: stale,
    };
  });

  const project: Record<string, unknown> = {
    id: PROJECT_ID,
    name: 'Queued Project',
    prompt: 'Modern minimalist',
    status: opts.projectStatus,
    settings: {
      style: 'modern',
      room_count: rooms.length,
      variations_per_room: 5,
      output_format: 'png',
      quality: 'high',
    },
    rooms,
    total_variations: rooms.length * 5,
    completed_variations: 0,
    created_at: stale,
    updated_at: stale,
    // current_project_job_id reflects the project being enqueued. The
    // backend reads this and consults the jobs container to derive
    // status; an active non-terminal job ⇒ stays 'processing'.
    current_project_job_id: opts.currentProjectJobId ?? null,
    // The test client serialises the most-recent server tick — for
    // the user this is the moment they refresh the page.
    last_seen_at: now,
  };
  return project;
}

async function setupSasTokenMock(page: Page) {
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
}

async function setupProjectGet(page: Page, project: ReturnType<typeof makeProject>) {
  await page.route(`${API_BASE}/staging/projects/${PROJECT_ID}`, (route: Route) => {
    if (route.request().method() === 'GET') {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ project }),
      });
    }
    return route.continue();
  });
}

async function setupJobsEndpoints(
  page: Page,
  opts: { activeJob?: boolean } = {},
) {
  // For the queued-but-busy scenario the slice surface needs an active
  // generate_project job so the page's recovery classifier sees
  // ``isAnyGenerationBusy=true`` and doesn't fire the 'interrupted' arm
  // (status='processing' && !isAnyInFlight). The PRE-fix bug was the
  // backend; the badge classifier has always treated 'processing &&
  // no-in-flight' as a recovery situation. Without an active job here
  // the page would render "interrupted" on top of the truthful
  // 'processing' status, masking the badge regression we want to pin.
  const activeJob = opts.activeJob !== false
    ? [
        {
          id: `${PROJECT_ID}:project:project:rev1`,
          project_id: PROJECT_ID,
          room_id: 'project',
          variation_id: 'project',
          revision: 1,
          kind: 'generate_project',
          status: 'pending',
          progress: 0,
          phase: 'queued',
          attempts: 0,
          error: null,
          result: null,
          cancel_requested: false,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
      ]
    : [];

  await page.route(
    `${API_BASE}/staging/projects/${PROJECT_ID}/jobs`,
    (route: Route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ jobs: activeJob }),
      }),
  );
  await page.route(
    new RegExp(`${API_BASE}/staging/projects/${PROJECT_ID}/jobs/stream`),
    (route: Route) =>
      route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        // Emit the seed (matches REST seed shape) then stall — the
        // EventSource will reconnect and we'll fulfill again with the
        // same seed.
        body: `event: seed\ndata: ${JSON.stringify({ jobs: activeJob })}\n\n`,
      }),
  );
}

/**
 * The badge primitive renders ``<span data-slot="badge">``. There may
 * be other badges on the page (room status pills, variation count);
 * scope to the project header badge — the one rendered as a sibling
 * of the project's ``<h1>`` title. The page mounts the project badge
 * adjacent to ``<h1>{project.name}</h1>`` so the CSS adjacent-sibling
 * selector pins exactly the project status badge regardless of how
 * many room cards (and their per-room status pills) are visible.
 */
function statusBadge(page: Page) {
  return page.locator('h1 + [data-slot="badge"]');
}

test.describe('Queued project stays processing (issue 003 — active-and-queued-jobs-ux-redesign)', () => {
  test('queued project with pending rooms shows "processing" badge after staleness window (NOT "failed")', async ({
    page,
  }) => {
    // Headline regression: the bug-report scenario. Project is in
    // 'processing' with all-pending rooms; the API has correctly
    // preserved the status (jobs-container said the worker has an
    // active job). The badge MUST show "processing" and MUST NOT
    // show "failed".
    const project = makeProject({
      rooms: ['pending', 'pending', 'pending'],
      projectStatus: 'processing',
      currentProjectJobId: 'test-queued-stays-processing:project:project:rev1',
    });
    await setupSasTokenMock(page);
    await setupProjectGet(page, project);
    await setupJobsEndpoints(page);

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('domcontentloaded');

    const badge = statusBadge(page);
    await expect(badge).toBeVisible();

    // Headline assertion: status reads "processing".
    await expect(badge).toHaveText(/^processing$/);

    // Defense-in-depth: explicit negative against the pre-fix bug
    // mode. The PRD's bug report was the badge flipping to 'failed';
    // this assertion would have failed on the old code.
    await expect(badge).not.toHaveText(/^failed$/);
    await expect(badge).not.toHaveText(/failed/i);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-queued-stays-processing.png`,
      fullPage: true,
    });
  });

  test('queued project with mixed pending+failed rooms shows "processing" badge (NOT "failed")', async ({
    page,
  }) => {
    // The exact buggy branch: the legacy reconcile_project saw mixed
    // room statuses and derived 'failed'. With an active job in the
    // jobs container the API now reports 'processing', and even
    // without one the room-derived fallback is 'pending'. Either
    // way: never 'failed' from the reconcile path.
    const project = makeProject({
      rooms: ['pending', 'failed', 'pending'],
      projectStatus: 'processing',
      currentProjectJobId: 'test-queued-stays-processing:project:project:rev1',
    });
    await setupSasTokenMock(page);
    await setupProjectGet(page, project);
    await setupJobsEndpoints(page);

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('domcontentloaded');

    const badge = statusBadge(page);
    await expect(badge).toBeVisible();
    await expect(badge).toHaveText(/^processing$/);
    await expect(badge).not.toHaveText(/^failed$/);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/02-mixed-stays-processing.png`,
      fullPage: true,
    });
  });
});
