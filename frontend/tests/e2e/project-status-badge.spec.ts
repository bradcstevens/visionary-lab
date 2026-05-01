import { test, expect, Page, Route } from '@playwright/test';

/**
 * Project status badge — issue 001 of the
 * `projects-page-improvements` PRD.
 *
 * The PRD's headlining bug ("Issue 1"): when a single image (room)
 * finished all its variations, the project header badge flipped to
 * `completed` even though several other uploaded images were still
 * untouched (`pending`). The user could not trust the status badge
 * to tell them whether their work was actually done.
 *
 * The fix replaces three duplicated, drift-prone inline status
 * branches with a single pure helper
 * `ProjectStatusCalculator.compute_status` on the backend. The
 * frontend already binds `<Badge>` directly to `project.status`, so
 * once the backend persists the truthful value the badge corrects
 * itself with no code change in the frontend except this spec.
 *
 * This spec mocks two project shapes and asserts the visible badge
 * text:
 *
 *   1. multi-room mid-generation (1 completed + 4 pending) →
 *      project.status === 'pending' →
 *      visible badge text reads "ready" (the page maps
 *      `pending` → "ready" for friendlier UX in the JSX at
 *      `app/projects/[id]/page.tsx` line ~532).
 *
 *   2. all-completed multi-room project (5 completed) →
 *      project.status === 'completed' →
 *      visible badge text reads "completed".
 *
 * No real generation flow is exercised; the test proves the binding
 * from `project.status` (the field the calculator now drives) to
 * the badge's user-visible text. The backend unit tests
 * (`tests/test_project_status_calculator.py` +
 * `tests/test_staging_pipeline.py::TestProjectStatusDelegatesToCalculator`)
 * prove the calculator returns the right value.
 *
 * The badge's selector strategy: there is exactly one element on
 * the page with the badge's variant classes anchored next to the
 * `<h1>` project name, but the safest cross-render-implementation
 * approach is to scope to the header `<h1>` neighborhood and assert
 * on the visible text via `getByText` with a tight regex. The badge
 * lives in `<div className="flex items-center gap-3">` next to the
 * <h1>{project.name}</h1>, so a `.locator('h1 + *')` reach is
 * brittle; instead this spec locates the badge by its predictable
 * text content within the header.
 */

const SCREENSHOT_DIR = 'test-results/screenshots/project-status-badge';
const PROJECT_ID = 'test-status-badge';
const API_BASE = 'http://localhost:8000/api/v1';

type RoomStatus = 'pending' | 'processing' | 'completed' | 'failed';
type ProjectStatus = 'uploading' | 'pending' | 'processing' | 'completed' | 'failed';

interface MockVariation {
  id: string;
  status: string;
  image_url?: string;
  created_at: string;
  updated_at: string;
}

interface MockRoom {
  id: string;
  label: string;
  original_image_url: string;
  status: RoomStatus;
  variations: MockVariation[];
  created_at: string;
  updated_at: string;
}

function makeProject(roomStatuses: RoomStatus[], projectStatus: ProjectStatus) {
  const now = new Date().toISOString();
  const rooms: MockRoom[] = roomStatuses.map((status, i) => {
    const idx = i + 1;
    return {
      id: `room-${idx}`,
      label: `Room ${idx}`,
      original_image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/originals/room-${idx}.png?sv=mock`,
      status,
      variations: Array.from({ length: 5 }, (_, v) => {
        const base: MockVariation = {
          id: `r${idx}-v${v}`,
          status: status === 'completed' ? 'completed' : 'pending',
          created_at: now,
          updated_at: now,
        };
        if (status === 'completed') {
          base.image_url = `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-${idx}/v${v}.png`;
        }
        return base;
      }),
      created_at: now,
      updated_at: now,
    };
  });

  const completedVariations = rooms.reduce(
    (sum, r) => sum + r.variations.filter((v) => v.status === 'completed').length,
    0,
  );

  return {
    id: PROJECT_ID,
    name: 'Status Badge Test',
    prompt: 'Modern minimalist',
    status: projectStatus,
    settings: {
      style: 'modern',
      room_count: rooms.length,
      variations_per_room: 5,
      output_format: 'png',
      quality: 'high',
    },
    rooms,
    total_variations: rooms.length * 5,
    completed_variations: completedVariations,
    created_at: now,
    updated_at: now,
  };
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

/**
 * Locate the project status badge — the small Badge primitive
 * rendered next to the `<h1>` project name in the header. The Badge
 * shadcn component renders a `<span data-slot="badge">` so we can
 * select it precisely without depending on transient class names or
 * sibling-position selectors. The header div may also contain a
 * `data-slot="badge"` for the variation count, so we additionally
 * scope to the badge whose text matches a known status keyword.
 */
function statusBadge(page: Page) {
  return page
    .locator('[data-slot="badge"]')
    .filter({ hasText: /^(ready|completed|failed|processing|uploading)$/ })
    .first();
}

test.describe('Project status badge (issue 001 — projects-page-improvements)', () => {
  test('multi-room with 1 completed + 4 pending shows "ready" badge (NOT "completed")', async ({
    page,
  }) => {
    // The Issue 1 bug case shape: 1 completed + 4 pending. Backend
    // calculator returns PENDING; the page maps `pending` → "ready"
    // for visible text. Pre-fix the badge would have shown
    // "completed" — the lying state the PRD calls out.
    const project = makeProject(
      ['completed', 'pending', 'pending', 'pending', 'pending'],
      'pending',
    );
    await setupSasTokenMock(page);
    await setupProjectGet(page, project);

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    const badge = statusBadge(page);
    await expect(badge).toBeVisible();
    // Visible text MUST be "ready" (the JSX maps pending → "ready").
    await expect(badge).toHaveText(/^ready$/);

    // Defense-in-depth: the badge MUST NOT show "completed" anywhere
    // in its text — that was the lie the bug produced.
    await expect(badge).not.toHaveText(/completed/i);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-mixed-pending-shows-ready.png`,
      fullPage: true,
    });
  });

  test('all-completed 5-room project shows "completed" badge', async ({ page }) => {
    // Once every room actually finishes, the calculator returns
    // COMPLETED and the badge matches.
    const project = makeProject(
      ['completed', 'completed', 'completed', 'completed', 'completed'],
      'completed',
    );
    await setupSasTokenMock(page);
    await setupProjectGet(page, project);

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    const badge = statusBadge(page);
    await expect(badge).toBeVisible();
    await expect(badge).toHaveText(/^completed$/);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/02-all-completed-shows-completed.png`,
      fullPage: true,
    });
  });

  test('all-pending 5-room project shows "ready" badge', async ({ page }) => {
    // Sanity third state — a brand-new project whose generation
    // hasn't started yet must not falsely show "completed" or any
    // terminal state.
    const project = makeProject(
      ['pending', 'pending', 'pending', 'pending', 'pending'],
      'pending',
    );
    await setupSasTokenMock(page);
    await setupProjectGet(page, project);

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    const badge = statusBadge(page);
    await expect(badge).toBeVisible();
    await expect(badge).toHaveText(/^ready$/);
    await expect(badge).not.toHaveText(/completed/i);
    await expect(badge).not.toHaveText(/failed/i);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/03-all-pending-shows-ready.png`,
      fullPage: true,
    });
  });
});
