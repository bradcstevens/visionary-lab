import { test, expect, Page, Route } from '@playwright/test';

/**
 * Header state-machine label — issue 002 of the
 * `per-room-generation-control` PRD.
 *
 * Drives the project detail page through four canonical project shapes
 * and asserts on the header CTA's rendered label / presence:
 *
 *   - all-pending                → "Generate" (no count, primary)
 *   - one completed + 12 pending → "Generate Remaining (12)" (outline)
 *   - all-completed              → header button is HIDDEN
 *   - in-flight (isGenerating)   → header button DISABLED + spinner
 *
 * Plus a fifth test that the duplicate `Regenerate all` overflow-menu
 * item is gone (only `Add more images` and `Delete project` remain).
 *
 * The header button is identified by its stable test ID
 * `project-header-action` (added in `frontend/app/projects/[id]/page.tsx`)
 * so per-row "Generate" / "Regenerate" buttons in `RoomGroup` (which
 * are siblings on the same page after issue 001 closed) don't collide
 * with the header query.
 */

const SCREENSHOT_DIR = 'test-results/screenshots/header-generate-remaining-label';
const PROJECT_ID = 'test-header-state';
const API_BASE = 'http://localhost:8000/api/v1';

type RoomStatus = 'pending' | 'processing' | 'completed' | 'failed';

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

function makeProject(roomStatuses: RoomStatus[]) {
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
    name: 'Header State Machine Test',
    prompt: 'Modern minimalist',
    status: roomStatuses.every((s) => s === 'completed') ? 'completed' : 'processing',
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

test.describe('Header state machine (issue 002)', () => {
  test('all-pending project shows "Generate" header button (no count, no Regenerate All)', async ({
    page,
  }) => {
    const project = makeProject(['pending', 'pending', 'pending']);
    await setupSasTokenMock(page);
    await setupProjectGet(page, project);

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    const headerBtn = page.getByTestId('project-header-action');
    await expect(headerBtn).toBeVisible();
    await expect(headerBtn).toBeEnabled();
    await expect(headerBtn).toHaveAccessibleName(/^Generate$/);

    // Old label gone; never accidentally rendered as "Regenerate All" or
    // "Generate Remaining (...)".
    await expect(page.getByRole('button', { name: /^Regenerate All$/ })).toHaveCount(0);
    await expect(page.getByRole('button', { name: /^Generate Remaining/ })).toHaveCount(0);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-all-pending.png`,
      fullPage: true,
    });
  });

  test('one completed + 12 pending shows "Generate Remaining (12)" header button', async ({
    page,
  }) => {
    const statuses: RoomStatus[] = [
      'completed',
      ...Array.from({ length: 12 }, () => 'pending' as const),
    ];
    const project = makeProject(statuses);
    await setupSasTokenMock(page);
    await setupProjectGet(page, project);

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    const headerBtn = page.getByTestId('project-header-action');
    await expect(headerBtn).toBeVisible();
    await expect(headerBtn).toBeEnabled();
    // Count MUST be exactly 12 (pending + failed; failed=0 here).
    await expect(headerBtn).toHaveAccessibleName(/^Generate Remaining \(12\)$/);

    // Old label gone.
    await expect(page.getByRole('button', { name: /^Regenerate All$/ })).toHaveCount(0);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/02-mixed-12-pending.png`,
      fullPage: true,
    });
  });

  test('all-completed project hides the header CTA entirely', async ({ page }) => {
    const project = makeProject(['completed', 'completed', 'completed']);
    await setupSasTokenMock(page);
    await setupProjectGet(page, project);

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // No header CTA at all.
    await expect(page.getByTestId('project-header-action')).toHaveCount(0);
    // Defense-in-depth: also no leftover "Regenerate All" anywhere on page.
    await expect(page.getByRole('button', { name: /^Regenerate All$/ })).toHaveCount(0);

    // The overflow trigger MUST still be present (More actions menu is
    // not gated on completion state — only the header CTA is hidden).
    await expect(page.getByRole('button', { name: /More actions/i })).toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/03-all-completed-hidden.png`,
      fullPage: true,
    });
  });

  test('header button is disabled and shows spinner while a stream is in flight', async ({
    page,
  }) => {
    const project = makeProject(['completed', 'pending', 'pending']);
    await setupSasTokenMock(page);
    await setupProjectGet(page, project);

    // Hold the SSE response so the test observes the in-flight state
    // deterministically.
    let releaseStream!: () => void;
    const released = new Promise<void>((resolve) => {
      releaseStream = resolve;
    });

    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/generate**`,
      async (route: Route) => {
        await released;
        return route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
          body:
            'event: project_completed\ndata: {"type":"project_completed"}\n\n' +
            'event: stream_ended\ndata: {"type":"stream_ended"}\n\n',
        });
      },
    );

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    const headerBtn = page.getByTestId('project-header-action');
    await expect(headerBtn).toBeEnabled();
    await expect(headerBtn).toHaveAccessibleName(/^Generate Remaining \(2\)$/);

    // Click → stream starts but is held open by the route handler above.
    await headerBtn.click();

    // While in-flight: button is DISABLED and contains a spinning Loader2.
    await expect(headerBtn).toBeDisabled();
    const spinner = headerBtn.locator('svg.animate-spin');
    await expect(spinner).toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/04-in-flight-disabled-spinner.png`,
      fullPage: true,
    });

    // Release so the test cleanly tears down.
    releaseStream();
  });

  test('overflow menu shows only "Add more images" and "Delete project" — no duplicate "Regenerate all"', async ({
    page,
  }) => {
    // Mixed state is the only state where today's overflow menu
    // ALSO renders the duplicate `Regenerate all` item — so this is
    // the discriminating fixture for the cleanup.
    const project = makeProject(['completed', 'pending', 'pending']);
    await setupSasTokenMock(page);
    await setupProjectGet(page, project);

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Open the overflow menu.
    await page.getByRole('button', { name: /More actions/i }).click();

    // Both legitimate items present.
    await expect(page.getByRole('menuitem', { name: /Add more images/i })).toBeVisible();
    await expect(page.getByRole('menuitem', { name: /Delete project/i })).toBeVisible();

    // The dup MUST be gone (case-insensitive, matches "Regenerate all"
    // and any future "Regenerate All" casing).
    await expect(page.getByRole('menuitem', { name: /Regenerate all/i })).toHaveCount(0);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/05-overflow-menu-no-dup.png`,
      fullPage: true,
    });
  });
});
