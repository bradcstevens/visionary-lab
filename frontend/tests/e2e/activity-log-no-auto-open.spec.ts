import { test, expect, Page, Route } from '@playwright/test';

/**
 * Activity log opens on demand only — issue 006 of
 * prds/2026-04-30-projects-page-improvements-prd.md.
 *
 * Validates that the right-side activity log panel does NOT auto-open
 * when the first log entry lands. The panel only opens in response to
 * a user click on the toggle. The notification dot on the toggle is
 * the unobtrusive cue that something happened.
 *
 * Source-of-truth signal for "panel open vs closed": the toggle button's
 * `title` attribute, which flips between "Show activity log" (closed) and
 * "Hide activity log" (open). Reading title is deterministic regardless
 * of the panel's collapsing-width animation; using `text=Activity` would
 * be unreliable because the panel content is in the DOM at all times
 * (only its parent's width animates between 0 and 420px).
 */

const SCREENSHOT_DIR = 'test-results/screenshots/activity-log-no-auto-open';
const PROJECT_ID = 'test-project-no-auto-open';
const API_BASE = 'http://localhost:8000/api/v1';

// ---------------------------------------------------------------------------
// Mock data + helpers
// ---------------------------------------------------------------------------

function makeProject() {
  return {
    id: PROJECT_ID,
    name: 'Backyard Redesign',
    prompt: 'Add drought-tolerant landscaping with native plants',
    status: 'pending',
    settings: {
      variations_per_room: 1,
      model: 'gpt-image-2',
      quality: 'high',
      size: '1024x1024',
    },
    rooms: [
      {
        id: 'room-1',
        label: 'Front Yard',
        original_image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/originals/room-1.png?sv=mock`,
        status: 'pending',
        variations: [
          {
            id: 'room-1-v0',
            status: 'pending',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          },
        ],
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
    ],
    total_variations: 1,
    completed_variations: 0,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
}

function makeProjectAfterGeneration() {
  const p = makeProject();
  p.status = 'completed';
  p.rooms[0].status = 'completed';
  Object.assign(p.rooms[0].variations[0], {
    status: 'completed',
    image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/v0.png?sv=mock`,
  });
  p.completed_variations = 1;
  return p;
}

function sseEvent(type: string, data: Record<string, unknown>): string {
  return `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
}

async function setupMockedRoutes(page: Page) {
  let getCount = 0;
  const initial = makeProject();
  const updated = makeProjectAfterGeneration();

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

  await page.route(`${API_BASE}/staging/projects/${PROJECT_ID}`, (route: Route) => {
    if (route.request().method() === 'GET') {
      getCount++;
      const data = getCount <= 1 ? initial : updated;
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ project: data }),
      });
    }
    return route.continue();
  });

  const sseBody =
    sseEvent('room_started', { type: 'room_started', room_id: 'room-1', label: 'Front Yard' })
    + sseEvent('variation_completed', {
        type: 'variation_completed',
        room_id: 'room-1',
        variation_index: 0,
        image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/v0.png`,
        elapsed_ms: 4500,
        tokens_used: 1300,
        model: 'gpt-image-2',
      })
    + sseEvent('project_completed', { type: 'project_completed', status: 'completed' });

  await page.route(`${API_BASE}/staging/projects/${PROJECT_ID}/generate`, (route: Route) =>
    route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
      body: sseBody,
    }),
  );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test.describe('Activity log opens on demand only (issue 006)', () => {
  test('panel stays closed during generation; opens after toggle click', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await setupMockedRoutes(page);
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Toggle starts in closed state ("Show activity log" title).
    const toggle = page.locator('button[title="Show activity log"], button[title="Hide activity log"]');
    await expect(toggle).toBeVisible();
    await expect(toggle).toHaveAttribute('title', 'Show activity log');

    // Kick off generation and wait for the full SSE response to be consumed.
    const generateBtn = page.getByRole('button', { name: /Generate 1 Variation/i });
    await expect(generateBtn).toBeVisible();
    const [postResp] = await Promise.all([
      page.waitForResponse((resp) => resp.url().includes('/generate') && resp.request().method() === 'POST'),
      generateBtn.click(),
    ]);
    expect(postResp.status()).toBe(200);

    // Wait for the entry count badge inside the toggle to appear — proves
    // at least one log entry was added by the SSE handler. The selector
    // matches the toggle in either open or closed state because we already
    // expressed that in the parent locator.
    await expect(toggle.locator('span.tabular-nums')).toBeVisible({ timeout: 8000 });

    // PRE-FIX (auto-open enabled): toggle title would have flipped to
    // "Hide activity log" by now, because the first log entry triggered
    // setIsOpen(true) inside ActivityLogProvider.log().
    // POST-FIX: the toggle stays in the closed state — entries logged,
    // panel intentionally still closed.
    await expect(toggle).toHaveAttribute('title', 'Show activity log');

    // The notification dot (blue, since no errors) is the unobtrusive cue
    // that activity has happened. AC: "existing notification-dot behavior
    // on the toggle is preserved unchanged (red for errors, blue for any
    // other activity)."
    const blueDot = toggle.locator('span.bg-blue-500.rounded-full');
    await expect(blueDot).toBeVisible();
    const redDot = toggle.locator('span.bg-red-500.rounded-full');
    await expect(redDot).not.toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-panel-closed-with-blue-dot.png`,
      fullPage: true,
    });

    // Click the toggle — the panel should now open.
    await toggle.click();
    await expect(toggle).toHaveAttribute('title', 'Hide activity log');

    // Panel content is now visible. The "Activity" heading lives inside
    // the previously-collapsed container.
    await expect(page.getByRole('heading', { name: 'Activity' })).toBeVisible();

    // The actual log entries rendered by the SSE handlers should be
    // visible in the panel now that it is open. Use text patterns that
    // match the existing handleStreamEvent log copy (see project page).
    await expect(page.getByText(/Starting generation for/).first()).toBeVisible();
    await expect(page.getByText(/Variation 1 saved/).first()).toBeVisible();

    // When the panel is open, the notification dot is hidden (its render
    // gate is `!isOpen`).
    await expect(blueDot).not.toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/02-panel-open-after-click.png`,
      fullPage: true,
    });

    expect(errors).toEqual([]);
  });
});
