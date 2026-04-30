import { test, expect, Page, Route } from '@playwright/test';

/**
 * Issue 002: regen failure preserves prior image
 *
 * Acceptance criteria (from issues/single-variation-regeneration/002):
 *   "Playwright test: simulate regen failure (e.g., via mocked endpoint),
 *    assert the prior image is still displayed in the thumbnail and lightbox."
 *
 * The backend rollback contract (process_single_variation): on failure, the
 * variation's `status` / `image_url` / `error` are restored to their pre-regen
 * values. So a project GET fetched AFTER a failed regen returns the variation
 * exactly as it was before the regen — completed, with the prior image_url.
 *
 * This test validates the frontend half of that contract: after a failed
 * regen, the thumbnail still shows the prior image (no red error tile, no
 * stale processing spinner).
 */

const SCREENSHOT_DIR = 'test-results/screenshots/regen-failure-preserves-prior-image';
const PROJECT_ID = 'test-regen-failure';
const API_BASE = 'http://localhost:8000/api/v1';
const PRIOR_IMAGE_URL =
  `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/prior.png`;

function sseEvent(type: string, data: Record<string, unknown>): string {
  return `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
}

function buildProjectWithCompletedVariation() {
  return {
    id: PROJECT_ID,
    name: 'Regen Failure Test',
    prompt: 'Modern minimalist',
    status: 'completed',
    settings: {
      style: 'modern',
      room_count: 1,
      variations_per_room: 1,
      output_format: 'png',
      quality: 'high',
    },
    rooms: [
      {
        id: 'room-1',
        label: 'Living Room',
        original_image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/originals/room-1.png?sv=mock`,
        status: 'completed',
        variations: [
          {
            id: 'r1-v0',
            status: 'completed',
            image_url: PRIOR_IMAGE_URL,
            generation_metadata: {
              model: 'gpt-image-2',
              adapted_prompt: 'A serene minimalist living room',
              generation_time_ms: 5000,
            },
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          },
        ],
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
    ],
    total_variations: 1,
    completed_variations: 1,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
}

async function setupMockedRoutes(page: Page, project: object) {
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

  // The GET endpoint always returns the same project — that's the rollback
  // contract: the variation looks the same before and after a failed regen.
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

test.describe('Regen failure preserves prior image (issue 002)', () => {
  test('thumbnail still shows prior image after a failed Retry Same Prompt', async ({ page }) => {
    const project = buildProjectWithCompletedVariation();
    await setupMockedRoutes(page, project);

    // Variation regen SSE emits a `variation_failed` event then `stream_ended`.
    // The frontend handler shows an error toast, calls loadProject(), and
    // since the GET returns the project unchanged (rollback contract),
    // the thumbnail must continue to render the prior image.
    let regenRequestCount = 0;
    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/room-1/variations/r1-v0/regenerate**`,
      (route: Route) => {
        regenRequestCount += 1;
        return route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
          body:
            sseEvent('variation_failed', {
              type: 'variation_failed',
              room_id: 'room-1',
              variation_index: 0,
              error: 'Simulated image-gen failure for issue 002 test',
              elapsed_ms: 1500,
              tokens_used: null,
              model: 'gpt-image-2',
            }) +
            sseEvent('stream_ended', { type: 'stream_ended' }),
        });
      },
    );

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Sanity: the prior image is showing before any regen.
    const thumbnailImg = page.locator(`img[alt="Variation 1"]`).first();
    await expect(thumbnailImg).toBeVisible();
    await expect(thumbnailImg).toHaveAttribute(
      'src',
      new RegExp(`prior\\.png`),
    );

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-before-regen.png`,
      fullPage: true,
    });

    // Hover the thumbnail so the regen dropdown trigger becomes visible
    // (it uses opacity-0 group-hover:opacity-100). Use force:true on the
    // click since hover-revealed buttons can be flaky.
    const thumbnailContainer = thumbnailImg.locator('..').locator('..');
    await thumbnailContainer.hover();

    const dropdownTrigger = thumbnailContainer.getByRole('button').first();
    await dropdownTrigger.click({ force: true });

    // Click "Retry Same Prompt" → triggers variation regen (which we mock to fail).
    await page.getByRole('menuitem', { name: /Retry Same Prompt/i }).click();

    // Wait for the regen request to fire and complete.
    await expect.poll(() => regenRequestCount, { timeout: 5000 }).toBe(1);

    // Wait for the post-stream loadProject() to settle.
    await page.waitForLoadState('networkidle');
    // Allow a small grace period for React to re-render after loadProject().
    await page.waitForTimeout(500);

    // Critical assertion: the thumbnail is still showing the prior image
    // (rollback preserved variation.status=completed and the prior URL).
    // It should NOT show the red error tile (failed status), nor a spinner.
    await expect(thumbnailImg).toBeVisible();
    await expect(thumbnailImg).toHaveAttribute(
      'src',
      new RegExp(`prior\\.png`),
    );

    // The red failure tile contains an AlertCircle icon labelled "failed";
    // assert no destructive badge is rendered for this variation.
    const failureBadge = page
      .locator('[class*="bg-destructive"]')
      .filter({ hasText: '1' });
    await expect(failureBadge).toHaveCount(0);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/02-after-failed-regen.png`,
      fullPage: true,
    });
  });
});
