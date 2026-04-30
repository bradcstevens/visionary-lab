import { test, expect, Route } from '@playwright/test';

/**
 * Issue 005 (single-variation-regeneration): Lightbox state-desync fix
 *
 * Acceptance criteria (from issues/single-variation-regeneration/005):
 *   "Open lightbox on Variation A, trigger regen, navigate to B with arrow
 *    keys, assert spinner is gone, navigate back to A, assert spinner is
 *    present until regen completes, then assert the new image is shown."
 *
 * The bug: `isRegenerating` was computed from `lightboxContext.variationIndex`
 * which only ever holds the variation we *opened* the lightbox on, never the
 * one currently displayed after arrow-key navigation. As a result the spinner
 * incorrectly followed the user to the sibling variation rather than staying
 * anchored to the regenerating one.
 *
 * The fix: drive both `isRegenerating` and the project-reload sync effect off
 * `lightboxImage` (the variation actually on screen). The sync effect also
 * refreshes `lightboxImage.variations` from the freshly-loaded project so
 * navigation arrows always read up-to-date URLs after a regen completes.
 */

const SCREENSHOT_DIR = 'test-results/screenshots/lightbox-regen-state-sync';
const PROJECT_ID = 'test-lightbox-regen';
const API_BASE = 'http://localhost:8000/api/v1';

// Distinct path segments so test assertions can disambiguate the urls.
const A_OLD_URL =
  `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/a-old.png`;
const A_NEW_URL =
  `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/a-new.png`;
const B_URL =
  `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/b.png`;

function sseEvent(type: string, data: Record<string, unknown>): string {
  return `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
}

function makeProject(aImageUrl: string) {
  const now = new Date().toISOString();
  return {
    id: PROJECT_ID,
    name: 'Lightbox Regen State Sync Test',
    prompt: 'Modern minimalist',
    status: 'completed',
    settings: {
      style: 'modern',
      room_count: 1,
      variations_per_room: 2,
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
            image_url: aImageUrl,
            generation_metadata: {
              model: 'gpt-image-2',
              adapted_prompt: 'A serene minimalist living room',
              generation_time_ms: 5000,
            },
            created_at: now,
            updated_at: now,
          },
          {
            id: 'r1-v1',
            status: 'completed',
            image_url: B_URL,
            generation_metadata: {
              model: 'gpt-image-2',
              adapted_prompt: 'A warm modern living room',
              generation_time_ms: 5000,
            },
            created_at: now,
            updated_at: now,
          },
        ],
        created_at: now,
        updated_at: now,
      },
    ],
    total_variations: 2,
    completed_variations: 2,
    created_at: now,
    updated_at: now,
  };
}

test.describe('Lightbox state-desync fix (issue 005)', () => {
  test('spinner stays anchored to regenerating variation while navigating siblings', async ({ page }) => {
    // Mutable project state — flipped to the "after regen" snapshot
    // immediately before we release the SSE response.
    let projectState = makeProject(A_OLD_URL);

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
      `${API_BASE}/staging/projects/${PROJECT_ID}`,
      (route: Route) => {
        if (route.request().method() === 'GET') {
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ project: projectState }),
          });
        }
        return route.continue();
      },
    );

    // Deferred SSE response: the regen request hangs until `releaseRegen()`
    // is called, giving the test a deterministic window to navigate while the
    // regen is in flight.
    let releaseRegen!: () => void;
    const regenReleased = new Promise<void>((resolve) => {
      releaseRegen = resolve;
    });

    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/room-1/variations/r1-v0/regenerate**`,
      async (route: Route) => {
        await regenReleased;
        return route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
          body:
            sseEvent('variation_completed', {
              type: 'variation_completed',
              room_id: 'room-1',
              variation_index: 0,
              elapsed_ms: 4500,
              tokens_used: 1234,
              model: 'gpt-image-2',
            }) +
            sseEvent('project_completed', { type: 'project_completed' }) +
            sseEvent('stream_ended', { type: 'stream_ended' }),
        });
      },
    );

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Open the lightbox on Variation 1 (A).
    //
    // We dispatch a synthetic click on the parent .group.cursor-pointer div
    // rather than calling .click(): the always-visible regenerate button
    // overlay is centered inside this same div and intercepts real clicks at
    // the element's center (it calls e.stopPropagation()). dispatchEvent
    // fires the React onClick on the parent directly, which is what an
    // off-center user click would do anyway. Validating the click-to-open
    // hit target is image-lightbox.spec.ts's job — this test cares about
    // the spinner-anchoring behavior once the lightbox is open.
    const completedImage = page.locator('.group.cursor-pointer').first();
    await expect(completedImage).toBeVisible();
    await completedImage.dispatchEvent('click');

    // Sanity: lightbox is open on Variation 1 with A's old URL.
    const lightboxLabel = page.locator('p').filter({ hasText: 'Living Room — Variation 1' });
    await expect(lightboxLabel).toBeVisible();

    const lightboxImg = page.locator('img[alt^="Living Room variation"]');
    await expect(lightboxImg).toHaveAttribute('src', /a-old\.png/);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-lightbox-open-on-a.png`,
      fullPage: true,
    });

    // Trigger a regen on A via the lightbox toolbar.
    const regenTrigger = page.getByLabel('Regenerate this variation');
    await regenTrigger.click();
    await page.getByRole('menuitem', { name: /Retry Same Prompt/i }).click();

    // Spinner must show on A — `regeneratingVariationId` is set synchronously
    // by `handleRegenerateVariation` before the SSE response arrives.
    const spinnerLabel = page.locator('text=Regenerating...');
    await expect(spinnerLabel).toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/02-spinner-on-a.png`,
      fullPage: true,
    });

    // Navigate to Variation 2 (B). The spinner MUST disappear because B is
    // not regenerating. Pre-fix, the spinner incorrectly followed.
    await page.keyboard.press('ArrowRight');
    await expect(
      page.locator('p').filter({ hasText: 'Living Room — Variation 2' }),
    ).toBeVisible();
    await expect(spinnerLabel).not.toBeVisible();
    await expect(lightboxImg).toHaveAttribute('src', /b\.png/);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/03-on-b-no-spinner.png`,
      fullPage: true,
    });

    // Navigate back to A. The spinner MUST reappear because A is still
    // regenerating.
    await page.keyboard.press('ArrowLeft');
    await expect(lightboxLabel).toBeVisible();
    await expect(spinnerLabel).toBeVisible();
    await expect(lightboxImg).toHaveAttribute('src', /a-old\.png/);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/04-back-on-a-spinner-restored.png`,
      fullPage: true,
    });

    // Flip the project snapshot to the "after regen" state, then release the
    // SSE response. The frontend will receive `project_completed`, clear
    // `regeneratingVariationId`, and call `loadProject()` which now returns
    // the new A URL.
    projectState = makeProject(A_NEW_URL);
    releaseRegen();

    await page.waitForLoadState('networkidle');

    // Spinner is gone, and because the user is still on A, the new image
    // is now visible (sync effect refreshed `lightboxImage.url`).
    await expect(spinnerLabel).not.toBeVisible();
    await expect(lightboxImg).toHaveAttribute('src', /a-new\.png/);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/05-regen-complete-new-image-on-a.png`,
      fullPage: true,
    });
  });
});
