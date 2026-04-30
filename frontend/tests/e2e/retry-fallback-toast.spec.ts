import { test, expect, Route } from '@playwright/test';

/**
 * Issue 004 (single-variation-regeneration): Retry-to-fresh fallback signaling
 *
 * When a user picks ``Retry Same Prompt`` on a variation that has no prior
 * ``adapted_prompt`` recorded in ``generation_metadata`` (legacy variation,
 * or one that errored before issue 001 closed the metadata-persistence gap),
 * the backend silently falls back to fresh prompt generation. This slice
 * surfaces the fallback as a dedicated ``variation_fallback`` SSE event so
 * the frontend can toast the user with a single info message.
 *
 * Acceptance:
 *  - On receipt of ``variation_fallback``, the project page renders ONE
 *    info toast: "No previous prompt found — generating a fresh take
 *    instead."
 *  - The toast does not block, dismiss, or otherwise interfere with the
 *    in-flight regen — ``variation_completed`` continues to arrive and
 *    the activity log records the success in the normal way.
 */

const SCREENSHOT_DIR = 'test-results/screenshots/retry-fallback-toast';
const PROJECT_ID = 'test-retry-fallback';
const API_BASE = 'http://localhost:8000/api/v1';

const VARIATION_OLD_URL =
  `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/old.png`;

function sseEvent(type: string, data: Record<string, unknown>): string {
  return `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
}

function makeProjectWithoutPriorPrompt() {
  const now = new Date().toISOString();
  return {
    id: PROJECT_ID,
    name: 'Retry Fallback Toast Test',
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
            image_url: VARIATION_OLD_URL,
            // CRITICAL: no ``adapted_prompt`` here — this is the legacy /
            // pre-issue-001 shape that the fallback signaling exists to
            // handle gracefully.
            generation_metadata: {
              model: 'gpt-image-2',
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
    total_variations: 1,
    completed_variations: 1,
    created_at: now,
    updated_at: now,
  };
}

test.describe('Retry-to-fresh fallback toast (issue 004)', () => {
  test('legacy variation: Retry Same Prompt shows fallback toast and regen completes', async ({ page }) => {
    const projectState = makeProjectWithoutPriorPrompt();

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

    // Regen SSE: emit fallback FIRST, then a normal variation_completed,
    // then project_completed. Mirrors the backend contract verified by
    // tests/test_staging_api.py::test_retry_no_prior_prompt_emits_variation_fallback_then_continues_normally
    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/room-1/variations/r1-v0/regenerate**`,
      async (route: Route) => {
        return route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
          body:
            sseEvent('variation_fallback', {
              type: 'variation_fallback',
              room_id: 'room-1',
              variation_id: 'r1-v0',
              reason: 'no_prior_prompt',
            }) +
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

    // Open the lightbox on the variation. Dispatch the click on the parent
    // .group.cursor-pointer (same pattern as lightbox-regen-state-sync.spec.ts):
    // the always-visible regenerate-button overlay is centered inside this div
    // and would intercept a center-click via stopPropagation; dispatchEvent
    // fires the React onClick directly, equivalent to an off-center user click.
    const completedImage = page.locator('.group.cursor-pointer').first();
    await expect(completedImage).toBeVisible();
    await completedImage.dispatchEvent('click');

    const lightboxLabel = page.locator('p').filter({ hasText: 'Living Room — Variation 1' });
    await expect(lightboxLabel).toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-lightbox-open.png`,
      fullPage: true,
    });

    // Trigger Retry Same Prompt.
    const regenTrigger = page.getByLabel('Regenerate this variation');
    await regenTrigger.click();
    await page.getByRole('menuitem', { name: /Retry Same Prompt/i }).click();

    // The fallback toast must appear with the expected copy.
    const fallbackToast = page.getByText(
      /No previous prompt found — generating a fresh take instead\./i,
    );
    await expect(fallbackToast).toBeVisible({ timeout: 10_000 });

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/02-fallback-toast-visible.png`,
      fullPage: true,
    });

    // The regen continues normally — issue 006 dropped the
    // ``toast.success('Variation regenerated!')`` on
    // ``project_completed``. Assert instead that the activity-log
    // entry shows the "(fresh — no prior prompt)" strategy label,
    // which is the post-issue-006 success signal.
    const successEntry = page.getByText(
      /Variation 1 regenerated \(fresh — no prior prompt\)/i,
    );
    await expect(successEntry).toBeVisible({ timeout: 10_000 });

    // And confirm the dropped toast does NOT fire — this is the
    // double-toast regression guard from issue 006.
    await expect(page.getByText(/Variation regenerated!/i)).toHaveCount(0);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/03-success-activity-log-entry.png`,
      fullPage: true,
    });
  });
});
