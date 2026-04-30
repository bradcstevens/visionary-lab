import { test, expect, Route } from '@playwright/test';

/**
 * Issue 006 (single-variation-regeneration): Activity log copy and
 * double-toast removal.
 *
 * Acceptance (per issue 006):
 *  - success activity log entry includes the strategy label
 *    `(retry)` / `(fresh)` / `(fresh — no prior prompt)`
 *  - success activity log detail includes a 60-char prompt snippet
 *    alongside model / tokens / elapsed
 *  - NO success toast on `project_completed`
 *  - NO toast on `stream_ended`
 *  - error toast still fires on `variation_failed`
 *  - failed regen + project_completed → exactly one toast (the error)
 *
 * The first toast in the failure regression test fires on
 * ``variation_failed`` and the SSE then closes with ``project_completed``;
 * pre-fix, ``project_completed`` produced a "Variation regenerated!"
 * success toast immediately after the error toast — the double-toast
 * bug. Post-fix, only the error toast remains.
 */

const SCREENSHOT_DIR = 'test-results/screenshots/activity-log-copy';
const PROJECT_ID = 'test-activity-log-copy';
const API_BASE = 'http://localhost:8000/api/v1';

const VARIATION_OLD_URL =
  `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/old.png`;
const VARIATION_NEW_URL =
  `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/new.png`;

// 78 chars — long enough that slice(0, 60) trims and adds the …
const ADAPTED_PROMPT_LONG =
  'Sculptural pendant lamp over a walnut dining table with brass accents, warm';
const SNIPPET_PORTION = 'Sculptural pendant lamp over a walnut dining table with bras';

function sseEvent(type: string, data: Record<string, unknown>): string {
  return `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
}

function makeProjectWithPriorPrompt() {
  const now = new Date().toISOString();
  return {
    id: PROJECT_ID,
    name: 'Activity Log Copy Test',
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
            generation_metadata: {
              model: 'gpt-image-2',
              adapted_prompt: 'A serene minimalist living room',
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

function makeProjectWithoutPriorPrompt() {
  const project = makeProjectWithPriorPrompt();
  // Strip the adapted_prompt — this is the legacy / pre-issue-001
  // shape that triggers the retry→fresh fallback signaling.
  project.rooms[0].variations[0].generation_metadata = {
    model: 'gpt-image-2',
    generation_time_ms: 5000,
  } as unknown as typeof project.rooms[0]['variations'][0]['generation_metadata'];
  return project;
}

async function setupCommonRoutes(page: import('@playwright/test').Page, project: ReturnType<typeof makeProjectWithPriorPrompt>) {
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
          body: JSON.stringify({ project }),
        });
      }
      return route.continue();
    },
  );
}

async function openLightboxOnVariation(page: import('@playwright/test').Page) {
  await page.goto(`/projects/${PROJECT_ID}`);
  await page.waitForLoadState('networkidle');

  // Match the click pattern documented in lightbox-regen-state-sync.spec.ts:
  // the always-visible regen button overlay is centered inside the
  // .group.cursor-pointer parent and intercepts a real .click() at the
  // element's center via stopPropagation. dispatchEvent fires the parent's
  // React onClick directly, equivalent to an off-center user click.
  const completedImage = page.locator('.group.cursor-pointer').first();
  await expect(completedImage).toBeVisible();
  await completedImage.dispatchEvent('click');

  const lightboxLabel = page.locator('p').filter({ hasText: 'Living Room — Variation 1' });
  await expect(lightboxLabel).toBeVisible();
}

test.describe('Activity log copy & double-toast removal (issue 006)', () => {
  test('successful Retry Same Prompt activity log shows (retry) label and prompt snippet', async ({ page }) => {
    await setupCommonRoutes(page, makeProjectWithPriorPrompt());

    // Backend SSE: variation_completed (with adapted_prompt) → project_completed.
    // Pre-fix this would emit a "Variation regenerated!" toast on
    // project_completed; post-fix it must not.
    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/room-1/variations/r1-v0/regenerate**`,
      async (route: Route) => {
        return route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
          body:
            sseEvent('variation_completed', {
              type: 'variation_completed',
              room_id: 'room-1',
              variation_index: 0,
              image_url: VARIATION_NEW_URL,
              elapsed_ms: 4500,
              tokens_used: 1234,
              model: 'gpt-image-2',
              adapted_prompt: ADAPTED_PROMPT_LONG,
            }) +
            sseEvent('project_completed', { type: 'project_completed', status: 'completed' }),
        });
      },
    );

    await openLightboxOnVariation(page);

    const regenTrigger = page.getByLabel('Regenerate this variation');
    await regenTrigger.click();
    await page.getByRole('menuitem', { name: /Retry Same Prompt/i }).click();

    // Activity-log entry: message must include the (retry) label.
    const retryEntry = page.getByText(/Variation 1 regenerated \(retry\)/i);
    await expect(retryEntry).toBeVisible({ timeout: 10_000 });

    // Detail must include the snippet — first 60 chars of the adapted_prompt.
    // Detail length (model · tokens · elapsed · 60-char snippet) > 80 chars
    // so LogEntryRow renders it as plain text rather than chip pills.
    await expect(page.getByText(SNIPPET_PORTION)).toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-retry-success-activity-log.png`,
      fullPage: true,
    });

    // No success toast must have fired.
    await expect(page.getByText(/Variation regenerated!/i)).toHaveCount(0);
  });

  test('successful Try Something New activity log shows (fresh) label and prompt snippet', async ({ page }) => {
    await setupCommonRoutes(page, makeProjectWithPriorPrompt());

    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/room-1/variations/r1-v0/regenerate**`,
      async (route: Route) => {
        return route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
          body:
            sseEvent('variation_completed', {
              type: 'variation_completed',
              room_id: 'room-1',
              variation_index: 0,
              image_url: VARIATION_NEW_URL,
              elapsed_ms: 6200,
              tokens_used: 1456,
              model: 'gpt-image-2',
              adapted_prompt: ADAPTED_PROMPT_LONG,
            }) +
            sseEvent('project_completed', { type: 'project_completed', status: 'completed' }),
        });
      },
    );

    await openLightboxOnVariation(page);

    const regenTrigger = page.getByLabel('Regenerate this variation');
    await regenTrigger.click();
    await page.getByRole('menuitem', { name: /Try Something New/i }).click();

    const freshEntry = page.getByText(/Variation 1 regenerated \(fresh\)/i);
    await expect(freshEntry).toBeVisible({ timeout: 10_000 });

    // Snippet must appear, AND it must NOT be the fallback variant.
    await expect(page.getByText(SNIPPET_PORTION)).toBeVisible();
    await expect(page.getByText(/no prior prompt/i)).toHaveCount(0);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/02-fresh-success-activity-log.png`,
      fullPage: true,
    });

    await expect(page.getByText(/Variation regenerated!/i)).toHaveCount(0);
  });

  test('Retry that falls back to fresh activity log shows (fresh — no prior prompt) label', async ({ page }) => {
    await setupCommonRoutes(page, makeProjectWithoutPriorPrompt());

    // SSE: variation_fallback (issue 004 contract) → variation_completed → project_completed.
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
              image_url: VARIATION_NEW_URL,
              elapsed_ms: 5800,
              tokens_used: 1678,
              model: 'gpt-image-2',
              adapted_prompt: ADAPTED_PROMPT_LONG,
            }) +
            sseEvent('project_completed', { type: 'project_completed', status: 'completed' }),
        });
      },
    );

    await openLightboxOnVariation(page);

    const regenTrigger = page.getByLabel('Regenerate this variation');
    await regenTrigger.click();
    await page.getByRole('menuitem', { name: /Retry Same Prompt/i }).click();

    // Activity-log entry: message must include the fallback label.
    const fallbackEntry = page.getByText(/Variation 1 regenerated \(fresh — no prior prompt\)/i);
    await expect(fallbackEntry).toBeVisible({ timeout: 10_000 });

    // The variation_fallback info toast still fires (issue 004 contract;
    // unaffected by issue 006).
    await expect(
      page.getByText(/No previous prompt found — generating a fresh take instead\./i),
    ).toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/03-fresh-fallback-activity-log.png`,
      fullPage: true,
    });

    // Even though there's a fallback toast, the success toast must NOT fire.
    await expect(page.getByText(/Variation regenerated!/i)).toHaveCount(0);
  });

  test('regen failure followed by project_completed shows exactly one toast (the error)', async ({ page }) => {
    await setupCommonRoutes(page, makeProjectWithPriorPrompt());

    // Realistic backend ordering: pipeline yields ``variation_failed``, then
    // the endpoint's ``finally`` emits ``project_completed``. Pre-fix, the
    // project_completed branch ran ``toast.success('Variation regenerated!')``
    // immediately AFTER the error toast — the double-toast bug.
    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/room-1/variations/r1-v0/regenerate**`,
      async (route: Route) => {
        return route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
          body:
            sseEvent('variation_failed', {
              type: 'variation_failed',
              room_id: 'room-1',
              variation_index: 0,
              error: 'simulated image-gen failure',
              elapsed_ms: 1200,
              tokens_used: null,
              model: 'gpt-image-2',
              adapted_prompt: ADAPTED_PROMPT_LONG,
            }) +
            sseEvent('project_completed', { type: 'project_completed', status: 'failed' }),
        });
      },
    );

    await openLightboxOnVariation(page);

    const regenTrigger = page.getByLabel('Regenerate this variation');
    await regenTrigger.click();
    await page.getByRole('menuitem', { name: /Retry Same Prompt/i }).click();

    // Error toast fires from ``variation_failed``.
    const errorToast = page.getByText(/Regeneration failed: simulated image-gen failure/i);
    await expect(errorToast).toBeVisible({ timeout: 10_000 });

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/04-error-toast-only.png`,
      fullPage: true,
    });

    // The success toast must NOT have fired despite project_completed.
    await expect(page.getByText(/Variation regenerated!/i)).toHaveCount(0);

    // Sanity hold: give the project_completed handler time to settle. If it
    // were going to flash a success toast, it would do so within a few
    // hundred ms after the error toast appears.
    await page.waitForTimeout(800);
    await expect(page.getByText(/Variation regenerated!/i)).toHaveCount(0);
  });
});
