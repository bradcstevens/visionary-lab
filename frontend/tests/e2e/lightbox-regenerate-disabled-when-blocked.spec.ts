import { test, expect, Page, Route } from '@playwright/test';

/**
 * Issue 001 (failed-variation-retry-queue): Lightbox Regenerate
 * disabled-with-tooltip during in-flight generation.
 *
 * Acceptance criteria (from issues/failed-variation-retry-queue/001):
 *   - When `isBlocked` is true (page-level: `isGenerating ||
 *     regeneratingVariationId !== null`), the lightbox's Regenerate
 *     menu/button is rendered visibly disabled (not hidden) with a
 *     tooltip explaining why.
 *   - When `isBlocked` is false, behavior is unchanged from today.
 *   - Clicking the disabled control does NOT fire a regen request.
 *
 * The PRD-specified tooltip copy:
 *   "Generating other variations… regenerate available when complete"
 *   (U+2026 ellipsis).
 *
 * Test scenario: trigger a regen on Variation A (held open via deferred
 * Promise), navigate to sibling B with arrow key while A is in flight,
 * assert B's regen control is visibly disabled, hover for tooltip, click
 * (force) and assert no regen POST fires for B. Releases A's regen at
 * the end so the test cleans up gracefully.
 *
 * Pattern follows lightbox-regen-state-sync.spec.ts (deferred SSE
 * Promise + arrow-key navigation between siblings) and the SSE-mocking
 * pattern from regen-failure-preserves-prior-image.spec.ts
 * (setupMockedRoutes helper, sseEvent helper).
 */

const SCREENSHOT_DIR =
  'test-results/screenshots/lightbox-regenerate-disabled-when-blocked';
const PROJECT_ID = 'test-lightbox-regen-blocked';
const API_BASE = 'http://localhost:8000/api/v1';

const A_URL =
  `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/a.png`;
const B_URL =
  `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/b.png`;

const BLOCKED_TOOLTIP_COPY =
  'Generating other variations… regenerate available when complete';

function sseEvent(type: string, data: Record<string, unknown>): string {
  return `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
}

function makeProject() {
  const now = new Date().toISOString();
  return {
    id: PROJECT_ID,
    name: 'Lightbox Blocked Regen Test',
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
            image_url: A_URL,
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

async function setupSasMock(page: Page) {
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

test.describe('Lightbox Regenerate disabled-with-tooltip while blocked', () => {
  test('disabled with tooltip while sibling variation is regenerating; click does not fire regen', async ({
    page,
  }) => {
    const project = makeProject();
    await setupSasMock(page);

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

    // A's regen response is held until releaseARegen() is called so the
    // in-flight window is observable for assertions.
    let releaseARegen!: () => void;
    const aRegenReleased = new Promise<void>((resolve) => {
      releaseARegen = resolve;
    });

    let aRegenPostCount = 0;
    let bRegenPostCount = 0;

    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/room-1/variations/r1-v0/regenerate**`,
      async (route: Route) => {
        aRegenPostCount += 1;
        await aRegenReleased;
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

    // B's regen MUST NOT be called — the blocking guard. If the
    // disabled-with-tooltip behavior regresses to silent-no-op or
    // (worse) actually fires, this counter will catch it.
    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/room-1/variations/r1-v1/regenerate**`,
      (route: Route) => {
        bRegenPostCount += 1;
        return route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
          body: sseEvent('stream_ended', { type: 'stream_ended' }),
        });
      },
    );

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Open the lightbox on Variation 1 (A). Use dispatchEvent('click')
    // because the always-visible regen icon's stopPropagation can
    // intercept real clicks at the centered hit-target — same pattern
    // documented in lightbox-regen-state-sync.spec.ts.
    const completedImage = page.locator('.group.cursor-pointer').first();
    await expect(completedImage).toBeVisible();
    await completedImage.dispatchEvent('click');

    await expect(
      page.locator('p').filter({ hasText: 'Living Room — Variation 1' }),
    ).toBeVisible();

    // Trigger a regen on A from the lightbox toolbar.
    const regenTrigger = page.getByLabel('Regenerate this variation');
    await regenTrigger.click();
    await page.getByRole('menuitem', { name: /Retry Same Prompt/i }).click();

    // Wait for the regen POST to actually arrive — without this, ArrowRight
    // could fire BEFORE handleRegenerateVariation runs and sets
    // regeneratingVariationId, leaving isBlocked=false and the test racy.
    await expect.poll(() => aRegenPostCount, { timeout: 5000 }).toBe(1);

    // Spinner visible on A — proves regeneratingVariationId is set.
    await expect(page.locator('text=Regenerating...')).toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-spinner-on-a.png`,
      fullPage: true,
    });

    // Navigate to sibling B with the arrow key. Now isRegenerating is
    // false on B (regeneratingVariationId is A's id, not B's), but
    // isBlocked is true — so B's regen control should be visibly
    // disabled with the tooltip.
    await page.keyboard.press('ArrowRight');
    await expect(
      page.locator('p').filter({ hasText: 'Living Room — Variation 2' }),
    ).toBeVisible();

    // The regen control on B must be visibly disabled (not hidden, not
    // a spinner — the spinner UI is reserved for the variation actively
    // being regenerated).
    const bRegenButton = page.getByLabel('Regenerate this variation');
    await expect(bRegenButton).toBeVisible();
    await expect(bRegenButton).toBeDisabled();

    // Hover to surface the tooltip (PRD copy).
    //
    // Playwright resolves `getByLabel('Regenerate this variation')` to the
    // inner disabled `<button>`. By design the disabled button has
    // `pointer-events: none` (Tailwind `disabled:pointer-events-none`),
    // so the wrapping `<span data-slot="tooltip-trigger">` is what
    // actually receives pointer events. Playwright's strict pointer-target
    // check sees this and refuses to hover without `force: true`. The
    // user's real hover lands on the span (which is geometrically the
    // same area as the button), the span fires Radix's onPointerEnter,
    // and the tooltip appears — exactly what we want to verify.
    await bRegenButton.hover({ force: true });
    await expect(page.getByText(BLOCKED_TOOLTIP_COPY)).toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/02-disabled-with-tooltip-on-b.png`,
      fullPage: true,
    });

    // Click (force, since Playwright refuses to click disabled by default).
    // The browser respects the `disabled` attribute and does NOT fire the
    // onClick — so no regen POST should land for B.
    await bRegenButton.click({ force: true });

    // Give any erroneously-dispatched regen POST a generous window to
    // arrive before asserting it didn't.
    await page.waitForTimeout(500);
    expect(bRegenPostCount).toBe(0);

    // Clean up: release A's regen so the page settles and any pending
    // SSE listeners get torn down.
    releaseARegen();
    await page.waitForLoadState('networkidle');
  });

  test('not blocked: dropdown opens normally and onRegenerate fires (regression guard)', async ({
    page,
  }) => {
    // Same project, but NO regen is in flight. The Regenerate menu in
    // the lightbox toolbar must behave exactly as it did before issue
    // 001 — opens a dropdown with two items, and clicking either fires
    // the variation-regen POST.
    const project = makeProject();
    await setupSasMock(page);

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

    let aRegenPostCount = 0;
    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/room-1/variations/r1-v0/regenerate**`,
      (route: Route) => {
        aRegenPostCount += 1;
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

    const completedImage = page.locator('.group.cursor-pointer').first();
    await completedImage.dispatchEvent('click');
    await expect(
      page.locator('p').filter({ hasText: 'Living Room — Variation 1' }),
    ).toBeVisible();

    // Trigger should be enabled (no in-flight regen, no global stream).
    const regenTrigger = page.getByLabel('Regenerate this variation');
    await expect(regenTrigger).toBeVisible();
    await expect(regenTrigger).toBeEnabled();

    // The blocked tooltip must NOT be present.
    await expect(page.getByText(BLOCKED_TOOLTIP_COPY)).toHaveCount(0);

    // Click opens the dropdown — both items present.
    await regenTrigger.click();
    await expect(
      page.getByRole('menuitem', { name: /Retry Same Prompt/i }),
    ).toBeVisible();
    await expect(
      page.getByRole('menuitem', { name: /Try Something New/i }),
    ).toBeVisible();

    // Selecting "Retry Same Prompt" fires the variation-regen POST.
    await page.getByRole('menuitem', { name: /Retry Same Prompt/i }).click();
    await expect.poll(() => aRegenPostCount, { timeout: 5000 }).toBe(1);

    await page.waitForLoadState('networkidle');
  });
});
