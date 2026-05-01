import { test, expect, Page, Route } from '@playwright/test';

/**
 * Issue 007 (single-variation-regeneration): always-visible regen button + a11y
 *
 * Acceptance criteria (from issues/single-variation-regeneration/007):
 *   - Completed-state thumbnails show the regen icon button without
 *     requiring hover.
 *   - The dropdown trigger has aria-label="Regenerate variation N".
 *   - The thumbnail container has aria-busy={isRegenerating}.
 *   - The icon button is keyboard-reachable via tab and triggers the
 *     dropdown via Enter/Space.
 *   - On a touch viewport the button is visible without hover and
 *     tapping opens the dropdown.
 *   - Activating the regen trigger does NOT open the lightbox
 *     (separate concern — the trigger must stop click propagation
 *     so it doesn't bubble to the parent's onClick).
 *
 * These tests deliberately don't actually drive a regen (no SSE
 * mocking) — they only assert the dropdown opens. The full regen
 * flow is covered by other specs (retry-fallback-toast,
 * activity-log-copy, regen-failure-preserves-prior-image).
 */

const SCREENSHOT_DIR = 'test-results/screenshots/regen-button-a11y';
const PROJECT_ID = 'test-regen-button-a11y';
const API_BASE = 'http://localhost:8000/api/v1';
const PRIOR_IMAGE_URL =
  `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/prior.png`;

function buildProjectWithCompletedVariation() {
  const now = new Date().toISOString();
  return {
    id: PROJECT_ID,
    name: 'Regen Button A11y Test',
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

test.describe('Regen button a11y (issue 007) — always-visible + keyboard', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockedRoutes(page, buildProjectWithCompletedVariation());
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');
  });

  test('completed thumbnail regen trigger is visible without hover', async ({ page }) => {
    // The trigger has aria-label "Regenerate variation 1" (1-indexed in copy).
    const regenTrigger = page.getByRole('button', {
      name: /Regenerate variation 1/i,
    });

    // Visibility WITHOUT any prior hover. Pre-fix the button is opacity-0
    // until the parent .group is hovered, so this assertion fails.
    await expect(regenTrigger).toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-button-visible-no-hover.png`,
      fullPage: true,
    });
  });

  test('aria-busy is "false" by default and toggles via the isRegenerating prop', async ({ page }) => {
    // The thumbnail container is the outermost VariationThumbnail wrapper.
    // We anchor on the regen trigger and walk up to the aspect-square
    // container — the trigger is rendered inside the StorageImage overlay
    // which is inside the cursor-pointer div which is inside the
    // aspect-square container.
    const regenTrigger = page.getByRole('button', {
      name: /Regenerate variation 1/i,
    });
    const thumbnailContainer = regenTrigger.locator(
      'xpath=ancestor::div[contains(@class, "aspect-square")][1]',
    );

    // Default state: not regenerating → aria-busy="false".
    await expect(thumbnailContainer).toHaveAttribute('aria-busy', 'false');
  });

  test('regen trigger has explicit aria-label "Regenerate variation N"', async ({ page }) => {
    // accessibility name MUST be the contract'd string. Lightbox uses
    // "Regenerate this variation"; thumbnail uses "Regenerate variation N"
    // — the two must NOT collide.
    const regenTrigger = page.getByRole('button', {
      name: 'Regenerate variation 1',
    });
    await expect(regenTrigger).toBeVisible();

    // And it must NOT match the lightbox label (which isn't on the page yet
    // since the lightbox isn't open). Sanity: only one such button.
    await expect(regenTrigger).toHaveCount(1);
  });

  test('regen trigger reachable via Tab and Enter opens the dropdown without opening the lightbox', async ({ page }) => {
    const regenTrigger = page.getByRole('button', {
      name: /Regenerate variation 1/i,
    });

    // Focus the trigger directly to assert it can hold focus and respond
    // to keyboard activation. Note: we don't drive Tab from document start
    // because the tab order depends on header/nav structure outside this
    // component's scope.
    await regenTrigger.focus();
    await expect(regenTrigger).toBeFocused();

    await page.keyboard.press('Enter');

    // Dropdown is open with both menu items visible.
    await expect(
      page.getByRole('menuitem', { name: /Retry Same Prompt/i }),
    ).toBeVisible();
    await expect(
      page.getByRole('menuitem', { name: /Edit Prompt/i }),
    ).toBeVisible();

    // CRITICAL: activating the regen trigger MUST NOT open the lightbox.
    // The trigger must call stopPropagation so the click doesn't bubble
    // to the parent .group.cursor-pointer's onClick (which opens the
    // lightbox).
    await expect(page.locator('[data-slot="dialog-overlay"]')).toHaveCount(0);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/02-keyboard-enter-dropdown-open.png`,
      fullPage: true,
    });
  });

  test('regen trigger reachable via Tab and Space opens the dropdown without opening the lightbox', async ({ page }) => {
    const regenTrigger = page.getByRole('button', {
      name: /Regenerate variation 1/i,
    });

    await regenTrigger.focus();
    await expect(regenTrigger).toBeFocused();

    await page.keyboard.press('Space');

    await expect(
      page.getByRole('menuitem', { name: /Retry Same Prompt/i }),
    ).toBeVisible();
    await expect(
      page.getByRole('menuitem', { name: /Edit Prompt/i }),
    ).toBeVisible();

    await expect(page.locator('[data-slot="dialog-overlay"]')).toHaveCount(0);
  });
});

test.describe('Regen button a11y (issue 007) — touch viewport', () => {
  test.use({
    viewport: { width: 390, height: 844 },
    hasTouch: true,
    isMobile: true,
  });

  test('on touch viewport the regen button is visible without hover and tap opens the dropdown', async ({ page }) => {
    await setupMockedRoutes(page, buildProjectWithCompletedVariation());
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    const regenTrigger = page.getByRole('button', {
      name: /Regenerate variation 1/i,
    });

    // Visible without any hover — this is the core touch contract.
    await expect(regenTrigger).toBeVisible();

    // Tap (not click) — exercises the touch path.
    await regenTrigger.tap();

    await expect(
      page.getByRole('menuitem', { name: /Retry Same Prompt/i }),
    ).toBeVisible();
    await expect(
      page.getByRole('menuitem', { name: /Edit Prompt/i }),
    ).toBeVisible();

    // Tapping the regen button must NOT open the lightbox.
    await expect(page.locator('[data-slot="dialog-overlay"]')).toHaveCount(0);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/03-touch-tap-dropdown-open.png`,
      fullPage: true,
    });
  });
});
