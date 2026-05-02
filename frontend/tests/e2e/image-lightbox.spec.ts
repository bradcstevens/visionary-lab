import { test, expect, Page, Route } from '@playwright/test';

/**
 * Image Lightbox E2E Tests
 *
 * Validates that clicking a completed variation opens an in-app lightbox
 * instead of a new browser tab. Covers open, close (X, Escape, backdrop),
 * and the "open in new tab" secondary action.
 *
 * Run with: npx playwright test tests/e2e/image-lightbox.spec.ts --headed
 */

const SCREENSHOT_DIR = 'test-results/screenshots/image-lightbox';
const PROJECT_ID = 'test-lightbox';
const API_BASE = 'http://localhost:8000/api/v1';

const VARIATION_IMAGE_URL =
  'https://storage.blob.core.windows.net/images/staging/test-lightbox/variations/room-1/v0.png';

// ---------------------------------------------------------------------------
// Mock helpers (mirrored from project-generation.spec.ts)
// ---------------------------------------------------------------------------

interface MockVariation {
  id: string;
  status: string;
  image_url?: string;
  error?: string;
  created_at: string;
  updated_at: string;
}

interface MockRoom {
  id: string;
  label: string;
  original_image_url: string;
  status: string;
  variations: MockVariation[];
  created_at: string;
  updated_at: string;
}

interface MockProject {
  id: string;
  name: string;
  prompt: string;
  status: string;
  settings: Record<string, unknown>;
  rooms: MockRoom[];
  total_variations: number;
  completed_variations: number;
  created_at: string;
  updated_at: string;
}

function makeCompletedProject(): MockProject {
  const now = new Date().toISOString();
  return {
    id: PROJECT_ID,
    name: 'Lightbox Test Project',
    prompt: 'Test prompt for lightbox',
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
            id: 'room-1-v0',
            status: 'completed',
            image_url: VARIATION_IMAGE_URL,
            created_at: now,
            updated_at: now,
          },
          {
            id: 'room-1-v1',
            status: 'completed',
            image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/v1.png`,
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

async function setupRoutes(page: Page) {
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

  await page.route(`${API_BASE}/staging/projects/${PROJECT_ID}`, (route: Route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ project: makeCompletedProject() }),
    }),
  );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test.describe('Image Lightbox', () => {

  test('clicking a completed variation opens the lightbox modal', async ({ page }) => {
    await setupRoutes(page);
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Find a completed variation thumbnail and click it
    const variationThumbnail = page.locator('[data-slot="dialog-overlay"]').first();
    // Verify no lightbox is open yet
    await expect(variationThumbnail).not.toBeVisible();

    // Click the first completed variation image (the group cursor-pointer div)
    const completedImage = page.locator('.group.cursor-pointer').first();
    await expect(completedImage).toBeVisible();
    await completedImage.click();

    // Lightbox dialog overlay should appear
    const overlay = page.locator('[data-slot="dialog-overlay"]');
    await expect(overlay).toBeVisible();

    // The lightbox should show the room label and variation number
    await expect(page.locator('p').filter({ hasText: 'Living Room — Variation 1' })).toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/lightbox-open.png`,
      fullPage: true,
    });
  });

  test('lightbox does NOT open a new tab', async ({ page, context }) => {
    await setupRoutes(page);
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Listen for new pages (tabs)
    const newPagePromise = context.waitForEvent('page', { timeout: 2000 }).catch(() => null);

    // Click the completed variation
    const completedImage = page.locator('.group.cursor-pointer').first();
    await expect(completedImage).toBeVisible();
    await completedImage.click();

    // No new tab should have been opened
    const newPage = await newPagePromise;
    expect(newPage).toBeNull();

    // But the lightbox should be visible
    const overlay = page.locator('[data-slot="dialog-overlay"]');
    await expect(overlay).toBeVisible();
  });

  test('lightbox closes when pressing Escape', async ({ page }) => {
    await setupRoutes(page);
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Open lightbox
    const completedImage = page.locator('.group.cursor-pointer').first();
    await completedImage.click();
    const overlay = page.locator('[data-slot="dialog-overlay"]');
    await expect(overlay).toBeVisible();

    // Press Escape
    await page.keyboard.press('Escape');

    // Lightbox should be gone
    await expect(overlay).not.toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/lightbox-closed-escape.png`,
      fullPage: true,
    });
  });

  test('lightbox closes when clicking the X button', async ({ page }) => {
    await setupRoutes(page);
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Open lightbox
    const completedImage = page.locator('.group.cursor-pointer').first();
    await completedImage.click();
    const overlay = page.locator('[data-slot="dialog-overlay"]');
    await expect(overlay).toBeVisible();

    // Click the close button (aria-label="Close")
    const closeButton = page.getByLabel('Close');
    await closeButton.click();

    // Lightbox should be gone
    await expect(overlay).not.toBeVisible();
  });

  test('lightbox has an "open in new tab" button', async ({ page }) => {
    await setupRoutes(page);
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Open lightbox
    const completedImage = page.locator('.group.cursor-pointer').first();
    await completedImage.click();

    // Should have an external link button
    const externalLink = page.getByLabel('Open full image in new tab');
    await expect(externalLink).toBeVisible();
  });

  test('lightbox displays image with accessible title', async ({ page }) => {
    await setupRoutes(page);
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Open lightbox
    const completedImage = page.locator('.group.cursor-pointer').first();
    await completedImage.click();

    // The dialog should have an accessible title (sr-only)
    const dialogTitle = page.locator('[data-slot="dialog-title"]');
    // The title text should contain the room label and variation number
    await expect(dialogTitle).toContainText('Living Room');
    await expect(dialogTitle).toContainText('Variation 1');
  });

  test('lightbox has accessible description (silences Radix warning)', async ({ page }) => {
    // Issue 001 of radix-dialog-body-lock-fix PRD: Radix DialogContent
    // emits a "Missing Description for DialogContent" warning on every
    // mount when no DialogDescription is rendered. We capture console
    // warnings during the open flow and assert none of them match the
    // missing-description pattern, AND assert the visually-hidden
    // description element itself is in the DOM with non-empty text.
    const consoleWarnings: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'warning' || msg.type() === 'error') {
        consoleWarnings.push(msg.text());
      }
    });

    await setupRoutes(page);
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Open lightbox
    const completedImage = page.locator('.group.cursor-pointer').first();
    await completedImage.click();

    // The dialog should have an accessible description (sr-only)
    const dialogDescription = page.locator('[data-slot="dialog-description"]');
    await expect(dialogDescription).toBeAttached();
    // Non-empty text — proves it's a real description, not an empty stub
    const descText = (await dialogDescription.textContent()) ?? '';
    expect(descText.trim().length).toBeGreaterThan(0);
    // The description must be visually hidden (sr-only) so the lightbox
    // visual layout is unchanged.
    await expect(dialogDescription).toHaveClass(/sr-only/);

    // Give Radix a beat to fire any aria warnings it would have fired
    // on mount (the warning is emitted synchronously inside DialogContent's
    // render path, so a microtask drain is enough).
    await page.waitForTimeout(100);

    const missingDescWarnings = consoleWarnings.filter((w) =>
      /Missing\s+`?Description`?\s+(?:or\s+`?aria-describedby={undefined}`?\s+)?for\s+(?:\{?DialogContent\}?|DialogContent)/i.test(
        w,
      ),
    );
    expect(missingDescWarnings, missingDescWarnings.join('\n---\n')).toEqual([]);
  });

  test('lightbox shows navigation arrows and navigates between variations', async ({ page }) => {
    await setupRoutes(page);
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Open the first completed variation
    const completedImage = page.locator('.group.cursor-pointer').first();
    await completedImage.click();

    // Should show "1 / 2" counter (2 completed variations)
    await expect(page.locator('text=1 / 2')).toBeVisible();

    // Should have a "next" arrow but no "previous" (we're on the first)
    const nextBtn = page.getByLabel('Next variation');
    const prevBtn = page.getByLabel('Previous variation');
    await expect(nextBtn).toBeVisible();
    await expect(prevBtn).not.toBeVisible();

    // Click next
    await nextBtn.click();

    // Counter should update to 2 / 2
    await expect(page.locator('text=2 / 2')).toBeVisible();
    // Label should now show Variation 2
    await expect(page.locator('p').filter({ hasText: 'Living Room — Variation 2' })).toBeVisible();

    // Now previous should be visible, next should be gone
    await expect(page.getByLabel('Previous variation')).toBeVisible();
    await expect(page.getByLabel('Next variation')).not.toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/lightbox-navigated.png`,
      fullPage: true,
    });
  });

  test('lightbox supports keyboard arrow navigation', async ({ page }) => {
    await setupRoutes(page);
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Open the first completed variation
    const completedImage = page.locator('.group.cursor-pointer').first();
    await completedImage.click();

    // Verify starting at variation 1
    await expect(page.locator('p').filter({ hasText: 'Living Room — Variation 1' })).toBeVisible();

    // Press right arrow to go to next
    await page.keyboard.press('ArrowRight');
    await expect(page.locator('p').filter({ hasText: 'Living Room — Variation 2' })).toBeVisible();

    // Press left arrow to go back
    await page.keyboard.press('ArrowLeft');
    await expect(page.locator('p').filter({ hasText: 'Living Room — Variation 1' })).toBeVisible();
  });

  // -------------------------------------------------------------------------
  // Issue 005 of radix-dialog-body-lock-fix PRD — close-then-interactive
  // regression. Mirrors the assertion shape from the project-settings-sheet
  // spec's Issue 004 block (commit a95576a). Pure e2e regression slice; no
  // implementation change to ImageLightbox or its consumers — the layout-
  // level body-lock-guard from issue 002 is what the assertion proves.
  //
  // Per close mechanism (✕ button, Escape key) we:
  //   1. open the lightbox on the first completed variation,
  //   2. navigate forward then back with the arrow keys (touches both
  //      completed variations — the user-natural review flow the PRD
  //      calls out),
  //   3. close via the mechanism under test,
  //   4. assert the page is interactive: no inline pointer-events:none
  //      on <body>, no data-scroll-locked attribute on <body>, AND a
  //      non-force click on a normal page element (the More-actions
  //      overflow trigger) succeeds.
  //
  // Failure messages on each assertion name the specific lock-leak family
  // being caught so a future CI failure points the next contributor at
  // the right Radix / scroll-lock layer to debug.
  // -------------------------------------------------------------------------

  type LightboxCloseMechanism = 'x-button' | 'escape';

  async function openLightboxAndReview(page: Page) {
    const completedImage = page.locator('.group.cursor-pointer').first();
    await completedImage.click();
    const overlay = page.locator('[data-slot="dialog-overlay"]');
    await expect(overlay).toBeVisible();

    // Navigate through both completed variations with arrow keys — the
    // "user-natural review flow" the PRD calls out. Two presses (right
    // then left) exercises both navigation directions and confirms we
    // touched both completed variations before closing.
    await expect(page.locator('p').filter({ hasText: 'Living Room — Variation 1' })).toBeVisible();
    await page.keyboard.press('ArrowRight');
    await expect(page.locator('p').filter({ hasText: 'Living Room — Variation 2' })).toBeVisible();
    await page.keyboard.press('ArrowLeft');
    await expect(page.locator('p').filter({ hasText: 'Living Room — Variation 1' })).toBeVisible();
  }

  async function closeLightbox(page: Page, mechanism: LightboxCloseMechanism) {
    const overlay = page.locator('[data-slot="dialog-overlay"]');
    if (mechanism === 'x-button') {
      await page.getByLabel('Close').click();
    } else if (mechanism === 'escape') {
      await page.keyboard.press('Escape');
    }
    await expect(overlay).not.toBeVisible();
  }

  /**
   * Assert <body> has no stuck Radix/react-remove-scroll lock state and
   * that a non-force click on a known page element below the closed
   * lightbox succeeds. Failure messages name the specific lock-leak
   * family being caught.
   */
  async function assertLightboxCloseThenInteractive(page: Page) {
    // 1. Radix Dialog body-lock signature.
    const inlinePointerEvents = await page.evaluate(
      () => document.body.style.pointerEvents,
    );
    expect(
      inlinePointerEvents,
      `Expected <body> to NOT have inline pointer-events:none after lightbox ` +
        `close; got "${inlinePointerEvents}". This is the Radix Dialog ` +
        `body-lock leak the layout-level body-lock-guard from issue 002 is ` +
        `supposed to clear.`,
    ).not.toBe('none');

    // 2. react-remove-scroll signature.
    const hasScrollLocked = await page.evaluate(() =>
      document.body.hasAttribute('data-scroll-locked'),
    );
    expect(
      hasScrollLocked,
      `Expected <body> to NOT carry data-scroll-locked attribute after ` +
        `lightbox close; found it set. This is the react-remove-scroll lock ` +
        `leak the issue-002 body-lock-guard is supposed to clear.`,
    ).toBe(false);

    // 3. Ground-truth interactivity check: a non-force click on the
    //    More-actions overflow trigger surfaces its menu. Catches any
    //    OTHER mechanism that could leave the page non-interactive
    //    (stale focus-trap, orphan portal, scheduler bug) beyond the
    //    two known signatures above.
    const overflowTrigger = page.getByRole('button', { name: /more actions/i });
    await overflowTrigger.click({ timeout: 3000 });
    await expect(
      page.getByTestId('overflow-menu-project-settings'),
      'After closing the lightbox, a non-force click on the More-actions ' +
        'overflow trigger must surface the menu. If this fails, the page is ' +
        'non-interactive after lightbox close — see the failure message ' +
        'above for the specific lock-leak family.',
    ).toBeVisible();
    // Dismiss the menu so this helper leaves no trailing UI state.
    await page.keyboard.press('Escape');
    await expect(
      page.getByTestId('overflow-menu-project-settings'),
    ).not.toBeVisible();
  }

  for (const mechanism of ['x-button', 'escape'] as LightboxCloseMechanism[]) {
    test(`close via ${mechanism} (after arrow-key review of both variations) leaves the page interactive`, async ({
      page,
    }) => {
      await setupRoutes(page);
      await page.goto(`/projects/${PROJECT_ID}`);
      await page.waitForLoadState('networkidle');

      await openLightboxAndReview(page);
      await closeLightbox(page, mechanism);

      await assertLightboxCloseThenInteractive(page);
    });
  }

  test('lightbox has frosted glass toolbar styling', async ({ page }) => {
    await setupRoutes(page);
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Open lightbox
    const completedImage = page.locator('.group.cursor-pointer').first();
    await completedImage.click();

    // Toolbar should have backdrop-blur styling (check the rounded-xl toolbar container)
    const toolbar = page.locator('.backdrop-blur-2xl').first();
    await expect(toolbar).toBeVisible();

    // Image should be in a framed container with shadow
    const imageFrame = page.locator('.ring-1').first();
    await expect(imageFrame).toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/lightbox-styled.png`,
      fullPage: true,
    });
  });

});
