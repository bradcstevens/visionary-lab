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
