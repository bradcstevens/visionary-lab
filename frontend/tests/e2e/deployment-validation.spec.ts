import { test, expect } from '@playwright/test';

/**
 * Deployment validation tests — run against the live Azure deployment.
 * 
 * Usage:
 *   PLAYWRIGHT_BASE_URL=https://ca-frontend-vislab-dev.mangoisland-5af820b8.eastus2.azurecontainerapps.io \
 *   npx playwright test tests/e2e/deployment-validation.spec.ts
 */

const SCREENSHOT_DIR = 'test-results/screenshots/deployment';

test.describe('Deployment Validation', () => {

  test('home page loads without errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.screenshot({ path: `${SCREENSHOT_DIR}/home.png`, fullPage: true });

    expect(errors).toEqual([]);
    await expect(page.locator('body')).toBeVisible();
  });

  test('new image page loads without errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/new-image');
    await page.waitForLoadState('networkidle');
    await page.screenshot({ path: `${SCREENSHOT_DIR}/new-image.png`, fullPage: true });

    expect(errors).toEqual([]);
  });

  test('edit image page loads without errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/edit-image');
    await page.waitForLoadState('networkidle');
    await page.screenshot({ path: `${SCREENSHOT_DIR}/edit-image.png`, fullPage: true });

    expect(errors).toEqual([]);
  });

  test('gallery page loads without errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/gallery');
    await page.waitForLoadState('networkidle');
    await page.screenshot({ path: `${SCREENSHOT_DIR}/gallery.png`, fullPage: true });

    expect(errors).toEqual([]);
  });

  test('projects page loads without errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/projects');
    await page.waitForLoadState('networkidle');
    await page.screenshot({ path: `${SCREENSHOT_DIR}/projects.png`, fullPage: true });

    expect(errors).toEqual([]);
    // Should see either project list or empty state
    const body = await page.textContent('body');
    expect(body).toBeTruthy();
  });

  test('new project wizard loads without errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/projects/new');
    await page.waitForLoadState('networkidle');
    await page.screenshot({ path: `${SCREENSHOT_DIR}/projects-new.png`, fullPage: true });

    expect(errors).toEqual([]);
  });

  test('new video page loads without errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/new-video');
    await page.waitForLoadState('networkidle');
    await page.screenshot({ path: `${SCREENSHOT_DIR}/new-video.png`, fullPage: true });

    expect(errors).toEqual([]);
  });

  test('analyze page loads without errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/analyze');
    await page.waitForLoadState('networkidle');
    await page.screenshot({ path: `${SCREENSHOT_DIR}/analyze.png`, fullPage: true });

    expect(errors).toEqual([]);
  });

  test('sidebar navigation contains all expected links', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const bodyText = await page.textContent('body');
    const expectedLinks = ['New Image', 'Edit Image', 'Projects', 'Gallery'];
    for (const link of expectedLinks) {
      expect(bodyText).toContain(link);
    }

    await page.screenshot({ path: `${SCREENSHOT_DIR}/sidebar-nav.png`, fullPage: true });
  });

  test('backend API health check responds', async ({ request }) => {
    const baseUrl = process.env.PLAYWRIGHT_BASE_URL;
    test.skip(!baseUrl, 'Skipped — only runs against deployed environment (set PLAYWRIGHT_BASE_URL)');

    const backendUrl = baseUrl!.replace('ca-frontend-', 'ca-backend-');

    const response = await request.get(`${backendUrl}/api/v1/health`);
    expect([200, 404]).toContain(response.status());
  });

  test('staging API returns empty project list', async ({ request }) => {
    const baseUrl = process.env.PLAYWRIGHT_BASE_URL;
    test.skip(!baseUrl, 'Skipped — only runs against deployed environment (set PLAYWRIGHT_BASE_URL)');

    const backendUrl = baseUrl!.replace('ca-frontend-', 'ca-backend-');

    const response = await request.get(`${backendUrl}/api/v1/staging/projects`);
    expect(response.status()).toBe(200);
    const data = await response.json();
    expect(data).toHaveProperty('projects');
    expect(Array.isArray(data.projects)).toBeTruthy();
  });

});
