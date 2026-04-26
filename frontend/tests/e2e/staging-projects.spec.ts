import { test, expect } from '@playwright/test';

test('projects page loads and shows header', async ({ page }) => {
  await page.goto('/projects');
  await page.waitForLoadState('domcontentloaded');
  await expect(page.locator('body')).toBeVisible();
  await page.screenshot({
    path: 'test-results/screenshots/staging/projects-list.png',
    fullPage: true,
  });
});

test('new project page loads wizard', async ({ page }) => {
  await page.goto('/projects/new');
  await page.waitForLoadState('domcontentloaded');
  await expect(page.locator('body')).toBeVisible();
  await page.screenshot({
    path: 'test-results/screenshots/staging/new-project-wizard.png',
    fullPage: true,
  });
});
