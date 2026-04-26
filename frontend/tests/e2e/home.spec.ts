import { test, expect } from '@playwright/test';

test('home page loads and renders', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('body')).toBeVisible();
  await page.screenshot({
    path: 'test-results/screenshots/home.png',
    fullPage: true,
  });
});
