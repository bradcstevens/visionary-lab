import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

test('ImageOverlay component source advertises gpt-image-2', async () => {
  const source = readFileSync(
    join(__dirname, '..', '..', 'components', 'ImageOverlay.tsx'),
    'utf-8',
  );
  expect(source).toContain('value="gpt-image-2"');
  expect(source).toContain('GPT-Image-2');
  expect(source).not.toMatch(/value="gpt-image-1\.5"/);
});

test('home renders content and screenshot is saved', async ({ page }) => {
  await page.goto('/');
  await page.waitForLoadState('domcontentloaded');
  await page.screenshot({
    path: 'test-results/screenshots/image-overlay-home.png',
    fullPage: true,
  });
  const html = await page.content();
  expect(html.length).toBeGreaterThan(500);
});
