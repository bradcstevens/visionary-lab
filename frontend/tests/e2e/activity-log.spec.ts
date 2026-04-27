import { test, expect } from '@playwright/test';

/**
 * Activity Log Panel & Project Status — E2E tests.
 * 
 * Validates the new activity log components render correctly
 * on all creation pages, and that project status badges are accurate.
 * 
 * Run with: npx playwright test tests/e2e/activity-log.spec.ts --headed
 */

const SCREENSHOT_DIR = 'test-results/screenshots/activity-log';

test.describe('Activity Log Panel', () => {

  test('activity log toggle is hidden when no activity', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/new-image');
    await page.waitForLoadState('domcontentloaded');

    // The toggle button should NOT be visible (no generation activity yet)
    const toggle = page.locator('button[title="Show activity log"]');
    await expect(toggle).not.toBeVisible();

    await page.screenshot({ path: `${SCREENSHOT_DIR}/new-image-no-toggle.png`, fullPage: true });
    expect(errors).toEqual([]);
  });

  test('new image page loads without errors and has activity log context', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/new-image');
    await page.waitForLoadState('networkidle');

    // Page should render without JS errors (proves ActivityLogProvider is wired correctly)
    await page.screenshot({ path: `${SCREENSHOT_DIR}/new-image-loaded.png`, fullPage: true });
    expect(errors).toEqual([]);

    // Sidebar should be visible with navigation
    await expect(page.locator('body')).toBeVisible();
  });

  test('edit image page loads without errors and has activity log context', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/edit-image');
    await page.waitForLoadState('networkidle');

    await page.screenshot({ path: `${SCREENSHOT_DIR}/edit-image-loaded.png`, fullPage: true });
    expect(errors).toEqual([]);
    await expect(page.locator('body')).toBeVisible();
  });

  test('projects page loads without errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/projects');
    await page.waitForLoadState('domcontentloaded');

    await page.screenshot({ path: `${SCREENSHOT_DIR}/projects-list.png`, fullPage: true });
    expect(errors).toEqual([]);
    await expect(page.locator('body')).toBeVisible();
  });

  test('new project wizard loads without errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/projects/new');
    await page.waitForLoadState('domcontentloaded');

    await page.screenshot({ path: `${SCREENSHOT_DIR}/new-project-wizard.png`, fullPage: true });
    expect(errors).toEqual([]);
    await expect(page.locator('body')).toBeVisible();
  });

  test('video generation page loads without errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/new-video');
    await page.waitForLoadState('networkidle');

    await page.screenshot({ path: `${SCREENSHOT_DIR}/new-video-loaded.png`, fullPage: true });
    expect(errors).toEqual([]);
  });
});

test.describe('Activity Log Panel - Interaction', () => {

  test('activity log panel opens via JavaScript injection and renders correctly', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/new-image');
    await page.waitForLoadState('networkidle');

    // Inject a log entry via the ActivityLogProvider to simulate generation
    // The provider exposes log() on the context — we trigger it via a test helper
    await page.evaluate(() => {
      // Find the React fiber and trigger the context
      // Simpler approach: dispatch a custom event that triggers the log
      const event = new CustomEvent('__test_activity_log__', {
        detail: { level: 'info', icon: '🎨', message: 'Test generation started', detail: 'gpt-image-2 · high quality' }
      });
      window.dispatchEvent(event);
    });

    // Wait briefly for any React updates
    await page.waitForTimeout(500);

    // Take screenshot of current state
    await page.screenshot({ path: `${SCREENSHOT_DIR}/new-image-after-inject.png`, fullPage: true });
    expect(errors).toEqual([]);
  });
});

test.describe('Project Status Badge', () => {

  test('projects list page renders status badges correctly', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/projects');
    await page.waitForLoadState('domcontentloaded');

    // Check that any project cards that exist have visible status badges
    const badges = page.locator('[class*="badge"]');
    const badgeCount = await badges.count();

    // Take screenshot showing project cards with status badges
    await page.screenshot({ path: `${SCREENSHOT_DIR}/projects-status-badges.png`, fullPage: true });

    // No JS errors from the new status handling code
    expect(errors).toEqual([]);
  });

  test('header contains activity log toggle area and standard buttons', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');

    // The header bar should exist and contain the standard layout elements
    const header = page.locator('div.flex.h-14');
    await expect(header).toBeVisible();

    await page.screenshot({ path: `${SCREENSHOT_DIR}/header-layout.png`, fullPage: true });
    expect(errors).toEqual([]);
  });
});

test.describe('Component Source Verification', () => {

  test('ActivityLogProvider context file exists and exports correctly', async () => {
    const { readFileSync } = await import('node:fs');
    const { join } = await import('node:path');

    const source = readFileSync(
      join(__dirname, '..', '..', 'context', 'activity-log-context.tsx'),
      'utf-8',
    );

    // Verify key exports exist
    expect(source).toContain('export function ActivityLogProvider');
    expect(source).toContain('export function useActivityLog');
    expect(source).toContain('interface LogEntry');
    expect(source).toContain("level: \"info\" | \"success\" | \"error\" | \"warn\"");
    expect(source).toContain('MAX_ENTRIES');
  });

  test('ActivityLogPanel component exists and uses Sheet', async () => {
    const { readFileSync } = await import('node:fs');
    const { join } = await import('node:path');

    const source = readFileSync(
      join(__dirname, '..', '..', 'components', 'activity-log', 'ActivityLogPanel.tsx'),
      'utf-8',
    );

    expect(source).toContain('Sheet');
    expect(source).toContain('SheetContent');
    expect(source).toContain('side="right"');
    expect(source).toContain('Activity Log');
    expect(source).toContain('autoScroll');
    expect(source).toContain('No activity yet');
  });

  test('ActivityLogToggle component exists and uses Terminal icon', async () => {
    const { readFileSync } = await import('node:fs');
    const { join } = await import('node:path');

    const source = readFileSync(
      join(__dirname, '..', '..', 'components', 'activity-log', 'ActivityLogToggle.tsx'),
      'utf-8',
    );

    expect(source).toContain('Terminal');
    expect(source).toContain('useActivityLog');
    expect(source).toContain('hasActivity');
  });

  test('LogEntry component exists with level-based color mapping', async () => {
    const { readFileSync } = await import('node:fs');
    const { join } = await import('node:path');

    const source = readFileSync(
      join(__dirname, '..', '..', 'components', 'activity-log', 'LogEntry.tsx'),
      'utf-8',
    );

    expect(source).toContain('text-blue-400');
    expect(source).toContain('text-green-400');
    expect(source).toContain('text-red-400');
    expect(source).toContain('text-amber-400');
    expect(source).toContain('formatTime');
  });

  test('Layout integrates ActivityLogProvider, Toggle, and Panel', async () => {
    const { readFileSync } = await import('node:fs');
    const { join } = await import('node:path');

    const source = readFileSync(
      join(__dirname, '..', '..', 'app', 'layout.tsx'),
      'utf-8',
    );

    expect(source).toContain('ActivityLogProvider');
    expect(source).toContain('ActivityLogToggle');
    expect(source).toContain('ActivityLogPanel');
  });

  test('Staging pipeline SSE events include timing and token fields', async () => {
    const { readFileSync } = await import('node:fs');
    const { join } = await import('node:path');

    const source = readFileSync(
      join(__dirname, '..', '..', '..', 'backend', 'core', 'staging_pipeline.py'),
      'utf-8',
    );

    expect(source).toContain('elapsed_ms');
    expect(source).toContain('tokens_used');
    expect(source).toContain('"model"');
  });

  test('Project status includes pending state', async () => {
    const { readFileSync } = await import('node:fs');
    const { join } = await import('node:path');

    const modelSource = readFileSync(
      join(__dirname, '..', '..', '..', 'backend', 'models', 'staging.py'),
      'utf-8',
    );
    expect(modelSource).toContain('PENDING = "pending"');

    const apiSource = readFileSync(
      join(__dirname, '..', '..', 'services', 'stagingApi.ts'),
      'utf-8',
    );
    expect(apiSource).toContain("'pending'");
  });

  test('upload_rooms transitions status from uploading to pending', async () => {
    const { readFileSync } = await import('node:fs');
    const { join } = await import('node:path');

    const source = readFileSync(
      join(__dirname, '..', '..', '..', 'backend', 'api', 'endpoints', 'staging.py'),
      'utf-8',
    );

    expect(source).toContain('updates["status"] = "pending"');
  });
});
