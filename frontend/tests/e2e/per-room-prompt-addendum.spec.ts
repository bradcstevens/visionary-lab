import { test, expect, Page, Route } from '@playwright/test';

/**
 * Per-room prompt addendum — issue 003 of the
 * `projects-page-improvements` PRD.
 *
 * Drives the pencil-icon → popover → textarea → Save flow on a one-room
 * project and asserts:
 *
 *   1. The pencil icon is rendered next to the room title.
 *   2. Clicking it opens a popover with an empty textarea (room has
 *      no addendum yet).
 *   3. Typing and clicking Save fires PATCH
 *      /staging/projects/{id}/rooms/{rid} with the addendum in the body.
 *   4. The popover closes after a successful save.
 *   5. The page reloads its project state from the PATCH response,
 *      so reopening the popover shows the previously-typed value.
 *   6. Saving the addendum does NOT trigger any generation — neither
 *      /generate nor any /regenerate route is hit.
 *
 * No backend / SSE flow is exercised. The composer integration that
 * makes the addendum actually appear in the prompt is covered by the
 * Python tests in `tests/test_prompt_composer.py`,
 * `tests/test_staging_pipeline.py::TestProcessRoomComposesAddendum`, and
 * `tests/test_staging_api.py::test_fresh_regen_composes_room_addendum_into_adapted_prompt`.
 */

const SCREENSHOT_DIR = 'test-results/screenshots/per-room-prompt-addendum';
const PROJECT_ID = 'test-prompt-addendum';
const ROOM_ID = 'room-1';
const API_BASE = 'http://localhost:8000/api/v1';

type RoomStatus = 'pending' | 'processing' | 'completed' | 'failed';

interface MockProject {
  id: string;
  name: string;
  prompt: string;
  status: string;
  settings: {
    style: string;
    room_count: number;
    variations_per_room: number;
    output_format: string;
    quality: string;
  };
  rooms: Array<{
    id: string;
    label: string;
    original_image_url: string;
    status: RoomStatus;
    prompt_addendum: string | null;
    variations: Array<{ id: string; status: string; created_at: string; updated_at: string }>;
    created_at: string;
    updated_at: string;
  }>;
  total_variations: number;
  completed_variations: number;
  created_at: string;
  updated_at: string;
}

function makeProject(promptAddendum: string | null = null): MockProject {
  const now = new Date().toISOString();
  return {
    id: PROJECT_ID,
    name: 'Addendum Test Project',
    prompt: 'modern minimalist',
    status: 'pending',
    settings: {
      style: 'modern',
      room_count: 1,
      variations_per_room: 5,
      output_format: 'png',
      quality: 'high',
    },
    rooms: [
      {
        id: ROOM_ID,
        label: 'Backyard',
        original_image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/originals/room-1.png?sv=mock`,
        status: 'pending',
        prompt_addendum: promptAddendum,
        variations: Array.from({ length: 5 }, (_, v) => ({
          id: `r1-v${v}`,
          status: 'pending',
          created_at: now,
          updated_at: now,
        })),
        created_at: now,
        updated_at: now,
      },
    ],
    total_variations: 5,
    completed_variations: 0,
    created_at: now,
    updated_at: now,
  };
}

async function setupSasTokenMock(page: Page) {
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

test.describe('Per-room prompt addendum (issue 003)', () => {
  test('pencil icon → popover → save flow updates the room addendum and re-renders', async ({
    page,
  }) => {
    let projectState = makeProject(null);
    const patchRequests: Array<{ url: string; body: Record<string, unknown> }> = [];
    const generateRequests: string[] = [];

    await setupSasTokenMock(page);

    // Mock GET project — returns whatever projectState currently is.
    await page.route(`${API_BASE}/staging/projects/${PROJECT_ID}`, (route: Route) => {
      const method = route.request().method();
      if (method === 'GET') {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ project: projectState }),
        });
      }
      return route.continue();
    });

    // Mock PATCH room — capture the body, persist into projectState, return updated.
    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/${ROOM_ID}`,
      async (route: Route) => {
        if (route.request().method() !== 'PATCH') {
          return route.continue();
        }
        const body = JSON.parse(route.request().postData() || '{}');
        patchRequests.push({ url: route.request().url(), body });
        // Mirror the backend's normalize-empty-to-null behavior so the
        // post-save state used by the second open is consistent.
        const raw = body.prompt_addendum;
        const normalized =
          raw === null || raw === undefined || (typeof raw === 'string' && raw.trim().length === 0)
            ? null
            : (raw as string).trim();
        projectState = {
          ...projectState,
          rooms: projectState.rooms.map((r) =>
            r.id === ROOM_ID ? { ...r, prompt_addendum: normalized } : r,
          ),
        };
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ project: projectState }),
        });
      },
    );

    // Tripwire: assert no generation routes are hit by the PATCH flow.
    await page.route(
      new RegExp(`${API_BASE}/staging/projects/${PROJECT_ID}/(generate|rooms/[^/]+/regenerate)`),
      (route: Route) => {
        generateRequests.push(route.request().url());
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ ok: true }),
        });
      },
    );

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // 1. Pencil-icon trigger is rendered next to the room title.
    const trigger = page.getByTestId(`room-addendum-trigger-${ROOM_ID}`);
    await expect(trigger).toBeVisible();
    await expect(trigger).toHaveAttribute('data-has-addendum', 'false');

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-pencil-icon-visible.png`,
      fullPage: true,
    });

    // 2. Click opens the popover with an empty textarea.
    await trigger.click();
    const textarea = page.getByTestId(`room-addendum-textarea-${ROOM_ID}`);
    await expect(textarea).toBeVisible();
    await expect(textarea).toHaveValue('');

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/02-popover-open-empty.png`,
      fullPage: true,
    });

    // 3. Type, save → PATCH fires with the body shape we expect.
    await textarea.fill('always in front of fence');
    const saveBtn = page.getByTestId(`room-addendum-save-${ROOM_ID}`);
    await Promise.all([
      page.waitForRequest(
        (req) =>
          req.method() === 'PATCH' &&
          req.url().endsWith(`/staging/projects/${PROJECT_ID}/rooms/${ROOM_ID}`),
      ),
      saveBtn.click(),
    ]);

    expect(patchRequests).toHaveLength(1);
    expect(patchRequests[0].body).toEqual({ prompt_addendum: 'always in front of fence' });

    // 4. Popover closes after the save resolves.
    await expect(textarea).not.toBeVisible();
    await expect(trigger).toHaveAttribute('data-has-addendum', 'true');

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/03-after-save-popover-closed.png`,
      fullPage: true,
    });

    // 5. Reopen — textarea is prefilled from the freshly-loaded room.
    await trigger.click();
    const reopenedTextarea = page.getByTestId(`room-addendum-textarea-${ROOM_ID}`);
    await expect(reopenedTextarea).toBeVisible();
    await expect(reopenedTextarea).toHaveValue('always in front of fence');

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/04-reopen-shows-saved-value.png`,
      fullPage: true,
    });

    // 6. PATCH did not trigger any generation route.
    expect(generateRequests).toEqual([]);
  });

  test('PATCH response replaces project state without losing SAS-resolved image URLs', async ({
    page,
  }) => {
    // Regression for the rubber-duck-flagged bug where the
    // handleUpdateRoomAddendum callback called setProject(updated) directly
    // without first running resolveImageUrls(). The PATCH response from the
    // backend contains BARE blob URLs (no `?sv=...` SAS token), so swapping
    // it into local state replaced the SAS-suffixed URLs that came from the
    // initial loadProject() — breaking <img> previews and the lightbox until
    // the next full reload.
    //
    // The fix runs resolveImageUrls(updated) before setProject, so the
    // post-save URLs still carry the SAS suffix. We pin both the original
    // image URL and one variation URL.
    const completedVariationUrl = `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/v0.png`;
    let projectState: MockProject = {
      ...makeProject(null),
      rooms: [
        {
          ...makeProject(null).rooms[0],
          status: 'completed',
          variations: Array.from({ length: 5 }, (_, v) => ({
            id: `r1-v${v}`,
            status: 'completed',
            // Bare blob URL — no `?sv=...`. resolveImageUrls() must add it.
            image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/v${v}.png`,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          })) as never,
        },
      ],
    };
    // Same bare URL for the original.
    projectState.rooms[0].original_image_url =
      `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/originals/room-1.png`;

    await setupSasTokenMock(page);

    await page.route(`${API_BASE}/staging/projects/${PROJECT_ID}`, (route: Route) => {
      if (route.request().method() === 'GET') {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ project: projectState }),
        });
      }
      return route.continue();
    });

    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/${ROOM_ID}`,
      async (route: Route) => {
        if (route.request().method() !== 'PATCH') return route.continue();
        const body = JSON.parse(route.request().postData() || '{}');
        // Backend echoes BARE-URL project (no SAS) — this is the test's
        // contract about what the backend actually returns.
        projectState = {
          ...projectState,
          rooms: projectState.rooms.map((r) =>
            r.id === ROOM_ID ? { ...r, prompt_addendum: (body.prompt_addendum as string | null) ?? null } : r,
          ),
        };
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ project: projectState }),
        });
      },
    );

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Sanity: an image is rendered and its src includes the SAS suffix
    // after the initial loadProject() resolves URLs.
    const firstImage = page.locator(`img[src*="${completedVariationUrl}"]`).first();
    await expect(firstImage).toBeVisible({ timeout: 10_000 });
    const srcBefore = await firstImage.getAttribute('src');
    expect(srcBefore).toContain('sv=mock');

    // Save an addendum.
    const trigger = page.getByTestId(`room-addendum-trigger-${ROOM_ID}`);
    await trigger.click();
    const textarea = page.getByTestId(`room-addendum-textarea-${ROOM_ID}`);
    await textarea.fill('does not matter');
    const saveBtn = page.getByTestId(`room-addendum-save-${ROOM_ID}`);
    await Promise.all([
      page.waitForRequest(
        (req) =>
          req.method() === 'PATCH' &&
          req.url().endsWith(`/staging/projects/${PROJECT_ID}/rooms/${ROOM_ID}`),
      ),
      saveBtn.click(),
    ]);

    // Post-save: the SAS suffix is STILL present on the image URLs. Without
    // the resolveImageUrls call this assertion fails because setProject
    // replaced the SAS-bearing URLs with the bare ones from the PATCH body.
    await expect(trigger).toHaveAttribute('data-has-addendum', 'true');
    const stillThere = page.locator(`img[src*="${completedVariationUrl}"]`).first();
    await expect(stillThere).toBeVisible();
    const srcAfter = await stillThere.getAttribute('src');
    expect(srcAfter).toContain('sv=mock');
  });

  test('saving an empty / whitespace-only addendum sends null to clear the field', async ({
    page,
  }) => {
    let projectState = makeProject('existing addendum to clear');
    const patchRequests: Array<{ body: Record<string, unknown> }> = [];

    await setupSasTokenMock(page);

    await page.route(`${API_BASE}/staging/projects/${PROJECT_ID}`, (route: Route) => {
      if (route.request().method() === 'GET') {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ project: projectState }),
        });
      }
      return route.continue();
    });

    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/${ROOM_ID}`,
      async (route: Route) => {
        if (route.request().method() !== 'PATCH') {
          return route.continue();
        }
        const body = JSON.parse(route.request().postData() || '{}');
        patchRequests.push({ body });
        projectState = {
          ...projectState,
          rooms: projectState.rooms.map((r) =>
            r.id === ROOM_ID ? { ...r, prompt_addendum: null } : r,
          ),
        };
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ project: projectState }),
        });
      },
    );

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    const trigger = page.getByTestId(`room-addendum-trigger-${ROOM_ID}`);
    await expect(trigger).toHaveAttribute('data-has-addendum', 'true');
    await trigger.click();

    // Existing value is prefilled.
    const textarea = page.getByTestId(`room-addendum-textarea-${ROOM_ID}`);
    await expect(textarea).toHaveValue('existing addendum to clear');

    // Clear and save → request body has `prompt_addendum: null`.
    await textarea.fill('   \n  ');
    const saveBtn = page.getByTestId(`room-addendum-save-${ROOM_ID}`);
    await Promise.all([
      page.waitForRequest(
        (req) =>
          req.method() === 'PATCH' &&
          req.url().endsWith(`/staging/projects/${PROJECT_ID}/rooms/${ROOM_ID}`),
      ),
      saveBtn.click(),
    ]);

    expect(patchRequests).toHaveLength(1);
    expect(patchRequests[0].body).toEqual({ prompt_addendum: null });
    await expect(trigger).toHaveAttribute('data-has-addendum', 'false');
  });
});
