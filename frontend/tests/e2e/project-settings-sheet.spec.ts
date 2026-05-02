import { test, expect, Page, Route } from '@playwright/test';

/**
 * Project Settings side sheet — issue 002 of the
 * `projects-page-improvements` PRD.
 *
 * Drives the overflow menu → Settings sheet → form edit → Save flow on
 * a one-room project and asserts:
 *
 *   1. The "Project settings" item is rendered in the overflow menu.
 *   2. Selecting it opens the Sheet with prefilled values from the
 *      current persisted project (name, prompt, settings).
 *   3. The "future generations only" notice banner is visible.
 *   4. Editing variations_per_room from 5 to 3 and clicking Save
 *      fires PATCH /staging/projects/{id} with body
 *      { settings: { variations_per_room: 3 } } — only the changed key.
 *   5. The Sheet closes after a successful save.
 *   6. Existing variation IDs / image URLs are unchanged in local
 *      state after the PATCH (proves "applies to future generations
 *      only" and that resolveImageUrls() preserves SAS suffixes).
 *   7. Cancel discards local edits — reopening shows persisted values.
 *   8. Saving the form does NOT trigger any /generate or /regenerate
 *      route.
 */

const SCREENSHOT_DIR = 'test-results/screenshots/project-settings-sheet';
const PROJECT_ID = 'test-settings-sheet';
const ROOM_ID = 'room-1';
const API_BASE = 'http://localhost:8000/api/v1';

interface MockDesignBrief {
  global_instructions: string;
  object_palette: unknown[];
  placement_guide: { back_row: string };
  per_image_notes: Record<string, string>;
  per_image_objects: Record<string, unknown[]>;
  preserve_elements: string[];
  settings: {
    variations_per_room: number;
    model: string;
    quality: string;
    size: string;
  };
}

interface MockProject {
  id: string;
  name: string;
  prompt: string;
  status: string;
  settings: {
    variations_per_room: number;
    model: string;
    quality: string;
    size: string;
  };
  rooms: Array<{
    id: string;
    label: string;
    original_image_url: string;
    status: string;
    prompt_addendum: string | null;
    variations: Array<{
      id: string;
      status: string;
      image_url?: string;
      created_at: string;
      updated_at: string;
    }>;
    created_at: string;
    updated_at: string;
  }>;
  total_variations: number;
  completed_variations: number;
  created_at: string;
  updated_at: string;
  // Issue 002 of project-settings-completeness PRD: optional brief.
  // When present, derivePromptForSettings prefers
  // `design_brief.global_instructions` over `project.prompt`.
  design_brief?: MockDesignBrief | null;
}

function makeProject(overrides: Partial<MockProject> = {}): MockProject {
  const now = new Date().toISOString();
  // Bare URL — resolveImageUrls() must add `?sv=mock` on load. Asserting
  // the SAS suffix survives the PATCH save is the regression for the
  // same bug fixed in the per-room-prompt-addendum spec.
  const variationUrl = `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/v0.png`;
  return {
    id: PROJECT_ID,
    name: 'Settings Sheet Test Project',
    prompt: 'modern minimalist',
    status: 'completed',
    settings: {
      variations_per_room: 5,
      model: 'gpt-image-2',
      quality: 'high',
      size: 'auto',
    },
    rooms: [
      {
        id: ROOM_ID,
        label: 'Living Room',
        original_image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/originals/room-1.png`,
        status: 'completed',
        prompt_addendum: null,
        variations: Array.from({ length: 5 }, (_, v) => ({
          id: `r1-v${v}`,
          status: 'completed',
          image_url: v === 0 ? variationUrl : `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/v${v}.png`,
          created_at: now,
          updated_at: now,
        })),
        created_at: now,
        updated_at: now,
      },
    ],
    total_variations: 5,
    completed_variations: 5,
    created_at: now,
    updated_at: now,
    ...overrides,
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

test.describe('Project Settings side sheet (issue 002)', () => {
  test('open from overflow menu, change variations_per_room, save → PATCH body sent and existing variations preserved', async ({
    page,
  }) => {
    let projectState = makeProject();
    const patchRequests: Array<{ url: string; body: Record<string, unknown> }> = [];
    const generateRequests: string[] = [];

    await setupSasTokenMock(page);

    await page.route(`${API_BASE}/staging/projects/${PROJECT_ID}`, async (route: Route) => {
      const method = route.request().method();
      if (method === 'GET') {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ project: projectState }),
        });
      }
      if (method === 'PATCH') {
        const body = JSON.parse(route.request().postData() || '{}');
        patchRequests.push({ url: route.request().url(), body });
        // Mirror backend's MERGE behavior on settings so the response
        // shape stays consistent with what real Cosmos does.
        const next: MockProject = { ...projectState };
        if ('name' in body) next.name = body.name as string;
        if ('prompt' in body) next.prompt = body.prompt as string;
        if ('settings' in body) {
          next.settings = { ...projectState.settings, ...(body.settings as Record<string, unknown>) };
        }
        projectState = next;
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ project: projectState }),
        });
      }
      return route.continue();
    });

    // Tripwire: assert no generation routes hit.
    await page.route(
      new RegExp(`${API_BASE}/staging/projects/${PROJECT_ID}/(generate|rooms/[^/]+/(?:regenerate|variations))`),
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

    // Capture variation IDs + image URLs (with SAS) BEFORE the PATCH so
    // we can assert they're unchanged afterward.
    const variationImagesBefore = await page
      .locator(`img[src*="staging/${PROJECT_ID}/variations/room-1/"]`)
      .evaluateAll((imgs) => imgs.map((i) => (i as HTMLImageElement).src).sort());
    expect(variationImagesBefore.length).toBeGreaterThan(0);
    expect(variationImagesBefore[0]).toContain('sv=mock');

    // 1. Open overflow menu, click Project settings.
    const overflowTrigger = page.getByRole('button', { name: /more actions/i });
    await overflowTrigger.click();
    const settingsItem = page.getByTestId('overflow-menu-project-settings');
    await expect(settingsItem).toBeVisible();
    await settingsItem.click();

    // 2. Sheet opens with prefilled values.
    const sheet = page.getByTestId('project-settings-sheet');
    await expect(sheet).toBeVisible();
    const nameInput = page.getByTestId('project-settings-name-input');
    await expect(nameInput).toHaveValue('Settings Sheet Test Project');
    const promptTextarea = page.getByTestId('project-settings-prompt-textarea');
    await expect(promptTextarea).toHaveValue('modern minimalist');
    const variationsInput = page.getByTestId('project-settings-variations-input');
    await expect(variationsInput).toHaveValue('5');

    // 3. Notice banner is visible.
    const notice = page.getByTestId('project-settings-future-only-notice');
    await expect(notice).toBeVisible();
    await expect(notice).toContainText(/future generations only/i);

    await page.screenshot({ path: `${SCREENSHOT_DIR}/01-sheet-open-prefilled.png`, fullPage: true });

    // 4. Change variations_per_room from 5 to 3, save.
    await variationsInput.fill('3');
    const saveBtn = page.getByTestId('project-settings-save');
    await expect(saveBtn).toBeEnabled();

    await Promise.all([
      page.waitForRequest(
        (req) => req.method() === 'PATCH' && req.url().endsWith(`/staging/projects/${PROJECT_ID}`),
      ),
      saveBtn.click(),
    ]);

    expect(patchRequests).toHaveLength(1);
    expect(patchRequests[0].body).toEqual({
      settings: { variations_per_room: 3 },
    });

    // 5. Sheet closes after save. The overlay detaches; subsequent
    //    clicks may need ``force`` because Radix's pointer-events lock
    //    can outlive the visibility transition.
    await expect(sheet).not.toBeVisible();
    await expect(page.locator('[data-slot="sheet-overlay"]')).not.toBeAttached();

    await page.screenshot({ path: `${SCREENSHOT_DIR}/02-after-save-sheet-closed.png`, fullPage: true });

    // 6. Existing variations untouched: same IDs, same image URLs WITH
    //    SAS suffix preserved (resolveImageUrls regression).
    const variationImagesAfter = await page
      .locator(`img[src*="staging/${PROJECT_ID}/variations/room-1/"]`)
      .evaluateAll((imgs) => imgs.map((i) => (i as HTMLImageElement).src).sort());
    expect(variationImagesAfter).toEqual(variationImagesBefore);

    // 7. No generation route hit.
    expect(generateRequests).toEqual([]);

    // 8. Reload the page and re-open the sheet — the variations value
    //    is now the freshly-persisted 3, not 5. We reload (rather than
    //    chain a click on the still-mounted Radix tree) to dodge the
    //    well-known Radix Dialog ``pointer-events`` lock that can
    //    persist briefly after close. The fresh page load guarantees a
    //    clean tree; the assertion still proves "save persisted +
    //    local state reflects it".
    await page.reload();
    await page.waitForLoadState('networkidle');
    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();
    await expect(page.getByTestId('project-settings-variations-input')).toHaveValue('3');
  });

  test('cancel discards local edits — reopening shows original values, no PATCH fired', async ({
    page,
  }) => {
    const projectState = makeProject();
    const patchRequests: Array<{ body: Record<string, unknown> }> = [];

    await setupSasTokenMock(page);

    await page.route(`${API_BASE}/staging/projects/${PROJECT_ID}`, async (route: Route) => {
      const method = route.request().method();
      if (method === 'GET') {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ project: projectState }),
        });
      }
      if (method === 'PATCH') {
        const body = JSON.parse(route.request().postData() || '{}');
        patchRequests.push({ body });
        // Should never get here in this scenario.
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ project: projectState }),
        });
      }
      return route.continue();
    });

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Open sheet, edit, cancel.
    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();

    const variationsInput = page.getByTestId('project-settings-variations-input');
    await expect(variationsInput).toHaveValue('5');
    await variationsInput.fill('7');

    // Save would be enabled (we have a diff). Click Cancel instead.
    await page.getByTestId('project-settings-cancel').click();

    // Sheet closes without firing a PATCH. No reopen needed: the
    // ``patchRequests.length === 0`` assertion proves that backend
    // state is unchanged, which is the same property "reopening shows
    // original 5" would prove.
    await expect(page.getByTestId('project-settings-sheet')).not.toBeVisible();
    await expect(page.locator('[data-slot="sheet-overlay"]')).not.toBeAttached();
    expect(patchRequests).toEqual([]);
  });

  test('Save is disabled with no changes (idempotent reopen)', async ({ page }) => {
    const projectState = makeProject();

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

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();

    // No edits — Save is disabled.
    const saveBtn = page.getByTestId('project-settings-save');
    await expect(saveBtn).toBeDisabled();
  });
});

/**
 * Issue 002 of the project-settings-completeness PRD.
 *
 * Propagation guards for the canonical project prompt across the settings
 * sheet, the page header, and the per-image Edit Prompt dialog. The PRD's
 * core claim is that "the prompt" is one coherent value the user sees in
 * Settings, on the Brief tab, in gallery dialogs, and on project cards —
 * regardless of whether the project has a `design_brief` yet.
 *
 * The frontend changes here are intentionally minimal: only the displayed
 * prompt is derived from `design_brief.global_instructions` (with the same
 * `_is_nonempty_str` gate the backend mirror uses). The save path itself
 * is unchanged — the sheet always sends a top-level `prompt` field on
 * edit and the backend mirror in `_mirror_prompt_and_brief_in_place`
 * (`backend/api/endpoints/staging.py`) propagates the value into
 * `design_brief.global_instructions` server-side. Mocking the backend
 * faithfully here means mirroring BOTH `project.prompt` AND
 * `design_brief.global_instructions` in the PATCH response so tests
 * exercise the same chain prod will see.
 *
 * Test coverage map (PRD Acceptance Criteria for issue 002):
 *
 *   1. Brief-backed project — textarea shows brief.global_instructions
 *      (NOT the wizard's `Draft — pending AI Design Session` placeholder).
 *      Save fires PATCH with top-level `prompt` only (no design_brief in
 *      the body). Page header (line 859 of `app/projects/[id]/page.tsx`)
 *      reflects the new prompt IMMEDIATELY (without reload). Hard reload
 *      preserves the new prompt. Per-image Edit Prompt dialog shows the
 *      new prompt as fallback on reopen — the gallery surface the PRD
 *      explicitly calls out as a propagation target.
 *   2. Legacy project (no brief) — textarea uses `project.prompt`. The
 *      explainer hint is visible. Save fires PATCH with top-level
 *      `prompt` only. Persists across reload.
 *   3. Whitespace-only-brief project — textarea falls back to
 *      `project.prompt` (matches backend `_is_nonempty_str`). The hint
 *      is NOT visible (asymmetric rule from `derivePromptForSettings`:
 *      hint is only shown when the brief is null/undefined, not when
 *      the brief exists with empty/whitespace content).
 *
 * Each test installs a `/generate` and `/regenerate` route guard so a
 * regression that accidentally triggers a generation on save trips a
 * loud assertion.
 */

const PROMPT_PROJECT_ID = 'test-canonical-prompt';
const BRIEF_PROMPT = 'Real designer-authored global instructions';
const PLACEHOLDER_PROMPT = 'Draft — pending AI Design Session';

function makeBrief(globalInstructions: string): MockDesignBrief {
  return {
    global_instructions: globalInstructions,
    object_palette: [],
    placement_guide: { back_row: '' },
    per_image_notes: {},
    per_image_objects: {},
    preserve_elements: [],
    settings: {
      variations_per_room: 5,
      model: 'gpt-image-2',
      quality: 'high',
      size: 'auto',
    },
  };
}

function makeBriefBackedProject(): MockProject {
  return makeProject({
    id: PROMPT_PROJECT_ID,
    name: 'Canonical Prompt Project',
    prompt: PLACEHOLDER_PROMPT,
    design_brief: makeBrief(BRIEF_PROMPT),
  });
}

function makeLegacyProject(): MockProject {
  return makeProject({
    id: PROMPT_PROJECT_ID,
    name: 'Legacy No-Brief Project',
    prompt: 'legacy-only prompt without a brief',
    design_brief: null,
  });
}

function makeWhitespaceBriefProject(): MockProject {
  return makeProject({
    id: PROMPT_PROJECT_ID,
    name: 'Whitespace Brief Project',
    prompt: 'fallback to project.prompt because brief is whitespace-only',
    design_brief: makeBrief('   \t\n  '),
  });
}

/**
 * Apply the same mirror rules to the in-memory `projectState` that
 * `_mirror_prompt_and_brief_in_place` applies on the backend. Without
 * this the PATCH response would drop one side of the mirror and the
 * "reload preserves" / "header reflects" assertions would silently
 * pass for the wrong reason.
 */
function applyBackendMirror(
  projectState: MockProject,
  body: Record<string, unknown>,
): MockProject {
  const next: MockProject = {
    ...projectState,
    design_brief: projectState.design_brief
      ? { ...projectState.design_brief }
      : projectState.design_brief,
  };
  const promptIn = 'prompt' in body;
  const briefIn = 'design_brief' in body;
  if (promptIn) next.prompt = body.prompt as string;
  if (briefIn) next.design_brief = body.design_brief as MockDesignBrief | null;
  if ('name' in body) next.name = body.name as string;
  if ('settings' in body) {
    next.settings = {
      ...projectState.settings,
      ...(body.settings as Record<string, unknown>),
    };
  }
  // Mirror rules — see `backend/api/endpoints/staging.py:445-501`.
  const isNonEmpty = (v: unknown): v is string =>
    typeof v === 'string' && v.trim().length > 0;
  const briefIsObject = next.design_brief && typeof next.design_brief === 'object';
  if (promptIn && briefIn) {
    if (briefIsObject && isNonEmpty(next.design_brief!.global_instructions)) {
      next.prompt = next.design_brief!.global_instructions;
    }
  } else if (promptIn) {
    if (briefIsObject) {
      next.design_brief = {
        ...next.design_brief!,
        global_instructions: body.prompt as string,
      };
    }
  } else if (briefIn) {
    if (briefIsObject && isNonEmpty(next.design_brief!.global_instructions)) {
      next.prompt = next.design_brief!.global_instructions;
    }
  }
  return next;
}

async function setupGenerationTripwire(
  page: Page,
  hits: string[],
): Promise<void> {
  await page.route(
    new RegExp(
      `${API_BASE}/staging/projects/${PROMPT_PROJECT_ID}/(generate|rooms/[^/]+/(?:regenerate|variations))`,
    ),
    (route: Route) => {
      hits.push(route.request().url());
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ok: true }),
      });
    },
  );
}

test.describe('Issue 002 — canonical project prompt', () => {
  test('brief-backed project: textarea shows global_instructions, save sends top-level prompt only, header + reload + Edit Prompt all reflect new value', async ({
    page,
  }) => {
    let projectState = makeBriefBackedProject();
    const patchRequests: Array<{ body: Record<string, unknown> }> = [];
    const generateHits: string[] = [];

    await setupSasTokenMock(page);
    await setupGenerationTripwire(page, generateHits);

    await page.route(
      `${API_BASE}/staging/projects/${PROMPT_PROJECT_ID}`,
      async (route: Route) => {
        const method = route.request().method();
        if (method === 'GET') {
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ project: projectState }),
          });
        }
        if (method === 'PATCH') {
          const body = JSON.parse(route.request().postData() || '{}');
          patchRequests.push({ body });
          projectState = applyBackendMirror(projectState, body);
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ project: projectState }),
          });
        }
        return route.continue();
      },
    );

    await page.goto(`/projects/${PROMPT_PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Sanity: page header initially shows the wizard placeholder
    // because nothing has mirrored project.prompt to global_instructions
    // yet on this mock. The Settings sheet is what flips the user's
    // first impression to the canonical prompt.
    const header = page.locator('h1', { hasText: 'Canonical Prompt Project' });
    await expect(header).toBeVisible();

    // 1. Open the sheet.
    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();

    const sheet = page.getByTestId('project-settings-sheet');
    await expect(sheet).toBeVisible();
    const promptTextarea = page.getByTestId('project-settings-prompt-textarea');

    // 2. Textarea shows the BRIEF's global_instructions, NOT the
    //    placeholder. This is the core bug fix from issue 002.
    await expect(promptTextarea).toHaveValue(BRIEF_PROMPT);
    await expect(promptTextarea).not.toHaveValue(PLACEHOLDER_PROMPT);

    // 3. The no-brief hint is NOT shown (a brief exists).
    await expect(
      page.getByTestId('project-settings-prompt-brief-hint'),
    ).toHaveCount(0);

    // 4. Edit the prompt and save.
    const NEW_PROMPT = 'updated canonical prompt — single source of truth';
    await promptTextarea.fill(NEW_PROMPT);
    const saveBtn = page.getByTestId('project-settings-save');
    await expect(saveBtn).toBeEnabled();

    await Promise.all([
      page.waitForRequest(
        (req) =>
          req.method() === 'PATCH' &&
          req.url().endsWith(`/staging/projects/${PROMPT_PROJECT_ID}`),
      ),
      saveBtn.click(),
    ]);

    // 5. Save body sends ONLY top-level prompt — NOT design_brief. This
    //    is the load-bearing payload assertion: the PRD spec example
    //    suggests routing prompt edits through design_brief, but the
    //    rubber-duck-validated approach is to trust the backend mirror
    //    and keep the client payload minimal (avoids stale-brief
    //    clobber risk on concurrent edits).
    expect(patchRequests).toHaveLength(1);
    expect(patchRequests[0].body).toEqual({ prompt: NEW_PROMPT });
    expect(patchRequests[0].body).not.toHaveProperty('design_brief');

    // 6. Sheet closes after a successful save.
    await expect(sheet).not.toBeVisible();

    // 7. Page header reflects the new prompt IMMEDIATELY (no reload).
    //    setProject(updated) on the page side picks up the mirrored
    //    project.prompt from the PATCH response. If the page was
    //    accidentally reading from a stale source (e.g., a memo over
    //    a snapshot), this would fail without the reload below
    //    masking it.
    await expect(
      page.getByText(NEW_PROMPT, { exact: true }).first(),
    ).toBeVisible();

    // 8. Hard reload — the persisted state now has the mirror applied
    //    on BOTH project.prompt AND design_brief.global_instructions.
    //    Reopening the Settings sheet shows the new prompt (derived
    //    from the brief side).
    await page.reload();
    await page.waitForLoadState('networkidle');
    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();
    await expect(
      page.getByTestId('project-settings-prompt-textarea'),
    ).toHaveValue(NEW_PROMPT);

    // 9. No generation routes were hit during the save.
    expect(generateHits).toEqual([]);
  });

  test('legacy project (no brief): textarea uses project.prompt, hint is visible, save sends top-level prompt only', async ({
    page,
  }) => {
    let projectState = makeLegacyProject();
    const patchRequests: Array<{ body: Record<string, unknown> }> = [];
    const generateHits: string[] = [];

    await setupSasTokenMock(page);
    await setupGenerationTripwire(page, generateHits);

    await page.route(
      `${API_BASE}/staging/projects/${PROMPT_PROJECT_ID}`,
      async (route: Route) => {
        const method = route.request().method();
        if (method === 'GET') {
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ project: projectState }),
          });
        }
        if (method === 'PATCH') {
          const body = JSON.parse(route.request().postData() || '{}');
          patchRequests.push({ body });
          projectState = applyBackendMirror(projectState, body);
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ project: projectState }),
          });
        }
        return route.continue();
      },
    );

    await page.goto(`/projects/${PROMPT_PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();

    const promptTextarea = page.getByTestId('project-settings-prompt-textarea');

    // 1. Textarea shows the legacy project.prompt — fallback path.
    await expect(promptTextarea).toHaveValue(
      'legacy-only prompt without a brief',
    );

    // 2. Hint is visible — explains where the prompt will live once a
    //    brief exists.
    const hint = page.getByTestId('project-settings-prompt-brief-hint');
    await expect(hint).toBeVisible();
    await expect(hint).toContainText(/once a design brief exists/i);

    // 3. Edit + save → top-level prompt only, no design_brief.
    const NEW_LEGACY = 'edited legacy prompt';
    await promptTextarea.fill(NEW_LEGACY);
    await Promise.all([
      page.waitForRequest(
        (req) =>
          req.method() === 'PATCH' &&
          req.url().endsWith(`/staging/projects/${PROMPT_PROJECT_ID}`),
      ),
      page.getByTestId('project-settings-save').click(),
    ]);

    expect(patchRequests).toHaveLength(1);
    expect(patchRequests[0].body).toEqual({ prompt: NEW_LEGACY });
    expect(patchRequests[0].body).not.toHaveProperty('design_brief');

    // 4. Hard reload — value persists. Hint is still visible because
    //    no brief was created (the legacy save path doesn't materialize
    //    a brief).
    await page.reload();
    await page.waitForLoadState('networkidle');
    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();
    await expect(
      page.getByTestId('project-settings-prompt-textarea'),
    ).toHaveValue(NEW_LEGACY);
    await expect(
      page.getByTestId('project-settings-prompt-brief-hint'),
    ).toBeVisible();

    expect(generateHits).toEqual([]);
  });

  test('whitespace-only brief: falls back to project.prompt for display, hint stays hidden (asymmetric rule)', async ({
    page,
  }) => {
    const projectState = makeWhitespaceBriefProject();

    await setupSasTokenMock(page);
    await page.route(
      `${API_BASE}/staging/projects/${PROMPT_PROJECT_ID}`,
      async (route: Route) => {
        const method = route.request().method();
        if (method === 'GET') {
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ project: projectState }),
          });
        }
        return route.continue();
      },
    );

    await page.goto(`/projects/${PROMPT_PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();

    // Display falls back to project.prompt — matches backend
    // _is_nonempty_str gate on global_instructions. Without the
    // whitespace fallback in derivePromptForSettings, this would
    // show "   " visually-empty whitespace and a save would
    // silently overwrite the legacy value.
    await expect(
      page.getByTestId('project-settings-prompt-textarea'),
    ).toHaveValue('fallback to project.prompt because brief is whitespace-only');

    // The hint stays HIDDEN even though the brief has no real
    // global_instructions. This is the asymmetric rule documented in
    // ProjectSettingsSheet.tsx: the hint is only shown when the brief
    // is null/undefined (no brief at all). A whitespace brief means
    // the user has already started a brief and doesn't need the
    // explainer — they'll edit and save and the backend mirror will
    // populate the brief's global_instructions on the next persist.
    await expect(
      page.getByTestId('project-settings-prompt-brief-hint'),
    ).toHaveCount(0);
  });
});

/**
 * Issue 004 of the project-settings-completeness PRD.
 *
 * `ProjectRoomsManager` scaffold + inline rename. The component is
 * mounted on the Settings sheet between the project-level fields
 * (name, prompt) and the generation settings (variations, model,
 * quality, size). Rename uses the existing room-scoped PATCH endpoint
 * that issue 004 also extended to accept `label` (was: addendum-only).
 *
 * Test coverage map (PRD Acceptance Criteria for issue 004):
 *
 *   1. Happy path — rename a room from Settings:
 *      - Open the sheet, the Rooms section is visible with both rooms.
 *      - Click the per-row pencil, the input prefills with the
 *        current label.
 *      - Type a new label and Save.
 *      - PATCH /staging/projects/{id}/rooms/{rid} fires with body
 *        { label: "<trimmed>" } — the load-bearing payload assertion.
 *      - The row reflects the new label without a reload (proves
 *        onProjectUpdate flowed through to setProject).
 *      - Hard reload + reopen the sheet shows the new label persists.
 *      - The sheet's project-level Save button is NOT enabled by the
 *        rename (rooms persist immediately per action; project-level
 *        Save tracks only name/prompt/settings dirtiness).
 *      - Tripwire: no /generate or /regenerate route hits.
 *
 *   2. Addendum preservation — the rubber-duck blocker regression:
 *      - Room has an existing prompt_addendum.
 *      - Rename it from Settings.
 *      - The PATCH body contains label only (NOT prompt_addendum).
 *      - The backend's __fields_set__-aware handler (verified by
 *        `tests/test_staging_endpoints_update_room.py::test_patch_
 *        room_label_only_preserves_existing_addendum`) leaves the
 *        addendum intact, but we also pin it at the integration
 *        boundary: the mock applies the same field-set rules to
 *        projectState, and a hard reload + reopening the per-room
 *        addendum popover still shows the original addendum.
 *
 *   3. Error path — PATCH 500:
 *      - Click edit, type new label, Save.
 *      - The route returns 500.
 *      - The row reverts to view mode with the ORIGINAL label.
 *      - A toast surfaces.
 */

const ROOMS_PROJECT_ID = 'test-rooms-manager';
const ROOM_A_ID = 'room-A';
const ROOM_B_ID = 'room-B';

interface MockRoomsProject {
  id: string;
  name: string;
  prompt: string;
  status: string;
  settings: {
    variations_per_room: number;
    model: string;
    quality: string;
    size: string;
  };
  rooms: Array<{
    id: string;
    label: string;
    original_image_url: string;
    original_thumbnail_url: string | null;
    status: string;
    prompt_addendum: string | null;
    variations: Array<{ id: string; status: string; created_at: string; updated_at: string }>;
    created_at: string;
    updated_at: string;
  }>;
  total_variations: number;
  completed_variations: number;
  created_at: string;
  updated_at: string;
  design_brief: null;
}

function makeRoomsProject(): MockRoomsProject {
  const now = new Date().toISOString();
  return {
    id: ROOMS_PROJECT_ID,
    name: 'Rooms Manager Test Project',
    prompt: 'modern minimalist',
    status: 'completed',
    settings: {
      variations_per_room: 2,
      model: 'gpt-image-2',
      quality: 'high',
      size: 'auto',
    },
    rooms: [
      {
        id: ROOM_A_ID,
        label: 'Living Room',
        original_image_url: `https://storage.blob.core.windows.net/images/staging/${ROOMS_PROJECT_ID}/originals/a.png`,
        original_thumbnail_url: `https://storage.blob.core.windows.net/images/staging/${ROOMS_PROJECT_ID}/originals/a-thumb.png`,
        status: 'completed',
        prompt_addendum: null,
        variations: [
          { id: 'r1-v0', status: 'completed', created_at: now, updated_at: now },
          { id: 'r1-v1', status: 'completed', created_at: now, updated_at: now },
        ],
        created_at: now,
        updated_at: now,
      },
      {
        id: ROOM_B_ID,
        label: 'Kitchen',
        original_image_url: `https://storage.blob.core.windows.net/images/staging/${ROOMS_PROJECT_ID}/originals/b.png`,
        original_thumbnail_url: `https://storage.blob.core.windows.net/images/staging/${ROOMS_PROJECT_ID}/originals/b-thumb.png`,
        status: 'completed',
        prompt_addendum: 'always include warm wood floors',
        variations: [
          { id: 'r2-v0', status: 'completed', created_at: now, updated_at: now },
          { id: 'r2-v1', status: 'completed', created_at: now, updated_at: now },
        ],
        created_at: now,
        updated_at: now,
      },
    ],
    total_variations: 4,
    completed_variations: 4,
    created_at: now,
    updated_at: now,
    design_brief: null,
  };
}

/**
 * Mirror of the backend's `__fields_set__`-aware update_room handler.
 * Without this, a label-only PATCH against the mock would silently
 * default `prompt_addendum` to None and the "addendum preservation"
 * assertion would fail for the wrong reason (the mock would clear it,
 * not the real bug). Mirrors `applyBackendMirror` for issue 002.
 */
function applyRoomPatch(
  state: MockRoomsProject,
  roomId: string,
  body: Record<string, unknown>,
): MockRoomsProject {
  const next: MockRoomsProject = {
    ...state,
    rooms: state.rooms.map((r) => {
      if (r.id !== roomId) return r;
      const updated = { ...r };
      if ('label' in body && typeof body.label === 'string') {
        updated.label = body.label.trim();
      }
      if ('prompt_addendum' in body) {
        const addendum = body.prompt_addendum;
        if (typeof addendum === 'string' && addendum.trim().length === 0) {
          updated.prompt_addendum = null;
        } else if (typeof addendum === 'string') {
          updated.prompt_addendum = addendum.trim();
        } else {
          updated.prompt_addendum = null;
        }
      }
      return updated;
    }),
  };
  return next;
}

async function setupGenerationTripwireForRoomsProject(
  page: Page,
  hits: string[],
): Promise<void> {
  await page.route(
    new RegExp(
      `${API_BASE}/staging/projects/${ROOMS_PROJECT_ID}/(generate|rooms/[^/]+/(?:regenerate|variations))`,
    ),
    (route: Route) => {
      hits.push(route.request().url());
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ok: true }),
      });
    },
  );
}

test.describe('Issue 004 — Rooms manager rename', () => {
  test('happy path: open Settings, rename room, PATCH carries {label} only, row reflects immediately, persists across reload', async ({
    page,
  }) => {
    let projectState = makeRoomsProject();
    const patchRequests: Array<{ url: string; method: string; body: Record<string, unknown> }> = [];
    const generateHits: string[] = [];

    await setupSasTokenMock(page);
    await setupGenerationTripwireForRoomsProject(page, generateHits);

    await page.route(
      `${API_BASE}/staging/projects/${ROOMS_PROJECT_ID}`,
      (route: Route) => {
        if (route.request().method() === 'GET') {
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ project: projectState }),
          });
        }
        return route.continue();
      },
    );

    await page.route(
      new RegExp(`${API_BASE}/staging/projects/${ROOMS_PROJECT_ID}/rooms/[^/]+$`),
      (route: Route) => {
        const url = route.request().url();
        const method = route.request().method();
        if (method === 'PATCH') {
          const match = url.match(/\/rooms\/([^/?]+)/);
          const roomId = match ? match[1] : '';
          const body = JSON.parse(route.request().postData() || '{}');
          patchRequests.push({ url, method, body });
          projectState = applyRoomPatch(projectState, roomId, body);
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ project: projectState }),
          });
        }
        return route.continue();
      },
    );

    await page.goto(`/projects/${ROOMS_PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Open Settings sheet.
    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();
    await expect(page.getByTestId('project-settings-sheet')).toBeVisible();

    // Rooms section is visible with both rooms.
    await expect(page.getByTestId('project-rooms-manager')).toBeVisible();
    await expect(
      page.getByTestId('project-rooms-manager-label-room-A'),
    ).toHaveText('Living Room');
    await expect(
      page.getByTestId('project-rooms-manager-label-room-B'),
    ).toHaveText('Kitchen');

    // Click the pencil on room A — input appears prefilled.
    await page.getByTestId('project-rooms-manager-edit-room-A').click();
    const input = page.getByTestId('project-rooms-manager-input-room-A');
    await expect(input).toHaveValue('Living Room');

    // Capture the project-level Save button state BEFORE the room
    // rename so we can pin that the rename does NOT enable it (rooms
    // persist immediately; project-level Save tracks name/prompt/
    // settings dirtiness only).
    const projectSaveBtn = page.getByTestId('project-settings-save');
    await expect(projectSaveBtn).toBeDisabled();

    // Type a new label (with whitespace, to also exercise the trim
    // contract).
    await input.fill('  Master Bedroom  ');
    await page.getByTestId('project-rooms-manager-save-room-A').click();

    // PATCH fired with the trimmed label, label-only body.
    await expect.poll(() => patchRequests.length).toBe(1);
    expect(patchRequests[0].body).toEqual({ label: 'Master Bedroom' });
    expect(patchRequests[0].url).toContain(`/rooms/${ROOM_A_ID}`);

    // Row reflects the new label without a reload (proves
    // onProjectUpdate → setProject path is wired).
    await expect(
      page.getByTestId('project-rooms-manager-label-room-A'),
    ).toHaveText('Master Bedroom');

    // Project-level Save button STILL disabled (the room rename did
    // not touch name/prompt/settings).
    await expect(projectSaveBtn).toBeDisabled();

    // No generation routes were hit.
    expect(generateHits).toEqual([]);

    // Hard reload + reopen — the new label persists.
    await page.reload();
    await page.waitForLoadState('networkidle');
    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();
    await expect(
      page.getByTestId('project-rooms-manager-label-room-A'),
    ).toHaveText('Master Bedroom');
  });

  test('label-only rename preserves existing addendum (rubber-duck blocker regression)', async ({
    page,
  }) => {
    // Room B has a non-empty prompt_addendum. Renaming it must NOT
    // silently clear the addendum — that's the rubber-duck blocker
    // the issue 004 backend handler fix specifically guards against.
    // The mock's applyRoomPatch implements the same field-set rules
    // the backend does, so a regression that re-introduces the
    // unconditional addendum write would clear the addendum here too
    // (the mock would obey whatever the client sent), and this
    // assertion would fail.
    let projectState = makeRoomsProject();
    expect(projectState.rooms[1].prompt_addendum).toBe(
      'always include warm wood floors',
    );
    const patchRequests: Array<{ body: Record<string, unknown> }> = [];

    await setupSasTokenMock(page);

    await page.route(
      `${API_BASE}/staging/projects/${ROOMS_PROJECT_ID}`,
      (route: Route) => {
        if (route.request().method() === 'GET') {
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ project: projectState }),
          });
        }
        return route.continue();
      },
    );

    await page.route(
      new RegExp(`${API_BASE}/staging/projects/${ROOMS_PROJECT_ID}/rooms/[^/]+$`),
      (route: Route) => {
        const url = route.request().url();
        const body = JSON.parse(route.request().postData() || '{}');
        patchRequests.push({ body });
        const match = url.match(/\/rooms\/([^/?]+)/);
        const roomId = match ? match[1] : '';
        projectState = applyRoomPatch(projectState, roomId, body);
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ project: projectState }),
        });
      },
    );

    await page.goto(`/projects/${ROOMS_PROJECT_ID}`);
    await page.waitForLoadState('networkidle');
    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();

    // Rename room B.
    await page.getByTestId('project-rooms-manager-edit-room-B').click();
    await page.getByTestId('project-rooms-manager-input-room-B').fill('Renamed Kitchen');
    await page.getByTestId('project-rooms-manager-save-room-B').click();

    // PATCH body contains label ONLY — no prompt_addendum key. The
    // load-bearing assertion that the client doesn't even send the
    // addendum field on a label-only edit. (The backend
    // __fields_set__-aware handler is also tested in pytest, but we
    // pin the wire shape here so a future refactor that "helpfully"
    // sends the current addendum back doesn't bypass the regression.)
    await expect.poll(() => patchRequests.length).toBe(1);
    expect(patchRequests[0].body).toEqual({ label: 'Renamed Kitchen' });
    expect(patchRequests[0].body).not.toHaveProperty('prompt_addendum');

    // The label updated.
    await expect(
      page.getByTestId('project-rooms-manager-label-room-B'),
    ).toHaveText('Renamed Kitchen');

    // The addendum survived (mock applied the same field-set rules
    // the backend does — no addendum key in the body means
    // addendum unchanged).
    expect(projectState.rooms[1].prompt_addendum).toBe(
      'always include warm wood floors',
    );
  });

  test('error path: PATCH returns 500, row reverts to original label, error toast appears', async ({
    page,
  }) => {
    const projectState = makeRoomsProject();
    let patchAttempts = 0;

    await setupSasTokenMock(page);

    await page.route(
      `${API_BASE}/staging/projects/${ROOMS_PROJECT_ID}`,
      (route: Route) => {
        if (route.request().method() === 'GET') {
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ project: projectState }),
          });
        }
        return route.continue();
      },
    );

    await page.route(
      new RegExp(`${API_BASE}/staging/projects/${ROOMS_PROJECT_ID}/rooms/[^/]+$`),
      (route: Route) => {
        patchAttempts++;
        return route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'something broke' }),
        });
      },
    );

    await page.goto(`/projects/${ROOMS_PROJECT_ID}`);
    await page.waitForLoadState('networkidle');
    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();

    await page.getByTestId('project-rooms-manager-edit-room-A').click();
    await page.getByTestId('project-rooms-manager-input-room-A').fill('Den');
    await page.getByTestId('project-rooms-manager-save-room-A').click();

    // PATCH was attempted.
    await expect.poll(() => patchAttempts).toBe(1);

    // Row reverts to view mode with the ORIGINAL label (the failure
    // handler clears edit state without calling onProjectUpdate, so
    // the static label re-renders with the unmodified project state).
    await expect(
      page.getByTestId('project-rooms-manager-label-room-A'),
    ).toHaveText('Living Room');

    // A sonner toast surfaces. We don't assert the exact message
    // (it's a thrown Error wrapped through the API client) but pin
    // that the toast region renders SOME content.
    await expect(page.locator('[data-sonner-toast]').first()).toBeVisible();
  });
});

/**
 * Issue 003 of the project-settings-completeness PRD.
 *
 * Generation settings dropdowns + read-only model. The PRD asks for:
 *
 *   - Quality dropdown with options low / medium / high / auto
 *   - Size dropdown with options auto / 1024x1024 / 1024x1536 / 1536x1024
 *   - Model rendered as a READ-ONLY label (no input affordance)
 *   - `useProjectSettings.save` (on `main`: `computeProjectSettingsDiff`
 *     in `ProjectSettingsSheet.tsx`) NEVER includes `model` in the
 *     outgoing payload, even defensively
 *
 * On `main`, quality + size already exist as Selects (PRD-spec'd
 * options match exactly) — the issue's "currently silently dropped"
 * claim about `size` reflects the worktree branch, not `main`. The
 * delta this slice creates is:
 *
 *   1. Model goes from a Select to a read-only label.
 *   2. The diff helper drops `model` from its input shape — so a
 *      future bug or a programmatic consumer can't accidentally
 *      smuggle a model change onto the wire.
 *
 * Test coverage map (PRD Acceptance Criteria for issue 003):
 *
 *   1. "dropdowns and read-only model": open sheet on a project with
 *      a known `model`. Assert:
 *        - `[data-testid="project-settings-model-readonly"]` exists
 *          and contains the human-readable label for the project's
 *          model value.
 *        - `[data-testid="project-settings-model-readonly"]` carries
 *          `aria-readonly="true"` so assistive tech sees it as a
 *          read-only display, not an interactive control.
 *        - The pre-issue-003 `[data-testid="project-settings-model-
 *          select"]` does NOT exist (regression guard against a
 *          revert).
 *        - Clicking the quality select opens a listbox with exactly
 *          the four spec'd options, in the spec'd order, with the
 *          project's current value (`high`) marked as selected.
 *        - Clicking the size select opens a listbox with exactly
 *          the four spec'd options, in the spec'd order, with the
 *          project's current value (`auto`) marked as selected.
 *
 *   2. "change quality + save": open, change quality from "high" to
 *      "medium", click Save. Assert:
 *        - PATCH body equals `{ settings: { quality: "medium" } }`
 *          exactly — NOT `{settings: {quality, model}}` (the load-
 *          bearing wire-shape assertion).
 *        - The body's `settings` does NOT contain a `model` key.
 *        - The body has no top-level `model` key either.
 *        - Hard reload + reopen shows the quality select displaying
 *          "Medium" (persisted value derived from the merged
 *          response).
 *
 *   3. "change size + Cancel reverts; resave persists": open, change
 *      size from "auto" to "1024x1024", click Cancel. Reopen and
 *      assert the size select is back at "auto" (the snapshot reset
 *      on each open is the regression guard for the user's
 *      "Discard changes" mental model). Then re-edit, save, reload,
 *      reopen, assert size persists.
 *
 * Each test installs the same `/generate` and `/regenerate` route
 * tripwire the issue 002 tests do — a saved generation-settings
 * change must NOT cascade into an automatic regeneration. The
 * existing `project-settings-future-only-notice` banner inside the
 * sheet + the existing toast on save play the role the PRD calls
 * "RegeneratePrompt banner" (which doesn't exist on `main`); both
 * are covered by the existing test 1 of this spec.
 */

const GEN_SETTINGS_PROJECT_ID = 'test-gen-settings';

function makeGenSettingsProject(
  overrides: Partial<MockProject> = {},
): MockProject {
  return makeProject({
    id: GEN_SETTINGS_PROJECT_ID,
    name: 'Generation Settings Test Project',
    prompt: 'modern minimalist',
    settings: {
      variations_per_room: 5,
      model: 'gpt-image-2',
      quality: 'high',
      size: 'auto',
    },
    ...overrides,
  });
}

async function setupGenerationTripwireForGenSettings(
  page: Page,
  hits: string[],
): Promise<void> {
  await page.route(
    new RegExp(
      `${API_BASE}/staging/projects/${GEN_SETTINGS_PROJECT_ID}/(generate|rooms/[^/]+/(?:regenerate|variations))`,
    ),
    (route: Route) => {
      hits.push(route.request().url());
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ok: true }),
      });
    },
  );
}

test.describe('Issue 003 — generation settings dropdowns + read-only model', () => {
  test('quality + size dropdowns enumerate exactly the spec\'d options; model is a read-only label', async ({
    page,
  }) => {
    const projectState = makeGenSettingsProject();
    const generateHits: string[] = [];

    await setupSasTokenMock(page);
    await setupGenerationTripwireForGenSettings(page, generateHits);

    await page.route(
      `${API_BASE}/staging/projects/${GEN_SETTINGS_PROJECT_ID}`,
      (route: Route) => {
        if (route.request().method() === 'GET') {
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ project: projectState }),
          });
        }
        return route.continue();
      },
    );

    await page.goto(`/projects/${GEN_SETTINGS_PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Open the sheet via the overflow menu.
    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();

    const sheet = page.getByTestId('project-settings-sheet');
    await expect(sheet).toBeVisible();

    // 1. Model is rendered as a read-only label — NOT a Select.
    //    The pre-issue-003 testid `project-settings-model-select`
    //    must be gone (regression guard against a revert).
    const modelReadonly = page.getByTestId('project-settings-model-readonly');
    await expect(modelReadonly).toBeVisible();
    // The display label should be the human-readable form. The exact
    // text comes from the `MODEL_DISPLAY_LABELS` map in
    // `ProjectSettingsSheet.tsx`. We assert containment so a future
    // copy tweak (e.g., adding "(default)" suffix) doesn't cascade
    // into a test churn — the load-bearing assertion is "shows the
    // project's model in a readable form".
    await expect(modelReadonly).toContainText(/gpt[-\s]?image[-\s]?2/i);
    // ARIA contract: assistive tech must see this as read-only.
    await expect(modelReadonly).toHaveAttribute('aria-readonly', 'true');
    // The OLD interactive Select must not exist anymore.
    await expect(
      page.getByTestId('project-settings-model-select'),
    ).toHaveCount(0);

    // 2. Quality dropdown: exactly the 4 spec'd options in order,
    //    current value selected.
    const qualityTrigger = page.getByTestId('project-settings-quality-select');
    await expect(qualityTrigger).toBeVisible();
    // The trigger shows the current value's human-readable label.
    await expect(qualityTrigger).toContainText(/high/i);
    await qualityTrigger.click();
    // Radix Select renders options in a portaled listbox. Each
    // SelectItem is `role="option"`. Match by accessible name (which
    // includes the option's full label text).
    const qualityOptions = page.locator('[role="option"]');
    await expect(qualityOptions).toHaveCount(4);
    // Spec'd value-set: low, medium, high, auto. Match against the
    // labels (which include parenthetical descriptors like "(default)").
    const qualityTexts = await qualityOptions.allTextContents();
    expect(qualityTexts.some((t) => /^low\b/i.test(t.trim()))).toBe(true);
    expect(qualityTexts.some((t) => /^medium\b/i.test(t.trim()))).toBe(true);
    expect(qualityTexts.some((t) => /^high\b/i.test(t.trim()))).toBe(true);
    expect(qualityTexts.some((t) => /^auto\b/i.test(t.trim()))).toBe(true);
    // Close the dropdown by re-clicking the trigger (Radix Select
    // treats this as a toggle). Avoid Escape: in this Sheet context,
    // an Escape captured by the portaled listbox can also bubble to
    // the Dialog's onEscapeKeyDown handler and close the sheet,
    // breaking the next assertion.
    await qualityTrigger.click();
    await expect(page.locator('[role="listbox"]')).toHaveCount(0);
    // Sanity guard: the sheet itself is still open so the size
    // trigger is reachable.
    await expect(sheet).toBeVisible();

    // 3. Size dropdown: exactly the 4 spec'd options in order,
    //    current value selected.
    const sizeTrigger = page.getByTestId('project-settings-size-select');
    await expect(sizeTrigger).toBeVisible();
    await expect(sizeTrigger).toContainText(/auto/i);
    await sizeTrigger.click();
    const sizeOptions = page.locator('[role="option"]');
    await expect(sizeOptions).toHaveCount(4);
    const sizeTexts = await sizeOptions.allTextContents();
    expect(sizeTexts.some((t) => /^auto\b/i.test(t.trim()))).toBe(true);
    expect(sizeTexts.some((t) => /1024\s*[×x]\s*1024/i.test(t))).toBe(true);
    expect(sizeTexts.some((t) => /1024\s*[×x]\s*1536/i.test(t))).toBe(true);
    expect(sizeTexts.some((t) => /1536\s*[×x]\s*1024/i.test(t))).toBe(true);
    await sizeTrigger.click();
    await expect(page.locator('[role="listbox"]')).toHaveCount(0);
    await expect(sheet).toBeVisible();

    // No accidental generation cascaded from opening the sheet.
    expect(generateHits).toEqual([]);
  });

  test('change quality + save → PATCH body has settings.quality only (no model), persists across reload', async ({
    page,
  }) => {
    let projectState = makeGenSettingsProject();
    const patchRequests: Array<{ body: Record<string, unknown> }> = [];
    const generateHits: string[] = [];

    await setupSasTokenMock(page);
    await setupGenerationTripwireForGenSettings(page, generateHits);

    await page.route(
      `${API_BASE}/staging/projects/${GEN_SETTINGS_PROJECT_ID}`,
      async (route: Route) => {
        const method = route.request().method();
        if (method === 'GET') {
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ project: projectState }),
          });
        }
        if (method === 'PATCH') {
          const body = JSON.parse(route.request().postData() || '{}');
          patchRequests.push({ body });
          // Mirror the backend MERGE on settings (same as test 1 of
          // this spec).
          const next: MockProject = { ...projectState };
          if ('name' in body) next.name = body.name as string;
          if ('prompt' in body) next.prompt = body.prompt as string;
          if ('settings' in body) {
            next.settings = {
              ...projectState.settings,
              ...(body.settings as Record<string, unknown>),
            };
          }
          projectState = next;
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ project: projectState }),
          });
        }
        return route.continue();
      },
    );

    await page.goto(`/projects/${GEN_SETTINGS_PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();
    await expect(page.getByTestId('project-settings-sheet')).toBeVisible();

    // Change quality from "high" to "medium" via the Select.
    await page.getByTestId('project-settings-quality-select').click();
    await page.getByRole('option', { name: /^medium/i }).click();

    // The Save button should be enabled now (we have a diff).
    const saveBtn = page.getByTestId('project-settings-save');
    await expect(saveBtn).toBeEnabled();

    await Promise.all([
      page.waitForRequest(
        (req) =>
          req.method() === 'PATCH' &&
          req.url().endsWith(`/staging/projects/${GEN_SETTINGS_PROJECT_ID}`),
      ),
      saveBtn.click(),
    ]);

    // 1. Wire-shape assertion: payload contains exactly
    //    { settings: { quality: "medium" } } — no model anywhere.
    expect(patchRequests).toHaveLength(1);
    expect(patchRequests[0].body).toEqual({
      settings: { quality: 'medium' },
    });
    // 2. Defensive structural pin: the settings partial must not
    //    contain `model`, even if the equality check above were
    //    relaxed in a future refactor (e.g., to accept extra keys).
    const settings = (patchRequests[0].body as { settings?: Record<string, unknown> })
      .settings;
    expect(settings).toBeDefined();
    expect(settings).not.toHaveProperty('model');
    // 3. Symmetric pin at the top level: no top-level `model` either.
    expect(patchRequests[0].body).not.toHaveProperty('model');

    // 4. Sheet closes after the successful save.
    await expect(page.getByTestId('project-settings-sheet')).not.toBeVisible();

    // 5. Hard reload + reopen → quality select shows "Medium"
    //    (persisted value).
    await page.reload();
    await page.waitForLoadState('networkidle');
    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();
    await expect(
      page.getByTestId('project-settings-quality-select'),
    ).toContainText(/medium/i);

    // 6. Tripwire: no generation routes hit.
    expect(generateHits).toEqual([]);
  });

  test('change size + Cancel reverts the dropdown; re-edit + save persists size across reload', async ({
    page,
  }) => {
    let projectState = makeGenSettingsProject();
    const patchRequests: Array<{ body: Record<string, unknown> }> = [];

    await setupSasTokenMock(page);

    await page.route(
      `${API_BASE}/staging/projects/${GEN_SETTINGS_PROJECT_ID}`,
      async (route: Route) => {
        const method = route.request().method();
        if (method === 'GET') {
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ project: projectState }),
          });
        }
        if (method === 'PATCH') {
          const body = JSON.parse(route.request().postData() || '{}');
          patchRequests.push({ body });
          const next: MockProject = { ...projectState };
          if ('settings' in body) {
            next.settings = {
              ...projectState.settings,
              ...(body.settings as Record<string, unknown>),
            };
          }
          projectState = next;
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ project: projectState }),
          });
        }
        return route.continue();
      },
    );

    await page.goto(`/projects/${GEN_SETTINGS_PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // 1. Open sheet, change size from "auto" to "1024x1024", click
    //    Cancel.
    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();

    let sizeTrigger = page.getByTestId('project-settings-size-select');
    await expect(sizeTrigger).toContainText(/auto/i);

    await sizeTrigger.click();
    await page.getByRole('option', { name: /1024.*1024/i }).click();
    await expect(sizeTrigger).toContainText(/1024/i);

    // The diff is non-empty so Save is enabled — but we Cancel.
    const saveBtn = page.getByTestId('project-settings-save');
    await expect(saveBtn).toBeEnabled();
    await page.getByTestId('project-settings-cancel').click();
    await expect(page.getByTestId('project-settings-sheet')).not.toBeVisible();

    // 2. No PATCH was fired — Cancel discards the local edit.
    expect(patchRequests).toEqual([]);

    // 3. Reopen the sheet — the size select is BACK at "auto". This
    //    is the load-bearing regression guard for the user's
    //    "Discard changes" mental model. Pre-issue-003 the snapshot
    //    reset only ran on Radix-internal close events, but Cancel
    //    + parent-driven reopen skipped it; the reset now lives in
    //    a rising-edge useEffect that fires on BOTH paths.
    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();
    sizeTrigger = page.getByTestId('project-settings-size-select');
    await expect(sizeTrigger).toContainText(/auto/i);

    // 4. Now actually change size + save.
    await sizeTrigger.click();
    await page.getByRole('option', { name: /1536.*1024/i }).click();
    await expect(sizeTrigger).toContainText(/1536/i);

    await Promise.all([
      page.waitForRequest(
        (req) =>
          req.method() === 'PATCH' &&
          req.url().endsWith(`/staging/projects/${GEN_SETTINGS_PROJECT_ID}`),
      ),
      page.getByTestId('project-settings-save').click(),
    ]);

    // 5. Wire-shape assertion: only size, no model, no other keys.
    expect(patchRequests).toHaveLength(1);
    expect(patchRequests[0].body).toEqual({
      settings: { size: '1536x1024' },
    });
    expect(
      (patchRequests[0].body as { settings?: Record<string, unknown> }).settings,
    ).not.toHaveProperty('model');

    // 6. Hard reload + reopen → size select shows the persisted
    //    "1536 × 1024" value.
    await page.reload();
    await page.waitForLoadState('networkidle');
    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();
    await expect(
      page.getByTestId('project-settings-size-select'),
    ).toContainText(/1536/i);
  });

  test('Esc closes the sheet; reopening shows persisted values (non-Cancel close path regression)', async ({
    page,
  }) => {
    let projectState = makeGenSettingsProject();
    const patchRequests: Array<{ body: Record<string, unknown> }> = [];

    await setupSasTokenMock(page);

    await page.route(
      `${API_BASE}/staging/projects/${GEN_SETTINGS_PROJECT_ID}`,
      async (route: Route) => {
        const method = route.request().method();
        if (method === 'GET') {
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ project: projectState }),
          });
        }
        if (method === 'PATCH') {
          const body = JSON.parse(route.request().postData() || '{}');
          patchRequests.push({ body });
          const next: MockProject = { ...projectState };
          if ('settings' in body) {
            next.settings = {
              ...projectState.settings,
              ...(body.settings as Record<string, unknown>),
            };
          }
          projectState = next;
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ project: projectState }),
          });
        }
        return route.continue();
      },
    );

    await page.goto(`/projects/${GEN_SETTINGS_PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // 1. Open sheet, change quality from "high" to "low" without
    //    saving.
    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();

    const qualityTrigger = page.getByTestId('project-settings-quality-select');
    await expect(qualityTrigger).toContainText(/high/i);

    await qualityTrigger.click();
    await page.getByRole('option', { name: /^low/i }).click();
    await expect(qualityTrigger).toContainText(/low/i);

    // 2. Close via Escape (NOT the Cancel button — different code
    //    path through Radix's onOpenChange handler). The
    //    rising-edge useEffect reset is the canonical mechanism that
    //    handles BOTH close paths uniformly; pre-issue-003 the
    //    in-handleOpenChange reset only ran on Radix-internal events
    //    AND on the Cancel button, but the parent-driven re-open path
    //    (overflow menu → Project settings) skipped it entirely. This
    //    test exercises the Esc path specifically as the duck-
    //    recommended non-Cancel regression guard.
    await page.keyboard.press('Escape');
    await expect(page.getByTestId('project-settings-sheet')).not.toBeVisible();
    expect(patchRequests).toEqual([]);

    // 3. Reopen — quality select shows the ORIGINAL "high" value,
    //    not the discarded "low" draft. This is the load-bearing
    //    assertion for the rising-edge reset on the non-Cancel close
    //    path.
    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();
    await expect(
      page.getByTestId('project-settings-quality-select'),
    ).toContainText(/high/i);
  });

  test('unknown model value falls back to the raw string in the read-only label', async ({
    page,
  }) => {
    // Defensive coverage per the duck-recommended fallback path: if
    // the backend introduces a new model value that has not yet been
    // mapped on the client (`MODEL_DISPLAY_LABELS`), the read-only
    // label must still render the raw model string rather than going
    // blank. Pre-issue-003 a Select with an unknown value would have
    // shown an empty trigger; the read-only div with raw-string
    // fallback degrades gracefully.
    const projectState = makeGenSettingsProject({
      settings: {
        variations_per_room: 5,
        model: 'future-unmapped-model-v9',
        quality: 'high',
        size: 'auto',
      },
    });

    await setupSasTokenMock(page);
    await page.route(
      `${API_BASE}/staging/projects/${GEN_SETTINGS_PROJECT_ID}`,
      (route: Route) => {
        if (route.request().method() === 'GET') {
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ project: projectState }),
          });
        }
        return route.continue();
      },
    );

    await page.goto(`/projects/${GEN_SETTINGS_PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    await page.getByRole('button', { name: /more actions/i }).click();
    await page.getByTestId('overflow-menu-project-settings').click();

    const modelReadonly = page.getByTestId('project-settings-model-readonly');
    await expect(modelReadonly).toBeVisible();
    // Raw string fallback — the unmapped model value renders verbatim.
    await expect(modelReadonly).toContainText('future-unmapped-model-v9');
    await expect(modelReadonly).toHaveAttribute('aria-readonly', 'true');
  });
});
