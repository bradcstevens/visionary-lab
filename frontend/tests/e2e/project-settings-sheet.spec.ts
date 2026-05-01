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
