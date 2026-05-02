import { describe, it, expect } from "vitest";
import {
  computeProjectSettingsDiff,
  derivePromptForSettings,
} from "../ProjectSettingsSheet";
import type { DesignBrief, StagingProject } from "@/services/stagingApi";

/**
 * Issue 002 of the project-settings-completeness PRD.
 *
 * Pure-function coverage of `derivePromptForSettings`. The PRD text says
 * "derive the displayed prompt as `design_brief.global_instructions ?? project.prompt`",
 * but the project intentionally extends "empty" to "missing OR empty
 * string OR whitespace-only" to match the backend mirror's `_is_nonempty_str`
 * gate (`backend/api/endpoints/staging.py:436-442`). Without that match,
 * a user could see a whitespace-only brief in the textarea (which renders
 * as visually empty) and silently overwrite the legacy `project.prompt`
 * on first save — the canonical-prompt symmetry would be violated for
 * legacy-with-whitespace-brief projects.
 *
 * Invariants asserted here:
 *
 *   - non-empty brief.global_instructions wins over project.prompt
 *   - empty / whitespace-only brief.global_instructions falls back to
 *     project.prompt (same gate as backend mirror)
 *   - missing or null design_brief falls back to project.prompt
 *   - missing project.prompt collapses to ""
 *   - undefined design_brief is treated identically to null (defensive)
 *
 * The corresponding e2e propagation guards live in
 * `frontend/tests/e2e/project-settings-sheet.spec.ts`'s "Issue 002 —
 * canonical project prompt" describe block.
 */

const EMPTY_BRIEF_FIELDS = {
  object_palette: [],
  placement_guide: { back_row: "" },
  per_image_notes: {},
  per_image_objects: {},
  preserve_elements: [],
  settings: {
    variations_per_room: 5,
    model: "gpt-image-2",
    quality: "high",
    size: "auto",
  },
} as const;

function makeBrief(globalInstructions: string): DesignBrief {
  return {
    global_instructions: globalInstructions,
    ...EMPTY_BRIEF_FIELDS,
  };
}

function makeProject(overrides: Partial<StagingProject> = {}): StagingProject {
  return {
    id: "proj-1",
    name: "Test Project",
    prompt: "legacy project prompt",
    status: "completed",
    settings: {
      variations_per_room: 5,
      model: "gpt-image-2",
      quality: "high",
      size: "auto",
    },
    rooms: [],
    ...overrides,
  };
}

describe("derivePromptForSettings — design_brief wins when non-empty", () => {
  it("returns design_brief.global_instructions when it has real content", () => {
    const project = makeProject({
      prompt: "legacy placeholder",
      design_brief: makeBrief("real designer-authored prompt"),
    });
    expect(derivePromptForSettings(project)).toBe(
      "real designer-authored prompt",
    );
  });

  it("returns design_brief.global_instructions even when project.prompt is also non-empty (brief takes precedence)", () => {
    const project = makeProject({
      prompt: "Draft — pending AI Design Session",
      design_brief: makeBrief("global instructions from the brief"),
    });
    expect(derivePromptForSettings(project)).toBe(
      "global instructions from the brief",
    );
  });

  it("preserves the brief's exact text (no trim) when non-empty", () => {
    const text = "  prompt with leading and trailing whitespace  ";
    const project = makeProject({
      design_brief: makeBrief(text),
    });
    expect(derivePromptForSettings(project)).toBe(text);
  });
});

describe("derivePromptForSettings — falls back to project.prompt when brief is empty-ish", () => {
  it("falls back when global_instructions is the empty string", () => {
    const project = makeProject({
      prompt: "legacy fallback",
      design_brief: makeBrief(""),
    });
    expect(derivePromptForSettings(project)).toBe("legacy fallback");
  });

  it("falls back when global_instructions is whitespace-only (mirrors backend _is_nonempty_str)", () => {
    const project = makeProject({
      prompt: "legacy fallback",
      design_brief: makeBrief("   \t\n  "),
    });
    expect(derivePromptForSettings(project)).toBe("legacy fallback");
  });
});

describe("derivePromptForSettings — falls back to project.prompt when no brief", () => {
  it("falls back when design_brief is null", () => {
    const project = makeProject({
      prompt: "legacy only",
      design_brief: null,
    });
    expect(derivePromptForSettings(project)).toBe("legacy only");
  });

  it("falls back when design_brief is undefined (defensive — same as null)", () => {
    const project = makeProject({ prompt: "legacy only" });
    delete project.design_brief;
    expect(derivePromptForSettings(project)).toBe("legacy only");
  });
});

describe("derivePromptForSettings — collapses to empty string when nothing available", () => {
  it("returns '' when no brief AND no project.prompt", () => {
    const project = makeProject({ prompt: "", design_brief: null });
    expect(derivePromptForSettings(project)).toBe("");
  });

  it("returns '' when no brief AND project.prompt is undefined", () => {
    const project = makeProject({ design_brief: null });
    // The StagingProject type marks prompt as required, but Cosmos can
    // return projects with missing prompt fields in legacy data — the
    // helper must not throw on that input.
    delete (project as unknown as { prompt?: string }).prompt;
    expect(derivePromptForSettings(project)).toBe("");
  });

  it("returns '' when brief.global_instructions is whitespace-only AND project.prompt is also empty (both gates fall through)", () => {
    const project = makeProject({
      prompt: "",
      design_brief: makeBrief("   "),
    });
    expect(derivePromptForSettings(project)).toBe("");
  });
});

/**
 * Issue 003 of the project-settings-completeness PRD.
 *
 * `computeProjectSettingsDiff` is the function that decides what
 * actually goes on the wire when the user clicks "Save". Issue 003
 * makes the model field READ-ONLY in the UI: the diff helper's input
 * shape no longer carries `model` at all, so even if a future bug or
 * a programmatic consumer toggled a "model" value, the diff helper
 * would NEVER include it in the outgoing payload. This is the
 * defensive structural guarantee the issue asks for.
 *
 * Coverage matrix:
 *
 *   - input shape excludes `model` entirely (compile-time guarantee
 *     verified by the test file building without a `model` key);
 *   - identical inputs produce an empty diff;
 *   - per-field changes (name, prompt, variations_per_room, quality,
 *     size) each produce the correct minimal payload;
 *   - settings-key changes are batched into one `settings` partial;
 *   - whitespace trim semantics on `name` and `prompt` match the
 *     existing handler contract;
 *   - the returned object is never `{ settings: {} }` — empty
 *     `settings` is omitted entirely so the backend's MERGE semantics
 *     are not invoked unnecessarily.
 *
 * The corresponding e2e propagation guards live in
 * `frontend/tests/e2e/project-settings-sheet.spec.ts`'s "Issue 003 —
 * generation settings dropdowns + read-only model" describe block,
 * which asserts the wire-shape `not.toHaveProperty(['settings',
 * 'model'])` against the actual PATCH body that crosses the network.
 */

const BASELINE = {
  name: "Test Project",
  prompt: "modern minimalist",
  variations_per_room: 5,
  quality: "high",
  size: "auto",
} as const;

describe("computeProjectSettingsDiff — empty diff when nothing changed", () => {
  it("returns {} when current === initial (identity)", () => {
    expect(computeProjectSettingsDiff({ ...BASELINE }, { ...BASELINE })).toEqual({});
  });

  it("returns {} when only whitespace-padded name/prompt that trim to identity", () => {
    expect(
      computeProjectSettingsDiff(
        { ...BASELINE },
        { ...BASELINE, name: "  Test Project  ", prompt: "  modern minimalist  " },
      ),
    ).toEqual({});
  });
});

describe("computeProjectSettingsDiff — single-field diffs", () => {
  it("returns name change as top-level trimmed string", () => {
    expect(
      computeProjectSettingsDiff(
        { ...BASELINE },
        { ...BASELINE, name: "  Renamed  " },
      ),
    ).toEqual({ name: "Renamed" });
  });

  it("returns prompt change as top-level trimmed string", () => {
    expect(
      computeProjectSettingsDiff(
        { ...BASELINE },
        { ...BASELINE, prompt: "  fresh prompt  " },
      ),
    ).toEqual({ prompt: "fresh prompt" });
  });

  it("returns variations_per_room change inside settings partial", () => {
    expect(
      computeProjectSettingsDiff(
        { ...BASELINE },
        { ...BASELINE, variations_per_room: 3 },
      ),
    ).toEqual({ settings: { variations_per_room: 3 } });
  });

  it("returns quality change inside settings partial", () => {
    expect(
      computeProjectSettingsDiff(
        { ...BASELINE },
        { ...BASELINE, quality: "medium" },
      ),
    ).toEqual({ settings: { quality: "medium" } });
  });

  it("returns size change inside settings partial", () => {
    expect(
      computeProjectSettingsDiff(
        { ...BASELINE },
        { ...BASELINE, size: "1024x1024" },
      ),
    ).toEqual({ settings: { size: "1024x1024" } });
  });
});

describe("computeProjectSettingsDiff — multi-field diffs batch settings", () => {
  it("batches multiple settings keys into one settings partial", () => {
    expect(
      computeProjectSettingsDiff(
        { ...BASELINE },
        { ...BASELINE, quality: "low", size: "1536x1024" },
      ),
    ).toEqual({ settings: { quality: "low", size: "1536x1024" } });
  });

  it("combines top-level and settings changes correctly", () => {
    expect(
      computeProjectSettingsDiff(
        { ...BASELINE },
        {
          ...BASELINE,
          name: "Renamed",
          prompt: "fresh prompt",
          variations_per_room: 2,
          quality: "auto",
          size: "1024x1536",
        },
      ),
    ).toEqual({
      name: "Renamed",
      prompt: "fresh prompt",
      settings: {
        variations_per_room: 2,
        quality: "auto",
        size: "1024x1536",
      },
    });
  });
});

describe("computeProjectSettingsDiff — never includes model", () => {
  /**
   * The strongest possible "never includes model" guarantee is
   * STRUCTURAL: the input shape itself excludes `model`, so a
   * `settings.model` key cannot reach the diff helper's output even
   * via a future programmatic mistake. The test file would fail to
   * type-check (and fail at runtime by `unknown` access) if a future
   * change re-added `model` to the helper's signature without also
   * adding the never-send guard. This pins the contract at the type
   * boundary, not just at the runtime branch.
   */
  it("the helper's input shape does not accept a `model` field", () => {
    const initial = { ...BASELINE };
    const current = { ...BASELINE };
    // @ts-expect-error — the input shape MUST NOT include `model`. If
    // a future change re-adds it, this directive becomes "unused" and
    // typescript-eslint-react fails the build, signaling that the
    // structural guarantee is broken.
    initial.model = "gpt-image-2";
    // @ts-expect-error — same as above for the `current` half.
    current.model = "flux-kontext-pro";
    const diff = computeProjectSettingsDiff(initial, current);
    expect(diff).toEqual({});
    expect(diff.settings).toBeUndefined();
  });

  it("a multi-field diff never produces a `settings.model` key in the output, even when other settings keys change", () => {
    const diff = computeProjectSettingsDiff(
      { ...BASELINE },
      { ...BASELINE, quality: "low", size: "1024x1024", variations_per_room: 1 },
    );
    expect(diff.settings).toBeDefined();
    expect(diff.settings).not.toHaveProperty("model");
    // Belt-and-suspenders: assert the exact key set so a future
    // change that adds keys won't pass this test silently.
    expect(Object.keys(diff.settings ?? {}).sort()).toEqual([
      "quality",
      "size",
      "variations_per_room",
    ]);
  });
});

describe("computeProjectSettingsDiff — empty settings is omitted", () => {
  it("does not emit `settings: {}` when only top-level fields changed", () => {
    const diff = computeProjectSettingsDiff(
      { ...BASELINE },
      { ...BASELINE, name: "Renamed" },
    );
    expect(diff).not.toHaveProperty("settings");
  });
});
