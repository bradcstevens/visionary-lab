import { describe, it, expect } from "vitest";
import { derivePromptForSettings } from "../ProjectSettingsSheet";
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
