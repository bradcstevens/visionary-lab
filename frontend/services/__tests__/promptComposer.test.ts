import { describe, it, expect } from "vitest";
import {
  composeBriefMarkdown,
  extractSections,
} from "../promptComposer";

/**
 * Frontend mirror of `backend/core/prompt_composer.py`'s
 * `compose_brief_markdown` / `extract_sections`. Issue 019 of the
 * image-pipeline-and-project-ux-overhaul PRD: the settings panel's
 * read-only Preview tab renders the composed markdown locally so the
 * user can see what will be sent to the model after each save —
 * without round-tripping through the backend on every edit.
 *
 * The TS implementation MUST agree byte-for-byte with the Python one
 * for any input the UI is allowed to produce.
 */

describe("composeBriefMarkdown — registry-ordered sections", () => {
  it("returns empty string for an empty sections map and no override", () => {
    expect(composeBriefMarkdown({})).toBe("");
  });

  it("renders one ## heading per non-empty section", () => {
    const md = composeBriefMarkdown({
      edit_task: "Add a stone path.",
    });
    expect(md).toBe("## Edit Task\nAdd a stone path.");
  });

  it("preserves registry order regardless of object insertion order", () => {
    const sections: Record<string, string> = {};
    sections.scale_fidelity = "highly detailed";
    sections.edit_task = "Add evergreens";
    sections.aesthetic_goal = "Pacific Northwest";
    const md = composeBriefMarkdown(sections);
    expect(md).toBe(
      "## Edit Task\nAdd evergreens\n\n" +
        "## Aesthetic Goal\nPacific Northwest\n\n" +
        "## Scale & Fidelity\nhighly detailed",
    );
  });

  it("omits sections with empty / whitespace-only content", () => {
    const md = composeBriefMarkdown({
      edit_task: "Add evergreens",
      edit_zone: "",
      do_not_alter: "   ",
      arrangement: "Tall in back",
    });
    expect(md).toBe(
      "## Edit Task\nAdd evergreens\n\n## Arrangement\nTall in back",
    );
  });

  it("strips leading and trailing whitespace inside section bodies", () => {
    const md = composeBriefMarkdown({ edit_task: "  trim me  \n" });
    expect(md).toBe("## Edit Task\ntrim me");
  });

  it("silently drops unknown section ids (not in the registry)", () => {
    const md = composeBriefMarkdown({
      edit_task: "Add evergreens",
      not_a_real_section: "should be dropped",
    });
    expect(md).toBe("## Edit Task\nAdd evergreens");
  });
});

describe("composeBriefMarkdown — raw_override precedence", () => {
  it("returns rawOverride verbatim when non-empty", () => {
    expect(
      composeBriefMarkdown({ edit_task: "ignored" }, "user-typed exactly this"),
    ).toBe("user-typed exactly this");
  });

  it("preserves intentional leading / trailing whitespace in rawOverride", () => {
    const override = "  power user prompt  \n";
    expect(composeBriefMarkdown({ edit_task: "ignored" }, override)).toBe(
      override,
    );
  });

  it("falls through to sections when rawOverride is empty / whitespace-only", () => {
    expect(composeBriefMarkdown({ edit_task: "Add" }, "")).toBe(
      "## Edit Task\nAdd",
    );
    expect(composeBriefMarkdown({ edit_task: "Add" }, "   ")).toBe(
      "## Edit Task\nAdd",
    );
  });

  it("falls through to sections when rawOverride is null / undefined", () => {
    expect(composeBriefMarkdown({ edit_task: "Add" }, null)).toBe(
      "## Edit Task\nAdd",
    );
    expect(composeBriefMarkdown({ edit_task: "Add" }, undefined)).toBe(
      "## Edit Task\nAdd",
    );
  });
});

describe("extractSections — round-trip with composeBriefMarkdown", () => {
  it("round-trips a populated sections map (keys preserved, bodies stripped)", () => {
    const original: Record<string, string> = {
      edit_task: "Add evergreens",
      edit_zone: "Back yard only",
      do_not_alter: "Existing fence",
      object_palette: "- 3x evergreen\n- 2x bench",
      arrangement: "Tall in back",
      regional_constraints: "Pacific Northwest hardiness",
      aesthetic_goal: "Modern minimalist",
      scale_fidelity: "Photorealistic",
    };
    const md = composeBriefMarkdown(original);
    const parsed = extractSections(md);
    expect(parsed).toEqual(original);
  });

  it("returns an empty object for an empty / null markdown", () => {
    expect(extractSections("")).toEqual({});
    expect(extractSections("free-form preamble with no headings")).toEqual({});
  });

  it("drops headings that don't match a registered section", () => {
    const md =
      "## Edit Task\nAdd evergreens\n\n## Random Heading\nignored body";
    expect(extractSections(md)).toEqual({ edit_task: "Add evergreens" });
  });

  it("drops free-form preamble before the first ## heading", () => {
    const md =
      "Some preamble text.\n\n## Edit Task\nAdd evergreens";
    expect(extractSections(md)).toEqual({ edit_task: "Add evergreens" });
  });

  it("LAST writer wins on duplicate headings", () => {
    const md =
      "## Edit Task\nfirst body\n\n## Edit Task\nsecond body";
    expect(extractSections(md)).toEqual({ edit_task: "second body" });
  });
});
