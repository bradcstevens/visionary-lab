import { describe, it, expect } from "vitest";
import {
  BRIEF_SECTIONS,
  sectionIds,
  getSection,
  titleToId,
} from "../briefSectionRegistry";

/**
 * Frontend mirror of `backend/core/brief_section_registry.py`. Issue
 * 019 of the image-pipeline-and-project-ux-overhaul PRD wants the
 * settings panel and the wizard driven by the same canonical list of
 * eight sections. The registry is the single source of truth on the
 * frontend.
 */

const EXPECTED_IDS = [
  "edit_task",
  "edit_zone",
  "do_not_alter",
  "object_palette",
  "arrangement",
  "regional_constraints",
  "aesthetic_goal",
  "scale_fidelity",
] as const;

describe("BriefSectionRegistry — canonical list", () => {
  it("ships exactly eight canonical sections", () => {
    expect(BRIEF_SECTIONS.length).toBe(8);
  });

  it("returns ids in registry order matching the backend", () => {
    expect(sectionIds()).toEqual(EXPECTED_IDS);
  });

  it("each section has id, title, and description", () => {
    for (const s of BRIEF_SECTIONS) {
      expect(typeof s.id).toBe("string");
      expect(s.id.length).toBeGreaterThan(0);
      expect(typeof s.title).toBe("string");
      expect(s.title.length).toBeGreaterThan(0);
      expect(typeof s.description).toBe("string");
    }
  });
});

describe("BriefSectionRegistry — getSection", () => {
  it("returns the BriefSection for a known id", () => {
    expect(getSection("edit_task").title).toBe("Edit Task");
    expect(getSection("scale_fidelity").title).toBe("Scale & Fidelity");
  });

  it("throws for an unknown id", () => {
    expect(() => getSection("nope")).toThrow();
  });
});

describe("BriefSectionRegistry — titleToId", () => {
  it("returns the id for an exact title match", () => {
    expect(titleToId("Edit Task")).toBe("edit_task");
  });

  it("matches case- and punctuation-insensitively for exact title round-trip", () => {
    // Mirrors the backend's normalisation: lowercase + drop
    // non-alphanumerics. So `## Scale & Fidelity` produced by
    // composeBriefMarkdown round-trips through extractSections.
    expect(titleToId("Scale & Fidelity")).toBe("scale_fidelity");
    expect(titleToId("SCALE & FIDELITY")).toBe("scale_fidelity");
    expect(titleToId("  Scale & Fidelity  ")).toBe("scale_fidelity");
    expect(titleToId("scale&fidelity")).toBe("scale_fidelity");
  });

  it("returns null for an unknown title", () => {
    expect(titleToId("not a real heading")).toBeNull();
  });
});
