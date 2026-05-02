/**
 * Frontend mirror of `backend/core/brief_section_registry.py`.
 *
 * Single source of truth for the eight canonical Design Brief sections.
 * The wizard (`NewProjectWizard`) and the settings panel
 * (`ProjectSettingsSheet`) both consume this registry so the two
 * surfaces always expose the same sections in the same order. Adding a
 * section is a one-line change here.
 *
 * Issue 019 of the image-pipeline-and-project-ux-overhaul PRD.
 *
 * Section content lives on `DesignBrief.sections` keyed by
 * `BriefSection.id`; rendering to the top-level prompt markdown lives
 * in `promptComposer.composeBriefMarkdown`.
 *
 * The list MUST stay byte-for-byte aligned with the backend
 * `SECTIONS` tuple (ids, order, titles). The round-trip test in
 * `promptComposer.test.ts` pins this contract from both sides.
 */

export interface BriefSection {
  /**
   * Stable machine identifier. Used as the key in
   * `DesignBrief.sections`. Lower snake_case. NEVER change once
   * shipped.
   */
  id: string;
  /**
   * Display title used as the rendered markdown `## <title>` heading
   * and as the wizard step / settings-panel tab label.
   */
  title: string;
  /**
   * One-sentence helper text for the wizard step / tab tooltip. Not
   * rendered into the prompt markdown.
   */
  description: string;
}

/**
 * The eight canonical sections, in the rendered + wizard order pinned
 * by the PRD. Order is part of the contract — both
 * `composeBriefMarkdown` output and the wizard step sequence follow
 * this list verbatim.
 */
export const BRIEF_SECTIONS: ReadonlyArray<BriefSection> = Object.freeze([
  {
    id: "edit_task",
    title: "Edit Task",
    description: "What overall change should the model make to the scene?",
  },
  {
    id: "edit_zone",
    title: "Edit Zone",
    description: "Which area of the image is in scope for editing?",
  },
  {
    id: "do_not_alter",
    title: "Do Not Alter",
    description: "Elements that must remain unchanged across renders.",
  },
  {
    id: "object_palette",
    title: "Object Palette",
    description: "The set of objects available for placement in the scene.",
  },
  {
    id: "arrangement",
    title: "Arrangement",
    description: "How objects should be composed and positioned.",
  },
  {
    id: "regional_constraints",
    title: "Regional Constraints",
    description: "Climate, plant-hardiness, code, or other regional rules.",
  },
  {
    id: "aesthetic_goal",
    title: "Aesthetic Goal",
    description: "The overall visual style, mood, or design intent.",
  },
  {
    id: "scale_fidelity",
    title: "Scale & Fidelity",
    description: "Sizing, level of detail, and rendering fidelity expectations.",
  },
]);

const BY_ID = new Map<string, BriefSection>(
  BRIEF_SECTIONS.map((s) => [s.id, s]),
);

function normaliseTitle(text: string): string {
  let out = "";
  for (const ch of text.toLowerCase()) {
    if ((ch >= "a" && ch <= "z") || (ch >= "0" && ch <= "9")) {
      out += ch;
    }
  }
  return out;
}

const BY_NORM_TITLE = new Map<string, BriefSection>(
  BRIEF_SECTIONS.map((s) => [normaliseTitle(s.title), s]),
);

/** Return the eight canonical section ids in registry order. */
export function sectionIds(): ReadonlyArray<string> {
  return BRIEF_SECTIONS.map((s) => s.id);
}

/**
 * Return the BriefSection for `id`. Throws if `id` is not a registered
 * section — the caller is asking for a section that does not exist,
 * which is a programming error, not user input.
 */
export function getSection(id: string): BriefSection {
  const s = BY_ID.get(id);
  if (!s) {
    throw new Error(`Unknown brief section id: ${id}`);
  }
  return s;
}

/**
 * Return the section id whose title matches `title`, or `null`.
 *
 * Match is case-insensitive and ignores punctuation / whitespace so
 * `## Scale & Fidelity` and `## scale and fidelity` both resolve to
 * `scale_fidelity`. `null` is returned for unknown titles so
 * extractors can route unregistered headings to a free-form bucket
 * without raising.
 */
export function titleToId(title: string): string | null {
  return BY_NORM_TITLE.get(normaliseTitle(title))?.id ?? null;
}
