/**
 * Frontend mirror of `backend/core/prompt_composer.py`'s
 * `compose_brief_markdown` and `extract_sections` helpers.
 *
 * Issue 019 of the image-pipeline-and-project-ux-overhaul PRD: the
 * settings panel renders a read-only Preview tab that shows the
 * composed prompt locally, updating after each save without a backend
 * round-trip. The TS port MUST agree byte-for-byte with the Python
 * implementation so the preview the user sees is identical to what
 * the backend will compose at generation time.
 *
 * Both helpers are pure: no I/O, no mutation, no logging.
 */
import { BRIEF_SECTIONS, titleToId } from "./briefSectionRegistry";

/**
 * Render the eight canonical sections to deterministic markdown.
 *
 * Output shape (one `## <Title>` heading per non-empty section,
 * registry-ordered, paragraph-break separated):
 *
 *     ## Edit Task
 *     <content>
 *
 *     ## Edit Zone
 *     <content>
 *
 * Determinism contract:
 *
 * - Iteration order is `BRIEF_SECTIONS`, NOT `Object.keys(sections)`.
 *   Same input → byte-identical output regardless of insertion order.
 * - Sections with `null` / missing / empty-after-strip values are
 *   OMITTED entirely (no empty heading body).
 * - Section bodies are stripped on the way out — leading and trailing
 *   whitespace in a section is dropped, internal newlines preserved.
 * - Unknown section ids (not in the registry) are silently dropped.
 *
 * `rawOverride` precedence:
 *
 * - If `rawOverride` is non-`null`/`undefined` AND non-empty after
 *   strip, it is returned VERBATIM (NOT stripped — preserves
 *   intentional leading/trailing whitespace) and `sections` is
 *   ignored entirely.
 * - Otherwise the composed-from-sections path runs.
 */
export function composeBriefMarkdown(
  sections: Readonly<Record<string, string | undefined | null>>,
  rawOverride?: string | null,
): string {
  if (rawOverride != null && rawOverride.trim().length > 0) {
    return rawOverride;
  }
  const parts: string[] = [];
  for (const section of BRIEF_SECTIONS) {
    const content = sections[section.id];
    if (content == null) continue;
    const stripped = content.trim();
    if (!stripped) continue;
    parts.push(`## ${section.title}\n${stripped}`);
  }
  return parts.join("\n\n");
}

/**
 * Parse markdown produced by `composeBriefMarkdown` back to a
 * `{section_id: content}` map.
 *
 * Round-trip contract — pinned by the round-trip test:
 *
 *     extractSections(composeBriefMarkdown(s)) === \\
 *       {k: v.trim() for k,v in s if non-empty}
 *
 * Robustness:
 *
 * - Headings whose title doesn't match any registered section are
 *   silently dropped.
 * - Any text before the first `## ` heading is dropped (free-form
 *   preamble has no slot in the structured schema).
 * - Title matching is case- and punctuation-insensitive (mirrors
 *   `titleToId`).
 * - Duplicate headings: LAST writer wins.
 */
export function extractSections(markdown: string): Record<string, string> {
  if (!markdown) return {};
  // Split on level-2 ATX headings on their own line. The capturing
  // group means the result interleaves [preamble, title_1, body_1,
  // title_2, body_2, ...]. Preamble is dropped.
  const headingRe = /^##[ \t]+(.+?)[ \t]*$/gm;
  const chunks = markdown.split(headingRe);
  if (chunks.length < 3) return {};
  const result: Record<string, string> = {};
  for (let i = 1; i < chunks.length; i += 2) {
    const title = chunks[i];
    const body = i + 1 < chunks.length ? chunks[i + 1] : "";
    const id = titleToId(title);
    if (id == null) continue;
    const stripped = body.trim();
    if (!stripped) continue;
    result[id] = stripped;
  }
  return result;
}
