"""Single source of truth for composing the final ``adapted_prompt``.

The staging pipeline has two prompt-generation source paths
(``BriefGeneratorService.brief_to_prompts`` for projects with a brief,
``StagingPipeline.adapt_prompt`` for legacy projects without one), plus
slice 004's incoming Edit Prompt path that takes a user-typed override.
All three sources need to honor the per-room ``prompt_addendum`` field
the same way: append it to the chosen base with a paragraph break.

This module exposes one pure helper that every call site delegates to.
The helper is pure: no I/O, no mutation, no logging, no Pydantic model
dependency — it operates on plain strings.

Per the projects-page-improvements PRD § Implementation Decisions →
Backend (PromptComposer bullet) and § Further Notes (addendum
precedence + Retry semantics).

Issue 015 of the image-pipeline-and-project-ux-overhaul PRD extends this
module with two pure helpers driven by ``BriefSectionRegistry``:

* ``compose_brief_markdown`` — render the eight canonical sections of a
  Design Brief into a deterministic top-level prompt markdown, honoring
  ``raw_override`` when set.
* ``extract_sections`` — parse the same markdown back into a section-id
  → content map, used by the legacy-brief lazy backfill (issue 016) and
  by the round-trip test in this slice.
"""

import re
from typing import Dict, Mapping, Optional

from backend.core.brief_section_registry import (
    SECTIONS,
    title_to_id,
)


# Matches a level-2 ATX heading on its own line, e.g. ``## Edit Task``.
# Body text under each heading runs until the next ``## `` or end-of-string.
# The split-then-walk approach in ``extract_sections`` uses this pattern in
# a single ``re.split`` call so we don't need a nested-state parser.
_HEADING_RE = re.compile(r"(?m)^##[ \t]+(.+?)[ \t]*$")


class PromptComposer:
    """Pure helper that assembles the final ``adapted_prompt``."""

    @staticmethod
    def compose(
        project_prompt: str,
        design_brief: Optional[str],
        room_addendum: Optional[str],
        variation_override: Optional[str] = None,
    ) -> str:
        """Return the final ``adapted_prompt`` from layered inputs.

        Base selection (highest to lowest precedence):

        1. ``variation_override`` (when non-empty after strip) — the
           user-typed prompt from slice 004's Edit Prompt path.
        2. ``design_brief`` (when non-empty after strip) — the typical
           per-variation prompt produced by ``brief_to_prompts`` or
           ``adapt_prompt``.
        3. ``project_prompt`` (stripped) — last-resort fallback for
           legacy projects that lack a brief.

        Once the base is chosen, ``room_addendum`` is **always**
        appended with a ``"\\n\\n"`` separator when it's non-empty after
        strip — even when the base came from ``variation_override``.
        This matches the PRD's Edit Prompt semantic: a user-typed
        prompt still respects the room's addendum constraint.

        Whitespace, ``None`` and empty inputs are all handled cleanly:
        every parameter is treated as "absent" if it strips to empty.

        The function is pure — no I/O, no mutation, no logging. The
        caller is responsible for assigning the returned string into
        ``Variation.generation_metadata.adapted_prompt``.
        """
        override_clean = (variation_override or "").strip()
        brief_clean = (design_brief or "").strip()
        project_clean = (project_prompt or "").strip()

        if override_clean:
            base = override_clean
        elif brief_clean:
            base = brief_clean
        else:
            base = project_clean

        addendum_clean = (room_addendum or "").strip()
        if not addendum_clean:
            return base
        if not base:
            return addendum_clean
        return f"{base}\n\n{addendum_clean}"

    @staticmethod
    def compose_brief_markdown(
        sections: Mapping[str, str],
        raw_override: Optional[str] = None,
    ) -> str:
        """Render the eight canonical sections to deterministic markdown.

        Output shape (one ``## <Title>`` heading per non-empty section,
        registry-ordered, paragraph-break separated):

            ## Edit Task
            <content>

            ## Edit Zone
            <content>

            ...

        Determinism contract — pinned by the round-trip test:

        * Iteration order is the order of ``BriefSectionRegistry.SECTIONS``,
          NOT ``sections.keys()`` order. Same input → byte-identical
          output regardless of dict insertion order.
        * Sections whose value is ``None`` / missing / empty after strip
          are OMITTED from the output entirely (no empty heading body).
        * Section bodies are stripped on the way out — leading and
          trailing whitespace inside a section is dropped, but internal
          newlines are preserved verbatim.
        * Unknown section ids (not in the registry) are silently dropped
          rather than rendered. This keeps a future schema where we
          remove a section from the registry from poisoning the output
          with stale content.

        ``raw_override`` precedence — pinned by the override test:

        * If ``raw_override`` is non-``None`` AND non-empty after strip,
          it is returned VERBATIM (NOT stripped — preserves any
          intentional leading / trailing whitespace the power user
          typed) and ``sections`` is ignored entirely.
        * If ``raw_override`` is ``None`` or empty / whitespace-only,
          the composed-from-sections path runs.

        Pure: no I/O, no mutation, no logging, no Pydantic dependency.
        """
        if raw_override is not None and raw_override.strip():
            return raw_override

        parts: list[str] = []
        for section in SECTIONS:
            content = sections.get(section.id)
            if content is None:
                continue
            stripped = content.strip()
            if not stripped:
                continue
            parts.append(f"## {section.title}\n{stripped}")
        return "\n\n".join(parts)

    @staticmethod
    def extract_sections(markdown: str) -> Dict[str, str]:
        """Parse markdown produced by ``compose_brief_markdown`` back to
        a ``{section_id: content}`` dict.

        Round-trip contract — pinned by the round-trip test:

            extract_sections(compose_brief_markdown(s)) == \\
                {k: v.strip() for k, v in s.items() if v and v.strip()}

        i.e. the round trip is lossy ONLY for empty sections (which
        ``compose_brief_markdown`` drops by design) and for inner
        leading / trailing whitespace inside a section body (which
        ``compose_brief_markdown`` strips).

        Robustness:

        * Headings whose title doesn't match any registered section are
          silently dropped — same posture as ``compose_brief_markdown``
          for unknown section ids. Keeps the lazy-backfill (issue 016)
          tolerant of legacy free-form briefs that contain unrelated
          ``##``-level headings.
        * Any text BEFORE the first ``## `` heading is dropped. Legacy
          briefs may have a free-form preamble; the registry-driven
          schema has no slot for it.
        * Title matching is case- and punctuation-insensitive (see
          ``brief_section_registry.title_to_id``) so ``## Scale &
          Fidelity`` and ``## scale and fidelity`` both round-trip to
          ``scale_fidelity``.
        * Duplicate headings: if the same section appears twice (which
          ``compose_brief_markdown`` will never produce, but a
          hand-edited legacy brief might), LAST writer wins. This
          matches the principle-of-least-surprise: the user editing the
          markdown bottom-up sees their latest edit reflected.

        Pure: no I/O, no mutation, no logging.
        """
        if not markdown:
            return {}

        # ``re.split`` with a capturing group yields:
        #   [preamble, title_1, body_1, title_2, body_2, ...]
        # The preamble (everything before the first heading) is dropped.
        chunks = _HEADING_RE.split(markdown)
        if len(chunks) < 3:
            return {}

        result: Dict[str, str] = {}
        # Walk title/body pairs starting at index 1.
        for i in range(1, len(chunks), 2):
            title = chunks[i]
            body = chunks[i + 1] if i + 1 < len(chunks) else ""
            section_id = title_to_id(title)
            if section_id is None:
                continue
            stripped_body = body.strip()
            if not stripped_body:
                continue
            result[section_id] = stripped_body
        return result
