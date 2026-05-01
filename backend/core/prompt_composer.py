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
"""

from typing import Optional


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
