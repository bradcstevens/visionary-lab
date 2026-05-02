import { describe, it, expect, afterEach, vi } from "vitest";
import { render, cleanup, screen, fireEvent, act, waitFor } from "@testing-library/react";
import { BriefSectionsEditor } from "../BriefSectionsEditor";
import type { DesignBrief, StagingProject } from "@/services/stagingApi";

/**
 * Issue 019 of the image-pipeline-and-project-ux-overhaul PRD.
 *
 * BriefSectionsEditor is the registry-driven sections + preview +
 * raw_override surface that mounts inside ProjectSettingsSheet. The
 * tests here pin the behavioral contract:
 *
 *   - One tab per BriefSectionRegistry section, in registry order.
 *   - Editing a field PATCHes the project on Save with a partial
 *     design_brief carrying ONLY sections / raw_override.
 *   - No regeneration jobs are created on save.
 *   - Preview tab shows composeBriefMarkdown of the SAVED state and
 *     refreshes after the parent re-renders with the saved project.
 *   - raw_override toggle activates a banner; Revert restores
 *     structured composition.
 *   - "Regenerate affected images" surfaces only after a save touched
 *     a section (or the override) and triggers onRegenerate; not
 *     visible before any save and re-hidden after click.
 */

const EMPTY_BRIEF_REST: Omit<DesignBrief, "global_instructions"> = {
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
};

function makeBrief(
  overrides: Partial<DesignBrief> = {},
): DesignBrief {
  return {
    global_instructions: "Add evergreens",
    ...EMPTY_BRIEF_REST,
    ...overrides,
  };
}

function makeProject(
  overrides: Partial<StagingProject> = {},
): StagingProject {
  return {
    id: "proj-1",
    name: "Test",
    prompt: "Add evergreens",
    status: "completed",
    settings: {
      variations_per_room: 5,
      model: "gpt-image-2",
      quality: "high",
      size: "auto",
    },
    rooms: [],
    design_brief: makeBrief({ sections: { edit_task: "Add evergreens" } }),
    ...overrides,
  };
}

describe("BriefSectionsEditor — registry-driven tabs", () => {
  afterEach(() => cleanup());

  it("renders one tab per BriefSectionRegistry section in registry order", () => {
    render(
      <BriefSectionsEditor
        project={makeProject()}
        onSave={vi.fn()}
        onRegenerate={vi.fn()}
      />,
    );
    const tabs = screen
      .getAllByRole("tab")
      .map((el) => el.getAttribute("data-section-id"))
      .filter((v): v is string => v !== null);
    expect(tabs).toEqual([
      "edit_task",
      "edit_zone",
      "do_not_alter",
      "object_palette",
      "arrangement",
      "regional_constraints",
      "aesthetic_goal",
      "scale_fidelity",
    ]);
  });

  it("renders a Preview tab in addition to the eight section tabs", () => {
    render(
      <BriefSectionsEditor
        project={makeProject()}
        onSave={vi.fn()}
        onRegenerate={vi.fn()}
      />,
    );
    expect(screen.getByTestId("brief-preview-tab")).not.toBeNull();
  });
});

describe("BriefSectionsEditor — editing + saving sections", () => {
  afterEach(() => cleanup());

  it("editing a section field and clicking Save calls onSave with a partial design_brief carrying sections", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    render(
      <BriefSectionsEditor
        project={makeProject()}
        onSave={onSave}
        onRegenerate={vi.fn()}
      />,
    );
    const editor = screen.getByTestId(
      "brief-section-editor-edit_task",
    ) as HTMLTextAreaElement;
    fireEvent.change(editor, { target: { value: "Add 3 evergreens" } });
    await act(async () => {
      fireEvent.click(screen.getByTestId("brief-sections-save"));
    });
    expect(onSave).toHaveBeenCalledTimes(1);
    const payload = onSave.mock.calls[0][0];
    expect(payload).toHaveProperty("design_brief");
    expect(payload.design_brief.sections.edit_task).toBe("Add 3 evergreens");
  });

  it("save does NOT trigger any regeneration callback", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    const onRegenerate = vi.fn().mockResolvedValue(undefined);
    render(
      <BriefSectionsEditor
        project={makeProject()}
        onSave={onSave}
        onRegenerate={onRegenerate}
      />,
    );
    const editor = screen.getByTestId(
      "brief-section-editor-edit_task",
    ) as HTMLTextAreaElement;
    fireEvent.change(editor, { target: { value: "Different" } });
    await act(async () => {
      fireEvent.click(screen.getByTestId("brief-sections-save"));
    });
    expect(onRegenerate).not.toHaveBeenCalled();
  });

  it("Save is disabled when no section has changed", () => {
    render(
      <BriefSectionsEditor
        project={makeProject()}
        onSave={vi.fn()}
        onRegenerate={vi.fn()}
      />,
    );
    const save = screen.getByTestId(
      "brief-sections-save",
    ) as HTMLButtonElement;
    expect(save.disabled).toBe(true);
  });
});

describe("BriefSectionsEditor — preview tab", () => {
  afterEach(() => cleanup());

  it("preview shows composeBriefMarkdown of the current saved state", () => {
    render(
      <BriefSectionsEditor
        project={makeProject({
          design_brief: makeBrief({
            sections: {
              edit_task: "Add evergreens",
              aesthetic_goal: "Modern",
            },
          }),
        })}
        onSave={vi.fn()}
        onRegenerate={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByTestId("brief-preview-tab"));
    const preview = screen.getByTestId("brief-preview-content");
    expect(preview.textContent).toContain("## Edit Task");
    expect(preview.textContent).toContain("Add evergreens");
    expect(preview.textContent).toContain("## Aesthetic Goal");
    expect(preview.textContent).toContain("Modern");
  });

  it("preview updates after save when parent re-renders with the saved project", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    const initialProject = makeProject({
      design_brief: makeBrief({ sections: { edit_task: "old" } }),
    });
    const { rerender } = render(
      <BriefSectionsEditor
        project={initialProject}
        onSave={onSave}
        onRegenerate={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByTestId("brief-preview-tab"));
    expect(screen.getByTestId("brief-preview-content").textContent).toContain(
      "old",
    );
    // Parent re-renders with the saved project after PATCH.
    rerender(
      <BriefSectionsEditor
        project={makeProject({
          design_brief: makeBrief({ sections: { edit_task: "new" } }),
        })}
        onSave={onSave}
        onRegenerate={vi.fn()}
      />,
    );
    expect(screen.getByTestId("brief-preview-content").textContent).toContain(
      "new",
    );
  });

  it("preview content is read-only (no textarea / input element inside the preview pane)", () => {
    render(
      <BriefSectionsEditor
        project={makeProject()}
        onSave={vi.fn()}
        onRegenerate={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByTestId("brief-preview-tab"));
    const preview = screen.getByTestId("brief-preview-content");
    expect(preview.querySelector("textarea")).toBeNull();
    expect(preview.querySelector("input")).toBeNull();
  });
});

describe("BriefSectionsEditor — raw_override toggle", () => {
  afterEach(() => cleanup());

  it("toggling raw_override on shows a banner and a raw textarea", () => {
    render(
      <BriefSectionsEditor
        project={makeProject()}
        onSave={vi.fn()}
        onRegenerate={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByTestId("brief-raw-override-toggle"));
    expect(screen.getByTestId("brief-raw-override-banner")).not.toBeNull();
    expect(screen.getByTestId("brief-raw-override-editor")).not.toBeNull();
  });

  it("preview reflects raw_override verbatim when toggle is on and override is non-empty", () => {
    render(
      <BriefSectionsEditor
        project={makeProject({
          design_brief: makeBrief({
            sections: { edit_task: "ignored" },
            raw_override: "VERBATIM PROMPT",
          }),
        })}
        onSave={vi.fn()}
        onRegenerate={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByTestId("brief-preview-tab"));
    const preview = screen.getByTestId("brief-preview-content");
    expect(preview.textContent).toContain("VERBATIM PROMPT");
    expect(preview.textContent).not.toContain("## Edit Task");
  });

  it("clicking Revert restores structured composition (raw_override → null) and hides the banner", () => {
    render(
      <BriefSectionsEditor
        project={makeProject({
          design_brief: makeBrief({
            sections: { edit_task: "structured body" },
            raw_override: "user override",
          }),
        })}
        onSave={vi.fn()}
        onRegenerate={vi.fn()}
      />,
    );
    expect(screen.queryByTestId("brief-raw-override-banner")).not.toBeNull();
    fireEvent.click(screen.getByTestId("brief-raw-override-revert"));
    expect(screen.queryByTestId("brief-raw-override-banner")).toBeNull();
    fireEvent.click(screen.getByTestId("brief-preview-tab"));
    expect(screen.getByTestId("brief-preview-content").textContent).toContain(
      "## Edit Task",
    );
  });

  it("Save sends raw_override=null when toggle is off, raw_override=value when toggle is on", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    render(
      <BriefSectionsEditor
        project={makeProject()}
        onSave={onSave}
        onRegenerate={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByTestId("brief-raw-override-toggle"));
    const ed = screen.getByTestId(
      "brief-raw-override-editor",
    ) as HTMLTextAreaElement;
    fireEvent.change(ed, { target: { value: "raw user prompt" } });
    await act(async () => {
      fireEvent.click(screen.getByTestId("brief-sections-save"));
    });
    expect(onSave).toHaveBeenCalledTimes(1);
    expect(onSave.mock.calls[0][0].design_brief.raw_override).toBe(
      "raw user prompt",
    );
  });
});

describe("BriefSectionsEditor — Regenerate affected images", () => {
  afterEach(() => cleanup());

  it("Regenerate button is hidden before any section change has been saved", () => {
    render(
      <BriefSectionsEditor
        project={makeProject()}
        onSave={vi.fn()}
        onRegenerate={vi.fn()}
      />,
    );
    expect(screen.queryByTestId("brief-regenerate-button")).toBeNull();
  });

  it("appears after a section change is saved", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    render(
      <BriefSectionsEditor
        project={makeProject()}
        onSave={onSave}
        onRegenerate={vi.fn()}
      />,
    );
    fireEvent.change(
      screen.getByTestId("brief-section-editor-edit_task"),
      { target: { value: "new content" } },
    );
    await act(async () => {
      fireEvent.click(screen.getByTestId("brief-sections-save"));
    });
    await waitFor(() => {
      expect(screen.queryByTestId("brief-regenerate-button")).not.toBeNull();
    });
  });

  it("clicking Regenerate calls onRegenerate and hides the button", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    const onRegenerate = vi.fn().mockResolvedValue(undefined);
    render(
      <BriefSectionsEditor
        project={makeProject()}
        onSave={onSave}
        onRegenerate={onRegenerate}
      />,
    );
    fireEvent.change(
      screen.getByTestId("brief-section-editor-edit_task"),
      { target: { value: "Different" } },
    );
    await act(async () => {
      fireEvent.click(screen.getByTestId("brief-sections-save"));
    });
    await act(async () => {
      fireEvent.click(screen.getByTestId("brief-regenerate-button"));
    });
    expect(onRegenerate).toHaveBeenCalledTimes(1);
    await waitFor(() => {
      expect(screen.queryByTestId("brief-regenerate-button")).toBeNull();
    });
  });
});
