import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import { ProjectRoomsManager } from "../ProjectRoomsManager";
import type { StagingProject } from "@/services/stagingApi";
import * as stagingApi from "@/services/stagingApi";
import * as sonner from "sonner";

/**
 * Vitest unit coverage for `ProjectRoomsManager` — issues 004 + 005
 * of the project-settings-completeness PRD.
 *
 * The component owns the inline rename + inline delete-with-confirm
 * UX on the Project Settings sheet. The Playwright integration spec
 * at `frontend/tests/e2e/project-settings-sheet.spec.ts` covers the
 * wiring across the sheet + page + backend; these tests exercise the
 * component's pure observable behavior at the React tree level:
 *
 *   1. Rendering behavior (rooms list, edit + delete affordances,
 *      fallback when empty).
 *   2. Edit-mode lifecycle (pencil reveals input pre-filled with
 *      current label; Cancel discards no API call; trimmed/no-op /
 *      empty / unchanged labels disable Save).
 *   3. Save lifecycle (server-confirmed: `updateRoom` then
 *      `onProjectUpdate`; rapid double-click does not double-fire).
 *   4. Rename error path (toast surfaces; row reverts to view mode).
 *   5. `disabled` prop forwarding (issues 004 + 005).
 *   6. Delete-confirm lifecycle (issue 005): trash reveals the
 *      inline confirm row with [Yes, delete] / [Cancel]; Cancel
 *      collapses without API call; Yes calls `removeRoom` then
 *      `onProjectUpdate`; rapid double-click guard; failure leaves
 *      the confirm row visible with INLINE error (NOT a toast that
 *      auto-dismisses); retry-after-failure clears the error;
 *      mutual exclusion with rename across rows.
 */

const STORAGE_HOST = "https://acct.blob.core.windows.net/images";

function makeProject(overrides: Partial<StagingProject> = {}): StagingProject {
  return {
    id: "proj-rooms",
    name: "Rooms Project",
    prompt: "modern minimalist",
    status: "completed",
    settings: {
      variations_per_room: 5,
      model: "gpt-image-2",
      quality: "high",
      size: "auto",
    },
    rooms: [
      {
        id: "room-A",
        label: "Living Room",
        original_image_url: `${STORAGE_HOST}/staging/proj/originals/a.png`,
        original_thumbnail_url: `${STORAGE_HOST}/staging/proj/originals/a-thumb.png`,
        status: "completed",
        prompt_addendum: null,
        variations: [],
      },
      {
        id: "room-B",
        label: "Kitchen",
        original_image_url: `${STORAGE_HOST}/staging/proj/originals/b.png`,
        original_thumbnail_url: `${STORAGE_HOST}/staging/proj/originals/b-thumb.png`,
        status: "completed",
        prompt_addendum: "B's existing addendum",
        variations: [],
      },
    ],
    created_at: "2026-05-01T00:00:00Z",
    updated_at: "2026-05-01T00:00:00Z",
    ...overrides,
  } as StagingProject;
}

let updateRoomSpy: ReturnType<typeof vi.spyOn>;
let removeRoomSpy: ReturnType<typeof vi.spyOn>;
let toastErrorSpy: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  updateRoomSpy = vi.spyOn(stagingApi, "updateRoom");
  removeRoomSpy = vi.spyOn(stagingApi, "removeRoom");
  toastErrorSpy = vi.spyOn(sonner.toast, "error").mockImplementation(() => "");
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("ProjectRoomsManager — rendering", () => {
  it("renders a row per room with the current label", () => {
    const project = makeProject();
    render(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={false} />,
    );
    expect(
      screen.getByTestId("project-rooms-manager-label-room-A").textContent,
    ).toBe("Living Room");
    expect(
      screen.getByTestId("project-rooms-manager-label-room-B").textContent,
    ).toBe("Kitchen");
    // Edit affordance per row.
    expect(screen.getByTestId("project-rooms-manager-edit-room-A")).toBeTruthy();
    expect(screen.getByTestId("project-rooms-manager-edit-room-B")).toBeTruthy();
  });

  it("renders an empty-state hint when project has no rooms", () => {
    const project = makeProject({ rooms: [] });
    render(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={false} />,
    );
    expect(screen.getByTestId("project-rooms-manager-empty")).toBeTruthy();
  });
});

describe("ProjectRoomsManager — edit-mode lifecycle", () => {
  it("clicking pencil reveals input prefilled with current label and hides the static label", () => {
    const project = makeProject();
    render(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={false} />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-edit-room-A"));

    const input = screen.getByTestId("project-rooms-manager-input-room-A") as HTMLInputElement;
    expect(input.value).toBe("Living Room");
    // Static label gone for the editing row.
    expect(screen.queryByTestId("project-rooms-manager-label-room-A")).toBeNull();
    // The other row is still in view mode (only one row is editable
    // at a time — keeps the implementation simple and the UX clear).
    expect(screen.getByTestId("project-rooms-manager-label-room-B")).toBeTruthy();
  });

  it("Cancel discards the local edit, returns to view mode, and never calls updateRoom", () => {
    const onProjectUpdate = vi.fn();
    const project = makeProject();
    render(
      <ProjectRoomsManager
        project={project}
        onProjectUpdate={onProjectUpdate}
        disabled={false}
      />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-edit-room-A"));
    const input = screen.getByTestId("project-rooms-manager-input-room-A") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "Den" } });
    fireEvent.click(screen.getByTestId("project-rooms-manager-cancel-room-A"));

    expect(updateRoomSpy).not.toHaveBeenCalled();
    expect(onProjectUpdate).not.toHaveBeenCalled();
    // Back in view mode with the original (un-renamed) label.
    expect(
      screen.getByTestId("project-rooms-manager-label-room-A").textContent,
    ).toBe("Living Room");
  });

  it("Save is disabled when the trimmed draft is empty", () => {
    const project = makeProject();
    render(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={false} />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-edit-room-A"));
    const input = screen.getByTestId("project-rooms-manager-input-room-A") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "   " } });

    const saveBtn = screen.getByTestId("project-rooms-manager-save-room-A") as HTMLButtonElement;
    expect(saveBtn.disabled).toBe(true);
  });

  it("Save is disabled when trimmed draft equals the current label (no-op rename)", () => {
    // Rubber-duck finding: a trimmed `" Living Room "` edit when the
    // persisted label is `"Living Room"` should be a no-op rather
    // than a wasted round-trip.
    const project = makeProject();
    render(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={false} />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-edit-room-A"));
    const input = screen.getByTestId("project-rooms-manager-input-room-A") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "  Living Room  " } });

    const saveBtn = screen.getByTestId("project-rooms-manager-save-room-A") as HTMLButtonElement;
    expect(saveBtn.disabled).toBe(true);
  });
});

describe("ProjectRoomsManager — save lifecycle", () => {
  it("Save calls updateRoom with the trimmed label, then onProjectUpdate with the response", async () => {
    const project = makeProject();
    const updatedFromServer = makeProject({
      rooms: [
        { ...project.rooms[0], label: "Master Bedroom" },
        project.rooms[1],
      ],
    });
    updateRoomSpy.mockResolvedValueOnce(updatedFromServer);
    const onProjectUpdate = vi.fn();

    render(
      <ProjectRoomsManager
        project={project}
        onProjectUpdate={onProjectUpdate}
        disabled={false}
      />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-edit-room-A"));
    fireEvent.change(screen.getByTestId("project-rooms-manager-input-room-A"), {
      target: { value: "  Master Bedroom  " },
    });
    fireEvent.click(screen.getByTestId("project-rooms-manager-save-room-A"));

    await waitFor(() => {
      expect(updateRoomSpy).toHaveBeenCalledWith("proj-rooms", "room-A", {
        label: "Master Bedroom",
      });
    });
    await waitFor(() => {
      expect(onProjectUpdate).toHaveBeenCalledWith(updatedFromServer);
    });
  });

  it("rapid double-click on Save does not fire updateRoom twice (per-row pending guard)", async () => {
    // Rubber-duck blind spot: without per-row pending state, a fast
    // user could trigger two PATCH requests for the same rename.
    const project = makeProject();
    let resolveFirstCall: (value: StagingProject) => void = () => {};
    const firstCallPromise = new Promise<StagingProject>((resolve) => {
      resolveFirstCall = resolve;
    });
    updateRoomSpy.mockReturnValueOnce(firstCallPromise);

    render(
      <ProjectRoomsManager
        project={project}
        onProjectUpdate={() => {}}
        disabled={false}
      />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-edit-room-A"));
    fireEvent.change(screen.getByTestId("project-rooms-manager-input-room-A"), {
      target: { value: "Den" },
    });
    const saveBtn = screen.getByTestId("project-rooms-manager-save-room-A");
    // Fire two clicks back-to-back BEFORE the first PATCH resolves.
    fireEvent.click(saveBtn);
    fireEvent.click(saveBtn);
    fireEvent.click(saveBtn);

    // Only one updateRoom call landed despite three clicks.
    expect(updateRoomSpy).toHaveBeenCalledTimes(1);

    // Drain the pending promise so React doesn't warn about an
    // unresolved act() boundary.
    resolveFirstCall(makeProject({ rooms: [{ ...project.rooms[0], label: "Den" }, project.rooms[1]] }));
    await waitFor(() => {
      // Still only one call after the resolve flushes.
      expect(updateRoomSpy).toHaveBeenCalledTimes(1);
    });
  });
});

describe("ProjectRoomsManager — error path", () => {
  it("on updateRoom failure, toasts the error and reverts to view mode with the original label", async () => {
    const project = makeProject();
    const failure = new Error("Network blew up");
    updateRoomSpy.mockRejectedValueOnce(failure);
    const onProjectUpdate = vi.fn();

    render(
      <ProjectRoomsManager
        project={project}
        onProjectUpdate={onProjectUpdate}
        disabled={false}
      />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-edit-room-A"));
    fireEvent.change(screen.getByTestId("project-rooms-manager-input-room-A"), {
      target: { value: "Den" },
    });
    fireEvent.click(screen.getByTestId("project-rooms-manager-save-room-A"));

    await waitFor(() => {
      expect(toastErrorSpy).toHaveBeenCalledWith("Network blew up");
    });
    // onProjectUpdate is never called because the server didn't ack.
    expect(onProjectUpdate).not.toHaveBeenCalled();
    // Row reverts to view mode with the ORIGINAL label — `project`
    // hasn't been replaced (server didn't update), so the static
    // label re-renders with "Living Room".
    await waitFor(() => {
      expect(
        screen.getByTestId("project-rooms-manager-label-room-A").textContent,
      ).toBe("Living Room");
    });
  });
});

describe("ProjectRoomsManager — disabled prop", () => {
  it("when disabled=true, every per-row pencil button is disabled", () => {
    const project = makeProject();
    render(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={true} />,
    );

    expect(
      (screen.getByTestId("project-rooms-manager-edit-room-A") as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(
      (screen.getByTestId("project-rooms-manager-edit-room-B") as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it("when disabled=true and the user is somehow in edit mode, Save and the input are disabled", () => {
    // Defense-in-depth: the edit affordance is disabled when
    // `disabled=true`, but if the prop flips mid-edit (e.g., a
    // generation kicks off after the user enters edit mode), the
    // input + save controls must respect the new disabled value.
    const project = makeProject();
    const { rerender } = render(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={false} />,
    );
    fireEvent.click(screen.getByTestId("project-rooms-manager-edit-room-A"));
    fireEvent.change(screen.getByTestId("project-rooms-manager-input-room-A"), {
      target: { value: "Den" },
    });

    // Simulate the parent flipping disabled to true mid-edit.
    rerender(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={true} />,
    );

    expect(
      (screen.getByTestId("project-rooms-manager-input-room-A") as HTMLInputElement).disabled,
    ).toBe(true);
    expect(
      (screen.getByTestId("project-rooms-manager-save-room-A") as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(
      (screen.getByTestId("project-rooms-manager-cancel-room-A") as HTMLButtonElement).disabled,
    ).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Issue 005 — inline delete-with-confirm
// ---------------------------------------------------------------------------

describe("ProjectRoomsManager — issue 005 — delete affordance rendering", () => {
  it("renders a Delete (trash) button for every room in view mode", () => {
    const project = makeProject();
    render(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={false} />,
    );
    expect(screen.getByTestId("project-rooms-manager-delete-room-A")).toBeTruthy();
    expect(screen.getByTestId("project-rooms-manager-delete-room-B")).toBeTruthy();
  });

  it("clicking the trash button reveals an inline confirm row in the SAME component (no portal/modal)", () => {
    const project = makeProject();
    render(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={false} />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-delete-room-A"));

    // The confirm row is an in-DOM descendant of the manager — not a
    // portal-mounted dialog elsewhere in the tree. Asserting the
    // confirm element lives INSIDE the manager's root pins the
    // PRD's "no modal popover from a different mount point — keep
    // the deep-module rule" requirement.
    const manager = screen.getByTestId("project-rooms-manager");
    const confirm = screen.getByTestId("project-rooms-manager-confirm-room-A");
    expect(manager.contains(confirm)).toBe(true);
    // [Yes, delete] + [Cancel] both rendered.
    expect(screen.getByTestId("project-rooms-manager-confirm-yes-room-A")).toBeTruthy();
    expect(screen.getByTestId("project-rooms-manager-confirm-cancel-room-A")).toBeTruthy();
    // The OTHER row stays in plain view mode.
    expect(screen.queryByTestId("project-rooms-manager-confirm-room-B")).toBeNull();
  });
});

describe("ProjectRoomsManager — issue 005 — confirm + cancel lifecycle", () => {
  it("clicking Cancel collapses the confirm row, calls neither removeRoom nor onProjectUpdate", () => {
    const onProjectUpdate = vi.fn();
    const project = makeProject();
    render(
      <ProjectRoomsManager
        project={project}
        onProjectUpdate={onProjectUpdate}
        disabled={false}
      />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-delete-room-A"));
    fireEvent.click(screen.getByTestId("project-rooms-manager-confirm-cancel-room-A"));

    // Confirm row is gone.
    expect(screen.queryByTestId("project-rooms-manager-confirm-room-A")).toBeNull();
    // Back to plain view mode for the row.
    expect(
      screen.getByTestId("project-rooms-manager-label-room-A").textContent,
    ).toBe("Living Room");
    // No API or callback invocation.
    expect(removeRoomSpy).not.toHaveBeenCalled();
    expect(onProjectUpdate).not.toHaveBeenCalled();
  });

  it("clicking 'Yes, delete' calls removeRoom then onProjectUpdate with the response", async () => {
    const project = makeProject();
    const updatedFromServer = makeProject({
      rooms: [project.rooms[1]],
    });
    removeRoomSpy.mockResolvedValueOnce(updatedFromServer);
    const onProjectUpdate = vi.fn();

    render(
      <ProjectRoomsManager
        project={project}
        onProjectUpdate={onProjectUpdate}
        disabled={false}
      />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-delete-room-A"));
    fireEvent.click(screen.getByTestId("project-rooms-manager-confirm-yes-room-A"));

    await waitFor(() => {
      expect(removeRoomSpy).toHaveBeenCalledWith("proj-rooms", "room-A");
    });
    await waitFor(() => {
      expect(onProjectUpdate).toHaveBeenCalledWith(updatedFromServer);
    });
  });

  it("rapid double-click on 'Yes, delete' fires removeRoom exactly once (per-row pending guard)", async () => {
    // Without per-row pending state, a fast user could trigger two
    // DELETE requests for the same room. The second one would 404
    // because the room is already gone — confusing UX.
    const project = makeProject();
    let resolveFirstCall: (value: StagingProject) => void = () => {};
    const firstCallPromise = new Promise<StagingProject>((resolve) => {
      resolveFirstCall = resolve;
    });
    removeRoomSpy.mockReturnValueOnce(firstCallPromise);

    render(
      <ProjectRoomsManager
        project={project}
        onProjectUpdate={() => {}}
        disabled={false}
      />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-delete-room-A"));
    const yesBtn = screen.getByTestId("project-rooms-manager-confirm-yes-room-A");
    // Three rapid clicks BEFORE the first DELETE resolves.
    fireEvent.click(yesBtn);
    fireEvent.click(yesBtn);
    fireEvent.click(yesBtn);

    // Only one removeRoom call landed.
    expect(removeRoomSpy).toHaveBeenCalledTimes(1);

    // Drain the pending promise so React doesn't warn about an
    // unresolved act() boundary.
    resolveFirstCall(makeProject({ rooms: [project.rooms[1]] }));
    await waitFor(() => {
      expect(removeRoomSpy).toHaveBeenCalledTimes(1);
    });
  });
});

describe("ProjectRoomsManager — issue 005 — error path (inline error, NOT a toast)", () => {
  it("on removeRoom failure, the confirm row stays visible with INLINE error and the row is preserved", async () => {
    const project = makeProject();
    const failure = new Error("Failed to remove room: 500 something broke");
    removeRoomSpy.mockRejectedValueOnce(failure);
    const onProjectUpdate = vi.fn();

    render(
      <ProjectRoomsManager
        project={project}
        onProjectUpdate={onProjectUpdate}
        disabled={false}
      />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-delete-room-A"));
    fireEvent.click(screen.getByTestId("project-rooms-manager-confirm-yes-room-A"));

    // Inline error appears (PRD: "the confirm row stays visible with
    // an inline error and the room row is preserved" — NOT a toast
    // that auto-dismisses).
    await waitFor(() => {
      expect(
        screen.getByTestId("project-rooms-manager-confirm-error-room-A").textContent,
      ).toContain("something broke");
    });
    // Confirm row STILL visible (not collapsed).
    expect(screen.getByTestId("project-rooms-manager-confirm-room-A")).toBeTruthy();
    // The Yes / Cancel buttons are still there for retry / abort.
    expect(screen.getByTestId("project-rooms-manager-confirm-yes-room-A")).toBeTruthy();
    expect(screen.getByTestId("project-rooms-manager-confirm-cancel-room-A")).toBeTruthy();
    // Row is preserved on the project state — onProjectUpdate NOT
    // called because the server didn't ack.
    expect(onProjectUpdate).not.toHaveBeenCalled();
    // Issue 005 deliberately uses INLINE error rather than a toast
    // so the failure stays visible until the user explicitly
    // retries or cancels (matches PRD's "the confirm row stays
    // visible with an inline error" rule).
    expect(toastErrorSpy).not.toHaveBeenCalled();
  });

  it("retry-after-failure: failed delete leaves error visible; clicking Yes again succeeds and clears the error", async () => {
    // Sticky-error-state regression guard (rubber-duck suggestion):
    // if a future refactor accidentally cleared the deleteError on
    // the next click but failed to flip the deleteConfirmRoomId,
    // this test would catch it.
    const project = makeProject();
    const updatedFromServer = makeProject({ rooms: [project.rooms[1]] });
    removeRoomSpy
      .mockRejectedValueOnce(new Error("Failed to remove room: 500 transient"))
      .mockResolvedValueOnce(updatedFromServer);
    const onProjectUpdate = vi.fn();

    render(
      <ProjectRoomsManager
        project={project}
        onProjectUpdate={onProjectUpdate}
        disabled={false}
      />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-delete-room-A"));
    fireEvent.click(screen.getByTestId("project-rooms-manager-confirm-yes-room-A"));

    await waitFor(() => {
      expect(
        screen.getByTestId("project-rooms-manager-confirm-error-room-A"),
      ).toBeTruthy();
    });

    // Click Yes again — second attempt succeeds.
    fireEvent.click(screen.getByTestId("project-rooms-manager-confirm-yes-room-A"));

    await waitFor(() => {
      expect(removeRoomSpy).toHaveBeenCalledTimes(2);
    });
    await waitFor(() => {
      expect(onProjectUpdate).toHaveBeenCalledWith(updatedFromServer);
    });
    // Error is cleared (the confirm row collapsed entirely on
    // success — no inline-error element survives).
    expect(
      screen.queryByTestId("project-rooms-manager-confirm-error-room-A"),
    ).toBeNull();
  });
});

describe("ProjectRoomsManager — issue 005 — disabled prop forwarding", () => {
  it("when disabled=true, every per-row trash button is disabled", () => {
    const project = makeProject();
    render(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={true} />,
    );

    expect(
      (screen.getByTestId("project-rooms-manager-delete-room-A") as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(
      (screen.getByTestId("project-rooms-manager-delete-room-B") as HTMLButtonElement).disabled,
    ).toBe(true);
  });
});

describe("ProjectRoomsManager — issue 005 — mutual exclusion across rows", () => {
  it("while one row is in delete-confirm, the OTHER row's pencil and trash are disabled", () => {
    // Symmetric mutual-exclusion (rubber-duck non-blocking finding):
    // only one row can be in any active mode at a time. Without this,
    // a user could open two delete-confirm prompts at once and lose
    // track of which one is active. Existing rename behavior already
    // disables the OTHER row's pencil while ANY row is saving; this
    // test pins the same rule for the delete-confirm phase.
    const project = makeProject();
    render(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={false} />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-delete-room-A"));

    // Row B is still in view mode but its action buttons are
    // disabled because Row A is in delete-confirm.
    expect(
      (screen.getByTestId("project-rooms-manager-edit-room-B") as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(
      (screen.getByTestId("project-rooms-manager-delete-room-B") as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it("while one row is in rename-edit, the OTHER row's trash is disabled", () => {
    // Symmetric: the existing rename-edit mode should also block
    // delete on other rows. Without this, a user mid-rename on row
    // A could click delete on row B, lose the rename draft on save,
    // and end up with a confusing transition.
    const project = makeProject();
    render(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={false} />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-edit-room-A"));

    expect(
      (screen.getByTestId("project-rooms-manager-delete-room-B") as HTMLButtonElement).disabled,
    ).toBe(true);
  });
});
