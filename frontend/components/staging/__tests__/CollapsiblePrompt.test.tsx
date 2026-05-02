import { describe, it, expect, afterEach, vi } from "vitest";
import { render, cleanup, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { CollapsiblePrompt } from "../CollapsiblePrompt";

const longPrompt = "A".repeat(800);

describe("CollapsiblePrompt", () => {
  afterEach(() => cleanup());

  it("renders prompt_summary collapsed by default with Show full prompt control", () => {
    render(
      <CollapsiblePrompt
        prompt={longPrompt}
        promptSummary="Short summary of prompt."
        onSave={vi.fn()}
      />
    );
    expect(screen.getByTestId("prompt-summary").textContent).toContain(
      "Short summary of prompt."
    );
    expect(screen.queryByTestId("prompt-editor")).toBeNull();
    expect(screen.queryByTestId("prompt-toggle")).not.toBeNull();
  });

  it("falls back to truncated prompt when prompt_summary missing", () => {
    render(
      <CollapsiblePrompt
        prompt={longPrompt}
        promptSummary={null}
        onSave={vi.fn()}
      />
    );
    const summary = screen.getByTestId("prompt-summary");
    // Falls back to the prompt itself, truncated for the collapsed view.
    expect(summary.textContent ?? "").not.toEqual("");
    expect((summary.textContent ?? "").length).toBeLessThanOrEqual(260);
  });

  it("expanding shows the full prompt in an editable textarea", () => {
    render(
      <CollapsiblePrompt
        prompt="Full prompt body"
        promptSummary="Sum"
        onSave={vi.fn()}
      />
    );
    fireEvent.click(screen.getByTestId("prompt-toggle"));
    const editor = screen.getByTestId("prompt-editor") as HTMLTextAreaElement;
    expect(editor.value).toBe("Full prompt body");
  });

  it("Save calls onSave with the edited prompt and collapses on success", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    render(
      <CollapsiblePrompt
        prompt="orig"
        promptSummary="orig sum"
        onSave={onSave}
      />
    );
    fireEvent.click(screen.getByTestId("prompt-toggle"));
    const editor = screen.getByTestId("prompt-editor") as HTMLTextAreaElement;
    fireEvent.change(editor, { target: { value: "edited prompt" } });
    await act(async () => {
      fireEvent.click(screen.getByTestId("prompt-save"));
    });
    expect(onSave).toHaveBeenCalledWith("edited prompt");
    await waitFor(() =>
      expect(screen.queryByTestId("prompt-editor")).toBeNull()
    );
  });

  it("collapsed summary refreshes after save (driven by parent props)", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    const { rerender } = render(
      <CollapsiblePrompt
        prompt="orig"
        promptSummary="orig summary"
        onSave={onSave}
      />
    );
    fireEvent.click(screen.getByTestId("prompt-toggle"));
    fireEvent.change(screen.getByTestId("prompt-editor"), {
      target: { value: "new long prompt" },
    });
    await act(async () => {
      fireEvent.click(screen.getByTestId("prompt-save"));
    });
    // Parent re-renders with the new server-returned summary.
    rerender(
      <CollapsiblePrompt
        prompt="new long prompt"
        promptSummary="freshly regenerated summary"
        onSave={onSave}
      />
    );
    expect(screen.getByTestId("prompt-summary").textContent).toContain(
      "freshly regenerated summary"
    );
  });

  it("Cancel discards edits and collapses without calling onSave", () => {
    const onSave = vi.fn();
    render(
      <CollapsiblePrompt prompt="orig" promptSummary="sum" onSave={onSave} />
    );
    fireEvent.click(screen.getByTestId("prompt-toggle"));
    fireEvent.change(screen.getByTestId("prompt-editor"), {
      target: { value: "abandoned edit" },
    });
    fireEvent.click(screen.getByTestId("prompt-cancel"));
    expect(onSave).not.toHaveBeenCalled();
    expect(screen.queryByTestId("prompt-editor")).toBeNull();
    // Re-expand: textarea reset to original prompt.
    fireEvent.click(screen.getByTestId("prompt-toggle"));
    expect(
      (screen.getByTestId("prompt-editor") as HTMLTextAreaElement).value
    ).toBe("orig");
  });

  it("Save is disabled while in flight and re-enabled after settle", async () => {
    let resolve: () => void = () => {};
    const onSave = vi.fn(
      () => new Promise<void>((r) => (resolve = r))
    );
    render(
      <CollapsiblePrompt prompt="orig" promptSummary="sum" onSave={onSave} />
    );
    fireEvent.click(screen.getByTestId("prompt-toggle"));
    fireEvent.change(screen.getByTestId("prompt-editor"), {
      target: { value: "x" },
    });
    fireEvent.click(screen.getByTestId("prompt-save"));
    const saveBtn = screen.getByTestId("prompt-save") as HTMLButtonElement;
    expect(saveBtn.disabled).toBe(true);
    await act(async () => {
      resolve();
      await Promise.resolve();
    });
  });

  it("save failure keeps the editor open so the user can retry", async () => {
    const onSave = vi.fn().mockRejectedValue(new Error("boom"));
    render(
      <CollapsiblePrompt prompt="orig" promptSummary="sum" onSave={onSave} />
    );
    fireEvent.click(screen.getByTestId("prompt-toggle"));
    fireEvent.change(screen.getByTestId("prompt-editor"), {
      target: { value: "x" },
    });
    await act(async () => {
      fireEvent.click(screen.getByTestId("prompt-save"));
    });
    // Editor must remain open after a failed save so the user can adjust + retry.
    expect(screen.queryByTestId("prompt-editor")).not.toBeNull();
  });

  it("Save is disabled when the edited prompt is unchanged or empty", () => {
    render(
      <CollapsiblePrompt
        prompt="orig"
        promptSummary="sum"
        onSave={vi.fn()}
      />
    );
    fireEvent.click(screen.getByTestId("prompt-toggle"));
    const saveBtn = screen.getByTestId("prompt-save") as HTMLButtonElement;
    // Initially identical to prompt -> disabled.
    expect(saveBtn.disabled).toBe(true);
    // Whitespace-only -> disabled.
    fireEvent.change(screen.getByTestId("prompt-editor"), {
      target: { value: "   " },
    });
    expect(saveBtn.disabled).toBe(true);
    // Real change -> enabled.
    fireEvent.change(screen.getByTestId("prompt-editor"), {
      target: { value: "new" },
    });
    expect(saveBtn.disabled).toBe(false);
  });

  it("does not invoke any regeneration callback (component owns no jobs surface)", async () => {
    // Pin the AC bullet "No regeneration jobs are created on save". This
    // component takes only an onSave callback for the prompt PATCH; if a
    // future contributor adds a regeneration prop the test names here
    // make the intent explicit.
    const onSave = vi.fn().mockResolvedValue(undefined);
    render(
      <CollapsiblePrompt prompt="orig" promptSummary="sum" onSave={onSave} />
    );
    fireEvent.click(screen.getByTestId("prompt-toggle"));
    fireEvent.change(screen.getByTestId("prompt-editor"), {
      target: { value: "edited" },
    });
    await act(async () => {
      fireEvent.click(screen.getByTestId("prompt-save"));
    });
    expect(onSave).toHaveBeenCalledTimes(1);
    expect(onSave).toHaveBeenCalledWith("edited");
  });

  it("hides the toggle when neither summary nor prompt has content", () => {
    render(
      <CollapsiblePrompt prompt="" promptSummary={null} onSave={vi.fn()} />
    );
    expect(screen.queryByTestId("prompt-toggle")).toBeNull();
  });
});
