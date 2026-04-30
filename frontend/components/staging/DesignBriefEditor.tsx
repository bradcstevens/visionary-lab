"use client"

import { useState } from "react";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { X } from "lucide-react";
import { ObjectPaletteTable } from "./ObjectPaletteTable";
import { BriefEditorTabs, type BriefEditorTabsImage } from "./BriefEditorTabs";
import { PerImageObjectTable } from "./PerImageObjectTable";
import type {
  DesignBrief,
  ImageObjectOverride,
  ObjectEntry,
} from "@/services/stagingApi";

interface DesignBriefEditorProps {
  brief: DesignBrief;
  onChange: (brief: DesignBrief) => void;
  // Image metadata for the per-image tabs. Replaces the previously-reserved
  // `imageLabels: Record<string, string>` prop now that issue 003 of the
  // per-image-object-quantities PRD actually consumes the data flow.
  images: BriefEditorTabsImage[];
}

// Remove any `per_image_objects[*]` entries whose `object_id` is no longer
// present in the new palette, AND drop any room keys that become empty.
// Critique catch: keeping `{ roomId: [] }` would accumulate noisy empty
// keys; the resolver treats absent and empty-list identically, but the map
// should stay sparse for clean serialization.
function prunePerImageObjects(
  perImage: Record<string, ImageObjectOverride[]>,
  validIds: ReadonlySet<string>,
): Record<string, ImageObjectOverride[]> {
  const result: Record<string, ImageObjectOverride[]> = {};
  for (const [roomId, list] of Object.entries(perImage)) {
    const kept = list.filter((o) => validIds.has(o.object_id));
    if (kept.length > 0) {
      result[roomId] = kept;
    }
  }
  return result;
}

export function DesignBriefEditor({ brief, onChange, images }: DesignBriefEditorProps) {
  const [newPreserve, setNewPreserve] = useState("");

  const updateField = <K extends keyof DesignBrief>(field: K, value: DesignBrief[K]) => {
    onChange({ ...brief, [field]: value });
  };

  // Palette mutation must also prune `per_image_objects` so deleted
  // ObjectEntry ids don't leak orphaned overrides into prompts.
  const handlePaletteChange = (objects: ObjectEntry[]) => {
    const validIds = new Set(objects.map((o) => o.id));
    const prevValidIds = new Set(brief.object_palette.map((o) => o.id));
    const wasDeleted = [...prevValidIds].some((id) => !validIds.has(id));
    const currentPerImage = brief.per_image_objects ?? {};
    const nextPerImage = wasDeleted
      ? prunePerImageObjects(currentPerImage, validIds)
      : currentPerImage;
    onChange({
      ...brief,
      object_palette: objects,
      per_image_objects: nextPerImage,
    });
  };

  const handlePerImageOverridesChange = (
    roomId: string,
    overrides: ImageObjectOverride[],
  ) => {
    const next: Record<string, ImageObjectOverride[]> = { ...(brief.per_image_objects ?? {}) };
    if (overrides.length === 0) {
      delete next[roomId];
    } else {
      next[roomId] = overrides;
    }
    updateField("per_image_objects", next);
  };

  const handlePerImageNoteChange = (roomId: string, raw: string) => {
    const trimmed = raw.trim();
    const next: Record<string, string> = { ...(brief.per_image_notes ?? {}) };
    if (trimmed === "") {
      delete next[roomId];
    } else {
      next[roomId] = raw;
    }
    updateField("per_image_notes", next);
  };

  const addPreserveElement = () => {
    if (!newPreserve.trim()) return;
    updateField("preserve_elements", [...brief.preserve_elements, newPreserve.trim()]);
    setNewPreserve("");
  };

  const removePreserveElement = (index: number) => {
    updateField("preserve_elements", brief.preserve_elements.filter((_, i) => i !== index));
  };

  const defaultTabContent = (
    <div className="space-y-6 max-w-4xl">
      <div className="space-y-2">
        <Label className="text-sm font-semibold">Global Instructions</Label>
        <Textarea
          value={brief.global_instructions}
          onChange={(e) => updateField("global_instructions", e.target.value)}
          rows={3}
          className="text-sm resize-none"
          placeholder="Overall styling direction..."
        />
      </div>

      <div className="space-y-2">
        <Label className="text-sm font-semibold">Object Palette</Label>
        <ObjectPaletteTable
          objects={brief.object_palette}
          onChange={handlePaletteChange}
        />
      </div>

      <div className="space-y-2">
        <Label className="text-sm font-semibold">Placement Guide</Label>
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Back Row (tall)</Label>
            <Input value={brief.placement_guide.back_row} onChange={(e) => updateField("placement_guide", { ...brief.placement_guide, back_row: e.target.value })} className="text-sm h-8" />
          </div>
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Middle Row (mid-height)</Label>
            <Input value={brief.placement_guide.middle_row ?? ""} onChange={(e) => updateField("placement_guide", { ...brief.placement_guide, middle_row: e.target.value || undefined })} className="text-sm h-8" />
          </div>
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Front Row (low)</Label>
            <Input value={brief.placement_guide.front_row ?? ""} onChange={(e) => updateField("placement_guide", { ...brief.placement_guide, front_row: e.target.value || undefined })} className="text-sm h-8" />
          </div>
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Accent Areas</Label>
            <Input value={brief.placement_guide.accent_areas ?? ""} onChange={(e) => updateField("placement_guide", { ...brief.placement_guide, accent_areas: e.target.value || undefined })} className="text-sm h-8" />
          </div>
        </div>
      </div>

      <div className="space-y-2">
        <Label className="text-sm font-semibold">Preserve (don&apos;t change)</Label>
        <div className="flex flex-wrap gap-1.5">
          {brief.preserve_elements.map((el, idx) => (
            <Badge key={idx} variant="secondary" className="text-xs gap-1">
              {el}
              <button onClick={() => removePreserveElement(idx)}>
                <X className="h-2.5 w-2.5" />
              </button>
            </Badge>
          ))}
        </div>
        <div className="flex gap-2">
          <Input
            value={newPreserve}
            onChange={(e) => setNewPreserve(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && addPreserveElement()}
            placeholder="e.g. existing patio"
            className="text-sm h-8"
          />
          <Button size="sm" variant="outline" onClick={addPreserveElement} className="h-8">Add</Button>
        </div>
      </div>

      <div className="space-y-2">
        <Label className="text-sm font-semibold">Generation Settings</Label>
        <div className="grid grid-cols-4 gap-3">
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Variations per image</Label>
            <Input type="number" value={brief.settings.variations_per_room} onChange={(e) => updateField("settings", { ...brief.settings, variations_per_room: parseInt(e.target.value) || 5 })} min={1} max={10} className="text-sm h-8" />
          </div>
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Model</Label>
            <Input value={brief.settings.model} disabled className="text-sm h-8" />
          </div>
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Quality</Label>
            <Input value={brief.settings.quality} disabled className="text-sm h-8" />
          </div>
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Size</Label>
            <Input value={brief.settings.size} disabled className="text-sm h-8" />
          </div>
        </div>
      </div>
    </div>
  );

  const renderImageTabContent = (image: BriefEditorTabsImage) => {
    const overrides = brief.per_image_objects?.[image.id] ?? [];
    const note = brief.per_image_notes?.[image.id] ?? "";
    return (
      <div className="space-y-6 max-w-4xl">
        <div className="space-y-2">
          <Label className="text-sm font-semibold">Per-Image Object Overrides — {image.label}</Label>
          <PerImageObjectTable
            palette={brief.object_palette}
            overrides={overrides}
            onChange={(next) => handlePerImageOverridesChange(image.id, next)}
          />
        </div>
        <div className="space-y-2">
          <Label className="text-sm font-semibold">Per-Image Note</Label>
          <Textarea
            value={note}
            onChange={(e) => handlePerImageNoteChange(image.id, e.target.value)}
            rows={2}
            className="text-sm resize-none"
            placeholder="Anything specific to this image..."
            data-testid={`per-image-note-${image.id}`}
          />
        </div>
      </div>
    );
  };

  return (
    <BriefEditorTabs
      images={images}
      defaultTabContent={defaultTabContent}
      renderImageTabContent={renderImageTabContent}
    />
  );
}
