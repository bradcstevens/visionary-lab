"use client"

import { useState } from "react";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { X } from "lucide-react";
import { PlantPaletteTable } from "./PlantPaletteTable";
import type { DesignBrief, PlantEntry } from "@/services/stagingApi";

interface DesignBriefEditorProps {
  brief: DesignBrief;
  onChange: (brief: DesignBrief) => void;
  imageLabels: Record<string, string>;
}

export function DesignBriefEditor({ brief, onChange, imageLabels }: DesignBriefEditorProps) {
  // imageLabels reserved for per-image notes rendering
  void imageLabels;
  const [newPreserve, setNewPreserve] = useState("");

  const updateField = <K extends keyof DesignBrief>(field: K, value: DesignBrief[K]) => {
    onChange({ ...brief, [field]: value });
  };

  const addPreserveElement = () => {
    if (!newPreserve.trim()) return;
    updateField("preserve_elements", [...brief.preserve_elements, newPreserve.trim()]);
    setNewPreserve("");
  };

  const removePreserveElement = (index: number) => {
    updateField("preserve_elements", brief.preserve_elements.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="space-y-2">
        <Label className="text-sm font-semibold">Global Instructions</Label>
        <Textarea value={brief.global_instructions} onChange={(e) => updateField("global_instructions", e.target.value)} rows={3} className="text-sm resize-none" placeholder="Overall styling direction..." />
      </div>

      <div className="space-y-2">
        <Label className="text-sm font-semibold">Plant Palette</Label>
        <PlantPaletteTable plants={brief.plant_palette} onChange={(plants: PlantEntry[]) => updateField("plant_palette", plants)} />
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
              <button onClick={() => removePreserveElement(idx)}><X className="h-2.5 w-2.5" /></button>
            </Badge>
          ))}
        </div>
        <div className="flex gap-2">
          <Input value={newPreserve} onChange={(e) => setNewPreserve(e.target.value)} onKeyDown={(e) => e.key === "Enter" && addPreserveElement()} placeholder="e.g. existing patio" className="text-sm h-8" />
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
}
