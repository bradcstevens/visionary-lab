"use client"

import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { PlantEntry } from "@/services/stagingApi";

interface PlantPaletteTableProps {
  plants: PlantEntry[];
  onChange: (plants: PlantEntry[]) => void;
}

export function PlantPaletteTable({ plants, onChange }: PlantPaletteTableProps) {
  const updatePlant = (index: number, field: keyof PlantEntry, value: string | number) => {
    const updated = [...plants];
    updated[index] = { ...updated[index], [field]: value };
    onChange(updated);
  };

  const addPlant = () => {
    onChange([...plants, { species: "", quantity: 1, size: "", placement: "" }]);
  };

  const removePlant = (index: number) => {
    onChange(plants.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-[1fr_120px_80px_1fr_1fr_40px] gap-2 text-xs font-medium text-muted-foreground">
        <div>Species</div>
        <div>Botanical Name</div>
        <div>Qty</div>
        <div>Size</div>
        <div>Placement</div>
        <div></div>
      </div>

      {plants.map((plant, idx) => (
        <div key={idx} className="grid grid-cols-[1fr_120px_80px_1fr_1fr_40px] gap-2">
          <Input value={plant.species} onChange={(e) => updatePlant(idx, "species", e.target.value)} placeholder="Species name" className="text-sm h-8" />
          <Input value={plant.botanical_name ?? ""} onChange={(e) => updatePlant(idx, "botanical_name", e.target.value)} placeholder="Latin name" className="text-sm h-8" />
          <Input type="number" value={plant.quantity} onChange={(e) => updatePlant(idx, "quantity", parseInt(e.target.value) || 1)} min={1} className="text-sm h-8" />
          <Input value={plant.size} onChange={(e) => updatePlant(idx, "size", e.target.value)} placeholder="e.g. 8-10 ft" className="text-sm h-8" />
          <Input value={plant.placement} onChange={(e) => updatePlant(idx, "placement", e.target.value)} placeholder="e.g. back row" className="text-sm h-8" />
          <Button size="sm" variant="ghost" onClick={() => removePlant(idx)} className="h-8 w-8 p-0">
            <Trash2 className="h-3.5 w-3.5 text-destructive" />
          </Button>
        </div>
      ))}

      <Button size="sm" variant="outline" onClick={addPlant} className="w-full">
        <Plus className="h-3.5 w-3.5 mr-1" /> Add Plant
      </Button>
    </div>
  );
}
