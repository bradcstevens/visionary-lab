"use client"

const ACTION_LABELS: Record<string, string> = {
  specify_species: "🌲 Specify plant species",
  choose_density: "📏 Set planting density",
  set_height_preference: "📐 Define height layers",
  define_placement: "📍 Describe placement",
  choose_style: "🎨 Choose a style",
  add_more_areas: "➕ Add more areas",
  generate_brief: "📋 Generate Design Brief",
  specify_quantity: "🔢 Specify quantities",
  add_more_species: "🌿 Add more species",
};

interface QuickReplyChipsProps {
  actions: string[];
  onSelect: (action: string) => void;
  disabled?: boolean;
}

export function QuickReplyChips({ actions, onSelect, disabled = false }: QuickReplyChipsProps) {
  if (!actions.length) return null;

  return (
    <div className="flex flex-wrap gap-2 mt-2">
      {actions.map((action) => (
        <button
          key={action}
          onClick={() => onSelect(action)}
          disabled={disabled}
          className="px-3 py-1.5 text-xs rounded-full border border-border bg-muted/50 
                     hover:bg-muted hover:border-primary/50 transition-colors
                     disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {ACTION_LABELS[action] ?? action}
        </button>
      ))}
    </div>
  );
}
