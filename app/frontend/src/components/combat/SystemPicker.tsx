import type { MechInventory, MechSystemState } from "../../lib/types/lancer";
import { Button } from "../ui";

export interface SystemPickerProps {
  inventory: MechInventory | null | undefined;
  onSelect: (systemId: string) => void;
  onCancel: () => void;
  isOpen: boolean;
}

export function SystemPicker({
  inventory,
  onSelect,
  onCancel,
  isOpen,
}: SystemPickerProps) {
  if (!isOpen) {
    return null;
  }

  const systems = inventory?.systems ?? [];

  if (systems.length === 0) {
    return (
      <div className="rounded-md border border-border bg-muted/30 p-3 space-y-2">
        <div className="text-sm font-medium text-foreground">Select System</div>
        <div className="text-xs text-muted-foreground">
          No systems available
        </div>
        <Button variant="ghost" size="sm" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    );
  }

  return (
    <div className="rounded-md border border-border bg-muted/30 p-3 space-y-2">
      <div className="text-sm font-medium text-foreground">Select System</div>
      <div className="space-y-1">
        {systems.map((system) => (
          <SystemItem
            key={system.system_id}
            system={system}
            onSelect={() => onSelect(system.system_id)}
          />
        ))}
      </div>
      <div className="pt-2">
        <Button variant="ghost" size="sm" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </div>
  );
}

interface SystemItemProps {
  system: MechSystemState;
  onSelect: () => void;
}

function SystemItem({ system, onSelect }: SystemItemProps) {
  const isDestroyed = system.destroyed === true;
  const hasCharges = system.limited_charges_remaining !== undefined;
  const noChargesLeft = hasCharges && (system.limited_charges_remaining ?? 0) <= 0;
  const isDisabled = isDestroyed || noChargesLeft;

  return (
    <button
      type="button"
      onClick={onSelect}
      disabled={isDisabled}
      className={`w-full text-left px-2 py-1.5 rounded text-sm transition-colors ${
        isDisabled
          ? "text-muted-foreground/50 cursor-not-allowed"
          : "hover:bg-primary/10 text-foreground cursor-pointer"
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              isDisabled ? "bg-muted-foreground/30" : "bg-primary"
            }`}
          />
          <span>{formatSystemId(system.system_id)}</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          {hasCharges && (
            <span>{system.limited_charges_remaining} charges</span>
          )}
          {isDestroyed && (
            <span className="text-destructive">Destroyed</span>
          )}
        </div>
      </div>
    </button>
  );
}

/**
 * Format system_id for display (e.g., "ms_personalizations" -> "Personalizations")
 */
function formatSystemId(systemId: string): string {
  // Remove common prefixes
  const cleaned = systemId
    .replace(/^ms_/, "")
    .replace(/^cs_/, "");

  // Convert snake_case to Title Case
  return cleaned
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
