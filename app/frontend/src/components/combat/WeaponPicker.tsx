import type { MechInventory, WeaponMountState, WeaponState } from "../../lib/types/lancer";
import { Button } from "../ui";

export interface WeaponPickerProps {
  inventory: MechInventory | null | undefined;
  onSelect: (weaponId: string) => void;
  onCancel: () => void;
  isOpen: boolean;
}

export function WeaponPicker({
  inventory,
  onSelect,
  onCancel,
  isOpen,
}: WeaponPickerProps) {
  if (!isOpen) {
    return null;
  }

  const mounts = inventory?.mounts ?? [];
  const weapons = flattenWeapons(mounts);

  if (weapons.length === 0) {
    return (
      <div className="rounded-md border border-border bg-muted/30 p-3 space-y-2">
        <div className="text-sm font-medium text-foreground">Select Weapon</div>
        <div className="text-xs text-muted-foreground">
          No weapons available
        </div>
        <Button variant="ghost" size="sm" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    );
  }

  return (
    <div className="rounded-md border border-border bg-muted/30 p-3 space-y-2">
      <div className="text-sm font-medium text-foreground">Select Weapon</div>
      <div className="space-y-1">
        {weapons.map((weapon) => (
          <WeaponItem
            key={weapon.weapon_id}
            weapon={weapon}
            onSelect={() => onSelect(weapon.weapon_id)}
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

interface WeaponItemProps {
  weapon: WeaponState;
  onSelect: () => void;
}

function WeaponItem({ weapon, onSelect }: WeaponItemProps) {
  const isDestroyed = weapon.destroyed === true;
  const hasCharges = weapon.limited_charges_remaining !== undefined;
  const noChargesLeft = hasCharges && (weapon.limited_charges_remaining ?? 0) <= 0;
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
          <span>{formatWeaponId(weapon.weapon_id)}</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          {hasCharges && (
            <span>{weapon.limited_charges_remaining} charges</span>
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
 * Flatten all weapons from all mounts into a single array.
 */
function flattenWeapons(mounts: WeaponMountState[]): WeaponState[] {
  const weapons: WeaponState[] = [];
  for (const mount of mounts) {
    if (mount.destroyed) {
      continue;
    }
    for (const weapon of mount.weapons ?? []) {
      weapons.push(weapon);
    }
  }
  return weapons;
}

/**
 * Format weapon_id for display (e.g., "mw_assault_rifle" -> "Assault Rifle")
 */
function formatWeaponId(weaponId: string): string {
  // Remove common prefixes
  const cleaned = weaponId
    .replace(/^mw_/, "")
    .replace(/^cw_/, "")
    .replace(/^heavy_/, "Heavy ")
    .replace(/^aux_/, "Aux ");

  // Convert snake_case to Title Case
  return cleaned
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
