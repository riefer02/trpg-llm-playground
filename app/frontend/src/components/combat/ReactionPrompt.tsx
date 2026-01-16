import { useState } from "react";

import type { ReactionRequest } from "../../lib/api/combat";
import type { MechInventory, WeaponMountState, WeaponState } from "../../lib/types/lancer";
import { Button } from "../ui";

export type ReactionTriggerType = "attack_incoming" | "enemy_movement";
export type ReactionType = "brace" | "overwatch";

export interface ReactionPromptProps {
  triggerType: ReactionTriggerType;
  reactorId: string;
  reactorName: string;
  triggeringActorName?: string;
  availableReactions: ReactionType[];
  inventory: MechInventory | null | undefined;
  validTargets?: { id: string; name: string }[];
  onSubmit: (reaction: ReactionRequest) => void;
  onDecline: () => void;
  isOpen: boolean;
  isSubmitting?: boolean;
}

export function ReactionPrompt({
  triggerType,
  reactorId,
  reactorName,
  triggeringActorName,
  availableReactions,
  inventory,
  validTargets,
  onSubmit,
  onDecline,
  isOpen,
  isSubmitting = false,
}: ReactionPromptProps) {
  const [selectedReaction, setSelectedReaction] = useState<ReactionType | null>(null);
  const [selectedWeaponId, setSelectedWeaponId] = useState<string | null>(null);
  const [selectedTargetId, setSelectedTargetId] = useState<string | null>(null);

  if (!isOpen) {
    return null;
  }

  const triggerDescription =
    triggerType === "attack_incoming"
      ? `${triggeringActorName ?? "An enemy"} is attacking ${reactorName}!`
      : `${triggeringActorName ?? "An enemy"} is moving within range of ${reactorName}!`;

  const weapons = flattenWeapons(inventory?.mounts ?? []);
  const needsWeapon = selectedReaction === "overwatch";
  const needsTarget = selectedReaction === "overwatch" && validTargets && validTargets.length > 0;

  const canSubmit =
    selectedReaction &&
    (!needsWeapon || selectedWeaponId) &&
    (!needsTarget || selectedTargetId);

  const handleSubmit = () => {
    if (!selectedReaction || !canSubmit) return;

    onSubmit({
      reactor_id: reactorId,
      reaction_type: selectedReaction,
      weapon_id: selectedWeaponId ?? undefined,
      target_ids: selectedTargetId ? [selectedTargetId] : [],
    });
  };

  const handleReactionSelect = (reaction: ReactionType) => {
    setSelectedReaction(reaction);
    // Reset dependent selections when reaction changes
    if (reaction !== "overwatch") {
      setSelectedWeaponId(null);
      setSelectedTargetId(null);
    }
  };

  return (
    <div className="rounded-md border border-amber-500/50 bg-amber-500/5 p-4 space-y-4">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-sm font-medium text-amber-500">
            Reaction Opportunity
          </div>
          <div className="text-xs text-muted-foreground mt-1">
            {triggerDescription}
          </div>
        </div>
        <span className="px-2 py-0.5 rounded text-xs bg-amber-500/20 text-amber-500">
          React?
        </span>
      </div>

      {/* Reaction type selection */}
      <div className="space-y-2">
        <div className="text-xs text-muted-foreground font-medium">
          Choose Reaction
        </div>
        <div className="flex gap-2">
          {availableReactions.includes("brace") && (
            <ReactionOption
              type="brace"
              label="Brace"
              description="Gain resistance to incoming attack"
              isSelected={selectedReaction === "brace"}
              onSelect={() => handleReactionSelect("brace")}
            />
          )}
          {availableReactions.includes("overwatch") && (
            <ReactionOption
              type="overwatch"
              label="Overwatch"
              description="Attack the moving enemy"
              isSelected={selectedReaction === "overwatch"}
              onSelect={() => handleReactionSelect("overwatch")}
            />
          )}
        </div>
      </div>

      {/* Weapon selection for overwatch */}
      {needsWeapon && (
        <div className="space-y-2">
          <div className="text-xs text-muted-foreground font-medium">
            Select Weapon
          </div>
          {weapons.length > 0 ? (
            <div className="space-y-1">
              {weapons.map((weapon) => (
                <WeaponOption
                  key={weapon.weapon_id}
                  weapon={weapon}
                  isSelected={selectedWeaponId === weapon.weapon_id}
                  onSelect={() => setSelectedWeaponId(weapon.weapon_id)}
                />
              ))}
            </div>
          ) : (
            <div className="text-xs text-muted-foreground">
              No weapons available
            </div>
          )}
        </div>
      )}

      {/* Target selection for overwatch */}
      {needsTarget && validTargets && validTargets.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs text-muted-foreground font-medium">
            Select Target
          </div>
          <div className="space-y-1">
            {validTargets.map((target) => (
              <button
                key={target.id}
                type="button"
                onClick={() => setSelectedTargetId(target.id)}
                className={`w-full text-left px-2 py-1.5 rounded text-sm transition-colors ${
                  selectedTargetId === target.id
                    ? "bg-primary/20 border border-primary/50 text-foreground"
                    : "bg-muted/30 border border-transparent hover:bg-primary/10 text-foreground"
                }`}
              >
                {target.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-2 pt-2">
        <Button
          variant="primary"
          size="sm"
          onClick={handleSubmit}
          disabled={!canSubmit || isSubmitting}
          className="bg-amber-500 hover:bg-amber-600"
        >
          {isSubmitting ? "Reacting..." : "React"}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={onDecline}
          disabled={isSubmitting}
        >
          Decline
        </Button>
      </div>
    </div>
  );
}

interface ReactionOptionProps {
  type: ReactionType;
  label: string;
  description: string;
  isSelected: boolean;
  onSelect: () => void;
}

function ReactionOption({
  label,
  description,
  isSelected,
  onSelect,
}: ReactionOptionProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`flex-1 p-2 rounded text-left transition-colors ${
        isSelected
          ? "bg-primary/20 border border-primary/50"
          : "bg-muted/30 border border-transparent hover:bg-primary/10"
      }`}
    >
      <div className={`text-sm font-medium ${isSelected ? "text-primary" : "text-foreground"}`}>
        {label}
      </div>
      <div className="text-xs text-muted-foreground">{description}</div>
    </button>
  );
}

interface WeaponOptionProps {
  weapon: WeaponState;
  isSelected: boolean;
  onSelect: () => void;
}

function WeaponOption({ weapon, isSelected, onSelect }: WeaponOptionProps) {
  const isDestroyed = weapon.destroyed === true;
  const isDisabled = isDestroyed;

  return (
    <button
      type="button"
      onClick={onSelect}
      disabled={isDisabled}
      className={`w-full text-left px-2 py-1.5 rounded text-sm transition-colors ${
        isDisabled
          ? "text-muted-foreground/50 cursor-not-allowed"
          : isSelected
            ? "bg-primary/20 border border-primary/50 text-foreground"
            : "bg-muted/30 border border-transparent hover:bg-primary/10 text-foreground"
      }`}
    >
      <div className="flex items-center justify-between">
        <span>{formatWeaponId(weapon.weapon_id)}</span>
        {isDestroyed && (
          <span className="text-xs text-destructive">Destroyed</span>
        )}
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
  const cleaned = weaponId
    .replace(/^mw_/, "")
    .replace(/^cw_/, "")
    .replace(/^heavy_/, "Heavy ")
    .replace(/^aux_/, "Aux ");

  return cleaned
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
