import { useState } from "react";

import type { DecisionSubmitRequest, PendingDecisionItem } from "../../lib/api/combat";
import type { MechInventory } from "../../lib/types/lancer";
import { Button } from "../ui";

type SelectionTarget = "mount" | "system";

export interface TraumaSelectionPromptProps {
  decision: PendingDecisionItem;
  combatantId: string;
  combatantName: string;
  inventory: MechInventory | null | undefined;
  onSubmit: (request: DecisionSubmitRequest) => void;
  onDecline: () => void;
  isOpen: boolean;
  isSubmitting?: boolean;
}

export function TraumaSelectionPrompt({
  decision,
  combatantId,
  combatantName,
  inventory,
  onSubmit,
  onDecline,
  isOpen,
  isSubmitting = false,
}: TraumaSelectionPromptProps) {
  const [selectedTarget, setSelectedTarget] = useState<SelectionTarget | null>(null);
  const [selectedMountIndex, setSelectedMountIndex] = useState<number | null>(null);
  const [selectedSystemId, setSelectedSystemId] = useState<string | null>(null);

  if (!isOpen) {
    return null;
  }

  const eligibleMounts = decision.eligible_mounts;
  const eligibleSystems = decision.eligible_systems;
  const hasMounts = eligibleMounts.length > 0;
  const hasSystems = eligibleSystems.length > 0;

  // Get mount details for display
  const mountDetails = inventory?.mounts
    ?.filter((m) => eligibleMounts.includes(m.mount_index))
    .map((m) => ({
      index: m.mount_index,
      weapons: m.weapons?.map((w) => formatWeaponId(w.weapon_id)).join(", ") ?? "Empty",
      slotType: m.slot_type ?? undefined,
    })) ?? [];

  // Get system details for display
  const systemDetails = inventory?.systems
    ?.filter((s) => eligibleSystems.includes(s.system_id))
    .map((s) => ({
      id: s.system_id,
      name: formatSystemId(s.system_id),
      charges: s.limited_charges_remaining ?? undefined,
    })) ?? [];

  const canSubmit =
    (selectedTarget === "mount" && selectedMountIndex !== null) ||
    (selectedTarget === "system" && selectedSystemId !== null);

  const handleSubmit = () => {
    if (!canSubmit) return;

    onSubmit({
      decision_id: decision.decision_id,
      combatant_id: combatantId,
      choice: "roll", // Not used for trauma, but required by the type
      selected_mount_index: selectedTarget === "mount" ? selectedMountIndex ?? undefined : undefined,
      selected_system_id: selectedTarget === "system" ? selectedSystemId ?? undefined : undefined,
    });
  };

  const handleTargetSelect = (target: SelectionTarget) => {
    setSelectedTarget(target);
    // Reset dependent selections
    if (target === "mount") {
      setSelectedSystemId(null);
    } else {
      setSelectedMountIndex(null);
    }
  };

  return (
    <div className="rounded-md border border-purple-500/50 bg-purple-500/5 p-4 space-y-4">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-sm font-medium text-purple-500">
            System Trauma
          </div>
          <div className="text-xs text-muted-foreground mt-1">
            {combatantName} has suffered system trauma! Choose what to destroy.
          </div>
        </div>
        <span className="px-2 py-0.5 rounded text-xs bg-purple-500/20 text-purple-500">
          Choose Target
        </span>
      </div>

      {/* Info text */}
      <div className="text-xs text-muted-foreground bg-muted/30 rounded px-2 py-1.5">
        Per PR2: "You choose what's destroyed, but systems or weapons with the limited tag and no charges left are not valid."
      </div>

      {/* Target type selection */}
      <div className="space-y-2">
        <div className="text-xs text-muted-foreground font-medium">
          What to Destroy
        </div>
        <div className="flex gap-2">
          {hasMounts && (
            <TargetTypeOption
              type="mount"
              label="Weapon Mount"
              description={`${eligibleMounts.length} mount(s) available`}
              isSelected={selectedTarget === "mount"}
              onSelect={() => handleTargetSelect("mount")}
            />
          )}
          {hasSystems && (
            <TargetTypeOption
              type="system"
              label="System"
              description={`${eligibleSystems.length} system(s) available`}
              isSelected={selectedTarget === "system"}
              onSelect={() => handleTargetSelect("system")}
            />
          )}
        </div>
      </div>

      {/* Mount selection */}
      {selectedTarget === "mount" && mountDetails.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs text-muted-foreground font-medium">
            Select Mount to Destroy
          </div>
          <div className="space-y-1">
            {mountDetails.map((mount) => (
              <MountOption
                key={mount.index}
                mountIndex={mount.index}
                weapons={mount.weapons}
                slotType={mount.slotType}
                isSelected={selectedMountIndex === mount.index}
                onSelect={() => setSelectedMountIndex(mount.index)}
              />
            ))}
          </div>
        </div>
      )}

      {/* System selection */}
      {selectedTarget === "system" && systemDetails.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs text-muted-foreground font-medium">
            Select System to Destroy
          </div>
          <div className="space-y-1">
            {systemDetails.map((system) => (
              <SystemOption
                key={system.id}
                systemId={system.id}
                systemName={system.name}
                charges={system.charges}
                isSelected={selectedSystemId === system.id}
                onSelect={() => setSelectedSystemId(system.id)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Warning */}
      <div className="text-xs text-red-500 bg-red-500/10 rounded px-2 py-1.5">
        This action is permanent! The selected item will be destroyed.
      </div>

      {/* Action buttons */}
      <div className="flex gap-2 pt-2">
        <Button
          variant="primary"
          size="sm"
          onClick={handleSubmit}
          disabled={!canSubmit || isSubmitting}
          className="bg-purple-500 hover:bg-purple-600"
        >
          {isSubmitting ? "Destroying..." : "Confirm Destruction"}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={onDecline}
          disabled={isSubmitting}
        >
          Cancel
        </Button>
      </div>
    </div>
  );
}

interface TargetTypeOptionProps {
  type: SelectionTarget;
  label: string;
  description: string;
  isSelected: boolean;
  onSelect: () => void;
}

function TargetTypeOption({
  label,
  description,
  isSelected,
  onSelect,
}: TargetTypeOptionProps) {
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

interface MountOptionProps {
  mountIndex: number;
  weapons: string;
  slotType?: string;
  isSelected: boolean;
  onSelect: () => void;
}

function MountOption({
  mountIndex,
  weapons,
  slotType,
  isSelected,
  onSelect,
}: MountOptionProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`w-full text-left px-2 py-1.5 rounded transition-colors ${
        isSelected
          ? "bg-primary/20 border border-primary/50 text-foreground"
          : "bg-muted/30 border border-transparent hover:bg-primary/10 text-foreground"
      }`}
    >
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">
          Mount {mountIndex + 1}
          {slotType && <span className="text-xs text-muted-foreground ml-1">({slotType})</span>}
        </span>
      </div>
      <div className="text-xs text-muted-foreground mt-0.5">
        {weapons}
      </div>
    </button>
  );
}

interface SystemOptionProps {
  systemId: string;
  systemName: string;
  charges?: number;
  isSelected: boolean;
  onSelect: () => void;
}

function SystemOption({
  systemName,
  charges,
  isSelected,
  onSelect,
}: SystemOptionProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`w-full text-left px-2 py-1.5 rounded transition-colors ${
        isSelected
          ? "bg-primary/20 border border-primary/50 text-foreground"
          : "bg-muted/30 border border-transparent hover:bg-primary/10 text-foreground"
      }`}
    >
      <div className="flex items-center justify-between">
        <span className="text-sm">{systemName}</span>
        {charges !== undefined && charges > 0 && (
          <span className="text-xs text-muted-foreground">{charges} charges</span>
        )}
      </div>
    </button>
  );
}

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

function formatSystemId(systemId: string): string {
  const cleaned = systemId
    .replace(/^ms_/, "")
    .replace(/^sys_/, "");

  return cleaned
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
