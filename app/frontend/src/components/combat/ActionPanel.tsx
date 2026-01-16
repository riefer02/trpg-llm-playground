import { useState } from "react";

import type {
  ActionEconomyState,
  ActionRequest,
  AvailableActionItem,
  AvailableActionsResponse,
} from "../../lib/api/combat";
import type { MechInventory } from "../../lib/types/lancer";
import { Button } from "../ui";
import { WeaponPicker } from "./WeaponPicker";

export type ActionPanelState =
  | "idle"
  | "selecting_target"
  | "selecting_weapon"
  | "confirming"
  | "executing";

export interface TargetMode {
  actionId: string;
  actionType: ActionRequest["action_type"];
  requiresTarget: boolean;
  requiresWeapon: boolean;
}

export interface ActionPanelProps {
  availableActions: AvailableActionsResponse | null;
  economy: ActionEconomyState | null;
  onActionSelect: (action: AvailableActionItem) => void;
  onExecuteAction: (request: ActionRequest) => void;
  onTargetModeChange: (mode: TargetMode | null) => void;
  isExecuting?: boolean;
  selectedTargetId?: string | null;
  actorInventory?: MechInventory | null;
}

export function ActionPanel({
  availableActions,
  economy,
  onActionSelect,
  onExecuteAction,
  onTargetModeChange,
  isExecuting = false,
  selectedTargetId,
  actorInventory,
}: ActionPanelProps) {
  const [panelState, setPanelState] = useState<ActionPanelState>("idle");
  const [selectedAction, setSelectedAction] = useState<AvailableActionItem | null>(null);
  const [selectedWeaponId, setSelectedWeaponId] = useState<string | null>(null);

  if (!availableActions || !economy) {
    return (
      <div className="rounded-md border border-border bg-muted/30 p-3">
        <div className="text-sm text-muted-foreground">
          Start turn to see available actions
        </div>
      </div>
    );
  }

  const handleActionClick = (action: AvailableActionItem) => {
    if (!action.is_available) return;

    setSelectedAction(action);
    setSelectedWeaponId(null);
    onActionSelect(action);

    // Determine initial state based on requirements
    if (action.requires_weapon) {
      // Weapon selection comes first
      setPanelState("selecting_weapon");
      onTargetModeChange(null);
    } else if (action.requires_target) {
      setPanelState("selecting_target");
      onTargetModeChange({
        actionId: action.action_id,
        actionType: action.action_type as ActionRequest["action_type"],
        requiresTarget: action.requires_target,
        requiresWeapon: action.requires_weapon,
      });
    } else {
      setPanelState("confirming");
      onTargetModeChange(null);
    }
  };

  const handleWeaponSelect = (weaponId: string) => {
    setSelectedWeaponId(weaponId);

    if (selectedAction?.requires_target) {
      // After weapon selection, move to target selection
      setPanelState("selecting_target");
      onTargetModeChange({
        actionId: selectedAction.action_id,
        actionType: selectedAction.action_type as ActionRequest["action_type"],
        requiresTarget: selectedAction.requires_target,
        requiresWeapon: selectedAction.requires_weapon,
      });
    } else {
      // No target needed, go straight to confirming
      setPanelState("confirming");
    }
  };

  const handleConfirmAction = () => {
    if (!selectedAction) return;

    setPanelState("executing");
    onExecuteAction({
      action_id: selectedAction.action_id,
      action_type: selectedAction.action_type as ActionRequest["action_type"],
      target_ids: selectedTargetId ? [selectedTargetId] : [],
      weapon_id: selectedWeaponId ?? undefined,
    });
  };

  const handleCancelAction = () => {
    setSelectedAction(null);
    setSelectedWeaponId(null);
    setPanelState("idle");
    onTargetModeChange(null);
  };

  // Reset state when action completes
  const resetAfterExecution = () => {
    setSelectedAction(null);
    setSelectedWeaponId(null);
    setPanelState("idle");
    onTargetModeChange(null);
  };

  // Show weapon picker
  if (panelState === "selecting_weapon") {
    return (
      <div className="space-y-3">
        {selectedAction && (
          <div className="rounded-md border border-border bg-muted/30 p-3">
            <div className="text-sm font-medium text-foreground mb-2">
              {selectedAction.action_name}
            </div>
            <div className="text-xs text-muted-foreground">
              {selectedAction.action_type} action - Select a weapon
            </div>
          </div>
        )}
        <WeaponPicker
          inventory={actorInventory}
          onSelect={handleWeaponSelect}
          onCancel={handleCancelAction}
          isOpen={true}
        />
      </div>
    );
  }

  // Show confirmation UI
  if (panelState === "confirming" || panelState === "selecting_target") {
    return (
      <div className="rounded-md border border-border bg-muted/30 p-3 space-y-3">
        <div className="text-sm font-medium text-foreground">
          {panelState === "selecting_target" ? "Select Target" : "Confirm Action"}
        </div>

        {selectedAction && (
          <div className="p-2 rounded bg-primary/10 border border-primary/30">
            <div className="font-medium text-primary">
              {selectedAction.action_name}
            </div>
            <div className="text-xs text-muted-foreground">
              {selectedAction.action_type} action
              {selectedWeaponId && ` - ${formatWeaponId(selectedWeaponId)}`}
            </div>
          </div>
        )}

        {panelState === "selecting_target" && (
          <div className="text-xs text-muted-foreground">
            Click a combatant on the canvas to select target
          </div>
        )}

        {selectedTargetId && (
          <div className="text-xs text-primary">
            Target selected: {selectedTargetId}
          </div>
        )}

        <div className="flex gap-2">
          <Button
            variant="primary"
            size="sm"
            onClick={handleConfirmAction}
            disabled={isExecuting || (selectedAction?.requires_target && !selectedTargetId)}
          >
            {isExecuting ? "Executing..." : "Execute"}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCancelAction}
            disabled={isExecuting}
          >
            Cancel
          </Button>
        </div>
      </div>
    );
  }

  // Show execution result briefly
  if (panelState === "executing" && !isExecuting) {
    setTimeout(resetAfterExecution, 500);
  }

  return (
    <div className="rounded-md border border-border bg-muted/30 p-3 space-y-3">
      <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
        Available Actions
      </div>

      {/* Full Actions */}
      {availableActions.full_actions.length > 0 && (
        <ActionSection
          title="Full Actions"
          actions={availableActions.full_actions}
          disabled={economy.full_action_used}
          onActionClick={handleActionClick}
        />
      )}

      {/* Quick Actions */}
      {availableActions.quick_actions.length > 0 && (
        <ActionSection
          title="Quick Actions"
          actions={availableActions.quick_actions}
          disabled={economy.quick_actions_available === 0}
          onActionClick={handleActionClick}
        />
      )}

      {/* Free Actions */}
      {availableActions.free_actions.length > 0 && (
        <ActionSection
          title="Free Actions"
          actions={availableActions.free_actions}
          onActionClick={handleActionClick}
        />
      )}

      {/* Protocols */}
      {availableActions.protocols.length > 0 && (
        <ActionSection
          title="Protocols"
          actions={availableActions.protocols}
          disabled={economy.protocol_used}
          onActionClick={handleActionClick}
        />
      )}

      {/* Overcharge */}
      {availableActions.can_overcharge && !economy.overcharge_used && (
        <div className="pt-2 border-t border-border">
          <Button
            variant="outline"
            size="sm"
            className="w-full text-amber-500 border-amber-500/50 hover:bg-amber-500/10"
            onClick={() => onExecuteAction({
              action_id: "overcharge",
              action_type: "free",
              is_overcharge: true,
            })}
            disabled={isExecuting}
          >
            Overcharge (+1 Quick, generates heat)
          </Button>
        </div>
      )}
    </div>
  );
}

interface ActionSectionProps {
  title: string;
  actions: AvailableActionItem[];
  disabled?: boolean;
  onActionClick: (action: AvailableActionItem) => void;
}

function ActionSection({ title, actions, disabled, onActionClick }: ActionSectionProps) {
  return (
    <div className="space-y-1">
      <div className="text-xs text-muted-foreground font-medium">{title}</div>
      <div className="space-y-0.5">
        {actions.map((action) => (
          <ActionItem
            key={action.action_id}
            action={action}
            disabled={disabled || !action.is_available}
            onClick={() => onActionClick(action)}
          />
        ))}
      </div>
    </div>
  );
}

interface ActionItemProps {
  action: AvailableActionItem;
  disabled?: boolean;
  onClick: () => void;
}

function ActionItem({ action, disabled, onClick }: ActionItemProps) {
  const isAvailable = action.is_available && !disabled;

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={!isAvailable}
      className={`w-full text-left px-2 py-1.5 rounded text-sm transition-colors ${
        isAvailable
          ? "hover:bg-primary/10 text-foreground cursor-pointer"
          : "text-muted-foreground/50 cursor-not-allowed"
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              isAvailable ? "bg-primary" : "bg-muted-foreground/30"
            }`}
          />
          <span>{action.action_name}</span>
        </div>
        {action.requires_target && (
          <span className="text-xs text-muted-foreground">target</span>
        )}
        {action.requires_weapon && (
          <span className="text-xs text-muted-foreground">weapon</span>
        )}
      </div>
      {action.unavailable_reason && (
        <div className="ml-4 text-xs text-muted-foreground/70">
          {action.unavailable_reason}
        </div>
      )}
    </button>
  );
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
