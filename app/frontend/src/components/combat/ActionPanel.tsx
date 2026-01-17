import { useEffect, useState } from "react";

import type {
  ActionEconomyState,
  ActionRequest,
  AvailableActionItem,
  AvailableActionsResponse,
} from "../../lib/api/combat";
import type { HexCoord, MechInventory } from "../../lib/types/lancer";
import { calculatePathDistance, hexEquals, isAdjacent } from "../../lib/combat-render/hex";
import { Button } from "../ui";
import { SystemPicker } from "./SystemPicker";
import { WeaponPicker } from "./WeaponPicker";

export type ActionPanelState =
  | "idle"
  | "selecting_target"
  | "selecting_weapon"
  | "selecting_system"
  | "selecting_path"
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
  onPathModeChange: (isActive: boolean, path: HexCoord[]) => void;
  isExecuting?: boolean;
  selectedTargetIds?: string[];
  actorInventory?: MechInventory | null;
  actorSpeed?: number;
  actorPosition?: HexCoord | null;
  /** Incoming hex click from canvas when in path mode */
  hexClickCoord?: HexCoord | null;
}

export function ActionPanel({
  availableActions,
  economy,
  onActionSelect,
  onExecuteAction,
  onTargetModeChange,
  onPathModeChange,
  isExecuting = false,
  selectedTargetIds = [],
  actorInventory,
  actorSpeed = 4,
  actorPosition,
  hexClickCoord,
}: ActionPanelProps) {
  const [panelState, setPanelState] = useState<ActionPanelState>("idle");
  const [selectedAction, setSelectedAction] = useState<AvailableActionItem | null>(null);
  const [selectedWeaponId, setSelectedWeaponId] = useState<string | null>(null);
  const [selectedSystemId, setSelectedSystemId] = useState<string | null>(null);
  const [movementPath, setMovementPath] = useState<HexCoord[]>([]);
  const [lastProcessedClick, setLastProcessedClick] = useState<HexCoord | null>(null);

  // Handle incoming hex clicks from the canvas when in path mode
  useEffect(() => {
    if (
      panelState !== "selecting_path" ||
      !hexClickCoord ||
      (lastProcessedClick && hexEquals(hexClickCoord, lastProcessedClick))
    ) {
      return;
    }

    const coord = hexClickCoord;
    setLastProcessedClick(coord);

    setMovementPath((currentPath) => {
      const lastHex = currentPath[currentPath.length - 1];

      // Click on last hex = undo (unless it's the only hex)
      if (lastHex && hexEquals(coord, lastHex)) {
        if (currentPath.length > 1) {
          const newPath = currentPath.slice(0, -1);
          onPathModeChange(true, newPath);
          return newPath;
        }
        return currentPath;
      }

      // Only allow adjacent hexes
      if (lastHex && !isAdjacent(lastHex, coord)) {
        return currentPath; // Ignore non-adjacent clicks
      }

      // Add to path
      const newPath = [...currentPath, coord];
      onPathModeChange(true, newPath);
      return newPath;
    });
  }, [hexClickCoord, panelState, lastProcessedClick, onPathModeChange]);

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
    setSelectedSystemId(null);
    setMovementPath([]);
    onActionSelect(action);

    // Determine initial state based on requirements
    if (action.requires_weapon) {
      // Weapon selection comes first
      setPanelState("selecting_weapon");
      onTargetModeChange(null);
      onPathModeChange(false, []);
    } else if (action.requires_system) {
      // System selection
      setPanelState("selecting_system");
      onTargetModeChange(null);
      onPathModeChange(false, []);
    } else if (action.requires_path) {
      // Path selection (for Move, Boost)
      const initialPath = actorPosition ? [actorPosition] : [];
      setMovementPath(initialPath);
      setPanelState("selecting_path");
      onTargetModeChange(null);
      onPathModeChange(true, initialPath);
    } else if (action.requires_target) {
      setPanelState("selecting_target");
      onTargetModeChange({
        actionId: action.action_id,
        actionType: action.action_type as ActionRequest["action_type"],
        requiresTarget: action.requires_target,
        requiresWeapon: action.requires_weapon,
      });
      onPathModeChange(false, []);
    } else {
      setPanelState("confirming");
      onTargetModeChange(null);
      onPathModeChange(false, []);
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

  const handleSystemSelect = (systemId: string) => {
    setSelectedSystemId(systemId);

    if (selectedAction?.requires_target) {
      // After system selection, move to target selection
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
      target_ids: selectedTargetIds,
      weapon_id: selectedWeaponId ?? undefined,
      system_id: selectedSystemId ?? undefined,
    });
  };

  const handleCancelAction = () => {
    setSelectedAction(null);
    setSelectedWeaponId(null);
    setSelectedSystemId(null);
    setMovementPath([]);
    setPanelState("idle");
    onTargetModeChange(null);
    onPathModeChange(false, []);
  };

  // Reset state when action completes
  const resetAfterExecution = () => {
    setSelectedAction(null);
    setSelectedWeaponId(null);
    setSelectedSystemId(null);
    setMovementPath([]);
    setPanelState("idle");
    onTargetModeChange(null);
    onPathModeChange(false, []);
  };

  // Handle confirm path for movement actions
  const handleConfirmPath = () => {
    if (!selectedAction || movementPath.length < 2) return;

    setPanelState("executing");
    onPathModeChange(false, []);

    onExecuteAction({
      action_id: selectedAction.action_id,
      action_type: selectedAction.action_type as ActionRequest["action_type"],
      movement_path: movementPath.map((c) => ({ coord: { q: c.q, r: c.r } })),
    });
  };

  // Handle clear path
  const handleClearPath = () => {
    const initialPath = actorPosition ? [actorPosition] : [];
    setMovementPath(initialPath);
    onPathModeChange(true, initialPath);
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

  // Show system picker
  if (panelState === "selecting_system") {
    return (
      <div className="space-y-3">
        {selectedAction && (
          <div className="rounded-md border border-border bg-muted/30 p-3">
            <div className="text-sm font-medium text-foreground mb-2">
              {selectedAction.action_name}
            </div>
            <div className="text-xs text-muted-foreground">
              {selectedAction.action_type} action - Select a system
            </div>
          </div>
        )}
        <SystemPicker
          inventory={actorInventory}
          onSelect={handleSystemSelect}
          onCancel={handleCancelAction}
          isOpen={true}
        />
      </div>
    );
  }

  // Show path selection UI
  if (panelState === "selecting_path") {
    // Boost grants extra movement (double speed)
    const effectiveSpeed = selectedAction?.action_id === "boost" ? actorSpeed * 2 : actorSpeed;
    const pathDistance = calculatePathDistance(movementPath);
    const remainingSpeed = effectiveSpeed - pathDistance;
    const isOverBudget = remainingSpeed < 0;

    return (
      <div className="rounded-md border border-border bg-muted/30 p-3 space-y-3">
        <div className="text-sm font-medium text-foreground">
          {selectedAction?.action_name} - Plot Movement
        </div>

        <div className="flex items-center gap-4 text-sm">
          <div>
            Distance: <span className="font-mono">{pathDistance}</span>
          </div>
          <div className={isOverBudget ? "text-destructive" : "text-muted-foreground"}>
            Remaining: <span className="font-mono">{remainingSpeed}</span>
          </div>
        </div>

        <div className="text-xs text-muted-foreground">
          Click hexes on the canvas to build path. Click last hex to undo.
        </div>

        <div className="flex gap-2">
          <Button
            variant="primary"
            size="sm"
            onClick={handleConfirmPath}
            disabled={isExecuting || movementPath.length < 2 || isOverBudget}
          >
            {isExecuting ? "Executing..." : "Confirm Path"}
          </Button>
          <Button variant="ghost" size="sm" onClick={handleClearPath}>
            Clear
          </Button>
          <Button variant="ghost" size="sm" onClick={handleCancelAction}>
            Cancel
          </Button>
        </div>
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
              {selectedSystemId && ` - ${formatSystemId(selectedSystemId)}`}
            </div>
          </div>
        )}

        {panelState === "selecting_target" && selectedAction && (
          <div className="text-xs text-muted-foreground">
            {selectedAction.max_targets > 1
              ? `Click combatants to select targets (${selectedTargetIds.length}/${selectedAction.max_targets})`
              : "Click a combatant on the canvas to select target"}
          </div>
        )}

        {selectedTargetIds.length > 0 && (
          <div className="text-xs text-primary">
            {selectedTargetIds.length === 1
              ? `Target: ${selectedTargetIds[0]}`
              : `Targets: ${selectedTargetIds.join(", ")}`}
          </div>
        )}

        <div className="flex gap-2">
          <Button
            variant="primary"
            size="sm"
            onClick={handleConfirmAction}
            disabled={isExecuting || (selectedAction?.requires_target && selectedTargetIds.length === 0)}
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
        {action.requires_system && (
          <span className="text-xs text-muted-foreground">system</span>
        )}
        {action.requires_path && (
          <span className="text-xs text-muted-foreground">path</span>
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
