import { useEffect, useState } from "react";

import type {
  ActionEconomyState,
  ActionRequest,
  AvailableActionItem,
  AvailableActionsResponse,
} from "../../lib/api/combat";
import type {
  FullTechOptionSelection,
  HexCoord,
  MechInventory,
  MechWeaponDefinition,
} from "../../lib/types/lancer";
import { calculatePathDistance, hexEquals, isAdjacent } from "../../lib/combat-render/hex";
import { Button } from "../ui";
import { SystemPicker } from "./SystemPicker";
import { WeaponPicker } from "./WeaponPicker";

type FullTechOption = FullTechOptionSelection["option"];
type FullTechStep = 1 | 2;

const FULL_TECH_OPTIONS: FullTechOption[] = ["scan", "bolster", "lock_on", "invade"];
const ATTACK_ACTION_IDS = new Set(["skirmish", "barrage", "fight"]);

export type ActionPanelState =
  | "idle"
  | "selecting_target"
  | "selecting_weapon"
  | "selecting_system"
  | "selecting_path"
  | "selecting_full_tech_option"
  | "selecting_full_tech_target"
  | "confirming_full_tech"
  | "confirming"
  | "executing";

interface FullTechSelectionState {
  firstOption?: FullTechOption;
  firstTargetId?: string;
  secondOption?: FullTechOption;
  secondTargetId?: string;
}

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
  weaponDefinitions?: Map<string, MechWeaponDefinition> | null;
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
  weaponDefinitions,
  actorSpeed = 4,
  actorPosition,
  hexClickCoord,
}: ActionPanelProps) {
  const [panelState, setPanelState] = useState<ActionPanelState>("idle");
  const [selectedAction, setSelectedAction] = useState<AvailableActionItem | null>(null);
  const [selectedWeaponId, setSelectedWeaponId] = useState<string | null>(null);
  const [selectedSystemId, setSelectedSystemId] = useState<string | null>(null);
  const [useThrown, setUseThrown] = useState(false);
  const [movementPath, setMovementPath] = useState<HexCoord[]>([]);
  const [lastProcessedClick, setLastProcessedClick] = useState<HexCoord | null>(null);
  const [fullTechStep, setFullTechStep] = useState<FullTechStep>(1);
  const [fullTechSelections, setFullTechSelections] = useState<FullTechSelectionState>({});

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

  const techOptions = FULL_TECH_OPTIONS.map((option) => ({
    id: option,
    name:
      availableActions.quick_actions.find((action) => action.action_id === option)?.action_name ??
      formatTechOption(option),
  }));

  const selectedWeaponDefinition =
    selectedWeaponId && weaponDefinitions ? weaponDefinitions.get(selectedWeaponId) : undefined;
  const canUseThrown = Boolean(
    selectedAction &&
      ATTACK_ACTION_IDS.has(selectedAction.action_id) &&
      selectedWeaponDefinition &&
      isMeleeWeapon(selectedWeaponDefinition) &&
      getThrownRange(selectedWeaponDefinition) !== null
  );
  const thrownRange = selectedWeaponDefinition
    ? getThrownRange(selectedWeaponDefinition)
    : null;

  useEffect(() => {
    if (!canUseThrown && useThrown) {
      setUseThrown(false);
    }
  }, [canUseThrown, useThrown]);

  const handleActionClick = (action: AvailableActionItem) => {
    if (!action.is_available) return;

    setSelectedAction(action);
    setSelectedWeaponId(null);
    setSelectedSystemId(null);
    setUseThrown(false);
    setMovementPath([]);
    setFullTechStep(1);
    setFullTechSelections({});
    onActionSelect(action);

    if (action.action_id === "full_tech") {
      setPanelState("selecting_full_tech_option");
      onTargetModeChange(null);
      onPathModeChange(false, []);
      return;
    }

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
    setUseThrown(false);

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

  const handleFullTechOptionSelect = (option: FullTechOption) => {
    if (fullTechStep === 1) {
      setFullTechSelections((prev) => ({ ...prev, firstOption: option }));
    } else {
      setFullTechSelections((prev) => ({ ...prev, secondOption: option }));
    }

    setPanelState("selecting_full_tech_target");
    onTargetModeChange({
      actionId: option,
      actionType: "full",
      requiresTarget: true,
      requiresWeapon: false,
    });
    onPathModeChange(false, []);
  };

  const handleFullTechTargetConfirm = () => {
    const targetId = selectedTargetIds[0];
    if (!targetId) return;

    if (fullTechStep === 1) {
      setFullTechSelections((prev) => ({ ...prev, firstTargetId: targetId }));
      setFullTechStep(2);
      setPanelState("selecting_full_tech_option");
      onTargetModeChange(null);
    } else {
      setFullTechSelections((prev) => ({ ...prev, secondTargetId: targetId }));
      setPanelState("confirming_full_tech");
      onTargetModeChange(null);
    }
  };

  const handleConfirmFullTech = () => {
    if (!selectedAction || selectedAction.action_id !== "full_tech") return;
    if (
      !fullTechSelections.firstOption ||
      !fullTechSelections.firstTargetId ||
      !fullTechSelections.secondOption ||
      !fullTechSelections.secondTargetId
    ) {
      return;
    }

    setPanelState("executing");
    onExecuteAction({
      action_id: selectedAction.action_id,
      action_type: selectedAction.action_type as ActionRequest["action_type"],
      full_tech_first: {
        option: fullTechSelections.firstOption,
        target_id: fullTechSelections.firstTargetId,
      },
      full_tech_second: {
        option: fullTechSelections.secondOption,
        target_id: fullTechSelections.secondTargetId,
      },
    });
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
      use_thrown: useThrown || undefined,
    });
  };

  const handleCancelAction = () => {
    setSelectedAction(null);
    setSelectedWeaponId(null);
    setSelectedSystemId(null);
    setUseThrown(false);
    setMovementPath([]);
    setFullTechStep(1);
    setFullTechSelections({});
    setPanelState("idle");
    onTargetModeChange(null);
    onPathModeChange(false, []);
  };

  // Reset state when action completes
  const resetAfterExecution = () => {
    setSelectedAction(null);
    setSelectedWeaponId(null);
    setSelectedSystemId(null);
    setUseThrown(false);
    setMovementPath([]);
    setFullTechStep(1);
    setFullTechSelections({});
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

  if (panelState === "selecting_full_tech_option") {
    return (
      <div className="rounded-md border border-border bg-muted/30 p-3 space-y-3">
        <div className="text-sm font-medium text-foreground">
          Full Tech - Choose {fullTechStep === 1 ? "First" : "Second"} Option
        </div>

        {fullTechStep === 2 && fullTechSelections.firstOption && fullTechSelections.firstTargetId && (
          <div className="text-xs text-muted-foreground">
            First: {formatTechOption(fullTechSelections.firstOption)} → {fullTechSelections.firstTargetId}
          </div>
        )}

        <div className="grid gap-2">
          {techOptions.map((option) => (
            <Button
              key={option.id}
              variant="outline"
              size="sm"
              onClick={() => handleFullTechOptionSelect(option.id)}
            >
              {option.name}
            </Button>
          ))}
        </div>

        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={handleCancelAction}>
            Cancel
          </Button>
        </div>
      </div>
    );
  }

  if (panelState === "selecting_full_tech_target") {
    const currentOption =
      fullTechStep === 1 ? fullTechSelections.firstOption : fullTechSelections.secondOption;

    return (
      <div className="rounded-md border border-border bg-muted/30 p-3 space-y-3">
        <div className="text-sm font-medium text-foreground">
          Select Target for {currentOption ? formatTechOption(currentOption) : "Tech Option"}
        </div>

        <div className="text-xs text-muted-foreground">
          Click a combatant on the canvas to select target
        </div>

        {selectedTargetIds.length > 0 && (
          <div className="text-xs text-primary">
            Target: {selectedTargetIds[0]}
          </div>
        )}

        <div className="flex gap-2">
          <Button
            variant="primary"
            size="sm"
            onClick={handleFullTechTargetConfirm}
            disabled={isExecuting || selectedTargetIds.length === 0}
          >
            Confirm Target
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setPanelState("selecting_full_tech_option");
              onTargetModeChange(null);
            }}
          >
            Back
          </Button>
          <Button variant="ghost" size="sm" onClick={handleCancelAction}>
            Cancel
          </Button>
        </div>
      </div>
    );
  }

  if (panelState === "confirming_full_tech") {
    const isReady =
      fullTechSelections.firstOption &&
      fullTechSelections.firstTargetId &&
      fullTechSelections.secondOption &&
      fullTechSelections.secondTargetId;

    return (
      <div className="rounded-md border border-border bg-muted/30 p-3 space-y-3">
        <div className="text-sm font-medium text-foreground">
          Confirm Full Tech
        </div>

        <div className="text-xs text-muted-foreground">
          First: {fullTechSelections.firstOption ? formatTechOption(fullTechSelections.firstOption) : "--"} →{" "}
          {fullTechSelections.firstTargetId ?? "--"}
        </div>
        <div className="text-xs text-muted-foreground">
          Second: {fullTechSelections.secondOption ? formatTechOption(fullTechSelections.secondOption) : "--"} →{" "}
          {fullTechSelections.secondTargetId ?? "--"}
        </div>

        <div className="flex gap-2">
          <Button
            variant="primary"
            size="sm"
            onClick={handleConfirmFullTech}
            disabled={isExecuting || !isReady}
          >
            {isExecuting ? "Executing..." : "Execute"}
          </Button>
          <Button variant="ghost" size="sm" onClick={handleCancelAction} disabled={isExecuting}>
            Cancel
          </Button>
        </div>
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

        {selectedWeaponId && canUseThrown && (
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            <input
              type="checkbox"
              className="accent-primary"
              checked={useThrown}
              onChange={(event) => setUseThrown(event.target.checked)}
              disabled={isExecuting}
            />
            Throw weapon{thrownRange ? ` (Range ${thrownRange})` : ""} - disarms until retrieved
          </label>
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

function formatTechOption(optionId: string): string {
  return optionId
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
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

function isMeleeWeapon(weapon: MechWeaponDefinition): boolean {
  return Boolean(weapon.ranges?.some((range) => range.range_type === "threat"));
}

function getThrownRange(weapon: MechWeaponDefinition): number | null {
  const thrownValues: number[] = [];
  for (const range of weapon.ranges ?? []) {
    if (range.range_type === "thrown") {
      thrownValues.push(range.value);
    }
  }
  for (const tag of weapon.tags ?? []) {
    if (tag.tag === "thrown" && tag.value !== undefined && tag.value !== null) {
      thrownValues.push(tag.value);
    }
  }
  if (!thrownValues.length) {
    return null;
  }
  return Math.max(...thrownValues);
}
