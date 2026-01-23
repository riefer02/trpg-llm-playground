import { useEffect, useState } from "react";

import type {
  ActionEconomyState,
  ActionRequest,
  AvailableActionItem,
  AvailableActionsResponse,
} from "../../lib/api/combat";
import type {
  AttackPatternDefinition,
  FullTechOptionSelection,
  HexCoord,
  MechInventory,
  MechWeaponDefinition,
} from "../../lib/types/lancer";
import { calculatePathDistance, hexEquals, isAdjacent } from "../../lib/combat-render/hex";
import { Button } from "../ui";
import { SystemPicker } from "./SystemPicker";
import { WeaponPicker } from "./WeaponPicker";
import { WeaponProfilePicker } from "./WeaponProfilePicker";

type FullTechOption = FullTechOptionSelection["option"];
type FullTechStep = 1 | 2;

const FULL_TECH_OPTIONS: FullTechOption[] = ["scan", "bolster", "lock_on", "invade"];
const ATTACK_ACTION_IDS = new Set(["skirmish", "barrage", "fight"]);
const AOE_PATTERN_TYPES = new Set(["line", "cone", "blast", "burst"]);

/**
 * Extracts AoE pattern from weapon definition or selected profile.
 * Returns null if weapon has no AoE pattern.
 */
function extractAreaPattern(
  weapon: MechWeaponDefinition | undefined,
  profileId: string | null
): AttackPatternDefinition | null {
  if (!weapon) return null;

  const profiles = weapon.dynamic?.profile_choice?.profiles;
  const profile = profileId && profiles
    ? profiles.find(p => p.profile_id === profileId)
    : null;

  const ranges = profile?.ranges ?? weapon.ranges ?? [];
  const tags = profile?.tags ?? weapon.tags ?? [];

  // Check ranges first (priority) - line/cone/blast/burst are range types
  for (const range of ranges) {
    if (AOE_PATTERN_TYPES.has(range.range_type)) {
      return {
        pattern: range.range_type as AttackPatternDefinition["pattern"],
        size: range.value,
      };
    }
  }

  // Check tags (some weapons use tags like "Blast 1")
  for (const tag of tags) {
    if (AOE_PATTERN_TYPES.has(tag.tag) && tag.value != null) {
      return {
        pattern: tag.tag as AttackPatternDefinition["pattern"],
        size: tag.value,
      };
    }
  }

  return null;
}

export type ActionPanelState =
  | "idle"
  | "selecting_target"
  | "selecting_weapon"
  | "selecting_weapon_profile"
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
  /** Callback to update AoE preview on the canvas when weapon is selected */
  onAreaPreviewChange?: (
    pattern: AttackPatternDefinition | null,
    origin: HexCoord | null,
    direction: HexCoord | null
  ) => void;
  /** Callback to show/hide movement range preview */
  onMovementRangeChange?: (show: boolean, speed: number) => void;
  isExecuting?: boolean;
  selectedTargetIds?: string[];
  actorInventory?: MechInventory | null;
  weaponDefinitions?: Map<string, MechWeaponDefinition> | null;
  actorSpeed?: number;
  actorPosition?: HexCoord | null;
  /** Incoming hex click from canvas when in path mode */
  hexClickCoord?: HexCoord | null;
  /** Action triggered externally (e.g., from ActionBar) */
  triggeredAction?: AvailableActionItem | null;
  /** Callback when triggered action is processed */
  onTriggeredActionProcessed?: () => void;
}

export function ActionPanel({
  availableActions,
  economy,
  onActionSelect,
  onExecuteAction,
  onTargetModeChange,
  onPathModeChange,
  onAreaPreviewChange,
  onMovementRangeChange,
  isExecuting = false,
  selectedTargetIds = [],
  actorInventory,
  weaponDefinitions,
  actorSpeed = 4,
  actorPosition,
  hexClickCoord,
  triggeredAction,
  onTriggeredActionProcessed,
}: ActionPanelProps) {
  const [panelState, setPanelState] = useState<ActionPanelState>("idle");
  const [selectedAction, setSelectedAction] = useState<AvailableActionItem | null>(null);
  const [selectedWeaponId, setSelectedWeaponId] = useState<string | null>(null);
  const [selectedWeaponProfileId, setSelectedWeaponProfileId] = useState<string | null>(null);
  const [selectedSystemId, setSelectedSystemId] = useState<string | null>(null);
  const [useThrown, setUseThrown] = useState(false);
  const [promptDangerousTerrain, setPromptDangerousTerrain] = useState(false);
  const [movementPath, setMovementPath] = useState<HexCoord[]>([]);
  const [lastProcessedClick, setLastProcessedClick] = useState<HexCoord | null>(null);
  const [fullTechStep, setFullTechStep] = useState<FullTechStep>(1);
  const [fullTechSelections, setFullTechSelections] = useState<FullTechSelectionState>({});

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

  // Reset thrown state when weapon changes and thrown is no longer valid
  useEffect(() => {
    if (!canUseThrown && useThrown) {
      setUseThrown(false);
    }
  }, [canUseThrown, useThrown]);

  // Track last processed triggered action to avoid re-processing
  const [lastTriggeredActionId, setLastTriggeredActionId] = useState<string | null>(null);

  // Process externally triggered actions (from ActionBar)
  useEffect(() => {
    if (!triggeredAction) return;
    if (triggeredAction.action_id === lastTriggeredActionId) return;
    if (!triggeredAction.is_available) return;

    // Mark as processed
    setLastTriggeredActionId(triggeredAction.action_id);

    // Reset state for new action
    setSelectedAction(triggeredAction);
    setSelectedWeaponId(null);
    setSelectedWeaponProfileId(null);
    setSelectedSystemId(null);
    setUseThrown(false);
    setMovementPath([]);
    setFullTechStep(1);
    setFullTechSelections({});
    onActionSelect(triggeredAction);
    onAreaPreviewChange?.(null, null, null);

    // Determine initial state based on requirements
    if (triggeredAction.action_id === "full_tech") {
      setPanelState("selecting_full_tech_option");
      onTargetModeChange(null);
      onPathModeChange(false, []);
      onMovementRangeChange?.(false, 0);
    } else if (triggeredAction.requires_weapon) {
      setPanelState("selecting_weapon");
      onTargetModeChange(null);
      onPathModeChange(false, []);
      onMovementRangeChange?.(false, 0);
    } else if (triggeredAction.requires_system) {
      setPanelState("selecting_system");
      onTargetModeChange(null);
      onPathModeChange(false, []);
      onMovementRangeChange?.(false, 0);
    } else if (triggeredAction.requires_path) {
      const initialPath = actorPosition ? [actorPosition] : [];
      setMovementPath(initialPath);
      setPanelState("selecting_path");
      onTargetModeChange(null);
      onPathModeChange(true, initialPath);
      // Show movement range preview - boost doubles speed
      const effectiveSpeed = triggeredAction.action_id === "boost" ? actorSpeed * 2 : actorSpeed;
      onMovementRangeChange?.(true, effectiveSpeed);
    } else if (triggeredAction.requires_target) {
      setPanelState("selecting_target");
      onTargetModeChange({
        actionId: triggeredAction.action_id,
        actionType: triggeredAction.action_type as ActionRequest["action_type"],
        requiresTarget: triggeredAction.requires_target,
        requiresWeapon: triggeredAction.requires_weapon,
      });
      onPathModeChange(false, []);
      onMovementRangeChange?.(false, 0);
    } else {
      setPanelState("confirming");
      onTargetModeChange(null);
      onPathModeChange(false, []);
      onMovementRangeChange?.(false, 0);
    }

    // Notify parent that we processed the action
    onTriggeredActionProcessed?.();
  }, [
    triggeredAction,
    lastTriggeredActionId,
    actorPosition,
    actorSpeed,
    onActionSelect,
    onAreaPreviewChange,
    onMovementRangeChange,
    onPathModeChange,
    onTargetModeChange,
    onTriggeredActionProcessed,
  ]);

  // Reset tracked action when panel returns to idle
  useEffect(() => {
    if (panelState === "idle") {
      setLastTriggeredActionId(null);
    }
  }, [panelState]);

  // Early return AFTER all hooks are called
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

  const handleActionClick = (action: AvailableActionItem) => {
    if (!action.is_available) return;

    setSelectedAction(action);
    setSelectedWeaponId(null);
    setSelectedWeaponProfileId(null);
    setSelectedSystemId(null);
    setUseThrown(false);
    setMovementPath([]);
    setFullTechStep(1);
    setFullTechSelections({});
    onActionSelect(action);
    // Clear any existing AoE preview - it will be re-set when weapon is selected
    onAreaPreviewChange?.(null, null, null);

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
      onMovementRangeChange?.(false, 0);
    } else if (action.requires_system) {
      // System selection
      setPanelState("selecting_system");
      onTargetModeChange(null);
      onPathModeChange(false, []);
      onMovementRangeChange?.(false, 0);
    } else if (action.requires_path) {
      // Path selection (for Move, Boost)
      const initialPath = actorPosition ? [actorPosition] : [];
      setMovementPath(initialPath);
      setPanelState("selecting_path");
      onTargetModeChange(null);
      onPathModeChange(true, initialPath);
      // Show movement range preview - boost doubles speed
      const effectiveSpeed = action.action_id === "boost" ? actorSpeed * 2 : actorSpeed;
      onMovementRangeChange?.(true, effectiveSpeed);
    } else if (action.requires_target) {
      setPanelState("selecting_target");
      onTargetModeChange({
        actionId: action.action_id,
        actionType: action.action_type as ActionRequest["action_type"],
        requiresTarget: action.requires_target,
        requiresWeapon: action.requires_weapon,
      });
      onPathModeChange(false, []);
      onMovementRangeChange?.(false, 0);
    } else {
      setPanelState("confirming");
      onTargetModeChange(null);
      onPathModeChange(false, []);
      onMovementRangeChange?.(false, 0);
    }
  };

  const handleWeaponSelect = (weaponId: string) => {
    setSelectedWeaponId(weaponId);
    setUseThrown(false);
    setSelectedWeaponProfileId(null);

    // Check if weapon has multiple profiles
    const weapon = weaponDefinitions?.get(weaponId);
    const profiles = weapon?.dynamic?.profile_choice?.profiles;

    if (profiles && profiles.length > 1) {
      // Need profile selection first - don't show preview until profile is selected
      setPanelState("selecting_weapon_profile");
      onAreaPreviewChange?.(null, null, null);
      return;
    }

    // Extract area pattern for preview (single profile or no profiles)
    const pattern = extractAreaPattern(weapon, null);
    if (pattern && onAreaPreviewChange) {
      if (pattern.pattern === "burst") {
        // Burst centered on attacker - show immediately
        onAreaPreviewChange(pattern, actorPosition ?? null, null);
      } else if (pattern.pattern === "blast") {
        // Blast - show pattern def, origin set on hover in parent
        onAreaPreviewChange(pattern, null, null);
      } else {
        // Line/cone - show from actor, direction set by DirectionPicker
        onAreaPreviewChange(pattern, actorPosition ?? null, null);
      }
    } else {
      onAreaPreviewChange?.(null, null, null);
    }

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

  const handleWeaponProfileSelect = (profileId: string) => {
    setSelectedWeaponProfileId(profileId);

    // Extract area pattern for preview with selected profile
    const weapon = selectedWeaponId ? weaponDefinitions?.get(selectedWeaponId) : undefined;
    const pattern = extractAreaPattern(weapon, profileId);
    if (pattern && onAreaPreviewChange) {
      if (pattern.pattern === "burst") {
        // Burst centered on attacker - show immediately
        onAreaPreviewChange(pattern, actorPosition ?? null, null);
      } else if (pattern.pattern === "blast") {
        // Blast - show pattern def, origin set on hover in parent
        onAreaPreviewChange(pattern, null, null);
      } else {
        // Line/cone - show from actor, direction set by DirectionPicker
        onAreaPreviewChange(pattern, actorPosition ?? null, null);
      }
    } else {
      onAreaPreviewChange?.(null, null, null);
    }

    if (selectedAction?.requires_target) {
      setPanelState("selecting_target");
      onTargetModeChange({
        actionId: selectedAction.action_id,
        actionType: selectedAction.action_type as ActionRequest["action_type"],
        requiresTarget: selectedAction.requires_target,
        requiresWeapon: selectedAction.requires_weapon,
      });
    } else {
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
      weapon_profile_id: selectedWeaponProfileId ?? undefined,
      system_id: selectedSystemId ?? undefined,
      use_thrown: useThrown || undefined,
    });
  };

  const handleCancelAction = () => {
    setSelectedAction(null);
    setSelectedWeaponId(null);
    setSelectedWeaponProfileId(null);
    setSelectedSystemId(null);
    setUseThrown(false);
    setMovementPath([]);
    setFullTechStep(1);
    setFullTechSelections({});
    setPanelState("idle");
    onTargetModeChange(null);
    onPathModeChange(false, []);
    onAreaPreviewChange?.(null, null, null);
    onMovementRangeChange?.(false, 0);
  };

  // Reset state when action completes
  const resetAfterExecution = () => {
    setSelectedAction(null);
    setSelectedWeaponId(null);
    setSelectedWeaponProfileId(null);
    setSelectedSystemId(null);
    setUseThrown(false);
    setMovementPath([]);
    setFullTechStep(1);
    setFullTechSelections({});
    setPanelState("idle");
    onTargetModeChange(null);
    onPathModeChange(false, []);
    onAreaPreviewChange?.(null, null, null);
    onMovementRangeChange?.(false, 0);
  };

  // Handle confirm path for movement actions
  const handleConfirmPath = () => {
    if (!selectedAction || movementPath.length < 2) return;

    setPanelState("executing");
    onPathModeChange(false, []);
    onMovementRangeChange?.(false, 0);

    onExecuteAction({
      action_id: selectedAction.action_id,
      action_type: selectedAction.action_type as ActionRequest["action_type"],
      movement_path: movementPath.map((c) => ({ coord: { q: c.q, r: c.r } })),
      prompt_dangerous_terrain: promptDangerousTerrain || undefined,
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

  // Show weapon profile picker
  if (panelState === "selecting_weapon_profile") {
    const weapon = selectedWeaponId ? weaponDefinitions?.get(selectedWeaponId) : null;
    return (
      <div className="space-y-3">
        {selectedAction && (
          <div className="rounded-md border border-border bg-muted/30 p-3">
            <div className="text-sm font-medium text-foreground mb-2">
              {selectedAction.action_name}
            </div>
            <div className="text-xs text-muted-foreground">
              {selectedAction.action_type} action - Select a profile
            </div>
          </div>
        )}
        <WeaponProfilePicker
          weapon={weapon ?? null}
          onSelect={handleWeaponProfileSelect}
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

        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          <input
            type="checkbox"
            className="accent-primary"
            checked={promptDangerousTerrain}
            onChange={(event) => setPromptDangerousTerrain(event.target.checked)}
            disabled={isExecuting}
          />
          Prompt for dangerous terrain checks (player only)
        </label>

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
          disabled={economy.full_actions_used > 0}
          onActionClick={handleActionClick}
        />
      )}

      {/* Quick Actions */}
      {availableActions.quick_actions.length > 0 && (
        <ActionSection
          title="Quick Actions"
          actions={availableActions.quick_actions}
          disabled={economy.quick_actions_used >= 2 + (economy.overcharge_used ? 1 : 0)}
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
