import { useEffect, useRef } from "react";
import {
  Footprints,
  Zap,
  Swords,
  Crosshair,
  ArrowBigRight,
  Grip,
  Cpu,
  Scan,
  Lock,
  Skull,
  Shield,
  Info,
  Eye,
  type LucideIcon,
} from "lucide-react";

import type { AvailableActionItem, AvailableActionsResponse } from "../../lib/api/combat";
import type { HexCoord } from "../../lib/types/lancer";

/**
 * Context menu for right-click interactions on the combat canvas.
 * Shows contextual actions based on what was clicked.
 */

// Icon mapping for menu items
const MENU_ICONS: Record<string, LucideIcon> = {
  move: Footprints,
  boost: Zap,
  skirmish: Swords,
  barrage: Crosshair,
  ram: ArrowBigRight,
  grapple: Grip,
  quick_tech: Cpu,
  scan: Scan,
  lock_on: Lock,
  invade: Skull,
  bolster: Shield,
  info: Info,
  view: Eye,
};

const DEFAULT_ICON = Info;

export type ContextMenuTarget =
  | { type: "empty"; coord: HexCoord }
  | { type: "enemy"; combatantId: string; combatantName: string; coord: HexCoord }
  | { type: "friendly"; combatantId: string; combatantName: string; coord: HexCoord }
  | { type: "deployable"; deployableId: string; deployableName: string; coord: HexCoord };

export interface ContextMenuOption {
  id: string;
  label: string;
  icon?: string;
  disabled?: boolean;
  disabledReason?: string;
  action?: AvailableActionItem;
}

export interface ContextMenuProps {
  /** Screen position to render the menu */
  position: { x: number; y: number };
  /** What was right-clicked */
  target: ContextMenuTarget;
  /** Available actions for the current turn */
  availableActions: AvailableActionsResponse | null;
  /** Current actor's position (for distance calculations) */
  actorPosition?: HexCoord | null;
  /** Callback when an option is selected */
  onSelect: (option: ContextMenuOption) => void;
  /** Callback to close the menu */
  onClose: () => void;
  /** Whether the turn is active (controls can be used) */
  isTurnActive: boolean;
}

/**
 * Calculate hex distance between two coordinates
 */
function hexDistance(a: HexCoord, b: HexCoord): number {
  return (Math.abs(a.q - b.q) + Math.abs(a.q + a.r - b.q - b.r) + Math.abs(a.r - b.r)) / 2;
}

export function ContextMenu({
  position,
  target,
  availableActions,
  actorPosition,
  onSelect,
  onClose,
  isTurnActive,
}: ContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    };

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };

    // Delay to avoid immediate close from the triggering right-click
    const timer = setTimeout(() => {
      document.addEventListener("mousedown", handleClickOutside);
      document.addEventListener("keydown", handleEscape);
    }, 10);

    return () => {
      clearTimeout(timer);
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [onClose]);

  // Build menu options based on target type
  const options = buildMenuOptions(target, availableActions, actorPosition, isTurnActive);

  // Calculate distance to target for display
  const distance = actorPosition ? hexDistance(actorPosition, target.coord) : null;

  // Adjust position to keep menu on screen
  const adjustedPosition = { ...position };
  if (typeof window !== "undefined") {
    const menuWidth = 200;
    const menuHeight = Math.min(options.length * 40 + 60, 300);
    if (position.x + menuWidth > window.innerWidth) {
      adjustedPosition.x = window.innerWidth - menuWidth - 10;
    }
    if (position.y + menuHeight > window.innerHeight) {
      adjustedPosition.y = window.innerHeight - menuHeight - 10;
    }
  }

  return (
    <div
      ref={menuRef}
      className="fixed z-50 bg-popover border border-border rounded-lg shadow-xl overflow-hidden min-w-[180px] max-w-[240px] animate-in fade-in zoom-in-95 duration-100"
      style={{ left: adjustedPosition.x, top: adjustedPosition.y }}
    >
      {/* Header */}
      <div className="px-3 py-2 border-b border-border bg-muted/50">
        <div className="font-medium text-sm text-foreground truncate">
          {getTargetLabel(target)}
        </div>
        <div className="text-xs text-muted-foreground">
          {formatCoord(target.coord)}
          {distance !== null && ` · ${distance} hex${distance !== 1 ? "es" : ""}`}
        </div>
      </div>

      {/* Options */}
      <div className="py-1 max-h-[240px] overflow-y-auto">
        {options.length === 0 ? (
          <div className="px-3 py-2 text-sm text-muted-foreground">
            No actions available
          </div>
        ) : (
          options.map((option) => {
            const Icon = MENU_ICONS[option.icon ?? ""] ?? DEFAULT_ICON;
            return (
              <button
                key={option.id}
                type="button"
                onClick={() => {
                  if (!option.disabled) {
                    onSelect(option);
                    onClose();
                  }
                }}
                disabled={option.disabled}
                className={`
                  w-full flex items-center gap-2 px-3 py-2 text-sm text-left transition-colors
                  ${option.disabled
                    ? "text-muted-foreground/50 cursor-not-allowed"
                    : "text-foreground hover:bg-primary/10 cursor-pointer"
                  }
                `}
              >
                <Icon className={`w-4 h-4 flex-shrink-0 ${option.disabled ? "opacity-50" : ""}`} />
                <div className="flex-1 min-w-0">
                  <div className="truncate">{option.label}</div>
                  {option.disabledReason && (
                    <div className="text-xs text-muted-foreground/70 truncate">
                      {option.disabledReason}
                    </div>
                  )}
                </div>
              </button>
            );
          })
        )}
      </div>
    </div>
  );
}

function getTargetLabel(target: ContextMenuTarget): string {
  switch (target.type) {
    case "empty":
      return "Empty Hex";
    case "enemy":
      return target.combatantName;
    case "friendly":
      return target.combatantName;
    case "deployable":
      return target.deployableName;
  }
}

function formatCoord(coord: HexCoord): string {
  return `(${coord.q}, ${coord.r})`;
}

function buildMenuOptions(
  target: ContextMenuTarget,
  availableActions: AvailableActionsResponse | null,
  actorPosition: HexCoord | null | undefined,
  isTurnActive: boolean
): ContextMenuOption[] {
  const options: ContextMenuOption[] = [];
  const allActions = [
    ...(availableActions?.full_actions ?? []),
    ...(availableActions?.quick_actions ?? []),
    ...(availableActions?.free_actions ?? []),
  ];

  const findAction = (id: string) => allActions.find((a) => a.action_id === id);

  switch (target.type) {
    case "empty": {
      // Move/Boost to empty hex
      const moveAction = findAction("move");
      const boostAction = findAction("boost");

      if (moveAction) {
        options.push({
          id: "move",
          label: "Move Here",
          icon: "move",
          disabled: !isTurnActive || !moveAction.is_available,
          disabledReason: !isTurnActive
            ? "Not your turn"
            : moveAction.unavailable_reason,
          action: moveAction,
        });
      }

      if (boostAction) {
        options.push({
          id: "boost",
          label: "Boost Here",
          icon: "boost",
          disabled: !isTurnActive || !boostAction.is_available,
          disabledReason: !isTurnActive
            ? "Not your turn"
            : boostAction.unavailable_reason,
          action: boostAction,
        });
      }

      // Always show hex info option
      options.push({
        id: "view_hex",
        label: "View Hex Info",
        icon: "info",
        disabled: false,
      });
      break;
    }

    case "enemy": {
      // Attack options
      const skirmishAction = findAction("skirmish");
      const barrageAction = findAction("barrage");

      if (skirmishAction) {
        options.push({
          id: "skirmish",
          label: "Skirmish",
          icon: "skirmish",
          disabled: !isTurnActive || !skirmishAction.is_available,
          disabledReason: !isTurnActive
            ? "Not your turn"
            : skirmishAction.unavailable_reason,
          action: skirmishAction,
        });
      }

      if (barrageAction) {
        options.push({
          id: "barrage",
          label: "Barrage",
          icon: "barrage",
          disabled: !isTurnActive || !barrageAction.is_available,
          disabledReason: !isTurnActive
            ? "Not your turn"
            : barrageAction.unavailable_reason,
          action: barrageAction,
        });
      }

      // Tech options
      const scanAction = findAction("scan");
      const lockOnAction = findAction("lock_on");
      const invadeAction = findAction("invade");

      if (scanAction) {
        options.push({
          id: "scan",
          label: "Scan",
          icon: "scan",
          disabled: !isTurnActive || !scanAction.is_available,
          disabledReason: !isTurnActive
            ? "Not your turn"
            : scanAction.unavailable_reason,
          action: scanAction,
        });
      }

      if (lockOnAction) {
        options.push({
          id: "lock_on",
          label: "Lock On",
          icon: "lock_on",
          disabled: !isTurnActive || !lockOnAction.is_available,
          disabledReason: !isTurnActive
            ? "Not your turn"
            : lockOnAction.unavailable_reason,
          action: lockOnAction,
        });
      }

      if (invadeAction) {
        options.push({
          id: "invade",
          label: "Invade",
          icon: "invade",
          disabled: !isTurnActive || !invadeAction.is_available,
          disabledReason: !isTurnActive
            ? "Not your turn"
            : invadeAction.unavailable_reason,
          action: invadeAction,
        });
      }

      // Melee options (check adjacency)
      const isAdjacent = actorPosition
        ? hexDistance(actorPosition, target.coord) === 1
        : false;

      const ramAction = findAction("ram");
      const grappleAction = findAction("grapple");

      if (ramAction) {
        options.push({
          id: "ram",
          label: "Ram",
          icon: "ram",
          disabled: !isTurnActive || !ramAction.is_available || !isAdjacent,
          disabledReason: !isTurnActive
            ? "Not your turn"
            : !isAdjacent
              ? "Not adjacent"
              : ramAction.unavailable_reason,
          action: ramAction,
        });
      }

      if (grappleAction) {
        options.push({
          id: "grapple",
          label: "Grapple",
          icon: "grapple",
          disabled: !isTurnActive || !grappleAction.is_available || !isAdjacent,
          disabledReason: !isTurnActive
            ? "Not your turn"
            : !isAdjacent
              ? "Not adjacent"
              : grappleAction.unavailable_reason,
          action: grappleAction,
        });
      }

      // View enemy info
      options.push({
        id: "view_enemy",
        label: "View Info",
        icon: "view",
        disabled: false,
      });
      break;
    }

    case "friendly": {
      // Bolster
      const bolsterAction = findAction("bolster");
      if (bolsterAction) {
        options.push({
          id: "bolster",
          label: "Bolster",
          icon: "bolster",
          disabled: !isTurnActive || !bolsterAction.is_available,
          disabledReason: !isTurnActive
            ? "Not your turn"
            : bolsterAction.unavailable_reason,
          action: bolsterAction,
        });
      }

      // View friendly info
      options.push({
        id: "view_friendly",
        label: "View Stats",
        icon: "view",
        disabled: false,
      });
      break;
    }

    case "deployable": {
      // Attack/destroy deployable
      const skirmishAction = findAction("skirmish");
      if (skirmishAction) {
        options.push({
          id: "attack_deployable",
          label: "Attack",
          icon: "skirmish",
          disabled: !isTurnActive || !skirmishAction.is_available,
          disabledReason: !isTurnActive
            ? "Not your turn"
            : skirmishAction.unavailable_reason,
          action: skirmishAction,
        });
      }

      // View deployable info
      options.push({
        id: "view_deployable",
        label: "View Info",
        icon: "view",
        disabled: false,
      });
      break;
    }
  }

  return options;
}
