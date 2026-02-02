import type { LucideIcon } from "lucide-react";
import {
  Footprints,
  Zap,
  Swords,
  Crosshair,
  Heart,
  Shield,
  Eye,
  Cpu,
  Radio,
  Scan,
  Lock,
  RotateCcw,
  Skull,
  Target,
  CircleDot,
  Flame,
  ArrowRight,
} from "lucide-react";

import type { ActionUse, CombatRound } from "../../lib/types/lancer";

export interface SelectedAction {
  roundIdx: number;
  turnIdx: number;
  actionIdx: number;
}

export interface ActionLogProps {
  rounds: CombatRound[];
  currentRound: number;
  currentTurnIndex: number;
  combatantNames: Map<string, string>;
  selectedAction: SelectedAction | null;
  onSelectAction: (roundIdx: number, turnIdx: number, actionIdx: number) => void;
}

// Map action IDs to icons and verbs
const ACTION_CONFIG: Record<string, { icon: LucideIcon; verb: string }> = {
  // Movement
  move: { icon: Footprints, verb: "moved" },
  boost: { icon: Zap, verb: "boosted" },
  // Attacks
  skirmish: { icon: Swords, verb: "attacked" },
  barrage: { icon: Crosshair, verb: "fired barrage at" },
  fight: { icon: Swords, verb: "fought" },
  ram: { icon: ArrowRight, verb: "rammed" },
  grapple: { icon: Target, verb: "grappled" },
  // Tech
  quick_tech: { icon: Cpu, verb: "used tech on" },
  full_tech: { icon: Radio, verb: "hacked" },
  scan: { icon: Scan, verb: "scanned" },
  lock_on: { icon: Lock, verb: "locked on to" },
  invade: { icon: Skull, verb: "invaded" },
  bolster: { icon: Shield, verb: "bolstered" },
  // Utility
  stabilize: { icon: Heart, verb: "stabilized" },
  reload: { icon: RotateCcw, verb: "reloaded" },
  activate_system: { icon: Cpu, verb: "activated system" },
  // Defensive
  overwatch: { icon: Eye, verb: "set overwatch" },
  brace: { icon: Shield, verb: "braced" },
  // Special
  overcharge: { icon: Flame, verb: "overcharged" },
  hide: { icon: Eye, verb: "hid" },
  search: { icon: Scan, verb: "searched" },
  prepare: { icon: Target, verb: "prepared action" },
  disengage: { icon: Footprints, verb: "disengaged" },
};

const DEFAULT_CONFIG = { icon: CircleDot, verb: "used" };

function formatPosition(pos: { coord?: { q: number; r: number } } | null | undefined): string {
  if (!pos?.coord) return "";
  return `(${pos.coord.q}, ${pos.coord.r})`;
}

function buildActionDescription(
  action: ActionUse,
  actorName: string,
  combatantNames: Map<string, string>,
): string {
  const config = ACTION_CONFIG[action.action_id] ?? DEFAULT_CONFIG;
  const verb = config.verb;

  // Handle movement actions
  if (action.action_id === "move" || action.action_id === "boost") {
    const destination = formatPosition(action.target_position);
    if (destination) {
      return `${actorName} ${verb} to ${destination}`;
    }
    return `${actorName} ${verb}`;
  }

  // Handle self-targeting actions
  if (
    action.action_id === "stabilize" ||
    action.action_id === "reload" ||
    action.action_id === "brace" ||
    action.action_id === "overcharge" ||
    action.action_id === "hide" ||
    action.action_id === "prepare" ||
    action.action_id === "disengage" ||
    action.action_id === "overwatch"
  ) {
    return `${actorName} ${verb}`;
  }

  // Handle actions with targets
  const targetIds = action.target_ids ?? (action.target_id ? [action.target_id] : []);
  if (targetIds.length > 0) {
    const targetNames = targetIds
      .map((id) => combatantNames.get(id) ?? id)
      .join(", ");
    return `${actorName} ${verb} ${targetNames}`;
  }

  // Handle actions with position targets (like area attacks)
  if (action.target_position) {
    const pos = formatPosition(action.target_position);
    return `${actorName} ${verb} at ${pos}`;
  }

  // Fallback
  return `${actorName} ${verb}`;
}

function buildStatusSummary(action: ActionUse): string[] {
  const effects = action.log_effects ?? [];
  const statuses: string[] = [];

  for (const effect of effects) {
    if (effect.type === "status_applied" && effect.status) {
      // Format status name nicely
      const statusName = effect.status
        .replace(/_/g, " ")
        .replace(/\b\w/g, (c) => c.toUpperCase());
      statuses.push(statusName);
    }
  }

  return statuses;
}

export function ActionLog({
  rounds,
  currentRound,
  currentTurnIndex,
  combatantNames,
  selectedAction,
  onSelectAction,
}: ActionLogProps) {
  if (!rounds.length) {
    return (
      <div className="text-xs text-muted-foreground italic py-2">
        No actions yet. Start your turn to begin.
      </div>
    );
  }

  // Flatten all actions into a chronological list
  const allEntries: Array<{
    type: "round" | "action";
    roundIdx: number;
    roundNumber: number;
    turnIdx?: number;
    actionIdx?: number;
    actorName?: string;
    description?: string;
    statuses?: string[];
    icon?: LucideIcon;
    isCurrent?: boolean;
  }> = [];

  for (let roundIdx = 0; roundIdx < rounds.length; roundIdx++) {
    const round = rounds[roundIdx];
    const roundNumber = round.round_index ?? roundIdx + 1;
    const isCurrent = roundNumber === currentRound;
    const turns = round.turns ?? [];

    // Add round header
    allEntries.push({
      type: "round",
      roundIdx,
      roundNumber,
      isCurrent,
    });

    // Add actions from each turn
    for (let turnIdx = 0; turnIdx < turns.length; turnIdx++) {
      const turn = turns[turnIdx];
      const actorName = combatantNames.get(turn.actor_id) ?? turn.actor_id;
      const actions = turn.actions ?? [];

      for (let actionIdx = 0; actionIdx < actions.length; actionIdx++) {
        const action = actions[actionIdx];
        const config = ACTION_CONFIG[action.action_id] ?? DEFAULT_CONFIG;
        const description = buildActionDescription(action, actorName, combatantNames);
        const statuses = buildStatusSummary(action);

        allEntries.push({
          type: "action",
          roundIdx,
          roundNumber,
          turnIdx,
          actionIdx,
          actorName,
          description,
          statuses,
          icon: config.icon,
          isCurrent: isCurrent && turnIdx === currentTurnIndex,
        });
      }
    }
  }

  // If no actions recorded yet, show a placeholder
  const hasActions = allEntries.some((e) => e.type === "action");
  if (!hasActions) {
    return (
      <div className="text-xs text-muted-foreground italic py-2">
        No actions taken yet.
      </div>
    );
  }

  return (
    <div className="space-y-1 text-xs">
      {allEntries.map((entry) => {
        if (entry.type === "round") {
          return (
            <div
              key={`round-${entry.roundIdx}`}
              className={`flex items-center gap-1.5 pt-1 ${
                entry.isCurrent ? "text-primary font-medium" : "text-muted-foreground"
              }`}
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  entry.isCurrent ? "bg-primary" : "bg-muted-foreground/40"
                }`}
              />
              <span>Round {entry.roundNumber}</span>
            </div>
          );
        }

        // Action entry
        const isSelected =
          selectedAction?.roundIdx === entry.roundIdx &&
          selectedAction?.turnIdx === entry.turnIdx &&
          selectedAction?.actionIdx === entry.actionIdx;

        const Icon = entry.icon ?? CircleDot;

        return (
          <button
            key={`action-${entry.roundIdx}-${entry.turnIdx}-${entry.actionIdx}`}
            type="button"
            onClick={() =>
              onSelectAction(entry.roundIdx, entry.turnIdx!, entry.actionIdx!)
            }
            className={`w-full text-left pl-3 pr-2 py-1 rounded flex items-start gap-2 transition-colors ${
              isSelected
                ? "bg-primary/15 text-primary"
                : "text-foreground/80 hover:bg-muted/60 hover:text-foreground"
            }`}
          >
            <Icon className="w-3.5 h-3.5 mt-0.5 flex-shrink-0 text-muted-foreground" />
            <div className="flex-1 min-w-0">
              <div className="truncate">{entry.description}</div>
              {entry.statuses && entry.statuses.length > 0 && (
                <div className="text-[10px] text-amber-500/80 truncate">
                  → {entry.statuses.join(", ")}
                </div>
              )}
            </div>
          </button>
        );
      })}
    </div>
  );
}
