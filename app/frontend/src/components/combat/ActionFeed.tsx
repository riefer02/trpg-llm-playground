import { useState, useEffect, useRef, useMemo } from "react";
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
  Flame,
  ArrowRight,
  CircleDot,
  ChevronUp,
  ChevronDown,
  type LucideIcon,
} from "lucide-react";

import type { CombatRound, ActionUse, ActionFeedEntry } from "../../lib/types/lancer";

// Icon mapping for actions
const ACTION_ICONS: Record<string, LucideIcon> = {
  move: Footprints,
  boost: Zap,
  skirmish: Swords,
  barrage: Crosshair,
  fight: Swords,
  ram: ArrowRight,
  grapple: Target,
  quick_tech: Cpu,
  full_tech: Radio,
  scan: Scan,
  lock_on: Lock,
  invade: Skull,
  bolster: Shield,
  stabilize: Heart,
  reload: RotateCcw,
  activate_system: Cpu,
  overwatch: Eye,
  brace: Shield,
  overcharge: Flame,
  hide: Eye,
  search: Scan,
  prepare: Target,
  disengage: Footprints,
};

const ACTION_VERBS: Record<string, string> = {
  move: "moved",
  boost: "boosted",
  skirmish: "attacked",
  barrage: "fired at",
  fight: "fought",
  ram: "rammed",
  grapple: "grappled",
  quick_tech: "used tech on",
  full_tech: "hacked",
  scan: "scanned",
  lock_on: "locked onto",
  invade: "invaded",
  bolster: "bolstered",
  stabilize: "stabilized",
  reload: "reloaded",
  activate_system: "activated",
  overwatch: "set overwatch",
  brace: "braced",
  overcharge: "overcharged",
  hide: "hid",
  search: "searched",
  prepare: "prepared",
  disengage: "disengaged",
};

interface FeedEntry {
  id: string;
  actorName: string;
  actorSide: "players" | "enemies";
  description: string;
  damageDealt?: number;
  statusApplied?: string[];
  icon: LucideIcon;
  timestamp: number;
}

interface ActionFeedProps {
  /** Pre-flattened action entries from the API (preferred) */
  recentActions?: ActionFeedEntry[];
  /** Total action count for "X more actions" display (from API) */
  totalActionCount?: number;
  /** @deprecated Use recentActions instead - kept for backwards compatibility */
  rounds?: CombatRound[];
  /** @deprecated Not currently used */
  currentRound?: number;
  /** @deprecated Use recentActions instead - needed only for legacy rounds prop */
  combatantNames?: Map<string, string>;
  /** @deprecated Use recentActions instead - needed only for legacy rounds prop */
  combatantSides?: Map<string, "players" | "enemies">;
  /** Max entries to show in collapsed state */
  maxVisibleEntries?: number;
  /** Whether to show expanded full history */
  expanded?: boolean;
  onToggleExpanded?: () => void;
}

/**
 * Convert a pre-flattened ActionFeedEntry from the API to our internal FeedEntry format.
 */
function convertApiEntry(entry: ActionFeedEntry): FeedEntry {
  const Icon = ACTION_ICONS[entry.action_id] ?? CircleDot;
  const verb = ACTION_VERBS[entry.action_id] ?? "used";

  // Build description from action name and targets
  let description = `${entry.actor_name} ${verb}`;
  if (entry.target_names && entry.target_names.length > 0) {
    description = `${entry.actor_name} ${verb} ${entry.target_names.join(", ")}`;
  }

  return {
    id: entry.id,
    actorName: entry.actor_name,
    actorSide: entry.actor_side === "hostiles" ? "enemies" : "players",
    description,
    damageDealt: entry.damage_dealt ?? undefined,
    statusApplied: entry.statuses_applied && entry.statuses_applied.length > 0
      ? entry.statuses_applied.map(s => s.replace(/_/g, " "))
      : undefined,
    icon: Icon,
    timestamp: entry.timestamp,
  };
}

/**
 * @deprecated Build a FeedEntry from raw ActionUse data (legacy path).
 */
function buildFeedEntry(
  action: ActionUse,
  actorId: string,
  actorName: string,
  actorSide: "players" | "enemies",
  combatantNames: Map<string, string>,
  entryId: string
): FeedEntry {
  const Icon = ACTION_ICONS[action.action_id] ?? CircleDot;
  const verb = ACTION_VERBS[action.action_id] ?? "used";

  // Build description
  let description = `${actorName} ${verb}`;

  // Handle targets
  const targetIds = action.target_ids ?? (action.target_id ? [action.target_id] : []);
  if (targetIds.length > 0) {
    const targetNames = targetIds.map((id) => combatantNames.get(id) ?? id);
    description = `${actorName} ${verb} ${targetNames.join(", ")}`;
  }

  // Handle movement
  if ((action.action_id === "move" || action.action_id === "boost") && action.target_position?.coord) {
    description = `${actorName} ${verb} to (${action.target_position.coord.q}, ${action.target_position.coord.r})`;
  }

  // Extract damage dealt from log_effects
  let damageDealt: number | undefined;
  const statusApplied: string[] = [];

  for (const effect of action.log_effects ?? []) {
    if (effect.type === "damage" && effect.amount) {
      damageDealt = (damageDealt ?? 0) + effect.amount;
    }
    if (effect.type === "status_applied" && effect.status) {
      statusApplied.push(effect.status.replace(/_/g, " "));
    }
  }

  return {
    id: entryId,
    actorName,
    actorSide,
    description,
    damageDealt,
    statusApplied: statusApplied.length > 0 ? statusApplied : undefined,
    icon: Icon,
    timestamp: Date.now(),
  };
}

export function ActionFeed({
  recentActions,
  totalActionCount,
  rounds,
  currentRound: _currentRound,
  combatantNames,
  combatantSides,
  maxVisibleEntries = 5,
  expanded = false,
  onToggleExpanded,
}: ActionFeedProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [localExpanded, setLocalExpanded] = useState(expanded);
  const isExpanded = expanded !== undefined ? expanded : localExpanded;

  // Build feed entries - prefer pre-flattened API data when available
  const allEntries = useMemo(() => {
    // Prefer the new API: pre-flattened recent_actions
    if (recentActions && recentActions.length > 0) {
      // API returns most recent first, but we want chronological order internally
      // (we reverse again in visibleEntries to show most recent first)
      return [...recentActions].reverse().map(convertApiEntry);
    }

    // Legacy fallback: build from nested rounds/turns/actions
    if (!rounds || !combatantNames || !combatantSides) {
      return [];
    }

    const entries: FeedEntry[] = [];

    for (let roundIdx = 0; roundIdx < rounds.length; roundIdx++) {
      const round = rounds[roundIdx];
      const turns = round.turns ?? [];

      for (let turnIdx = 0; turnIdx < turns.length; turnIdx++) {
        const turn = turns[turnIdx];
        const actorName = combatantNames.get(turn.actor_id) ?? turn.actor_id;
        const actorSide = combatantSides.get(turn.actor_id) ?? "enemies";
        const actions = turn.actions ?? [];

        for (let actionIdx = 0; actionIdx < actions.length; actionIdx++) {
          const action = actions[actionIdx];
          const entryId = `${roundIdx}-${turnIdx}-${actionIdx}`;
          entries.push(
            buildFeedEntry(action, turn.actor_id, actorName, actorSide, combatantNames, entryId)
          );
        }
      }
    }

    return entries;
  }, [recentActions, rounds, combatantNames, combatantSides]);

  // Get visible entries (most recent first)
  const visibleEntries = useMemo(() => {
    const reversed = [...allEntries].reverse();
    return isExpanded ? reversed : reversed.slice(0, maxVisibleEntries);
  }, [allEntries, isExpanded, maxVisibleEntries]);

  // Auto-scroll to bottom when new entries arrive
  useEffect(() => {
    if (containerRef.current && isExpanded) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [allEntries.length, isExpanded]);

  const handleToggle = () => {
    if (onToggleExpanded) {
      onToggleExpanded();
    } else {
      setLocalExpanded(!localExpanded);
    }
  };

  // Nothing to show - either no entries computed or totalActionCount is 0
  if (allEntries.length === 0 && (totalActionCount ?? 0) === 0) {
    return null;
  }

  return (
    <div className="absolute bottom-4 left-4 z-20 w-72 pointer-events-auto">
      <div className="bg-background/90 backdrop-blur-sm rounded-lg border border-border shadow-xl overflow-hidden">
        {/* Header */}
        <button
          type="button"
          onClick={handleToggle}
          className="w-full px-3 py-2 flex items-center justify-between text-xs font-medium text-muted-foreground hover:bg-muted/50 transition-colors"
        >
          <span>Action Feed</span>
          <div className="flex items-center gap-2">
            <span className="text-[10px]">
              {totalActionCount ?? allEntries.length} action{(totalActionCount ?? allEntries.length) !== 1 ? "s" : ""}
            </span>
            {isExpanded ? (
              <ChevronDown className="w-3.5 h-3.5" />
            ) : (
              <ChevronUp className="w-3.5 h-3.5" />
            )}
          </div>
        </button>

        {/* Feed entries */}
        <div
          ref={containerRef}
          className={`overflow-y-auto transition-all duration-200 ${
            isExpanded ? "max-h-64" : "max-h-40"
          }`}
        >
          <div className="px-2 pb-2 space-y-1">
            {visibleEntries.map((entry, index) => {
              const Icon = entry.icon;
              const isPlayer = entry.actorSide === "players";
              const isRecent = index < 2;

              return (
                <div
                  key={entry.id}
                  className={`flex items-start gap-2 px-2 py-1.5 rounded transition-all ${
                    isRecent
                      ? "bg-muted/60"
                      : "bg-transparent opacity-75"
                  }`}
                  style={{
                    animation: isRecent ? "slideIn 0.3s ease-out" : undefined,
                  }}
                >
                  <div
                    className={`mt-0.5 w-5 h-5 rounded flex items-center justify-center flex-shrink-0 ${
                      isPlayer
                        ? "bg-blue-500/20 text-blue-400"
                        : "bg-red-500/20 text-red-400"
                    }`}
                  >
                    <Icon className="w-3 h-3" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-xs text-foreground/90 leading-tight">
                      {entry.description}
                    </div>
                    {/* Damage dealt */}
                    {entry.damageDealt !== undefined && entry.damageDealt > 0 && (
                      <div className="text-[10px] text-red-400 font-medium">
                        {entry.damageDealt} damage
                      </div>
                    )}
                    {/* Status effects */}
                    {entry.statusApplied && entry.statusApplied.length > 0 && (
                      <div className="text-[10px] text-amber-400">
                        {entry.statusApplied.join(", ")}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Show more indicator */}
        {!isExpanded && allEntries.length > maxVisibleEntries && (
          <div className="px-3 py-1.5 text-center text-[10px] text-muted-foreground border-t border-border/50">
            + {(totalActionCount ?? allEntries.length) - maxVisibleEntries} more actions
          </div>
        )}
      </div>

      {/* CSS for slide-in animation */}
      <style>{`
        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
}
