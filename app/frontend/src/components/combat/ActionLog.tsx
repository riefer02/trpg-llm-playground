import type { LucideIcon } from "lucide-react";
import {
  AlertTriangle,
  Anchor,
  ArrowDown,
  Ban,
  Bomb,
  Bot,
  CircleDot,
  Crosshair,
  EyeOff,
  Ghost,
  Package,
  Power,
  Scissors,
  Send,
  Shield,
  Snail,
  Sun,
  Swords,
  Undo2,
  Zap,
} from "lucide-react";

import {
  resolveActionLogEffectIcon,
  type IconName,
} from "../../lib/combat-effects";
import type { ActionLogEffect, ActionUse, CombatRound } from "../../lib/types/lancer";

const EFFECT_ICON_COMPONENTS: Record<IconName, LucideIcon> = {
  send: Send,
  undo2: Undo2,
  crosshair: Crosshair,
  alertTriangle: AlertTriangle,
  scissors: Scissors,
  ban: Ban,
  sun: Sun,
  anchor: Anchor,
  snail: Snail,
  zap: Zap,
  eyeOff: EyeOff,
  ghost: Ghost,
  arrowDown: ArrowDown,
  shield: Shield,
  swords: Swords,
  power: Power,
  circleDot: CircleDot,
  bomb: Bomb,
  bot: Bot,
  package: Package,
};

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
      <div className="text-sm text-muted-foreground">No rounds recorded yet.</div>
    );
  }

  return (
    <div className="space-y-1 text-sm">
      {rounds.map((round, roundIdx) => {
        const roundNumber = round.round_index ?? roundIdx + 1;
        const isCurrent = roundNumber === currentRound;
        const turns = round.turns ?? [];

        return (
          <div key={roundIdx}>
            <div
              className={`flex items-center gap-2 py-1 ${
                isCurrent ? "text-primary font-medium" : "text-foreground"
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  isCurrent ? "bg-primary" : "bg-muted-foreground/40"
                }`}
              />
              Round {roundNumber}
              {isCurrent && (
                <span className="text-xs text-muted-foreground">(current)</span>
              )}
            </div>

            <div className="ml-4 border-l border-border pl-3 space-y-0.5">
              {turns.map((turn, turnIdx) => {
                const actorName =
                  combatantNames.get(turn.actor_id) ?? turn.actor_id;
                const actions = turn.actions ?? [];
                const isCurrentTurn = isCurrent && turnIdx === currentTurnIndex;

                return (
                  <div key={turnIdx}>
                    <div
                      className={`py-0.5 ${
                        isCurrentTurn
                          ? "text-primary/90 font-medium"
                          : "text-muted-foreground"
                      }`}
                    >
                      Turn {turnIdx + 1} · {actorName}
                      {isCurrentTurn && (
                        <span className="text-xs ml-1">(active)</span>
                      )}
                    </div>

                    {actions.length > 0 && (
                      <div className="ml-3 space-y-0.5">
                        {actions.map((action, actionIdx) => {
                          const isSelected =
                            selectedAction?.roundIdx === roundIdx &&
                            selectedAction?.turnIdx === turnIdx &&
                            selectedAction?.actionIdx === actionIdx;

                          return (
                            <ActionItem
                              key={actionIdx}
                              action={action}
                              isSelected={isSelected}
                              onClick={() =>
                                onSelectAction(roundIdx, turnIdx, actionIdx)
                              }
                            />
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })}
              {!turns.length && (
                <div className="py-0.5 text-muted-foreground/60 text-xs">
                  No turns yet
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ActionItem({
  action,
  isSelected,
  onClick,
}: {
  action: ActionUse;
  isSelected: boolean;
  onClick: () => void;
}) {
  const effects = dedupeActionEffects(action.log_effects ?? []);
  return (
    <button
      type="button"
      onClick={onClick}
      className={`w-full text-left px-2 py-0.5 rounded text-xs transition-colors ${
        isSelected
          ? "bg-primary/15 text-primary"
          : "text-muted-foreground hover:bg-muted/60 hover:text-foreground"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="truncate">
          {action.action_id} ({action.action_type})
        </span>
        {effects.length > 0 && (
          <span className="flex items-center gap-1">
            {effects.map((effect, index) => (
              <EffectIcon key={`${effect.type}-${effect.status ?? index}`} effect={effect} />
            ))}
          </span>
        )}
      </div>
    </button>
  );
}

function dedupeActionEffects(effects: ActionLogEffect[]): ActionLogEffect[] {
  const seen = new Set<string>();
  const result: ActionLogEffect[] = [];
  for (const effect of effects) {
    const key =
      effect.type === "status_applied"
        ? `${effect.type}:${effect.status ?? "unknown"}`
        : effect.type;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    result.push(effect);
  }
  return result;
}

function EffectIcon({ effect }: { effect: ActionLogEffect }) {
  const config = resolveActionLogEffectIcon(effect);
  const Icon = EFFECT_ICON_COMPONENTS[config.icon] ?? CircleDot;
  return (
    <span
      className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-border/70 bg-background/80"
      title={config.label}
      style={{ color: config.color }}
    >
      <Icon className="h-3 w-3" />
    </span>
  );
}
