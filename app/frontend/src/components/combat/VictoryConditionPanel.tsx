import type { SitrepResolution, SitrepVictoryCondition } from "../../lib/types/lancer";

export interface VictoryConditionPanelProps {
  sitrepResolution: SitrepResolution | null | undefined;
}

const CONDITION_TYPE_LABELS: Record<string, string> = {
  control_zones: "Control Zones",
  extract_objective: "Extraction",
  survive_rounds: "Survive",
  score_above_threshold: "Score Target",
  outnumber_enemies: "Outnumber",
  eliminate_target: "Eliminate Target",
  protect_target: "Protect Target",
};

export function VictoryConditionPanel({ sitrepResolution }: VictoryConditionPanelProps) {
  if (!sitrepResolution) {
    return null;
  }

  const { victory_conditions, outcome } = sitrepResolution;
  const current_round = sitrepResolution.current_round ?? 1;
  const max_rounds = sitrepResolution.max_rounds ?? 6;
  const player_score = sitrepResolution.player_score ?? 0;
  const enemy_score = sitrepResolution.enemy_score ?? 0;

  if (!victory_conditions || victory_conditions.length === 0) {
    return null;
  }

  return (
    <div className="rounded-md border border-border bg-muted/30 p-3 space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-sm font-medium text-foreground">Victory Conditions</div>
        {outcome && (
          <span
            className={`px-2 py-0.5 rounded text-xs font-medium ${
              outcome === "players_win"
                ? "bg-green-500/20 text-green-500"
                : outcome === "enemies_win"
                  ? "bg-destructive/20 text-destructive"
                  : outcome === "draw"
                    ? "bg-amber-500/20 text-amber-500"
                    : "bg-muted text-muted-foreground"
            }`}
          >
            {outcome === "players_win"
              ? "Victory"
              : outcome === "enemies_win"
                ? "Defeat"
                : outcome === "draw"
                  ? "Draw"
                  : "Ongoing"}
          </span>
        )}
      </div>

      {/* Round Progress */}
      <div className="flex items-center gap-3 text-xs">
        <div className="text-muted-foreground">
          Round {current_round}/{max_rounds}
        </div>
        <div className="flex-1 h-1 bg-muted rounded-full overflow-hidden">
          <div
            className="h-full bg-primary transition-all"
            style={{ width: `${(current_round / max_rounds) * 100}%` }}
          />
        </div>
      </div>

      {/* Score Display (if applicable) */}
      {(player_score > 0 || enemy_score > 0) && (
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-2">
            <span className="text-primary font-medium">{player_score}</span>
            <span className="text-muted-foreground">Players</span>
          </div>
          <div className="text-muted-foreground">vs</div>
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground">Enemies</span>
            <span className="text-destructive font-medium">{enemy_score}</span>
          </div>
        </div>
      )}

      {/* Victory Conditions List */}
      <div className="space-y-2">
        {victory_conditions.map((vc, index) => (
          <VictoryConditionItem key={`${vc.condition_type}-${index}`} condition={vc} />
        ))}
      </div>
    </div>
  );
}

interface VictoryConditionItemProps {
  condition: SitrepVictoryCondition;
}

function VictoryConditionItem({ condition }: VictoryConditionItemProps) {
  const { condition_type, description, is_met } = condition;
  const current_value = condition.current_value ?? 0;
  const target_value = condition.target_value;

  const label = CONDITION_TYPE_LABELS[condition_type] || condition_type;
  const progress = target_value ? Math.min((current_value / target_value) * 100, 100) : 0;

  return (
    <div
      className={`rounded-md border px-3 py-2 ${
        is_met
          ? "border-green-500/40 bg-green-500/10"
          : "border-border/60 bg-background"
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              is_met ? "bg-green-500" : "bg-muted-foreground"
            }`}
          />
          <span className="text-sm font-medium text-foreground">{label}</span>
        </div>
        {target_value && (
          <span className="text-xs text-muted-foreground">
            {current_value}/{target_value}
          </span>
        )}
      </div>
      <div className="text-xs text-muted-foreground mt-1">{description}</div>
      {target_value && (
        <div className="mt-2 h-1 bg-muted rounded-full overflow-hidden">
          <div
            className={`h-full transition-all ${is_met ? "bg-green-500" : "bg-primary"}`}
            style={{ width: `${progress}%` }}
          />
        </div>
      )}
    </div>
  );
}
