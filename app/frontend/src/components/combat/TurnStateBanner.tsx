import { Cpu, Crosshair, Footprints, Target } from "lucide-react";
import type { ActionEconomyState } from "../../lib/api/combat";

export type TurnBannerState =
  | "your_turn"
  | "enemy_turn"
  | "select_target"
  | "select_destination"
  | "select_weapon"
  | "ai_thinking"
  | "waiting";

export interface TurnStateBannerProps {
  state: TurnBannerState;
  actorName?: string;
  mechName?: string;
  economy?: ActionEconomyState | null;
  /** Additional context like "4 spaces" for movement range */
  contextHint?: string;
  /** Whether the current actor is AI-controlled */
  isAiControlled?: boolean;
}

export function TurnStateBanner({
  state,
  actorName = "Unknown",
  mechName,
  economy,
  contextHint,
  isAiControlled = false,
}: TurnStateBannerProps) {
  // Economy pips
  const fullRemaining = economy ? 1 - economy.full_actions_used : 0;
  const quickTotal = economy ? 2 + (economy.overcharge_used ? 1 : 0) : 2;
  const quickRemaining = economy ? quickTotal - economy.quick_actions_used : quickTotal;
  const reactRemaining = economy ? 1 - economy.reactions_used_this_turn : 1;

  const renderEconomyPips = () => {
    if (!economy) return null;
    return (
      <div className="flex items-center gap-3">
        {/* Full */}
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-blue-400 uppercase font-medium">Full</span>
          <div className="w-3 h-3 rounded-full bg-blue-500" style={{ opacity: fullRemaining > 0 ? 1 : 0.2 }} />
        </div>
        {/* Quick */}
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-green-400 uppercase font-medium">Quick</span>
          <div className="flex gap-0.5">
            {Array.from({ length: quickTotal }).map((_, i) => (
              <div
                key={i}
                className="w-3 h-3 rounded-full bg-green-500"
                style={{ opacity: i < quickRemaining ? 1 : 0.2 }}
              />
            ))}
          </div>
        </div>
        {/* React */}
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-amber-400 uppercase font-medium">React</span>
          <div className="w-3 h-3 rounded-full bg-amber-500" style={{ opacity: reactRemaining > 0 ? 1 : 0.2 }} />
        </div>
      </div>
    );
  };

  // Content based on state
  switch (state) {
    case "your_turn":
      return (
        <div className="flex items-center justify-between w-full px-4 py-2 rounded-lg bg-gradient-to-r from-primary/20 to-primary/5 border border-primary/30">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center">
              <Target className="w-5 h-5 text-primary" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-primary uppercase tracking-wider">
                  Your Turn
                </span>
              </div>
              <div className="text-sm font-medium text-foreground">
                {actorName}
                {mechName && <span className="text-muted-foreground">'s {mechName}</span>}
              </div>
            </div>
          </div>
          {renderEconomyPips()}
        </div>
      );

    case "enemy_turn":
      return (
        <div className="flex items-center justify-between w-full px-4 py-2 rounded-lg bg-gradient-to-r from-destructive/20 to-destructive/5 border border-destructive/30">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-destructive/20 flex items-center justify-center">
              <Cpu className="w-5 h-5 text-destructive" />
            </div>
            <div>
              <div className="text-xs font-bold text-destructive uppercase tracking-wider">
                Enemy Turn
              </div>
              <div className="text-sm font-medium text-foreground">
                {actorName}
              </div>
            </div>
          </div>
          {isAiControlled && (
            <span className="px-2 py-1 rounded text-xs bg-destructive/20 text-destructive font-medium">
              AI Controlled
            </span>
          )}
        </div>
      );

    case "select_target":
      return (
        <div className="flex items-center justify-center w-full px-4 py-3 rounded-lg bg-gradient-to-r from-amber-500/20 to-amber-500/5 border border-amber-500/30 animate-pulse">
          <div className="flex items-center gap-3">
            <Crosshair className="w-6 h-6 text-amber-500" />
            <div>
              <div className="text-base font-bold text-amber-400">
                Select Target
              </div>
              {contextHint && (
                <div className="text-xs text-amber-400/70">
                  {contextHint}
                </div>
              )}
            </div>
          </div>
        </div>
      );

    case "select_destination":
      return (
        <div className="flex items-center justify-center w-full px-4 py-3 rounded-lg bg-gradient-to-r from-blue-500/20 to-blue-500/5 border border-blue-500/30 animate-pulse">
          <div className="flex items-center gap-3">
            <Footprints className="w-6 h-6 text-blue-500" />
            <div>
              <div className="text-base font-bold text-blue-400">
                Select Destination
              </div>
              {contextHint && (
                <div className="text-xs text-blue-400/70">
                  {contextHint}
                </div>
              )}
            </div>
          </div>
        </div>
      );

    case "select_weapon":
      return (
        <div className="flex items-center justify-center w-full px-4 py-3 rounded-lg bg-gradient-to-r from-green-500/20 to-green-500/5 border border-green-500/30">
          <div className="flex items-center gap-3">
            <Target className="w-6 h-6 text-green-500" />
            <div>
              <div className="text-base font-bold text-green-400">
                Select Weapon
              </div>
              {contextHint && (
                <div className="text-xs text-green-400/70">
                  {contextHint}
                </div>
              )}
            </div>
          </div>
        </div>
      );

    case "ai_thinking":
      return (
        <div className="flex items-center justify-center w-full px-4 py-3 rounded-lg bg-gradient-to-r from-purple-500/20 to-purple-500/5 border border-purple-500/30">
          <div className="flex items-center gap-3">
            <Cpu className="w-6 h-6 text-purple-500 animate-pulse" />
            <div>
              <div className="text-base font-bold text-purple-400">
                AI Thinking...
              </div>
              <div className="text-xs text-purple-400/70">
                {actorName} is deciding
              </div>
            </div>
          </div>
        </div>
      );

    case "waiting":
    default:
      return (
        <div className="flex items-center justify-center w-full px-4 py-2 rounded-lg bg-muted/50 border border-border">
          <div className="text-sm text-muted-foreground">
            Waiting for turn to start...
          </div>
        </div>
      );
  }
}
