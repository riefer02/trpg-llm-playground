import type { ActionEconomyState } from "../../lib/api/combat";
import { getOverchargeCost } from "./OverchargeConfirm";

export interface EconomyDisplayProps {
  economy: ActionEconomyState;
  canOvercharge: boolean;
  overchargeLevel?: number;
  /** When true, shows greyed-out state (not your turn) */
  disabled?: boolean;
}

export function EconomyDisplay({
  economy,
  canOvercharge,
  overchargeLevel = 0,
  disabled = false,
}: EconomyDisplayProps) {
  // Base quick actions is 2, +1 if overcharged
  const baseQuickActions = 2;
  const bonusFromOvercharge = economy.overcharge_used ? 1 : 0;
  const quickSlotsTotal = baseQuickActions + bonusFromOvercharge;
  const quickSlotsRemaining = Math.max(0, quickSlotsTotal - economy.quick_actions_used);

  const fullActionAvailable = economy.full_actions_used === 0;
  const reactionAvailable = economy.reactions_used_this_turn === 0;

  return (
    <div className={`rounded-md border p-3 space-y-2 transition-opacity ${
      disabled
        ? "border-border/50 bg-muted/10 opacity-50"
        : "border-border bg-muted/30"
    }`}>
      <div className="flex items-center justify-between">
        <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
          Action Economy
        </div>
        {disabled && (
          <span className="text-[10px] text-muted-foreground/60">Not your turn</span>
        )}
      </div>

      <div className="flex items-center gap-3">
        {/* Full Action */}
        <div className="flex items-center gap-1.5">
          <ActionSlot
            filled={fullActionAvailable}
            label="Full"
            variant="full"
          />
        </div>

        {/* Quick Actions */}
        <div className="flex items-center gap-1">
          {Array.from({ length: quickSlotsTotal }).map((_, idx) => (
            <ActionSlot
              key={idx}
              filled={idx < quickSlotsRemaining}
              label={idx === 0 ? "Quick" : undefined}
              variant="quick"
            />
          ))}
        </div>

        {/* Reaction */}
        <div className="flex items-center gap-1.5">
          <ActionSlot
            filled={reactionAvailable}
            label="React"
            variant="reaction"
          />
        </div>
      </div>

      {/* Overcharge status */}
      <div className="flex flex-wrap gap-2 text-xs">
        {canOvercharge && !economy.overcharge_used && (
          <span className="px-2 py-0.5 rounded bg-primary/20 text-primary">
            Overcharge Available
          </span>
        )}
        {economy.overcharge_used && (
          <span className="px-2 py-0.5 rounded bg-destructive/20 text-destructive">
            Overcharged (+1 Quick)
          </span>
        )}
        {overchargeLevel > 0 && (
          <span className="px-2 py-0.5 rounded bg-amber-500/20 text-amber-500">
            OC Level {overchargeLevel}: {getOverchargeCost(overchargeLevel)}
          </span>
        )}
      </div>
    </div>
  );
}

interface ActionSlotProps {
  filled: boolean;
  label?: string;
  variant: "full" | "quick" | "reaction";
}

function ActionSlot({ filled, label, variant }: ActionSlotProps) {
  const baseClasses = "w-4 h-4 rounded-full border-2 transition-colors";
  const variantClasses = {
    full: filled
      ? "bg-primary border-primary"
      : "bg-transparent border-muted-foreground/40",
    quick: filled
      ? "bg-secondary border-secondary-foreground/60"
      : "bg-transparent border-muted-foreground/40",
    reaction: filled
      ? "bg-amber-500 border-amber-600"
      : "bg-transparent border-muted-foreground/40",
  };

  return (
    <div className="flex items-center gap-1">
      <div className={`${baseClasses} ${variantClasses[variant]}`} />
      {label && (
        <span className="text-xs text-muted-foreground">{label}</span>
      )}
    </div>
  );
}
