import type { ActionEconomyState } from "../../lib/api/combat";

export interface EconomyDisplayProps {
  economy: ActionEconomyState;
  canOvercharge: boolean;
}

export function EconomyDisplay({ economy, canOvercharge }: EconomyDisplayProps) {
  const quickSlotsTotal = economy.quick_actions_available + economy.quick_actions_used;
  const quickSlotsRemaining = economy.quick_actions_available;

  return (
    <div className="rounded-md border border-border bg-muted/30 p-3 space-y-2">
      <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
        Action Economy
      </div>

      <div className="flex items-center gap-3">
        {/* Full Action */}
        <div className="flex items-center gap-1.5">
          <ActionSlot
            filled={!economy.full_action_used}
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
            filled={!economy.reaction_used}
            label="React"
            variant="reaction"
          />
        </div>
      </div>

      {/* Protocol and Overcharge status */}
      <div className="flex flex-wrap gap-2 text-xs">
        {!economy.protocol_used && (
          <span className="px-2 py-0.5 rounded bg-secondary text-secondary-foreground">
            Protocol Available
          </span>
        )}
        {canOvercharge && !economy.overcharge_used && (
          <span className="px-2 py-0.5 rounded bg-primary/20 text-primary">
            Overcharge Available
          </span>
        )}
        {economy.overcharge_used && (
          <span className="px-2 py-0.5 rounded bg-destructive/20 text-destructive">
            Overcharged
          </span>
        )}
      </div>

      {/* Movement */}
      {economy.movement_available > 0 && (
        <div className="text-xs text-muted-foreground">
          Movement: {economy.movement_available - economy.movement_used} / {economy.movement_available} spaces
        </div>
      )}

      {/* Free actions used */}
      {economy.free_actions_used.length > 0 && (
        <div className="text-xs text-muted-foreground">
          Free actions used: {economy.free_actions_used.join(", ")}
        </div>
      )}
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
