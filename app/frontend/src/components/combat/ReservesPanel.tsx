import type { ReservePlanEntry } from "../../lib/types/lancer";
import { Button } from "../ui";

export type ReservePlanStatus = "planned" | "spent" | "earned";

export interface ReservesPanelProps {
  reserves: ReservePlanEntry[] | null | undefined;
  onSpendReserve: (reserveId: string) => void;
  isSpending: boolean;
}

const STATUS_STYLES: Record<ReservePlanStatus, { label: string; color: string }> = {
  planned: { label: "Available", color: "bg-green-500/20 text-green-500" },
  spent: { label: "Spent", color: "bg-muted text-muted-foreground" },
  earned: { label: "Earned", color: "bg-amber-500/20 text-amber-500" },
};

export function ReservesPanel({ reserves, onSpendReserve, isSpending }: ReservesPanelProps) {
  if (!reserves || reserves.length === 0) {
    return null;
  }

  const plannedCount = reserves.filter((r) => r.status === "planned").length;
  const spentCount = reserves.filter((r) => r.status === "spent").length;

  return (
    <div className="rounded-md border border-border bg-muted/30 p-3 space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-sm font-medium text-foreground">Mission Reserves</div>
        <span className="text-xs text-muted-foreground">
          {plannedCount} available / {spentCount} spent
        </span>
      </div>

      <div className="space-y-2">
        {reserves.map((reserve, index) => (
          <ReserveItem
            key={reserve.reserve_id || `reserve-${index}`}
            reserve={reserve}
            onSpend={onSpendReserve}
            isSpending={isSpending}
          />
        ))}
      </div>
    </div>
  );
}

interface ReserveItemProps {
  reserve: ReservePlanEntry;
  onSpend: (reserveId: string) => void;
  isSpending: boolean;
}

function ReserveItem({ reserve, onSpend, isSpending }: ReserveItemProps) {
  const status = (reserve.status ?? "planned") as ReservePlanStatus;
  const statusConfig = STATUS_STYLES[status] || STATUS_STYLES.planned;
  const isPlanned = status === "planned";

  const handleSpend = () => {
    if (reserve.reserve_id) {
      onSpend(reserve.reserve_id);
    }
  };

  return (
    <div className="rounded-md border border-border/60 bg-background px-3 py-2">
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${isPlanned ? "bg-green-500" : "bg-muted-foreground"}`}
            />
            <span className="text-sm font-medium text-foreground">
              {reserve.usage_notes || reserve.reserve_id}
            </span>
          </div>

          {reserve.assigned_character_id && (
            <div className="text-xs text-muted-foreground mt-1 ml-4">
              Assigned: {reserve.assigned_character_id}
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          <span className={`px-1.5 py-0.5 rounded text-xs ${statusConfig.color}`}>
            {statusConfig.label}
          </span>

          {isPlanned && (
            <Button
              size="sm"
              variant="outline"
              onClick={handleSpend}
              disabled={isSpending}
            >
              {isSpending ? "..." : "Spend"}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
