import { useState, useMemo } from "react";
import { Button } from "../ui";
import type { MissionOutcome, CombatCompleteRequest } from "../../lib/api";
import type { ReservePlanEntry } from "../../lib/types/lancer";

export interface MissionCompleteModalProps {
  isOpen: boolean;
  onComplete: (request: CombatCompleteRequest) => void;
  onCancel: () => void;
  isSubmitting: boolean;
  campaignId: string | null;
  missionReserves?: ReservePlanEntry[] | null;
}

const OUTCOME_OPTIONS: { value: MissionOutcome; label: string; description: string }[] = [
  {
    value: "success",
    label: "Success",
    description: "All primary objectives completed. Full rewards.",
  },
  {
    value: "partial",
    label: "Partial Success",
    description: "Some objectives completed. Reduced rewards.",
  },
  {
    value: "failure",
    label: "Failure",
    description: "Mission failed but squad extracted. No rewards.",
  },
  {
    value: "catastrophic",
    label: "Catastrophic",
    description: "Total mission failure. Narrative consequences.",
  },
];

export function MissionCompleteModal({
  isOpen,
  onComplete,
  onCancel,
  isSubmitting,
  campaignId,
  missionReserves,
}: MissionCompleteModalProps) {
  const [outcome, setOutcome] = useState<MissionOutcome>("success");
  const [completionScore, setCompletionScore] = useState(1.0);
  const [debriefNotes, setDebriefNotes] = useState("");

  // Compute spent reserves from mission_reserves
  const spentReserves = useMemo(() => {
    if (!missionReserves) return [];
    return missionReserves.filter((r) => r.status === "spent");
  }, [missionReserves]);

  if (!isOpen) {
    return null;
  }

  const handleSubmit = () => {
    // Auto-populate reserves_spent from reserves with status === "spent"
    const reservesSpentData = spentReserves.map((r) => ({
      reserve_id: r.reserve_id,
      usage_notes: r.usage_notes,
      assigned_character_id: r.assigned_character_id,
    }));

    const request: CombatCompleteRequest = {
      outcome,
      completion_score: completionScore,
      debrief_notes: debriefNotes.trim() || undefined,
      reserves_spent: reservesSpentData.length > 0 ? reservesSpentData : undefined,
    };
    onComplete(request);
  };

  const selectedOption = OUTCOME_OPTIONS.find((opt) => opt.value === outcome);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-background/80 backdrop-blur-sm"
        onClick={onCancel}
      />
      <div className="relative w-full max-w-md rounded-lg border border-border bg-card p-6 shadow-lg">
        <div className="space-y-4">
          <div>
            <h2 className="text-lg font-semibold text-foreground">
              End Mission
            </h2>
            <p className="text-sm text-muted-foreground">
              Complete the combat session and record the mission outcome.
            </p>
          </div>

          <div className="space-y-3">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">
                Mission Outcome
              </label>
              <div className="grid grid-cols-2 gap-2">
                {OUTCOME_OPTIONS.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => setOutcome(option.value)}
                    className={`rounded-md border px-3 py-2 text-left transition-colors ${
                      outcome === option.value
                        ? "border-primary bg-primary/10 text-foreground"
                        : "border-border bg-background text-muted-foreground hover:border-primary/50"
                    }`}
                  >
                    <div className="text-sm font-medium">{option.label}</div>
                  </button>
                ))}
              </div>
              {selectedOption && (
                <p className="text-xs text-muted-foreground">
                  {selectedOption.description}
                </p>
              )}
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">
                Completion Score
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.1}
                  value={completionScore}
                  onChange={(e) => setCompletionScore(parseFloat(e.target.value))}
                  className="flex-1"
                />
                <span className="w-12 text-sm text-muted-foreground text-right">
                  {Math.round(completionScore * 100)}%
                </span>
              </div>
              <p className="text-xs text-muted-foreground">
                How much of the mission objectives were achieved
              </p>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">
                Debrief Notes
              </label>
              <textarea
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                rows={3}
                value={debriefNotes}
                onChange={(e) => setDebriefNotes(e.target.value)}
                placeholder="Key events, player highlights, consequences..."
              />
            </div>

            {/* Spent Reserves Summary */}
            {spentReserves.length > 0 && (
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">
                  Reserves Spent ({spentReserves.length})
                </label>
                <div className="rounded-md border border-border/60 bg-muted/30 px-3 py-2 space-y-1">
                  {spentReserves.map((reserve, index) => (
                    <div
                      key={reserve.reserve_id || `spent-${index}`}
                      className="text-xs text-muted-foreground flex items-center gap-2"
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground" />
                      <span>{reserve.usage_notes || reserve.reserve_id}</span>
                    </div>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground">
                  These reserves will be recorded as spent in the mission outcome.
                </p>
              </div>
            )}

            {campaignId && (
              <div className="rounded-md border border-border/60 bg-muted/30 px-3 py-2">
                <p className="text-xs text-muted-foreground">
                  This outcome will be recorded to the campaign history.
                </p>
              </div>
            )}
          </div>

          <div className="flex gap-2 pt-2">
            <Button
              variant="primary"
              onClick={handleSubmit}
              disabled={isSubmitting}
            >
              {isSubmitting ? "Completing..." : "Complete Mission"}
            </Button>
            <Button
              variant="ghost"
              onClick={onCancel}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
