import { Button } from "../ui";
import type {
  MissionObjectiveBrief,
  MissionStakesBrief,
  ReservePlanEntry,
} from "../../lib/types/lancer";
import {
  EnemyRosterPreview,
  type EnemyRosterPreviewProps,
} from "./EnemyRosterPreview";

export interface MissionBriefingModalProps {
  isOpen: boolean;
  onClose: () => void;
  missionName: string | null | undefined;
  briefingNotes: string | null | undefined;
  stakes: MissionStakesBrief | null | undefined;
  objectives: MissionObjectiveBrief[] | null | undefined;
  supportAssets: string[] | null | undefined;
  reserves: ReservePlanEntry[] | null | undefined;
  enemyForcePreview: EnemyRosterPreviewProps["preview"] | null | undefined;
}

const PRIORITY_LABELS: Record<string, { label: string; color: string }> = {
  primary: { label: "Primary", color: "bg-primary/20 text-primary" },
  secondary: { label: "Secondary", color: "bg-muted text-muted-foreground" },
  optional: { label: "Optional", color: "bg-amber-500/20 text-amber-500" },
};

const STAKES_TYPE_LABELS: Record<string, string> = {
  personal: "Personal Stakes",
  faction: "Faction Stakes",
  immediate: "Immediate Stakes",
  gradual: "Gradual Stakes",
  custom: "Custom Stakes",
};

export function MissionBriefingModal({
  isOpen,
  onClose,
  missionName,
  briefingNotes,
  stakes,
  objectives,
  supportAssets,
  reserves,
  enemyForcePreview,
}: MissionBriefingModalProps) {
  if (!isOpen) {
    return null;
  }

  const hasContent =
    missionName ||
    briefingNotes ||
    stakes ||
    (objectives && objectives.length > 0) ||
    (supportAssets && supportAssets.length > 0) ||
    (reserves && reserves.length > 0) ||
    enemyForcePreview;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-background/80 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className="relative w-full max-w-2xl max-h-[80vh] overflow-y-auto rounded-lg border border-border bg-card p-6 shadow-lg">
        <div className="space-y-6">
          {/* Header */}
          <div className="flex items-start justify-between">
            <div>
              <h2 className="text-xl font-semibold text-foreground">
                {missionName || "Mission Briefing"}
              </h2>
              <p className="text-sm text-muted-foreground">
                Review mission details before launch
              </p>
            </div>
            <Button variant="ghost" size="sm" onClick={onClose}>
              Close
            </Button>
          </div>

          {!hasContent && (
            <div className="rounded-md border border-border/60 bg-muted/30 px-4 py-6 text-center">
              <p className="text-muted-foreground">
                No mission briefing has been prepared yet.
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                The GM can add mission details in the Mission Lobby section.
              </p>
            </div>
          )}

          {/* Briefing Notes */}
          {briefingNotes && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-foreground">Briefing</h3>
              <div className="rounded-md border border-border/60 bg-muted/30 px-4 py-3">
                <p className="text-sm text-foreground whitespace-pre-wrap">
                  {briefingNotes}
                </p>
              </div>
            </div>
          )}

          {/* Stakes */}
          {stakes && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-foreground">
                {STAKES_TYPE_LABELS[stakes.stakes_type] || "Stakes"}
              </h3>
              <div className="rounded-md border border-border/60 bg-muted/30 px-4 py-3">
                <p className="text-sm text-foreground">{stakes.summary}</p>
              </div>
            </div>
          )}

          {/* Objectives */}
          {objectives && objectives.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-foreground">Objectives</h3>
              <div className="space-y-2">
                {objectives.map((objective, index) => {
                  const priorityConfig =
                    PRIORITY_LABELS[objective.priority ?? "primary"] ||
                    PRIORITY_LABELS.primary;
                  return (
                    <div
                      key={objective.id || `obj-${index}`}
                      className="rounded-md border border-border/60 bg-background px-4 py-3"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-foreground">
                              {objective.title || `Objective ${index + 1}`}
                            </span>
                            <span
                              className={`px-1.5 py-0.5 rounded text-xs ${priorityConfig.color}`}
                            >
                              {priorityConfig.label}
                            </span>
                          </div>
                          {objective.success_condition && (
                            <p className="text-sm text-muted-foreground mt-1">
                              {objective.success_condition}
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Enemy Force Preview */}
          {enemyForcePreview && (
            <EnemyRosterPreview preview={enemyForcePreview} />
          )}

          {/* Support Assets */}
          {supportAssets && supportAssets.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-foreground">Support Assets</h3>
              <div className="rounded-md border border-border/60 bg-muted/30 px-4 py-3">
                <ul className="space-y-1">
                  {supportAssets.map((asset, index) => (
                    <li
                      key={`asset-${index}`}
                      className="text-sm text-foreground flex items-center gap-2"
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-primary" />
                      {asset}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Reserves */}
          {reserves && reserves.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-foreground">Reserves</h3>
              <div className="rounded-md border border-border/60 bg-muted/30 px-4 py-3">
                <ul className="space-y-2">
                  {reserves.map((reserve, index) => (
                    <li
                      key={`reserve-${index}`}
                      className="text-sm text-foreground flex items-center justify-between"
                    >
                      <span className="flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
                        {reserve.usage_notes || reserve.reserve_id}
                      </span>
                      <span className="text-xs text-muted-foreground px-1.5 py-0.5 rounded bg-muted">
                        {reserve.status}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Footer */}
          <div className="flex justify-end pt-2">
            <Button onClick={onClose}>Understood</Button>
          </div>
        </div>
      </div>
    </div>
  );
}
