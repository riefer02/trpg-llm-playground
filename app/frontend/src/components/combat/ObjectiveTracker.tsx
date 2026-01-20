import type { MissionObjectiveBrief } from "../../lib/types/lancer";

export interface ObjectiveTrackerProps {
  objectives: MissionObjectiveBrief[] | null | undefined;
}

const PRIORITY_LABELS: Record<string, { label: string; color: string }> = {
  primary: { label: "Primary", color: "bg-primary/20 text-primary" },
  secondary: { label: "Secondary", color: "bg-muted text-muted-foreground" },
  optional: { label: "Optional", color: "bg-amber-500/20 text-amber-500" },
};

export function ObjectiveTracker({ objectives }: ObjectiveTrackerProps) {
  if (!objectives || objectives.length === 0) {
    return null;
  }

  // Sort objectives: primary first, then secondary, then optional
  const sortedObjectives = [...objectives].sort((a, b) => {
    const priorityOrder: Record<string, number> = { primary: 0, secondary: 1, optional: 2 };
    const aPriority = priorityOrder[a.priority ?? "primary"] ?? 1;
    const bPriority = priorityOrder[b.priority ?? "primary"] ?? 1;
    return aPriority - bPriority;
  });

  const totalPrimary = objectives.filter((obj) => obj.priority !== "optional").length;

  return (
    <div className="rounded-md border border-border bg-muted/30 p-3 space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-sm font-medium text-foreground">Mission Objectives</div>
        <span className="text-xs text-muted-foreground">
          {totalPrimary} objectives
        </span>
      </div>

      {/* Objectives list */}
      <div className="space-y-2">
        {sortedObjectives.map((objective, index) => (
          <ObjectiveItem key={objective.id || `obj-${index}`} objective={objective} />
        ))}
      </div>
    </div>
  );
}

interface ObjectiveItemProps {
  objective: MissionObjectiveBrief;
}

function ObjectiveItem({ objective }: ObjectiveItemProps) {
  const priority = objective.priority ?? "primary";
  const priorityConfig = PRIORITY_LABELS[priority] || PRIORITY_LABELS.primary;
  const isOptional = priority === "optional";

  return (
    <div className="rounded-md border border-border/60 bg-background px-3 py-2">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-muted-foreground" />
          <span className="text-sm font-medium text-foreground">
            {objective.title}
          </span>
        </div>
        <div className="flex items-center gap-1">
          {isOptional && (
            <span className="px-1.5 py-0.5 rounded text-xs bg-muted text-muted-foreground">
              Optional
            </span>
          )}
          <span className={`px-1.5 py-0.5 rounded text-xs ${priorityConfig.color}`}>
            {priorityConfig.label}
          </span>
        </div>
      </div>

      {/* Success condition */}
      {objective.success_condition && (
        <div className="text-xs text-muted-foreground mt-1 ml-4">
          {objective.success_condition}
        </div>
      )}

      {/* Related objective */}
      {objective.related_objective_id && (
        <div className="text-xs text-muted-foreground mt-1 ml-4">
          Related to: {objective.related_objective_id}
        </div>
      )}
    </div>
  );
}
