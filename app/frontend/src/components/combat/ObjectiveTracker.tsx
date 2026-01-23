import type { MissionObjective } from "../../lib/types/lancer";

export interface ObjectiveTrackerProps {
  objectives: MissionObjective[] | null | undefined;
}

// MissionObjective uses numeric priorities (0 = highest)
const PRIORITY_LABELS: Record<number, { label: string; color: string }> = {
  0: { label: "Primary", color: "bg-primary/20 text-primary" },
  1: { label: "Secondary", color: "bg-muted text-muted-foreground" },
  2: { label: "Optional", color: "bg-amber-500/20 text-amber-500" },
};

const DEFAULT_PRIORITY_CONFIG = { label: "Primary", color: "bg-primary/20 text-primary" };

export function ObjectiveTracker({ objectives }: ObjectiveTrackerProps) {
  if (!objectives || objectives.length === 0) {
    return null;
  }

  // Sort objectives by numeric priority (lower = higher priority)
  const sortedObjectives = [...objectives].sort((a, b) => {
    const aPriority = a.priority ?? 0;
    const bPriority = b.priority ?? 0;
    return aPriority - bPriority;
  });

  const totalPrimary = objectives.filter((obj) => !obj.is_optional).length;

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
  objective: MissionObjective;
}

function ObjectiveItem({ objective }: ObjectiveItemProps) {
  const priority = objective.priority ?? 0;
  const priorityConfig = PRIORITY_LABELS[priority] ?? DEFAULT_PRIORITY_CONFIG;
  const isOptional = objective.is_optional ?? false;

  return (
    <div className="rounded-md border border-border/60 bg-background px-3 py-2">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-muted-foreground" />
          <span className="text-sm font-medium text-foreground">
            {objective.description}
          </span>
        </div>
        <div className="flex items-center gap-1">
          {objective.status && (
            <span className="px-1.5 py-0.5 rounded text-xs bg-muted text-muted-foreground">
              {objective.status}
            </span>
          )}
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

      {/* Completion criteria */}
      {objective.completion_criteria && objective.completion_criteria.length > 0 && (
        <div className="text-xs text-muted-foreground mt-1 ml-4">
          {objective.completion_criteria.map((criterion, i) => (
            <div key={i}>{criterion.description}</div>
          ))}
        </div>
      )}

      {/* Dependencies */}
      {objective.depends_on && objective.depends_on.length > 0 && (
        <div className="text-xs text-muted-foreground mt-1 ml-4">
          Depends on: {objective.depends_on.join(", ")}
        </div>
      )}
    </div>
  );
}
