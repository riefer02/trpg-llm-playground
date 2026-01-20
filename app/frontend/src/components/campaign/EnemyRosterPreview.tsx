/**
 * Props interface matching the serialized EnemyForcePreview from core.
 * We define this locally since the generated type is a permissive dict.
 */
interface EnemyForcePreviewData {
  total_victory_points: number;
  initial_victory_points: number;
  reserve_victory_points: number;
  initial_count: number;
  reserve_count: number;
  composition?: Array<{
    template_id: string;
    name: string;
    count: number;
    victory_points: number;
  }>;
  difficulty?: string;
  sitrep_type?: string;
}

export interface EnemyRosterPreviewProps {
  preview: EnemyForcePreviewData;
}

const DIFFICULTY_LABELS: Record<string, { label: string; color: string }> = {
  trivial: { label: "Trivial", color: "text-muted-foreground" },
  easy: { label: "Easy", color: "text-green-500" },
  standard: { label: "Standard", color: "text-foreground" },
  hard: { label: "Hard", color: "text-amber-500" },
  extreme: { label: "Extreme", color: "text-destructive" },
};

export function EnemyRosterPreview({ preview }: EnemyRosterPreviewProps) {
  const difficultyConfig =
    DIFFICULTY_LABELS[preview.difficulty || "standard"] ||
    DIFFICULTY_LABELS.standard;

  const composition = preview.composition || [];

  return (
    <div className="space-y-3">
      {/* Header with total VP */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-foreground">Enemy Force</h3>
        <div className="flex items-center gap-2">
          {preview.difficulty && (
            <span
              className={`text-xs font-medium ${difficultyConfig.color}`}
            >
              {difficultyConfig.label}
            </span>
          )}
          <span className="text-xs text-muted-foreground px-1.5 py-0.5 rounded bg-muted">
            {preview.total_victory_points.toFixed(1)} VP
          </span>
        </div>
      </div>

      {/* Deployment breakdown */}
      <div className="rounded-md border border-border/60 bg-muted/30 px-4 py-3 space-y-3">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-xs text-muted-foreground mb-1">
              Initial Deployment
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-lg font-semibold text-foreground">
                {preview.initial_count}
              </span>
              <span className="text-sm text-muted-foreground">
                units ({preview.initial_victory_points.toFixed(1)} VP)
              </span>
            </div>
          </div>
          <div>
            <div className="text-xs text-muted-foreground mb-1">Reserves</div>
            <div className="flex items-baseline gap-2">
              <span className="text-lg font-semibold text-foreground">
                {preview.reserve_count}
              </span>
              <span className="text-sm text-muted-foreground">
                units ({preview.reserve_victory_points.toFixed(1)} VP)
              </span>
            </div>
          </div>
        </div>

        {/* Composition breakdown */}
        {composition.length > 0 && (
          <div className="border-t border-border/40 pt-3">
            <div className="text-xs text-muted-foreground mb-2">Composition</div>
            <ul className="space-y-1">
              {composition.map((entry, index) => (
                <li
                  key={entry.template_id || `entry-${index}`}
                  className="flex items-center justify-between text-sm"
                >
                  <span className="flex items-center gap-2 text-foreground">
                    <span className="w-1.5 h-1.5 rounded-full bg-destructive/70" />
                    {entry.count}x {entry.name}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {entry.victory_points.toFixed(1)} VP
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Empty state when no composition */}
        {composition.length === 0 &&
          (preview.initial_count > 0 || preview.reserve_count > 0) && (
            <div className="border-t border-border/40 pt-3">
              <p className="text-xs text-muted-foreground italic">
                Enemy types will be revealed at mission start
              </p>
            </div>
          )}

        {/* Zero enemies state */}
        {preview.initial_count === 0 && preview.reserve_count === 0 && (
          <p className="text-xs text-muted-foreground italic">
            No enemies configured for this mission
          </p>
        )}
      </div>

      {/* SITREP type info */}
      {preview.sitrep_type && (
        <div className="text-xs text-muted-foreground">
          SITREP: <span className="font-medium">{preview.sitrep_type}</span>
        </div>
      )}
    </div>
  );
}
