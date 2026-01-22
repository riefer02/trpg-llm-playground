/**
 * Level selector for character creation.
 *
 * Allows selection of License Level 0-3 with progression info.
 */

interface LevelSelectorProps {
  value: number;
  onChange: (level: number) => void;
}

// Progression values from core/pilot/progression.py
const LEVEL_PROGRESSION = {
  0: { licensePoints: 0, skillPoints: 2, talentPoints: 3, triggerPoints: 4 },
  1: { licensePoints: 1, skillPoints: 3, talentPoints: 4, triggerPoints: 4 },
  2: { licensePoints: 2, skillPoints: 4, talentPoints: 5, triggerPoints: 4 },
  3: { licensePoints: 3, skillPoints: 5, talentPoints: 6, triggerPoints: 4 },
} as const;

export function LevelSelector({ value, onChange }: LevelSelectorProps) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {([0, 1, 2, 3] as const).map((level) => {
        const prog = LEVEL_PROGRESSION[level];
        const isSelected = value === level;

        return (
          <button
            key={level}
            type="button"
            onClick={() => onChange(level)}
            className={`p-4 text-left border rounded-lg transition-colors ${
              isSelected
                ? "bg-primary/20 border-primary"
                : "hover:bg-primary/10 hover:border-primary/50"
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-lg font-semibold">LL{level}</span>
              {level === 0 && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
                  Recommended
                </span>
              )}
            </div>
            <div className="text-sm text-muted-foreground space-y-1">
              <div className="flex justify-between">
                <span>License Points</span>
                <span className="font-medium text-foreground">
                  {prog.licensePoints}
                </span>
              </div>
              <div className="flex justify-between">
                <span>HASE Points</span>
                <span className="font-medium text-foreground">
                  {prog.skillPoints}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Talent Points</span>
                <span className="font-medium text-foreground">
                  {prog.talentPoints}
                </span>
              </div>
            </div>
            {level > 0 && (
              <div className="mt-2 text-xs text-primary">
                Unlock manufacturer frames
              </div>
            )}
          </button>
        );
      })}
    </div>
  );
}

export { LEVEL_PROGRESSION };
