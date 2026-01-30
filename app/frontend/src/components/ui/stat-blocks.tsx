/**
 * UI components for displaying stats and skills.
 */

export interface StatBlockProps {
  label: string;
  value: number | string;
}

export function StatBlock({ label, value }: StatBlockProps) {
  return (
    <div>
      <div className="text-muted-foreground text-xs uppercase">{label}</div>
      <div className="font-semibold text-lg">{value}</div>
    </div>
  );
}

export interface SkillBlockProps {
  label: string;
  value: number;
}

export function SkillBlock({ label, value }: SkillBlockProps) {
  return (
    <div className="p-2 bg-card-foreground/5 rounded text-center">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-semibold">+{value}</div>
    </div>
  );
}