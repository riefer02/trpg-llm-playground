/**
 * ValidationSummary component for displaying loadout validation issues.
 */

import type { LoadoutValidationIssue } from "../../lib/validation/loadout";

interface ValidationSummaryProps {
  issues: LoadoutValidationIssue[];
}

/**
 * Displays validation errors and warnings in a compact summary format.
 * Used during loadout editing to provide real-time feedback.
 */
export function ValidationSummary({ issues }: ValidationSummaryProps) {
  const errors = issues.filter((i) => i.severity === "error");
  const warnings = issues.filter((i) => i.severity === "warning");

  if (issues.length === 0) return null;

  return (
    <div className="space-y-1" role="alert" aria-live="polite">
      {errors.map((issue, i) => (
        <div
          key={`error-${i}`}
          className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive"
        >
          <span className="font-mono">[{issue.code}]</span> {issue.message}
        </div>
      ))}
      {warnings.map((issue, i) => (
        <div
          key={`warning-${i}`}
          className="rounded-md border border-accent/40 bg-accent/10 px-3 py-2 text-xs text-accent"
        >
          <span className="font-mono">[{issue.code}]</span> {issue.message}
        </div>
      ))}
    </div>
  );
}
