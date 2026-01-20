import { useState } from "react";

import type { DecisionSubmitRequest, PendingDecisionItem, DecisionChoice } from "../../lib/api/combat";
import { Button } from "../ui";

export interface SaveCheckPromptProps {
  decision: PendingDecisionItem;
  combatantId: string;
  combatantName: string;
  onSubmit: (request: DecisionSubmitRequest) => void;
  onDecline: () => void;
  isOpen: boolean;
  isSubmitting?: boolean;
}

export function SaveCheckPrompt({
  decision,
  combatantId,
  combatantName,
  onSubmit,
  onDecline,
  isOpen,
  isSubmitting = false,
}: SaveCheckPromptProps) {
  const [selectedChoice, setSelectedChoice] = useState<DecisionChoice | null>(null);

  if (!isOpen) {
    return null;
  }

  const saveTypeLabel = formatSaveType(decision.save_type);
  const triggerDescription = getTriggerDescription(decision, combatantName);
  const warningMessage = getWarningMessage(decision);

  const canSubmit = selectedChoice !== null;

  const handleSubmit = () => {
    if (!selectedChoice || !canSubmit) return;

    onSubmit({
      decision_id: decision.decision_id,
      combatant_id: combatantId,
      choice: selectedChoice,
    });
  };

  const borderColor = decision.decision_type === "hull_save"
    ? "border-red-500/50"
    : "border-orange-500/50";
  const bgColor = decision.decision_type === "hull_save"
    ? "bg-red-500/5"
    : "bg-orange-500/5";
  const accentColor = decision.decision_type === "hull_save"
    ? "text-red-500"
    : "text-orange-500";
  const badgeColor = decision.decision_type === "hull_save"
    ? "bg-red-500/20 text-red-500"
    : "bg-orange-500/20 text-orange-500";

  return (
    <div className={`rounded-md border ${borderColor} ${bgColor} p-4 space-y-4`}>
      <div className="flex items-start justify-between">
        <div>
          <div className={`text-sm font-medium ${accentColor}`}>
            {saveTypeLabel} Required
          </div>
          <div className="text-xs text-muted-foreground mt-1">
            {triggerDescription}
          </div>
        </div>
        <span className={`px-2 py-0.5 rounded text-xs ${badgeColor}`}>
          DC {decision.save_target ?? 10}
        </span>
      </div>

      {/* Save info */}
      <div className="flex items-center gap-4 text-sm">
        <div className="flex items-center gap-1">
          <span className="text-muted-foreground">Bonus:</span>
          <span className="font-medium">+{decision.save_bonus}</span>
        </div>
        {decision.reroll_available && (
          <div className="flex items-center gap-1">
            <span className="text-green-500 text-xs">Reroll Available</span>
            {decision.reroll_source && (
              <span className="text-xs text-muted-foreground">({decision.reroll_source})</span>
            )}
          </div>
        )}
      </div>

      {/* Warning message */}
      {warningMessage && (
        <div className="text-xs text-red-500 bg-red-500/10 rounded px-2 py-1.5">
          {warningMessage}
        </div>
      )}

      {/* Choice selection */}
      <div className="space-y-2">
        <div className="text-xs text-muted-foreground font-medium">
          Choose Action
        </div>
        <div className="space-y-1">
          <ChoiceOption
            choice="roll"
            label="Make Save"
            description={`Roll 1d20 + ${decision.save_bonus} vs DC ${decision.save_target ?? 10}`}
            isSelected={selectedChoice === "roll"}
            onSelect={() => setSelectedChoice("roll")}
          />
          {decision.reroll_available && (
            <ChoiceOption
              choice="use_reroll"
              label="Make Save with Reroll"
              description="Roll twice and keep the better result"
              isSelected={selectedChoice === "use_reroll"}
              onSelect={() => setSelectedChoice("use_reroll")}
            />
          )}
          <ChoiceOption
            choice="voluntary_fail"
            label="Voluntarily Fail"
            description="Automatically fail the save (per PR2 1370)"
            isSelected={selectedChoice === "voluntary_fail"}
            onSelect={() => setSelectedChoice("voluntary_fail")}
            variant="danger"
          />
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex gap-2 pt-2">
        <Button
          variant="primary"
          size="sm"
          onClick={handleSubmit}
          disabled={!canSubmit || isSubmitting}
          className={decision.decision_type === "hull_save"
            ? "bg-red-500 hover:bg-red-600"
            : "bg-orange-500 hover:bg-orange-600"
          }
        >
          {isSubmitting ? "Resolving..." : "Confirm"}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={onDecline}
          disabled={isSubmitting}
        >
          Cancel
        </Button>
      </div>
    </div>
  );
}

interface ChoiceOptionProps {
  choice: DecisionChoice;
  label: string;
  description: string;
  isSelected: boolean;
  onSelect: () => void;
  variant?: "default" | "danger";
}

function ChoiceOption({
  label,
  description,
  isSelected,
  onSelect,
  variant = "default",
}: ChoiceOptionProps) {
  const borderClass = isSelected
    ? variant === "danger"
      ? "bg-red-500/20 border border-red-500/50"
      : "bg-primary/20 border border-primary/50"
    : "bg-muted/30 border border-transparent hover:bg-primary/10";

  const textClass = isSelected
    ? variant === "danger"
      ? "text-red-500"
      : "text-primary"
    : "text-foreground";

  return (
    <button
      type="button"
      onClick={onSelect}
      className={`w-full p-2 rounded text-left transition-colors ${borderClass}`}
    >
      <div className={`text-sm font-medium ${textClass}`}>
        {label}
      </div>
      <div className="text-xs text-muted-foreground">{description}</div>
    </button>
  );
}

function formatSaveType(saveType: string | undefined): string {
  switch (saveType) {
    case "hull":
      return "Hull Save";
    case "engineering":
      return "Engineering Save";
    case "agility":
      return "Agility Save";
    case "systems":
      return "Systems Save";
    default:
      return "Save";
  }
}

function getTriggerDescription(decision: PendingDecisionItem, combatantName: string): string {
  const source = decision.trigger_source;

  if (source === "structure_cascade") {
    return `${combatantName} has suffered a Direct Hit at 2 structure! A hull save is required to survive.`;
  }
  if (source === "meltdown") {
    return `${combatantName}'s reactor is destabilizing! An engineering save is required to prevent meltdown.`;
  }
  if (source.startsWith("dangerous_terrain:")) {
    const terrainType = source.replace("dangerous_terrain:", "");
    return `${combatantName} is entering dangerous terrain (${terrainType})! An engineering check is required.`;
  }

  return `${combatantName} must make a ${decision.decision_type.replace("_", " ")}!`;
}

function getWarningMessage(decision: PendingDecisionItem): string | null {
  if (decision.decision_type === "hull_save") {
    return "Failure means mech destruction!";
  }
  if (decision.decision_type === "engineering_save" && decision.trigger_source === "meltdown") {
    return "Failure starts a meltdown countdown!";
  }
  if (decision.decision_type === "engineering_check") {
    return "Failure results in damage from the terrain!";
  }
  return null;
}
