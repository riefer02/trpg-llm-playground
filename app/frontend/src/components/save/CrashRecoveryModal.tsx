/**
 * Crash Recovery Modal
 * 
 * Shown when a player continues a game that has a mission in progress.
 * Offers options to resume the mission or return to quarters.
 */

import { Button } from "../ui/button";
import { Modal } from "../ui/modal";
import { Play, Home, AlertTriangle } from "lucide-react";
import type { MissionState } from "../../lib/save/saveSystem";

interface CrashRecoveryModalProps {
  /** Whether the modal is open */
  isOpen: boolean;
  /** Mission state to display */
  missionState: MissionState;
  /** Called when user chooses to resume the mission */
  onResume: () => void;
  /** Called when user chooses to return to quarters */
  onReturnToQuarters: () => void;
  /** Called when modal is closed without action */
  onClose: () => void;
}

export function CrashRecoveryModal({
  isOpen,
  missionState,
  onResume,
  onReturnToQuarters,
  onClose,
}: CrashRecoveryModalProps) {
  const formatSitrep = (sitrep: string) => {
    return sitrep.replace(/_/g, " ").toUpperCase();
  };

  const difficultyStars = "★".repeat(missionState.difficulty) + "☆".repeat(3 - missionState.difficulty);

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Mission in Progress"
      size="md"
      ariaLabel="Crash recovery dialog"
    >
      <div className="space-y-6">
        {/* Warning header */}
        <div className="flex items-start gap-3 p-4 bg-warning/10 border border-warning/20 rounded-lg">
          <AlertTriangle className="w-5 h-5 text-warning flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="font-semibold text-warning">Previous session interrupted</h3>
            <p className="text-sm text-muted-foreground mt-1">
              You were in the middle of a mission when you last played. 
              You can resume from where you left off or return to your quarters.
            </p>
          </div>
        </div>

        {/* Mission details */}
        <div className="p-4 bg-muted rounded-lg space-y-3">
          <h4 className="font-semibold text-lg">{missionState.missionName}</h4>
          <div className="flex flex-wrap gap-2 text-sm">
            <span className="px-2 py-1 bg-primary/10 rounded">
              {formatSitrep(missionState.sitrep)}
            </span>
            <span className="px-2 py-1 bg-primary/10 rounded">
              Difficulty: {difficultyStars}
            </span>
          </div>
          {missionState.combatSessionId && (
            <p className="text-sm text-muted-foreground">
              Combat session active - ready to resume immediately
            </p>
          )}
        </div>

        {/* Action buttons */}
        <div className="flex flex-col sm:flex-row gap-3">
          <Button
            variant="primary"
            onClick={onResume}
            className="flex-1 flex items-center justify-center gap-2"
            autoFocus
          >
            <Play className="w-4 h-4" />
            Resume Mission
          </Button>
          <Button
            variant="outline"
            onClick={onReturnToQuarters}
            className="flex-1 flex items-center justify-center gap-2"
          >
            <Home className="w-4 h-4" />
            Return to Quarters
          </Button>
        </div>

        <p className="text-xs text-muted-foreground text-center">
          Your progress is automatically saved. You can resume this mission at any time.
        </p>
      </div>
    </Modal>
  );
}
