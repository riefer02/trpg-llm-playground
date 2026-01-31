import { AlertTriangle } from "lucide-react";
import { Button, Modal } from "../ui";
import { useCallback, useEffect } from "react";

export interface ForfeitConfirmationModalProps {
  /** Whether the modal is open */
  isOpen: boolean;
  /** Whether the forfeit is currently being processed */
  isSubmitting?: boolean;
  /** Callback when user confirms forfeit */
  onConfirm: () => void;
  /** Callback when user cancels */
  onCancel: () => void;
}

export function ForfeitConfirmationModal({
  isOpen,
  isSubmitting = false,
  onConfirm,
  onCancel,
}: ForfeitConfirmationModalProps) {
  // Keyboard shortcuts: Enter to confirm, Escape to cancel
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onCancel();
      }
      if (e.key === "Enter" && !isSubmitting) {
        e.preventDefault();
        onConfirm();
      }
      // Ctrl+Q also triggers confirmation (same as button)
      if (e.ctrlKey && e.key === "q" && !isSubmitting) {
        e.preventDefault();
        onConfirm();
      }
    },
    [isSubmitting, onCancel, onConfirm]
  );

  useEffect(() => {
    if (isOpen) {
      document.addEventListener("keydown", handleKeyDown);
      return () => {
        document.removeEventListener("keydown", handleKeyDown);
      };
    }
  }, [isOpen, handleKeyDown]);

  return (
    <Modal isOpen={isOpen} onClose={onCancel} ariaLabel="Forfeit mission confirmation">
      <div className="bg-background rounded-lg border shadow-xl w-full overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            <div>
              <h3 className="font-semibold text-foreground">Forfeit Mission</h3>
              <p className="text-sm text-muted-foreground">
                This will count as a defeat.
              </p>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="px-6 py-4 space-y-4">
          <div className="text-sm">
            <p className="text-foreground">
              Are you sure you want to forfeit the mission?
            </p>
            <ul className="mt-3 pl-5 space-y-2 list-disc text-muted-foreground">
              <li>This counts as a <strong>DEFEAT</strong> outcome</li>
              <li>You will receive partial salvage based on enemies defeated so far</li>
              <li>No experience points will be awarded</li>
              <li>You will return to the debrief screen</li>
            </ul>
            <div className="mt-4 p-3 rounded bg-destructive/10 border border-destructive/20">
              <p className="text-sm text-destructive-foreground">
                <strong>Warning:</strong> This action cannot be undone. You can only forfeit during your turn.
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t flex items-center justify-between">
          <div className="text-xs text-muted-foreground">
            {!isSubmitting && (
              <>
                Press{" "}
                <kbd className="px-1.5 py-0.5 rounded bg-muted font-mono text-[0.7rem]">
                  Enter
                </kbd>{" "}
                or{" "}
                <kbd className="px-1.5 py-0.5 rounded bg-muted font-mono text-[0.7rem]">
                  Ctrl+Q
                </kbd>{" "}
                to forfeit,{" "}
                <kbd className="px-1.5 py-0.5 rounded bg-muted font-mono text-[0.7rem]">
                  Esc
                </kbd>{" "}
                to cancel.
              </>
            )}
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={onCancel}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={onConfirm}
              disabled={isSubmitting}
            >
              {isSubmitting ? "Forfeiting..." : "Forfeit Mission"}
            </Button>
          </div>
        </div>
      </div>
    </Modal>
  );
}