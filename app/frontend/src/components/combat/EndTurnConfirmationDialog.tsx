import { AlertTriangle, CheckCircle, XCircle } from "lucide-react";
import { Button } from "../ui";
import { useEffect, useCallback } from "react";

export interface EndTurnConfirmationDialogProps {
  /** Whether the dialog is open */
  isOpen: boolean;
  /** Remaining full actions (0-1) */
  fullRemaining: number;
  /** Remaining quick actions (0-2+) */
  quickRemaining: number;
  /** Remaining reactions (0-1) */
  reactRemaining: number;
  /** Whether overcharge is available (not used yet) */
  canOvercharge?: boolean;
  /** Overcharge level if used */
  overchargeLevel?: number;
  /** Whether the confirmation is currently being processed */
  isProcessing?: boolean;
  /** Callback when user confirms ending turn */
  onConfirm: () => void;
  /** Callback when user cancels */
  onCancel: () => void;
}

export function EndTurnConfirmationDialog({
  isOpen,
  fullRemaining,
  quickRemaining,
  reactRemaining,
  canOvercharge = false,
  overchargeLevel = 0,
  isProcessing = false,
  onConfirm,
  onCancel,
}: EndTurnConfirmationDialogProps) {
  // ALL HOOKS MUST BE CALLED BEFORE ANY EARLY RETURNS
  // Keyboard shortcuts
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (!isOpen) return; // Guard inside callback instead
    if (e.key === 'Escape') {
      e.preventDefault();
      onCancel();
    }
    if (e.key === 'Enter' && !isProcessing) {
      e.preventDefault();
      onConfirm();
    }
  }, [isOpen, isProcessing, onCancel, onConfirm]);

  useEffect(() => {
    if (!isOpen) return; // Guard inside effect instead
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, handleKeyDown]);

  if (!isOpen) {
    return null;
  }

  const renderActionList = () => {
    const items = [];
    if (fullRemaining > 0) {
      items.push(`${fullRemaining} full action${fullRemaining > 1 ? 's' : ''}`);
    }
    if (quickRemaining > 0) {
      items.push(`${quickRemaining} quick action${quickRemaining > 1 ? 's' : ''}`);
    }
    if (reactRemaining > 0) {
      items.push(`${reactRemaining} reaction${reactRemaining > 1 ? 's' : ''}`);
    }
    if (canOvercharge) {
      items.push('overcharge available');
    }
    if (overchargeLevel > 0) {
      items.push(`overcharge level ${overchargeLevel} already used`);
    }

    if (items.length === 0) {
      return <p className="text-sm text-muted-foreground">No remaining actions.</p>;
    }

    return (
      <ul className="list-disc pl-5 space-y-1">
        {items.map((item, idx) => (
          <li key={idx} className="text-sm">
            {item}
          </li>
        ))}
      </ul>
    );
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-background rounded-lg border shadow-xl w-full max-w-md mx-4 overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        {/* Header */}
        <div className="px-6 py-4 border-b">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            <div>
              <h3 className="font-semibold text-foreground">
                End Turn Confirmation
              </h3>
              <p className="text-sm text-muted-foreground">
                You have unused actions remaining.
              </p>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="px-6 py-4 space-y-4">
          <div className="text-sm">
            <p className="text-foreground">
              Are you sure you want to end your turn? You still have:
            </p>
            <div className="mt-3 p-3 rounded bg-muted/50">
              {renderActionList()}
            </div>
            <p className="mt-3 text-muted-foreground">
              Ending your turn now will forfeit these actions.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t flex items-center justify-between">
          <div className="text-xs text-muted-foreground">
            {!isProcessing && (
              <>Press <kbd className="px-1.5 py-0.5 rounded bg-muted font-mono text-[0.7rem]">Enter</kbd> to end anyway, <kbd className="px-1.5 py-0.5 rounded bg-muted font-mono text-[0.7rem]">Esc</kbd> to cancel.</>
            )}
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={onCancel}
              disabled={isProcessing}
              className="gap-2"
            >
              <XCircle className="w-4 h-4" />
              Cancel
            </Button>
            <Button
              variant="secondary"
              onClick={onConfirm}
              disabled={isProcessing}
              className="gap-2"
            >
              {isProcessing ? (
                <>
                  <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                  Ending...
                </>
              ) : (
                <>
                  <CheckCircle className="w-4 h-4" />
                  End Anyway
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}