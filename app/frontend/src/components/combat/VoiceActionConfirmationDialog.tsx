import { CheckCircle, AlertCircle, Move, Sword, Zap, Target, Navigation } from "lucide-react";
import { Button } from "../ui";
import { useEffect, useCallback } from "react";

/**
 * Enhanced confirmation dialog for voice-parsed actions.
 * Displays action details in a user-friendly format instead of raw JSON.
 */

export interface VoiceActionConfirmationDialogProps {
  /** Whether the dialog is open */
  isOpen: boolean;
  /** Raw transcript from speech recognition */
  transcript: string;
  /** Parsed action data (ActionRequest object) */
  parsedAction: Record<string, unknown> | null;
  /** Error message if parsing failed */
  error: string | null;
  /** Whether the action is currently being executed */
  isExecuting: boolean;
  /** Callback when dialog should close (cancel) */
  onClose: () => void;
  /** Callback when user confirms the action */
  onConfirm: (action: Record<string, unknown>) => void;
  /** Optional function to get combatant display name from ID */
  getCombatantName?: (id: string) => string;
  /** Optional function to get weapon display name from ID */
  getWeaponName?: (id: string) => string;
}

export function VoiceActionConfirmationDialog({
  isOpen,
  transcript,
  parsedAction,
  error,
  isExecuting,
  onClose,
  onConfirm,
  getCombatantName = (id) => id,
  getWeaponName = (id) => id,
}: VoiceActionConfirmationDialogProps) {
  const hasError = error !== null;
  const hasAction = parsedAction !== null && !hasError;

  // ALL HOOKS MUST BE CALLED BEFORE ANY EARLY RETURNS
  // Keyboard shortcuts
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (!isOpen) return; // Guard inside callback instead
    if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    }
    if (e.key === 'Enter' && hasAction && !isExecuting) {
      e.preventDefault();
      onConfirm(parsedAction!);
    }
  }, [isOpen, hasAction, isExecuting, onClose, onConfirm, parsedAction]);

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

  // Helper to get action type display
  const getActionTypeDisplay = (actionType: string): { label: string; icon: React.ReactNode; color: string } => {
    switch (actionType) {
      case "full":
        return { label: "Full Action", icon: <Sword className="w-4 h-4" />, color: "text-purple-500" };
      case "quick":
        return { label: "Quick Action", icon: <Zap className="w-4 h-4" />, color: "text-blue-500" };
      case "free":
        return { label: "Free Action", icon: <CheckCircle className="w-4 h-4" />, color: "text-green-500" };
      case "move":
        return { label: "Move", icon: <Move className="w-4 h-4" />, color: "text-amber-500" };
      case "protocol":
        return { label: "Protocol", icon: <CheckCircle className="w-4 h-4" />, color: "text-cyan-500" };
      case "reaction":
        return { label: "Reaction", icon: <AlertCircle className="w-4 h-4" />, color: "text-orange-500" };
      default:
        return { label: actionType, icon: <Target className="w-4 h-4" />, color: "text-gray-500" };
    }
  };

  // Format action details for display
  const renderActionDetails = () => {
    if (!parsedAction) return null;

    const actionType = parsedAction.action_type as string;
    const actionId = parsedAction.action_id as string;
    const targetIds = parsedAction.target_ids as string[] | undefined;
    const weaponId = parsedAction.weapon_id as string | undefined;
    const systemId = parsedAction.system_id as string | undefined;
    const movementPath = parsedAction.movement_path as any[] | undefined;

    const typeDisplay = getActionTypeDisplay(actionType);

    return (
      <div className="space-y-3">
        {/* Action Type Badge */}
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-md ${typeDisplay.color.replace('text-', 'bg-')}/10 border ${typeDisplay.color.replace('text-', 'border-')}/20`}>
          <div className={typeDisplay.color}>{typeDisplay.icon}</div>
          <div className="font-medium text-sm">{typeDisplay.label}</div>
        </div>

        {/* Action ID (humanized) */}
        <div className="text-sm">
          <div className="font-medium text-muted-foreground">Command:</div>
          <div className="font-medium capitalize">{actionId.replace(/_/g, ' ')}</div>
        </div>

        {/* Targets */}
        {targetIds && targetIds.length > 0 && (
          <div className="text-sm">
            <div className="font-medium text-muted-foreground">Targets:</div>
            <div className="flex flex-wrap gap-1">
              {targetIds.map((id) => (
                <span key={id} className="px-2 py-1 rounded bg-muted text-xs font-medium">
                  {getCombatantName(id)}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Weapon/System */}
        {weaponId && (
          <div className="text-sm">
            <div className="font-medium text-muted-foreground">Weapon:</div>
            <div className="font-medium">{getWeaponName(weaponId)}</div>
          </div>
        )}
        {systemId && (
          <div className="text-sm">
            <div className="font-medium text-muted-foreground">System:</div>
            <div className="font-medium">{systemId}</div>
          </div>
        )}

        {/* Movement Path */}
        {movementPath && movementPath.length > 0 && (
          <div className="text-sm">
            <div className="font-medium text-muted-foreground">Movement Path:</div>
            <div className="flex flex-wrap gap-1">
              {movementPath.map((step, idx) => (
                <span key={idx} className="px-2 py-1 rounded bg-muted text-xs font-medium flex items-center gap-1">
                  <Navigation className="w-3 h-3" />
                  {step.q},{step.r}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Raw JSON (collapsible for debugging) */}
        <details className="text-xs">
          <summary className="cursor-pointer text-muted-foreground hover:text-foreground">Show raw data</summary>
          <pre className="mt-2 p-2 bg-muted rounded overflow-x-auto">
            {JSON.stringify(parsedAction, null, 2)}
          </pre>
        </details>
      </div>
    );
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-background rounded-lg border shadow-xl w-full max-w-md mx-4 overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        {/* Header */}
        <div className="px-6 py-4 border-b">
          <div className="flex items-center gap-3">
            {hasError ? (
              <AlertCircle className="w-5 h-5 text-destructive" />
            ) : hasAction ? (
              <CheckCircle className="w-5 h-5 text-green-500" />
            ) : (
              <AlertCircle className="w-5 h-5 text-amber-500" />
            )}
            <div>
              <h3 className="font-semibold text-foreground">
                {hasError ? "Voice Command Error" : "Confirm Voice Action"}
              </h3>
              <p className="text-sm text-muted-foreground">
                {hasError
                  ? "The voice command could not be understood."
                  : "Please verify the action before executing."}
              </p>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="px-6 py-4 space-y-4 max-h-[60vh] overflow-y-auto">
          {/* Transcript */}
          <div className="text-sm">
            <div className="font-medium text-muted-foreground">You said:</div>
            <div className="mt-1 p-2 bg-muted rounded font-medium italic">"{transcript}"</div>
          </div>

          {/* Error Message */}
          {hasError && (
            <div className="p-3 rounded-md bg-destructive/10 border border-destructive/20">
              <div className="flex items-center gap-2 text-destructive">
                <AlertCircle className="w-4 h-4" />
                <div className="font-medium">Error</div>
              </div>
              <div className="mt-1 text-sm">{error}</div>
            </div>
          )}

          {/* Parsed Action Details */}
          {hasAction && renderActionDetails()}

          {/* No Action Found (but no error) */}
          {!hasAction && !hasError && (
            <div className="p-3 rounded-md bg-amber-500/10 border border-amber-500/20">
              <div className="flex items-center gap-2 text-amber-500">
                <AlertCircle className="w-4 h-4" />
                <div className="font-medium">No Action Found</div>
              </div>
              <div className="mt-1 text-sm">
                The voice command was recognized but no specific action could be determined. Try rephrasing your command.
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t flex items-center justify-between">
          <div className="text-xs text-muted-foreground">
            {hasAction && !isExecuting && (
              <>Press <kbd className="px-1.5 py-0.5 rounded bg-muted font-mono text-[0.7rem]">Enter</kbd> to confirm, <kbd className="px-1.5 py-0.5 rounded bg-muted font-mono text-[0.7rem]">Esc</kbd> to cancel. Or say "Yes" or "No".</>
            )}
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={onClose}
              disabled={isExecuting}
            >
              Cancel
            </Button>
            {hasAction && (
              <Button
                onClick={() => onConfirm(parsedAction!)}
                disabled={isExecuting}
                className="gap-2"
              >
                {isExecuting ? (
                  <>
                    <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                    Executing...
                  </>
                ) : (
                  <>
                    <CheckCircle className="w-4 h-4" />
                    Confirm Action
                  </>
                )}
              </Button>
            )}
            {!hasAction && !hasError && (
              <Button
                variant="secondary"
                onClick={onClose}
                disabled={isExecuting}
              >
                Close
              </Button>
            )}
            {hasError && (
              <Button
                variant="secondary"
                onClick={onClose}
                disabled={isExecuting}
              >
                Dismiss
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}