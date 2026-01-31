import { Settings, HelpCircle, Flag, Play } from "lucide-react";
import { Button, Modal } from "../ui";
import { useCallback, useEffect } from "react";

export interface PauseMenuProps {
  /** Whether the modal is open */
  isOpen: boolean;
  /** Callback when user resumes (closes pause menu) */
  onResume: () => void;
  /** Callback when user opens settings */
  onOpenSettings: () => void;
  /** Callback when user opens help */
  onOpenHelp: () => void;
  /** Callback when user opens forfeit confirmation */
  onOpenForfeit: () => void;
  /** Whether the game is currently paused (for overlay) */
  isPaused?: boolean;
}

export function PauseMenu({
  isOpen,
  onResume,
  onOpenSettings,
  onOpenHelp,
  onOpenForfeit,
  isPaused = false,
}: PauseMenuProps) {
  // Keyboard shortcuts: Escape to resume, Enter for resume (default button)
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape" || e.key === " ") {
        e.preventDefault();
        onResume();
      }
      // Optional: number keys for quick selection (1: Resume, 2: Settings, 3: Forfeit, 4: Help)
      if (e.key === "1" || e.key === "Enter") {
        e.preventDefault();
        onResume();
      }
      if (e.key === "2") {
        e.preventDefault();
        onOpenSettings();
      }
      if (e.key === "3") {
        e.preventDefault();
        onOpenForfeit();
      }
      if (e.key === "4") {
        e.preventDefault();
        onOpenHelp();
      }
    },
    [onResume, onOpenSettings, onOpenForfeit, onOpenHelp]
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
    <>
      {/* Pause overlay that dims the combat view */}
      {isPaused && (
        <div className="fixed inset-0 z-40 bg-black/40 backdrop-blur-[2px] pointer-events-none" />
      )}
      <Modal isOpen={isOpen} onClose={onResume} ariaLabel="Pause menu">
        <div className="bg-background rounded-lg border shadow-xl w-full max-w-md overflow-hidden">
          {/* Header */}
          <div className="px-6 py-4 border-b">
            <div className="flex items-center gap-3">
              <div className="w-5 h-5 rounded-full bg-primary/20 flex items-center justify-center">
                <Play className="w-3 h-3 text-primary fill-primary" />
              </div>
              <div>
                <h3 className="font-semibold text-foreground">Game Paused</h3>
                <p className="text-sm text-muted-foreground">
                  Combat is paused. Select an option below.
                </p>
              </div>
            </div>
          </div>

          {/* Menu Options */}
          <div className="px-6 py-4 space-y-2">
            <Button
              variant="outline"
              size="lg"
              className="w-full justify-start h-auto py-3 px-4"
              onClick={onResume}
              autoFocus
            >
              <Play className="w-4 h-4 mr-3" />
              <div className="text-left">
                <div className="font-medium">Resume</div>
                <div className="text-xs text-muted-foreground">
                  Return to combat (Escape or Space)
                </div>
              </div>
            </Button>

            <Button
              variant="outline"
              size="lg"
              className="w-full justify-start h-auto py-3 px-4"
              onClick={onOpenSettings}
            >
              <Settings className="w-4 h-4 mr-3" />
              <div className="text-left">
                <div className="font-medium">Settings</div>
                <div className="text-xs text-muted-foreground">
                  Audio, voice, display, accessibility
                </div>
              </div>
            </Button>

            <Button
              variant="outline"
              size="lg"
              className="w-full justify-start h-auto py-3 px-4"
              onClick={onOpenForfeit}
            >
              <Flag className="w-4 h-4 mr-3" />
              <div className="text-left">
                <div className="font-medium">Forfeit Mission</div>
                <div className="text-xs text-muted-foreground">
                  Surrender and count as defeat
                </div>
              </div>
            </Button>

            <Button
              variant="outline"
              size="lg"
              className="w-full justify-start h-auto py-3 px-4"
              onClick={onOpenHelp}
            >
              <HelpCircle className="w-4 h-4 mr-3" />
              <div className="text-left">
                <div className="font-medium">Help</div>
                <div className="text-xs text-muted-foreground">
                  Keyboard shortcuts and combat guide
                </div>
              </div>
            </Button>
          </div>

          {/* Footer with quick keys */}
          <div className="px-6 py-3 border-t bg-muted/30 text-xs text-muted-foreground">
            <div className="flex items-center justify-between">
              <div>
                Quick keys:{" "}
                <kbd className="px-1.5 py-0.5 rounded bg-muted font-mono text-[0.7rem]">
                  1
                </kbd>{" "}
                Resume ·{" "}
                <kbd className="px-1.5 py-0.5 rounded bg-muted font-mono text-[0.7rem]">
                  2
                </kbd>{" "}
                Settings ·{" "}
                <kbd className="px-1.5 py-0.5 rounded bg-muted font-mono text-[0.7rem]">
                  3
                </kbd>{" "}
                Forfeit ·{" "}
                <kbd className="px-1.5 py-0.5 rounded bg-muted font-mono text-[0.7rem]">
                  4
                </kbd>{" "}
                Help
              </div>
              <div>
                <kbd className="px-1.5 py-0.5 rounded bg-muted font-mono text-[0.7rem]">
                  Esc
                </kbd>{" "}
                or{" "}
                <kbd className="px-1.5 py-0.5 rounded bg-muted font-mono text-[0.7rem]">
                  Space
                </kbd>{" "}
                to resume
              </div>
            </div>
          </div>
        </div>
      </Modal>
    </>
  );
}