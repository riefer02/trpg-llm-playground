import { useState, useEffect } from "react";
import { Modal } from "../ui/modal";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { 
  Swords, 
  Clock, 
  Zap, 
  Footprints,
  Target,
  Mic,
  Keyboard,
  ChevronRight,
  ChevronLeft,
  X
} from "lucide-react";

export interface FirstCombatTutorialProps {
  isOpen: boolean;
  onClose: () => void;
  onDontShowAgain?: () => void;
}

interface TutorialStep {
  id: string;
  title: string;
  icon: React.ReactNode;
  content: React.ReactNode;
}

export function FirstCombatTutorial({ 
  isOpen, 
  onClose,
  onDontShowAgain 
}: FirstCombatTutorialProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [dontShowAgain, setDontShowAgain] = useState(false);
  const titleId = "first-combat-tutorial-title";

  // Reset step when opened
  useEffect(() => {
    if (isOpen) {
      setCurrentStep(0);
      setDontShowAgain(false);
    }
  }, [isOpen]);

  const steps: TutorialStep[] = [
    {
      id: "welcome",
      title: "Welcome to Combat",
      icon: <Swords className="w-6 h-6 text-primary" />,
      content: (
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">
            You're about to engage in tactical mech combat. This brief tutorial will cover the basics to get you started.
          </p>
          <div className="rounded-lg border border-primary/30 bg-primary/10 p-3">
            <p className="text-sm">
              <strong>Tip:</strong> You can reopen this tutorial anytime by pressing the ? key during combat.
            </p>
          </div>
        </div>
      )
    },
    {
      id: "actions",
      title: "Actions & Economy",
      icon: <Clock className="w-6 h-6 text-blue-500" />,
      content: (
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">
            Each turn, your mech can perform:
          </p>
          <div className="grid grid-cols-3 gap-2">
            <div className="p-3 rounded-md bg-blue-500/20 border border-blue-500/50 text-center">
              <div className="text-lg font-bold text-blue-500">1</div>
              <div className="text-xs text-muted-foreground">Full Action</div>
              <div className="text-xs text-muted-foreground mt-1">Attack, Tech</div>
            </div>
            <div className="p-3 rounded-md bg-green-500/20 border border-green-500/50 text-center">
              <div className="text-lg font-bold text-green-500">2</div>
              <div className="text-xs text-muted-foreground">Quick Actions</div>
              <div className="text-xs text-muted-foreground mt-1">Move, Scan</div>
            </div>
            <div className="p-3 rounded-md bg-amber-500/20 border border-amber-500/50 text-center">
              <div className="text-lg font-bold text-amber-500">1</div>
              <div className="text-xs text-muted-foreground">Reaction</div>
              <div className="text-xs text-muted-foreground mt-1">Overwatch</div>
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            Actions are shown at the bottom of the screen. Press 1-0 to select them quickly.
          </p>
        </div>
      )
    },
    {
      id: "movement",
      title: "Movement & Positioning",
      icon: <Footprints className="w-6 h-6 text-green-500" />,
      content: (
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">
            Move your mech on the hex grid:
          </p>
          <ul className="text-sm text-muted-foreground space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-primary">•</span>
              <span><strong>Move</strong> - Travel up to your speed (Quick action)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary">•</span>
              <span><strong>Boost</strong> - Double speed, but generates heat (Full action)</span>
            </li>
          </ul>
          <div className="rounded-lg border border-border p-3">
            <p className="text-sm">
              Click hexes on the map to plot your path. Click the last hex again to undo. Position matters for cover and line of sight!
            </p>
          </div>
        </div>
      )
    },
    {
      id: "combat",
      title: "Attacking",
      icon: <Target className="w-6 h-6 text-red-500" />,
      content: (
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">
            Engage enemies with your weapons:
          </p>
          <ul className="text-sm text-muted-foreground space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-primary">•</span>
              <span><strong>Skirmish</strong> - Attack with one weapon (Full action)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary">•</span>
              <span><strong>Barrage</strong> - Fire all weapons at one target (Full action)</span>
            </li>
          </ul>
          <div className="rounded-lg border border-border bg-muted/30 p-3">
            <p className="text-sm">
              Select a target by clicking on an enemy. Hover over actions to see hit chances and damage previews.
            </p>
          </div>
        </div>
      )
    },
    {
      id: "voice",
      title: "Voice Commands",
      icon: <Mic className="w-6 h-6 text-purple-500" />,
      content: (
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">
            You can control combat with your voice:
          </p>
          <div className="rounded-lg border border-border bg-muted/30 p-3">
            <p className="text-sm font-medium mb-2">Hold Spacebar and say:</p>
            <ul className="text-sm text-muted-foreground space-y-1">
              <li>"Attack striker with rifle"</li>
              <li>"Move forward"</li>
              <li>"Use scan on tank"</li>
              <li>"End turn"</li>
            </ul>
          </div>
          <p className="text-xs text-muted-foreground">
            Release Spacebar to submit your command. Say "yes" or "no" to confirm actions.
          </p>
        </div>
      )
    },
    {
      id: "shortcuts",
      title: "Keyboard Shortcuts",
      icon: <Keyboard className="w-6 h-6 text-orange-500" />,
      content: (
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">
            Quick keys for faster play:
          </p>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="p-2 rounded bg-muted">
              <kbd className="font-mono bg-background px-1 rounded">1-0</kbd>
              <span className="ml-2 text-muted-foreground">Select actions</span>
            </div>
            <div className="p-2 rounded bg-muted">
              <kbd className="font-mono bg-background px-1 rounded">Space</kbd>
              <span className="ml-2 text-muted-foreground">Voice input</span>
            </div>
            <div className="p-2 rounded bg-muted">
              <kbd className="font-mono bg-background px-1 rounded">Escape</kbd>
              <span className="ml-2 text-muted-foreground">Cancel targeting</span>
            </div>
            <div className="p-2 rounded bg-muted">
              <kbd className="font-mono bg-background px-1 rounded">?</kbd>
              <span className="ml-2 text-muted-foreground">Help overlay</span>
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            Shortcuts are disabled when typing in input fields.
          </p>
        </div>
      )
    },
    {
      id: "ready",
      title: "You're Ready!",
      icon: <Zap className="w-6 h-6 text-yellow-500" />,
      content: (
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">
            That's the basics! Remember:
          </p>
          <ul className="text-sm text-muted-foreground space-y-1">
            <li>• Use your actions wisely (1 Full + 2 Quick per turn)</li>
            <li>• Position matters - use cover and high ground</li>
            <li>• Watch your heat - stabilize if needed</li>
            <li>• Press ? anytime for help</li>
          </ul>
          <div className="rounded-lg border border-primary/30 bg-primary/10 p-3">
            <p className="text-sm font-medium">
              Good luck, pilot!
            </p>
          </div>
        </div>
      )
    }
  ];

  const currentStepData = steps[currentStep];
  const isFirstStep = currentStep === 0;
  const isLastStep = currentStep === steps.length - 1;

  const handleNext = () => {
    if (isLastStep) {
      if (dontShowAgain && onDontShowAgain) {
        onDontShowAgain();
      }
      onClose();
    } else {
      setCurrentStep(prev => prev + 1);
    }
  };

  const handlePrevious = () => {
    if (!isFirstStep) {
      setCurrentStep(prev => prev - 1);
    }
  };

  const handleSkip = () => {
    if (dontShowAgain && onDontShowAgain) {
      onDontShowAgain();
    }
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onClose={handleSkip} ariaLabelledBy={titleId}>
      <Card className="max-w-md">
        <CardHeader className="flex flex-row items-center justify-between">
          <div className="flex items-center gap-3">
            {currentStepData.icon}
            <CardTitle id={titleId} className="text-lg">
              {currentStepData.title}
            </CardTitle>
          </div>
          <button
            onClick={handleSkip}
            className="p-1 rounded hover:bg-muted text-muted-foreground"
            aria-label="Skip tutorial"
          >
            <X className="w-4 h-4" />
          </button>
        </CardHeader>
        <CardContent className="space-y-4">
          {currentStepData.content}

          {/* Progress dots */}
          <div className="flex justify-center gap-1 pt-2">
            {steps.map((_, idx) => (
              <div
                key={idx}
                className={`w-2 h-2 rounded-full transition-colors ${
                  idx === currentStep ? 'bg-primary' : 'bg-muted-foreground/30'
                }`}
              />
            ))}
          </div>

          {/* Navigation */}
          <div className="flex items-center justify-between pt-4 border-t border-border">
            <div className="flex items-center gap-2">
              <button
                onClick={handlePrevious}
                disabled={isFirstStep}
                className="p-2 rounded hover:bg-muted disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="text-sm text-muted-foreground">
                {currentStep + 1} / {steps.length}
              </span>
              <button
                onClick={handleNext}
                className="p-2 rounded hover:bg-muted"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            <div className="flex items-center gap-2">
              {isLastStep && (
                <label className="flex items-center gap-2 text-xs text-muted-foreground mr-2">
                  <input
                    type="checkbox"
                    checked={dontShowAgain}
                    onChange={(e) => setDontShowAgain(e.target.checked)}
                    className="accent-primary"
                  />
                  Don't show again
                </label>
              )}
              <Button onClick={handleNext}>
                {isLastStep ? "Start Combat" : "Next"}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </Modal>
  );
}
