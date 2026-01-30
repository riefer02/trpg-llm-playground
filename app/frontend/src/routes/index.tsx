/**
 * Title screen for Lancer Tactics AI.
 * Entry point for the entire game.
 */

import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Button } from "../components/ui/button";
import { useSavedPilot } from "../lib/api/title";
import { useCreateDemoCombat } from "../lib/api";

export const Route = createFileRoute("/" as const)({
  component: TitleScreen,
});

function TitleScreen() {
  const navigate = useNavigate();
  const { hasSavedPilot, isLoading } = useSavedPilot();
  const createDemo = useCreateDemoCombat();

  const handleNewPilot = () => {
    navigate({ to: "/characters/new" });
  };

  const handleContinue = () => {
    navigate({ to: "/quarters" });
  };

  const handleQuickBattle = async () => {
    try {
      const session = await createDemo.mutateAsync("skirmish");
      navigate({ to: `/combat/${session.id}` });
    } catch (error) {
      console.error("Failed to create quick battle:", error);
    }
  };

  const handleSettings = () => {
    navigate({ to: "/settings" });
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-b from-background to-muted/30 p-6">
      <div className="max-w-2xl w-full text-center space-y-12">
        {/* Title and Tagline */}
        <div className="space-y-4">
          <h1 className="text-6xl font-bold tracking-tight text-foreground">
            LANCER TACTICS AI
          </h1>
          <p className="text-xl text-muted-foreground">
            Your mech. Your voice. An AI that fights back.
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col gap-4 max-w-md mx-auto">
          <Button
            variant="primary"
            size="lg"
            onClick={handleNewPilot}
            className="w-full py-6 text-lg"
            aria-label="Create new pilot"
            autoFocus
          >
            New Pilot
          </Button>
          <Button
            variant="secondary"
            size="lg"
            onClick={handleContinue}
            disabled={!hasSavedPilot || isLoading}
            className="w-full py-6 text-lg"
            aria-label={hasSavedPilot ? "Continue with saved pilot" : isLoading ? "Checking for saved pilot..." : "No saved pilot available"}
          >
            {isLoading ? (
              <>
                <span className="inline-block w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin mr-2" />
                Checking...
              </>
            ) : (
              <>
                Continue
                {!hasSavedPilot && (
                  <span className="ml-2 text-xs opacity-70">(No saved pilot)</span>
                )}
              </>
            )}
          </Button>
          <Button
            variant="outline"
            size="lg"
            onClick={handleQuickBattle}
            className="w-full py-6 text-lg"
            aria-label="Start quick battle with preset pilot and mech"
            disabled={createDemo.isPending}
          >
            {createDemo.isPending ? (
              <>
                <span className="inline-block w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin mr-2" />
                Setting up combat...
              </>
            ) : (
              "Quick Battle"
            )}
          </Button>
          <Button
            variant="ghost"
            size="lg"
            onClick={handleSettings}
            className="w-full py-6 text-lg"
            aria-label="Open settings"
          >
            Settings
          </Button>
        </div>

        {/* Accessibility note */}
        <div className="text-sm text-muted-foreground pt-8 border-t border-border">
          <p>
            All buttons are keyboard accessible. Use <kbd>Tab</kbd> to navigate,{" "}
            <kbd>Enter</kbd> to activate. Quick Battle launches a combat scenario with preset units for testing.
          </p>
        </div>
      </div>
    </div>
  );
}


