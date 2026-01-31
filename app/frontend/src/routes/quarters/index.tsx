/**
 * Pilot Quarters Hub.
 * Central hub screen showing pilot/mech summary with navigation to missions, loadout, and compendium.
 */

import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useRef, useEffect } from "react";
import { Button } from "../../components/ui/button";
import { useActiveCharacter, useMissionCount } from "../../lib/api/quarters";
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui";

export const Route = createFileRoute("/quarters/" as const)({
  component: QuartersHub,
});

function QuartersHub() {
  const navigate = useNavigate();
  const { character, isLoading, error } = useActiveCharacter();

  // Redirect to title if no active pilot
  if (!isLoading && !character) {
    navigate({ to: "/" });
    return null;
  }

  // Show loading state
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="inline-block w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin" />
          <p className="text-muted-foreground">Loading pilot quarters...</p>
        </div>
      </div>
    );
  }

  // Show error state
  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center space-y-4">
          <p className="text-destructive">Failed to load pilot data</p>
          <Button variant="outline" onClick={() => window.location.reload()}>
            Retry
          </Button>
        </div>
      </div>
    );
  }

  // Mission count (placeholder)
  const { count: missionCount } = useMissionCount();

  // Refs for keyboard navigation
  const pilotButtonRef = useRef<HTMLButtonElement>(null);
  const mechButtonRef = useRef<HTMLButtonElement>(null);
  const licensesButtonRef = useRef<HTMLButtonElement>(null);
  const missionsButtonRef = useRef<HTMLButtonElement>(null);
  const compendiumButtonRef = useRef<HTMLButtonElement>(null);
  const deployButtonRef = useRef<HTMLButtonElement>(null);
  const buttonRefs = [pilotButtonRef, mechButtonRef, licensesButtonRef, missionsButtonRef, compendiumButtonRef, deployButtonRef];

  // Keyboard navigation with arrow keys
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        const currentIndex = buttonRefs.findIndex(ref => ref.current === document.activeElement);
        const nextIndex = (currentIndex + 1) % buttonRefs.length;
        buttonRefs[nextIndex]?.current?.focus();
      } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        const currentIndex = buttonRefs.findIndex(ref => ref.current === document.activeElement);
        const prevIndex = (currentIndex - 1 + buttonRefs.length) % buttonRefs.length;
        buttonRefs[prevIndex]?.current?.focus();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Extract pilot and mech info
  const pilotName = character?.callsign || "Unknown";
  const licenseLevel = character?.level || 0;
  const salvage = character?.pilot?.salvage || 0;
  
  // Find active mech frame name
  let mechFrame = "No active mech";
  let mechHP = "N/A";
  let mechArmor = "N/A";
  
  if (character?.active_mech_id && character?.mechs) {
    const activeMech = character.mechs.find(m => m.id === character.active_mech_id);
    if (activeMech) {
      mechFrame = activeMech.frame_id; // TODO: get frame name from compendium
      mechHP = character.active_mech_stats?.hp?.toString() || "N/A";
      mechArmor = character.active_mech_stats?.armor?.toString() || "N/A";
    }
  }

  // Navigation handlers
  const handlePilot = () => navigate({ to: "/quarters/pilot" });
  const handleMech = () => navigate({ to: "/quarters/mech" });
  const handleLicenses = () => navigate({ to: "/quarters/licenses" });
  const handleMissions = () => navigate({ to: "/missions" });
  const handleCompendium = () => navigate({ to: "/compendium" });
  const handleDeploy = () => {
    // TODO: navigate to selected mission briefing or mission select
    navigate({ to: "/missions" });
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/30 p-6">
      <div className="max-w-4xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold tracking-tight text-foreground">
            Pilot Quarters
          </h1>
          <p className="text-xl text-muted-foreground">
            Mission control and loadout management
          </p>
        </div>

        {/* Pilot/Mech Summary Card */}
        <Card className="dashboard-surface">
          <CardHeader>
            <CardTitle>Current Status</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
               <div className="space-y-2">
                 <h3 className="text-sm font-medium text-muted-foreground">
                   Pilot
                 </h3>
                 <p className="text-2xl font-bold">{pilotName}</p>
                 <p className="text-sm">License Level: LL{licenseLevel}</p>
                 <p className="text-sm">Salvage: {salvage} ⚙️</p>
               </div>
              <div className="space-y-2">
                <h3 className="text-sm font-medium text-muted-foreground">
                  Mech
                </h3>
                <p className="text-2xl font-bold">{mechFrame}</p>
                <p className="text-sm">
                  HP: {mechHP} | Armor: {mechArmor}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Navigation Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          <Button
            variant="secondary"
            size="lg"
            onClick={handlePilot}
            ref={pilotButtonRef}
            className="h-24 flex flex-col items-center justify-center p-4"
            aria-label="View pilot details"
          >
            <span className="text-2xl mb-2">👤</span>
            <span>Pilot</span>
          </Button>
          <Button
            variant="secondary"
            size="lg"
            onClick={handleMech}
            ref={mechButtonRef}
            className="h-24 flex flex-col items-center justify-center p-4"
            aria-label="View mech loadout"
          >
            <span className="text-2xl mb-2">🤖</span>
            <span>Mech</span>
          </Button>
          <Button
            variant="secondary"
            size="lg"
            onClick={handleLicenses}
            ref={licensesButtonRef}
            className="h-24 flex flex-col items-center justify-center p-4"
            aria-label="Unlock licenses"
          >
            <span className="text-2xl mb-2">🔓</span>
            <span>Licenses</span>
          </Button>
          <Button
            variant="secondary"
            size="lg"
            onClick={handleMissions}
            ref={missionsButtonRef}
            className="h-24 flex flex-col items-center justify-center p-4 relative"
            aria-label="View available missions"
          >
            <span className="text-2xl mb-2">🎯</span>
            <span>Missions</span>
            {/* Mission count badge - placeholder */}
             <span className="absolute top-2 right-2 bg-primary text-primary-foreground text-xs rounded-full w-6 h-6 flex items-center justify-center">
               {missionCount}
             </span>
          </Button>
          <Button
            variant="secondary"
            size="lg"
            onClick={handleCompendium}
            ref={compendiumButtonRef}
            className="h-24 flex flex-col items-center justify-center p-4"
            aria-label="Browse compendium"
          >
            <span className="text-2xl mb-2">📚</span>
            <span>Compendium</span>
          </Button>
        </div>

        {/* Deploy to Mission Button */}
        <div className="text-center">
           <Button
            variant="primary"
            size="xl"
            onClick={handleDeploy}
            ref={deployButtonRef}
            disabled={true} // TODO: disable if no mission selected
            className="px-12 py-6 text-lg"
            aria-label="Deploy to selected mission"
          >
            DEPLOY TO MISSION
          </Button>
          <p className="text-sm text-muted-foreground mt-2">
            Select a mission from the Missions screen first
          </p>
        </div>

        {/* Keyboard navigation note */}
        <div className="text-sm text-muted-foreground pt-8 border-t border-border">
          <p>
            Use <kbd>Tab</kbd> to navigate between buttons, <kbd>Enter</kbd> to
            activate. Arrow keys cycle through navigation buttons.
          </p>
        </div>
      </div>
    </div>
  );
}