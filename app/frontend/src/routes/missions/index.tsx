/**
 * Mission Select screen.
 * Displays 3 available missions with difficulty rating, SITREP type, terrain, and enemy count.
 * Player selects one to view briefing.
 */

import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useRef, useEffect, useState } from "react";
import { Button } from "../../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui";
import { useMissions, useActiveCharacter } from "../../lib/api";
import { Lock, Star, ArrowLeft } from "lucide-react";

export const Route = createFileRoute("/missions/" as const)({
  component: MissionSelect,
});

function MissionSelect() {
  const navigate = useNavigate();
  const { missions, isLoading, error } = useMissions();
  const { character } = useActiveCharacter();
  const pilotLevel = character?.level ?? 0;

  // Refs for keyboard navigation between mission cards
  const missionRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const backButtonRef = useRef<HTMLButtonElement>(null);
  const [focusedIndex, setFocusedIndex] = useState<number>(-1);

  // ALL HOOKS MUST BE CALLED BEFORE ANY EARLY RETURNS
  // Navigation handlers (defined here so useEffect can reference them)
  const handleSelectMission = (missionId: string) => {
    navigate({ to: "/missions/$missionId/briefing", params: { missionId } });
  };

  // Keyboard navigation with arrow keys
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Arrow navigation between mission cards
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        const nextIndex = focusedIndex === -1 ? 0 : (focusedIndex + 1) % missions.length;
        missionRefs.current[nextIndex]?.focus();
        setFocusedIndex(nextIndex);
      } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        const prevIndex = focusedIndex === -1 ? missions.length - 1 : (focusedIndex - 1 + missions.length) % missions.length;
        missionRefs.current[prevIndex]?.focus();
        setFocusedIndex(prevIndex);
      } else if (e.key === "Escape") {
        e.preventDefault();
        backButtonRef.current?.focus();
      } else if (e.key === "Enter" && focusedIndex >= 0) {
        e.preventDefault();
        const mission = missions[focusedIndex];
        if (!mission.locked) {
          handleSelectMission(mission.id);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [focusedIndex, missions]);

  // Focus first mission on mount
  useEffect(() => {
    if (missions.length > 0 && focusedIndex === -1) {
      missionRefs.current[0]?.focus();
      setFocusedIndex(0);
    }
  }, [missions]);

  // Redirect to quarters if no active pilot (should not happen due to quarters redirect)
  if (!isLoading && !character) {
    navigate({ to: "/quarters" });
    return null;
  }

  // Show loading state
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="inline-block w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin" />
          <p className="text-muted-foreground">Loading available missions...</p>
        </div>
      </div>
    );
  }

  // Show error state
  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center space-y-4">
          <p className="text-destructive">Failed to load mission data</p>
          <Button variant="outline" onClick={() => window.location.reload()}>
            Retry
          </Button>
        </div>
      </div>
    );
  }

  // Navigation handlers
  const handleBack = () => navigate({ to: "/quarters" });

  // Render stars for difficulty
  const renderStars = (difficulty: number) => {
    return (
      <div className="flex gap-1">
        {[1, 2, 3].map((star) => (
          <Star
            key={star}
            className={`w-4 h-4 ${star <= difficulty ? "fill-yellow-400 text-yellow-400" : "fill-muted text-muted"}`}
          />
        ))}
      </div>
    );
  };

  // Format SITREP type for display
  const formatSitrep = (sitrep: string) => {
    const map: Record<string, string> = {
      control: "Control",
      escort: "Escort",
      extract: "Extract",
      hold_out: "Holdout",
      gauntlet: "Gauntlet",
      recon: "Recon",
    };
    return map[sitrep] || sitrep;
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/30 p-6">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold tracking-tight text-foreground">
            Available Missions
          </h1>
          <p className="text-xl text-muted-foreground">
            Select a mission to view briefing and deploy
          </p>
          <p className="text-sm text-muted-foreground">
            Pilot License Level: LL{pilotLevel}
          </p>
        </div>

        {/* Back button */}
        <div>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleBack}
            ref={backButtonRef}
            className="flex items-center gap-2"
            aria-label="Return to pilot quarters"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Quarters
          </Button>
        </div>

        {/* Mission grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {missions.map((mission, index) => (
            <Card
              key={mission.id}
              className={`dashboard-surface relative overflow-hidden transition-all ${
                mission.locked
                  ? "opacity-70 cursor-not-allowed"
                  : "hover:shadow-lg hover:scale-[1.02] cursor-pointer"
              }`}
              aria-label={`Mission: ${mission.name}, Difficulty: ${mission.difficulty} stars, SITREP: ${mission.sitrep}, Terrain: ${mission.terrain}, Enemy count: ${mission.enemyCount}`}
            >
              {/* Lock overlay */}
              {mission.locked && (
                <div className="absolute inset-0 bg-background/80 flex items-center justify-center z-10">
                  <div className="text-center space-y-2">
                    <Lock className="w-12 h-12 mx-auto text-muted-foreground" />
                    <p className="font-bold">Mission Locked</p>
                    <p className="text-sm text-muted-foreground">
                      Requires License Level {mission.difficulty - 1}
                    </p>
                  </div>
                </div>
              )}

              <button
                ref={(el) => {
                  missionRefs.current[index] = el;
                }}
                onClick={() => !mission.locked && handleSelectMission(mission.id)}
                disabled={mission.locked}
                className="w-full text-left focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 rounded-lg"
                aria-disabled={mission.locked}
                onFocus={() => setFocusedIndex(index)}
              >
                <CardHeader>
                  <div className="flex justify-between items-start">
                    <CardTitle className="text-xl">{mission.name}</CardTitle>
                    {renderStars(mission.difficulty)}
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium text-muted-foreground">SITREP</span>
                      <span className="font-semibold">{formatSitrep(mission.sitrep)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium text-muted-foreground">Terrain</span>
                      <span className="font-semibold capitalize">{mission.terrain}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium text-muted-foreground">Enemy Count</span>
                      <span className="font-semibold">{mission.enemyCount}</span>
                    </div>
                  </div>
                  {mission.description && (
                    <p className="text-sm text-muted-foreground pt-2 border-t border-border">
                      {mission.description}
                    </p>
                  )}
                </CardContent>
              </button>
            </Card>
          ))}
        </div>

        {/* Keyboard navigation note */}
        <div className="text-sm text-muted-foreground pt-8 border-t border-border">
          <p>
            Use <kbd>Tab</kbd> to navigate between missions, <kbd>Arrow keys</kbd> to cycle,{" "}
            <kbd>Enter</kbd> to select a mission, <kbd>Escape</kbd> to focus back button.
          </p>
        </div>
      </div>
    </div>
  );
}