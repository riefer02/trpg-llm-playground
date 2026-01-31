/**
 * Mission Debrief placeholder.
 * Post-mission screen showing outcome, narrative epilogue, rewards earned, and statistics.
 * Static content for MVP.
 */

import { createFileRoute, useNavigate, useParams, useSearch } from "@tanstack/react-router";
import { useEffect, useRef } from 'react';
import { Button } from "../../../components/ui/button";
import { ArrowLeft, Award, Target, Users, Zap, Shield, Heart, TrendingUp } from "lucide-react";
import { useMission } from "../../../lib/api/missions";
import { useActiveCharacter } from "../../../lib/api/quarters";
import { useAutoSave } from "../../../lib/save/useSaveSlots";

interface DebriefSearch {
  outcome?: "victory" | "defeat";
  turns?: number;
  damageDealt?: number;
  damageReceived?: number;
  xp?: number;
  salvage?: number;
}

export const Route = createFileRoute("/missions/$missionId/debrief" as const)({
  component: MissionDebrief,
  validateSearch: (search: Record<string, unknown>): DebriefSearch => ({
    outcome: search.outcome === "defeat" ? "defeat" : "victory",
    turns: typeof search.turns === "number" ? search.turns : 8,
    damageDealt: typeof search.damageDealt === "number" ? search.damageDealt : 2450,
    damageReceived: typeof search.damageReceived === "number" ? search.damageReceived : 1200,
    xp: typeof search.xp === "number" ? search.xp : undefined,
    salvage: typeof search.salvage === "number" ? search.salvage : undefined,
  }),
});

function MissionDebrief() {
  const navigate = useNavigate();
  const { missionId } = useParams({ from: "/missions/$missionId/debrief" });
  const search = useSearch({ from: "/missions/$missionId/debrief" });
  const { mission, isLoading, error } = useMission(missionId);
  const { character } = useActiveCharacter();
  const { triggerAutoSave } = useAutoSave();
  const hasAutoSaved = useRef(false);

  useEffect(() => {
    if (search.salvage !== undefined && character && !hasAutoSaved.current) {
      triggerAutoSave(character);
      hasAutoSaved.current = true;
    }
  }, [search.salvage, character, triggerAutoSave]);

  const handleContinue = () => navigate({ to: "/quarters" });
  const handleBack = () => navigate({ to: "/missions" });

  // Epilogue text based on outcome
  const epilogueVictory = `The enemy forces have been neutralized and the objective secured. Union command commends your performance in the ${mission?.terrain ?? "mission"} theater. Your mech sustained moderate damage but remains operational for future deployments.\n\nSalvage teams recovered valuable components from the wreckage, adding to your reserves. The success of this operation strengthens Union's position in the sector.`;

  const epilogueDefeat = `Despite valiant efforts, the mission could not be completed. Enemy reinforcements overwhelmed your position, forcing an emergency extraction. The ${mission?.terrain ?? "mission"} theater remains under hostile control.\n\nYour mech sustained significant damage and will require extensive repairs. Union command has noted the setback but acknowledges the difficult circumstances.`;

  const epilogue = search.outcome === "victory" ? epilogueVictory : epilogueDefeat;

  // Reward values (use actual awarded amounts if provided, otherwise placeholder)
  const xpEarned = search.xp ?? (mission?.difficulty ? mission.difficulty * 250 : 500);
  const salvageEarned = search.salvage ?? (mission?.difficulty ? mission.difficulty * 150 : 300);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin mb-4" />
          <p className="text-muted-foreground">Loading mission debrief...</p>
        </div>
      </div>
    );
  }

  if (error || !mission) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center space-y-4">
          <h1 className="text-2xl font-bold text-destructive">Mission Not Found</h1>
          <p className="text-muted-foreground">
            The requested mission debrief could not be loaded.
          </p>
          <Button variant="outline" onClick={handleBack}>
            Return to Mission Select
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/30 p-6">
      <div className="max-w-5xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold tracking-tight text-foreground">
            Mission Debrief
          </h1>
          <div className="flex flex-wrap items-center justify-center gap-4 text-lg text-muted-foreground">
            <span className="font-semibold text-foreground">{mission.name}</span>
            <span className="flex items-center gap-1">
              <Shield className="w-4 h-4" />
              {mission.sitrep.toUpperCase()} SITREP
            </span>
            <span className="px-2 py-1 text-xs font-medium rounded-full border border-border bg-muted/50">
              {mission.terrain.toUpperCase()} terrain
            </span>
          </div>
        </div>

        {/* Outcome Banner */}
        <div className={`rounded-lg border p-6 text-center ${search.outcome === "victory" ? "bg-success/10 border-success/30" : "bg-destructive/10 border-destructive/30"}`}>
          <div className="flex items-center justify-center gap-3">
            {search.outcome === "victory" ? (
              <>
                <Award className="w-8 h-8 text-success" />
                <h2 className="text-3xl font-bold text-success">MISSION ACCOMPLISHED</h2>
              </>
            ) : (
              <>
                <Shield className="w-8 h-8 text-destructive" />
                <h2 className="text-3xl font-bold text-destructive">MISSION FAILED</h2>
              </>
            )}
          </div>
          <p className="text-muted-foreground mt-2">
            {search.outcome === "victory"
              ? "All primary objectives completed successfully."
              : "Mission objectives could not be completed."}
          </p>
        </div>

        {/* Back button */}
        <div>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleBack}
            className="flex items-center gap-2"
            aria-label="Return to mission select"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Mission Select
          </Button>
        </div>

        {/* Main content grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left column: Epilogue and Statistics */}
          <div className="lg:col-span-2 space-y-8">
            {/* Epilogue */}
            <section className="dashboard-surface p-6">
              <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                <Target className="w-5 h-5" />
                After-Action Report
              </h2>
              <div className="prose prose-invert max-w-none">
                {epilogue.split("\n\n").map((paragraph, idx) => (
                  <p key={idx} className="mb-4 text-foreground/90">
                    {paragraph}
                  </p>
                ))}
              </div>
            </section>

            {/* Statistics */}
            <section className="dashboard-surface p-6">
              <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                Combat Statistics
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-muted/30 rounded-lg p-4 text-center">
                  <div className="text-3xl font-bold text-foreground">{search.turns}</div>
                  <div className="text-sm text-muted-foreground">Turns Taken</div>
                </div>
                <div className="bg-muted/30 rounded-lg p-4 text-center">
                  <div className="text-3xl font-bold text-foreground">{search.damageDealt.toLocaleString()}</div>
                  <div className="text-sm text-muted-foreground">Damage Dealt</div>
                </div>
                <div className="bg-muted/30 rounded-lg p-4 text-center">
                  <div className="text-3xl font-bold text-foreground">{search.damageReceived.toLocaleString()}</div>
                  <div className="text-sm text-muted-foreground">Damage Received</div>
                </div>
                <div className="bg-muted/30 rounded-lg p-4 text-center">
                  <div className="text-3xl font-bold text-foreground">{mission.enemyCount}</div>
                  <div className="text-sm text-muted-foreground">Enemies Engaged</div>
                </div>
              </div>
            </section>
          </div>

          {/* Right column: Rewards and Continue */}
          <div className="space-y-8">
            {/* Rewards */}
            <section className="dashboard-surface p-6">
              <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                <Award className="w-5 h-5" />
                Rewards Earned
              </h2>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-muted/30 rounded-lg">
                  <div className="flex items-center gap-3">
                    <Zap className="w-5 h-5 text-primary" />
                    <div>
                      <div className="font-semibold text-foreground">Experience</div>
                      <div className="text-sm text-muted-foreground">Pilot progression</div>
                    </div>
                  </div>
                  <div className="text-2xl font-bold text-primary">+{xpEarned} XP</div>
                </div>
                <div className="flex items-center justify-between p-4 bg-muted/30 rounded-lg">
                  <div className="flex items-center gap-3">
                    <Heart className="w-5 h-5 text-amber-500" />
                    <div>
                      <div className="font-semibold text-foreground">Salvage</div>
                      <div className="text-sm text-muted-foreground">Parts & materials</div>
                    </div>
                  </div>
                  <div className="text-2xl font-bold text-amber-500">+{salvageEarned}</div>
                </div>
              </div>
              <p className="text-sm text-muted-foreground mt-4">
                Rewards have been added to your pilot's inventory.
              </p>
            </section>

            {/* Continue button */}
            <div className="dashboard-surface p-6 text-center">
              <Button
                variant="primary"
                size="xl"
                onClick={handleContinue}
                className="w-full py-6 text-lg"
                aria-label="Continue to Pilot Quarters"
              >
                CONTINUE TO QUARTERS
              </Button>
              <p className="text-sm text-muted-foreground mt-2">
                Return to your hub to manage your pilot and select the next mission.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}