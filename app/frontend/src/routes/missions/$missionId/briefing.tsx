/**
 * Mission Briefing placeholder.
 * Pre-mission screen with narrative briefing, map preview, enemy intel, objectives, and launch button.
 * Static content for MVP.
 */

import { createFileRoute, useNavigate, useParams } from "@tanstack/react-router";
import { Button } from "../../../components/ui/button";
import { ArrowLeft, Shield, Target, Users, Map } from "lucide-react";
import { useMission } from "../../../lib/api/missions";
import { useActiveCharacter } from "../../../lib/api/quarters";
import { useCreateDemoCombat, DemoScenarioType } from "../../../lib/api/combat";

export const Route = createFileRoute("/missions/$missionId/briefing" as const)({
  component: MissionBriefing,
});

function MissionBriefing() {
  const navigate = useNavigate();
  const { missionId } = useParams({ from: "/missions/$missionId/briefing" });
  const { mission, isLoading, error } = useMission(missionId);
  const { character } = useActiveCharacter();
  const createDemo = useCreateDemoCombat();

  const handleBack = () => navigate({ to: "/missions" });
  const handleLaunch = async () => {
    if (!mission) return;
    
    // Map SITREP to demo scenario type
    const sitrepToScenario: Record<string, DemoScenarioType> = {
      control: "control",
      extract: "skirmish",
      gauntlet: "boss",
      hold_out: "skirmish",
      recon: "skirmish",
      escort: "skirmish",
    };
    
    const scenarioType = sitrepToScenario[mission.sitrep] || "skirmish";
    
    try {
      const session = await createDemo.mutateAsync(scenarioType);
      navigate({ to: `/combat/${session.id}` });
    } catch (error) {
      console.error("Failed to launch mission:", error);
      alert(`Failed to launch mission: ${error instanceof Error ? error.message : "Unknown error"}`);
    }
  };

  const handleViewLoadout = () => navigate({ to: "/quarters/mech" });

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin mb-4" />
          <p className="text-muted-foreground">Loading mission briefing...</p>
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
            The requested mission could not be loaded.
          </p>
          <Button variant="outline" onClick={handleBack}>
            Return to Mission Select
          </Button>
        </div>
      </div>
    );
  }

  // Format difficulty stars
  const difficultyStars = "★".repeat(mission.difficulty) + "☆".repeat(3 - mission.difficulty);

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/30 p-6">
      <div className="max-w-5xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold tracking-tight text-foreground">
            Mission Briefing
          </h1>
          <div className="flex flex-wrap items-center justify-center gap-4 text-lg text-muted-foreground">
            <span className="font-semibold text-foreground">{mission.name}</span>
            <span className="flex items-center gap-1">
              <Shield className="w-4 h-4" />
              {mission.sitrep.toUpperCase()} SITREP
            </span>
            <span className="flex items-center gap-1">
              <Target className="w-4 h-4" />
              Difficulty: {difficultyStars}
            </span>
            <span className="flex items-center gap-1">
              <Users className="w-4 h-4" />
              {mission.enemyCount} enemy mechs
            </span>
            <span className="px-2 py-1 text-xs font-medium rounded-full border border-border bg-muted/50">
              {mission.terrain.toUpperCase()}
            </span>
          </div>
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
          {/* Left column: Briefing and Objectives */}
          <div className="lg:col-span-2 space-y-8">
            {/* Situation/Briefing */}
            <section className="dashboard-surface p-6">
              <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                <Shield className="w-5 h-5" />
                Situation Briefing
              </h2>
              <div className="prose prose-invert max-w-none">
                {mission.briefing.split("\n\n").map((paragraph, idx) => (
                  <p key={idx} className="mb-4 text-foreground/90">
                    {paragraph}
                  </p>
                ))}
              </div>
            </section>

            {/* Objectives */}
            <section className="dashboard-surface p-6">
              <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                <Target className="w-5 h-5" />
                Mission Objectives
              </h2>
              <ul className="space-y-3">
                {mission.objectives.map((obj, idx) => (
                  <li key={idx} className="flex items-start gap-3">
                    <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center mt-0.5">
                      <span className="text-sm font-semibold text-primary">{idx + 1}</span>
                    </div>
                    <span className="text-foreground/90">{obj}</span>
                  </li>
                ))}
              </ul>
            </section>

            {/* Enemy Intel */}
            <section className="dashboard-surface p-6">
              <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                <Users className="w-5 h-5" />
                Enemy Intel
              </h2>
              <div className="prose prose-invert max-w-none">
                <p className="text-foreground/90">{mission.enemyIntel}</p>
              </div>
              <div className="mt-4 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <p className="text-sm font-medium text-destructive">
                  Threat Assessment: {mission.difficulty === 1 ? "MODERATE" : mission.difficulty === 2 ? "HIGH" : "SEVERE"}
                </p>
              </div>
            </section>
          </div>

          {/* Right column: Map preview and Loadout */}
          <div className="space-y-8">
            {/* Map Preview */}
            <section className="dashboard-surface p-6">
              <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                <Map className="w-5 h-5" />
                Map Preview
              </h2>
              <div className="aspect-video bg-muted rounded-lg overflow-hidden">
                <img
                  src={mission.mapPreviewUrl || "https://placehold.co/600x400/1e293b/94a3b8?text=Map+Preview"}
                  alt={`${mission.terrain} terrain map`}
                  className="w-full h-full object-cover"
                />
              </div>
              <p className="text-sm text-muted-foreground mt-2 text-center">
                {mission.terrain.toUpperCase()} terrain - {mission.sitrep.toUpperCase()} SITREP layout
              </p>
            </section>

            {/* Loadout Check */}
            <section className="dashboard-surface p-6">
              <h2 className="text-2xl font-bold mb-4">Loadout Check</h2>
              {character?.active_mech ? (
                <div className="space-y-4">
                  <div className="p-4 bg-muted/30 rounded-lg">
                    <h3 className="font-semibold text-lg">{character.active_mech.frame?.name || "Unknown Frame"}</h3>
                    <p className="text-sm text-muted-foreground">
                      {character.active_mech.weapons?.length || 0} weapons, {character.active_mech.systems?.length || 0} systems equipped
                    </p>
                  </div>
                  <div className="text-sm space-y-2">
                    <p className="font-medium">Pilot: {character.callsign || "Unknown"}</p>
                    <p className="text-muted-foreground">
                      License Level: LL{character.level || 0}
                    </p>
                  </div>
                </div>
              ) : (
                <div className="p-4 bg-warning/10 border border-warning/20 rounded-lg">
                  <p className="text-warning font-medium">No active mech configured</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Visit the Mech Loadout screen to configure your mech before deployment.
                  </p>
                </div>
              )}
              <Button variant="outline" size="sm" className="w-full mt-4" onClick={handleViewLoadout}>
                View Full Loadout
              </Button>
            </section>

            {/* Launch button */}
            <div className="dashboard-surface p-6 text-center">
              <Button
                variant="primary"
                size="xl"
                onClick={handleLaunch}
                className="w-full py-6 text-lg"
                aria-label="Launch mission"
                disabled={!character?.active_mech || createDemo.isPending}
               >
                {createDemo.isPending ? (
                  <>
                    <span className="inline-block w-5 h-5 border-2 border-current border-t-transparent rounded-full animate-spin mr-2" />
                    Launching...
                  </>
                ) : (
                  "LAUNCH MISSION"
                )}
              </Button>
              <p className="text-sm text-muted-foreground mt-2">
                {!character?.active_mech
                  ? "Configure a mech loadout before launching"
                  : `This will start combat with ${mission.sitrep.toUpperCase()} scenario.`}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}