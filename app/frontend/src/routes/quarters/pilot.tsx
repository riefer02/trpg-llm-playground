/**
 * Pilot detail view for quarters.
 * Shows pilot stats, skills, talents, triggers, licenses, and core bonuses.
 */

import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Button } from "../../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui";
import { PilotDisplay } from "../../components/quarters/PilotDisplay";
import { useActiveCharacter } from "../../lib/api/quarters";

export const Route = createFileRoute("/quarters/pilot" as const)({
  component: PilotDetailPage,
});

function PilotDetailPage() {
  const navigate = useNavigate();
  const { character, isLoading, error } = useActiveCharacter();

  const handleBack = () => navigate({ to: "/quarters" });

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-background to-muted/30 p-6">
        <div className="max-w-4xl mx-auto space-y-8">
          <div className="text-center space-y-4">
            <h1 className="text-4xl font-bold tracking-tight text-foreground">
              Pilot Details
            </h1>
            <p className="text-xl text-muted-foreground">Loading pilot...</p>
          </div>
          <div className="text-center">
            <Button variant="outline" onClick={handleBack}>
              Back to Quarters
            </Button>
          </div>
        </div>
      </div>
    );
  }

  if (error || !character) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-background to-muted/30 p-6">
        <div className="max-w-4xl mx-auto space-y-8">
          <div className="text-center space-y-4">
            <h1 className="text-4xl font-bold tracking-tight text-foreground">
              Pilot Details
            </h1>
            <p className="text-xl text-muted-foreground">
              {error?.message || "No active pilot found."}
            </p>
          </div>
          <Card className="dashboard-surface">
            <CardHeader>
              <CardTitle>Error</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                Please create a pilot first from the title screen.
              </p>
            </CardContent>
          </Card>
          <div className="text-center">
            <Button variant="outline" onClick={handleBack}>
              Back to Quarters
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/30 p-6">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold tracking-tight text-foreground">
            Pilot Details
          </h1>
          <p className="text-xl text-muted-foreground">
            {character.callsign} - License Level {character.level}
          </p>
        </div>

        {/* Back button */}
        <div className="flex justify-start">
          <Button variant="outline" onClick={handleBack}>
            ← Back to Quarters
          </Button>
        </div>

        {/* Pilot Display */}
        <PilotDisplay character={character} />
      </div>
    </div>
  );
}