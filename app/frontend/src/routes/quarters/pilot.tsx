/**
 * Pilot detail view (placeholder).
 * Will be implemented in E1-US-006.
 */

import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Button } from "../../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui";

export const Route = createFileRoute("/quarters/pilot" as const)({
  component: PilotDetailPlaceholder,
});

function PilotDetailPlaceholder() {
  const navigate = useNavigate();

  const handleBack = () => navigate({ to: "/quarters" });

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/30 p-6">
      <div className="max-w-4xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold tracking-tight text-foreground">
            Pilot Details
          </h1>
          <p className="text-xl text-muted-foreground">
            Coming soon in E1-US-006
          </p>
        </div>

        {/* Placeholder content */}
        <Card className="dashboard-surface">
          <CardHeader>
            <CardTitle>Pilot Stats</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              This screen will show pilot skills, talents, triggers, and background.
            </p>
          </CardContent>
        </Card>

        {/* Back button */}
        <div className="text-center">
          <Button variant="outline" onClick={handleBack}>
            Back to Quarters
          </Button>
        </div>
      </div>
    </div>
  );
}