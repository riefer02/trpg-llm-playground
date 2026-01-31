/**
 * License unlock screen for quarters.
 * Shows manufacturers with license trees, allows spending LP to unlock license ranks.
 */

import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Button } from "../../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui";
import { useActiveCharacter } from "../../lib/api/quarters";
import { useLicenses } from "../../lib/api/compendium";
import { LicenseUnlock } from "../../components/quarters/LicenseUnlock";

export const Route = createFileRoute("/quarters/licenses" as const)({
  component: LicenseUnlockPage,
});

function LicenseUnlockPage() {
  const navigate = useNavigate();
  const { character, isLoading: characterLoading, error: characterError } = useActiveCharacter();
  const { data: licensesData, isLoading: licensesLoading, error: licensesError } = useLicenses();

  const handleBack = () => navigate({ to: "/quarters" });

  if (characterLoading || licensesLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-background to-muted/30 p-6">
        <div className="max-w-7xl mx-auto space-y-8">
          <div className="text-center space-y-4">
            <h1 className="text-4xl font-bold tracking-tight text-foreground">
              License Unlock
            </h1>
            <p className="text-xl text-muted-foreground">Loading licenses...</p>
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

  if (characterError || !character) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-background to-muted/30 p-6">
        <div className="max-w-7xl mx-auto space-y-8">
          <div className="text-center space-y-4">
            <h1 className="text-4xl font-bold tracking-tight text-foreground">
              License Unlock
            </h1>
            <p className="text-xl text-muted-foreground">
              {characterError?.message || "No active pilot found."}
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

  if (licensesError) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-background to-muted/30 p-6">
        <div className="max-w-7xl mx-auto space-y-8">
          <div className="text-center space-y-4">
            <h1 className="text-4xl font-bold tracking-tight text-foreground">
              License Unlock
            </h1>
            <p className="text-xl text-muted-foreground">
              Failed to load license data.
            </p>
          </div>
          <Card className="dashboard-surface">
            <CardHeader>
              <CardTitle>Error</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                {licensesError.message}
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

  const totalLicenseLevels = character.licenses.reduce((sum, lic) => sum + lic.rank, 0);
  const maxLicenseLevels = character.level; // LL0 has 0, LL1 has 1, etc.
  const availableLP = maxLicenseLevels - totalLicenseLevels;

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/30 p-6">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold tracking-tight text-foreground">
            License Unlock
          </h1>
          <p className="text-xl text-muted-foreground">
            {character.callsign} - License Level {character.level}
          </p>
          <div className="flex justify-center items-center gap-8">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">{availableLP}</div>
              <div className="text-sm text-muted-foreground">Available License Points</div>
            </div>
            <div className="text-center">
               <div className="text-2xl font-bold">{totalLicenseLevels}</div>
              <div className="text-sm text-muted-foreground">Spent License Points</div>
            </div>
            <div className="text-center">
               <div className="text-2xl font-bold">{maxLicenseLevels}</div>
              <div className="text-sm text-muted-foreground">Total License Points</div>
            </div>
          </div>
        </div>

        {/* Back button */}
        <div className="flex justify-start">
          <Button variant="outline" onClick={handleBack}>
            ← Back to Quarters
          </Button>
        </div>

        {/* License Unlock Component */}
        <LicenseUnlock 
          character={character} 
          licenses={licensesData || []} 
        />
      </div>
    </div>
  );
}