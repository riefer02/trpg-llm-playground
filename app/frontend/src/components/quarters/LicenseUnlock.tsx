/**
 * License unlock component for quarters.
 * Shows manufacturers with license trees, allows spending LP to unlock license ranks.
 */

import { useState, useEffect } from "react";
import { Button } from "../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../ui";
import { useUpdateCharacter } from "../../lib/api/characters";
import type { CharacterResponse } from "../../lib/api";
import type { License } from "../../lib/api/compendium";

// Manufacturer names mapping
const MANUFACTURER_NAMES: Record<string, string> = {
  "GMS": "General Massive Systems",
  "IPS-N": "Interplanetary Shipping-Northstar",
  "SSC": "Smith-Shimano Corpro",
  "HORUS": "HORUS",
  "HA": "Harrison Armory",
};

// Manufacturer order for display
const MANUFACTURER_ORDER = ["GMS", "IPS-N", "SSC", "HORUS", "HA"];

interface LicenseUnlockProps {
  character: CharacterResponse;
  licenses: License[];
}

interface LicenseState {
  license_id: string;
  rank: number;
}

export function LicenseUnlock({ character, licenses }: LicenseUnlockProps) {
  // Current license state (editable)
  const [licenseStates, setLicenseStates] = useState<LicenseState[]>([]);
  const [availableLP, setAvailableLP] = useState(0);
  
  // Mutation for saving license changes
  const updateCharacterMutation = useUpdateCharacter();

  // Initialize from character licenses
  useEffect(() => {
    const initialLicenses = character.licenses.map(lic => ({
      license_id: lic.license_id,
      rank: lic.rank,
    }));
    setLicenseStates(initialLicenses);
    
    // Compute available LP
    const totalSpent = initialLicenses.reduce((sum, lic) => sum + lic.rank, 0);
    const maxLP = character.level;
    setAvailableLP(maxLP - totalSpent);
  }, [character]);

  // Group licenses by manufacturer
  const licensesByManufacturer: Record<string, License[]> = {};
  licenses.forEach(license => {
    if (!licensesByManufacturer[license.manufacturer]) {
      licensesByManufacturer[license.manufacturer] = [];
    }
    licensesByManufacturer[license.manufacturer].push(license);
  });

  // Get current rank for a license
  const getLicenseRank = (licenseId: string): number => {
    const found = licenseStates.find(lic => lic.license_id === licenseId);
    return found?.rank || 0;
  };

  // Check if license can be increased (has available LP and rank < 3)
  const canIncrease = (licenseId: string): boolean => {
    if (availableLP <= 0) return false;
    const currentRank = getLicenseRank(licenseId);
    return currentRank < 3;
  };

  // Check if license can be decreased (rank > 0 and won't violate validation)
  const canDecrease = (licenseId: string): boolean => {
    const currentRank = getLicenseRank(licenseId);
    return currentRank > 0;
  };

  // Handle increasing a license rank
  const handleIncrease = (licenseId: string) => {
    if (!canIncrease(licenseId)) return;

    setLicenseStates(prev => 
      prev.map(lic => 
        lic.license_id === licenseId 
          ? { ...lic, rank: lic.rank + 1 }
          : lic
      ).concat(
        !prev.some(lic => lic.license_id === licenseId)
          ? [{ license_id: licenseId, rank: 1 }]
          : []
      )
    );
    setAvailableLP(prev => prev - 1);
  };

  // Handle decreasing a license rank
  const handleDecrease = (licenseId: string) => {
    if (!canDecrease(licenseId)) return;

    setLicenseStates(prev => 
      prev.map(lic => 
        lic.license_id === licenseId 
          ? { ...lic, rank: lic.rank - 1 }
          : lic
      ).filter(lic => lic.rank > 0)
    );
    setAvailableLP(prev => prev + 1);
  };

  // Get license definition by ID
  const getLicenseDefinition = (licenseId: string): License | undefined => {
    return licenses.find(lic => lic.id === licenseId);
  };

  // Calculate total spent LP
  const totalSpent = licenseStates.reduce((sum, lic) => sum + lic.rank, 0);

  // Handle save
  const handleSave = () => {
    if (availableLP < 0) {
      alert("Cannot save: You have allocated more License Points than available.");
      return;
    }
    
    // Convert to format expected by API
    const licenseUpdates = licenseStates.map(lic => ({
      license_id: lic.license_id,
      rank: lic.rank,
    }));
    
    updateCharacterMutation.mutate(
      {
        id: character.id,
        data: { licenses: licenseUpdates },
      },
      {
        onSuccess: (updatedCharacter) => {
          // Success - state is already updated via React Query cache
          console.log("Licenses updated successfully", updatedCharacter);
        },
        onError: (error) => {
          console.error("Failed to update licenses:", error);
          alert(`Failed to save license changes: ${error.message}`);
        },
      }
    );
  };

  // Handle reset
  const handleReset = () => {
    const initialLicenses = character.licenses.map(lic => ({
      license_id: lic.license_id,
      rank: lic.rank,
    }));
    setLicenseStates(initialLicenses);
    const totalSpent = initialLicenses.reduce((sum, lic) => sum + lic.rank, 0);
    setAvailableLP(character.level - totalSpent);
  };

  // Check if there are changes
  const hasChanges = () => {
    const currentMap = new Map(licenseStates.map(lic => [lic.license_id, lic.rank]));
    const originalMap = new Map(character.licenses.map(lic => [lic.license_id, lic.rank]));
    
    if (currentMap.size !== originalMap.size) return true;
    
    for (const [id, rank] of currentMap) {
      if (originalMap.get(id) !== rank) return true;
    }
    
    return false;
  };

  return (
    <div className="space-y-8">
      {/* Summary card */}
      <Card>
        <CardHeader>
          <CardTitle>License Points</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center p-4 bg-primary/10 rounded-lg">
              <div className="text-3xl font-bold text-primary">{availableLP}</div>
              <div className="text-sm text-muted-foreground">Available</div>
            </div>
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="text-3xl font-bold">{totalSpent}</div>
              <div className="text-sm text-muted-foreground">Spent</div>
            </div>
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="text-3xl font-bold">{character.level}</div>
              <div className="text-sm text-muted-foreground">Total</div>
            </div>
          </div>
          <p className="text-sm text-muted-foreground mt-4">
            Each license level (LL) grants 1 License Point. Licenses have 3 ranks.
            Rank 1 unlocks basic gear, Rank 2 unlocks advanced systems, Rank 3 unlocks the frame.
          </p>
        </CardContent>
      </Card>

      {/* Manufacturer sections */}
      {MANUFACTURER_ORDER.map(manufacturer => {
        const manufacturerLicenses = licensesByManufacturer[manufacturer];
        if (!manufacturerLicenses || manufacturerLicenses.length === 0) return null;

        return (
          <Card key={manufacturer}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <span>{manufacturer}</span>
                <span className="text-muted-foreground text-sm font-normal">
                  {MANUFACTURER_NAMES[manufacturer]}
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {manufacturerLicenses.map(license => {
                  const currentRank = getLicenseRank(license.id);
                  const canInc = canIncrease(license.id);
                  const canDec = canDecrease(license.id);
                  const licenseDef = getLicenseDefinition(license.id);

                  return (
                    <div 
                      key={license.id} 
                      className={`border rounded-lg p-4 ${currentRank > 0 ? 'border-primary/50 bg-primary/5' : 'border-border bg-card'}`}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <h4 className="font-bold">{license.name}</h4>
                          <p className="text-xs text-muted-foreground">
                            Frame: {license.frame_id.replace(/_/g, ' ')}
                          </p>
                        </div>
                        <div className="text-right">
                          <div className="text-2xl font-bold">{currentRank}</div>
                          <div className="text-xs text-muted-foreground">/ 3</div>
                        </div>
                      </div>

                      {/* Rank indicators */}
                      <div className="flex gap-1 mb-4">
                        {[1, 2, 3].map(rank => (
                          <div
                            key={rank}
                            className={`h-2 flex-1 rounded ${currentRank >= rank ? 'bg-primary' : 'bg-muted'}`}
                            title={`Rank ${rank}`}
                          />
                        ))}
                      </div>

                      {/* Rank description */}
                      <div className="text-xs text-muted-foreground mb-4">
                        {currentRank === 0 && "Locked"}
                        {currentRank === 1 && "Unlocks basic weapons/systems"}
                        {currentRank === 2 && "Unlocks advanced systems"}
                        {currentRank === 3 && "Unlocks frame + signature gear"}
                      </div>

                      {/* Controls */}
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleDecrease(license.id)}
                          disabled={!canDec || updateCharacterMutation.isPending}
                          className="flex-1"
                          aria-label={`Decrease ${license.name} rank`}
                        >
                          -
                        </Button>
                        <div className="flex-1 text-center py-2 px-4 bg-muted rounded">
                          Rank {currentRank}
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleIncrease(license.id)}
                          disabled={!canInc || updateCharacterMutation.isPending}
                          className="flex-1"
                          aria-label={`Increase ${license.name} rank`}
                        >
                          +
                        </Button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        );
      })}

      {/* Action buttons */}
      <div className="flex justify-end gap-4 pt-6 border-t">
        <Button
          variant="outline"
          onClick={handleReset}
          disabled={!hasChanges() || updateCharacterMutation.isPending}
        >
          Reset Changes
        </Button>
        <Button
          variant="primary"
          onClick={handleSave}
          disabled={!hasChanges() || availableLP < 0 || updateCharacterMutation.isPending}
        >
          {updateCharacterMutation.isPending ? "Saving..." : "Save License Changes"}
        </Button>
      </div>

      {/* Error message */}
      {updateCharacterMutation.error && (
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <p className="text-destructive">
              Failed to save license changes: {updateCharacterMutation.error.message}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Validation message */}
      {availableLP < 0 && (
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <p className="text-destructive">
              Error: You have allocated more License Points than available. 
              Please reduce some license ranks.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}