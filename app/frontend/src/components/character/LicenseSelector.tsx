/**
 * License selector for character creation.
 *
 * Allows pilots to allocate license points to manufacturer licenses.
 * Each license can have ranks 1-3, and total ranks must equal available points.
 */

import type { CompendiumLicense } from "../../lib/api";

// Manufacturer display info
const MANUFACTURER_INFO = {
  "IPS-N": {
    name: "IPS-N",
    fullName: "Interplanetary Shipping-Northstar",
    description: "Durable brawlers and defensive specialists",
    color: "text-orange-500",
  },
  SSC: {
    name: "SSC",
    fullName: "Smith-Shimano Corpro",
    description: "Fast strikers and mobility specialists",
    color: "text-blue-500",
  },
  HORUS: {
    name: "HORUS",
    fullName: "HORUS",
    description: "Electronic warfare and hacking specialists",
    color: "text-green-500",
  },
  HA: {
    name: "HA",
    fullName: "Harrison Armory",
    description: "Heavy firepower and area control",
    color: "text-red-500",
  },
} as const;

type Manufacturer = keyof typeof MANUFACTURER_INFO;

export interface LicenseAllocation {
  licenseId: string;
  rank: number;
}

interface LicenseSelectorProps {
  licenses: CompendiumLicense[];
  allocations: LicenseAllocation[];
  availablePoints: number;
  onChange: (allocations: LicenseAllocation[]) => void;
}

export function LicenseSelector({
  licenses,
  allocations,
  availablePoints,
  onChange,
}: LicenseSelectorProps) {
  const usedPoints = allocations.reduce((sum, a) => sum + a.rank, 0);
  const remainingPoints = availablePoints - usedPoints;

  // Group licenses by manufacturer (exclude GMS)
  const byManufacturer = licenses.reduce(
    (acc, lic) => {
      // Skip GMS licenses - they're always available
      if (lic.manufacturer === "GMS") return acc;
      const mfr = lic.manufacturer as Manufacturer;
      if (MANUFACTURER_INFO[mfr]) {
        if (!acc[mfr]) acc[mfr] = [];
        acc[mfr].push(lic);
      }
      return acc;
    },
    {} as Record<Manufacturer, CompendiumLicense[]>
  );

  const getAllocation = (licenseId: string): number => {
    return allocations.find((a) => a.licenseId === licenseId)?.rank ?? 0;
  };

  const setAllocation = (licenseId: string, rank: number) => {
    const filtered = allocations.filter((a) => a.licenseId !== licenseId);
    if (rank > 0) {
      onChange([...filtered, { licenseId, rank }]);
    } else {
      onChange(filtered);
    }
  };

  const incrementRank = (licenseId: string) => {
    const current = getAllocation(licenseId);
    if (current < 3 && remainingPoints > 0) {
      setAllocation(licenseId, current + 1);
    }
  };

  const decrementRank = (licenseId: string) => {
    const current = getAllocation(licenseId);
    if (current > 0) {
      setAllocation(licenseId, current - 1);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between p-3 rounded-lg border border-border bg-muted/50">
        <div>
          <div className="text-sm font-medium">License Points</div>
          <div className="text-xs text-muted-foreground">
            Allocate points to unlock manufacturer equipment
          </div>
        </div>
        <div className="text-right">
          <div
            className={`text-lg font-semibold ${
              remainingPoints === 0
                ? "text-primary"
                : remainingPoints < 0
                  ? "text-destructive"
                  : "text-foreground"
            }`}
          >
            {usedPoints} / {availablePoints}
          </div>
          <div className="text-xs text-muted-foreground">
            {remainingPoints > 0
              ? `${remainingPoints} remaining`
              : remainingPoints === 0
                ? "All allocated"
                : "Over budget!"}
          </div>
        </div>
      </div>

      {(["IPS-N", "SSC", "HORUS", "HA"] as const).map((mfr) => {
        const mfrLicenses = byManufacturer[mfr] ?? [];
        const info = MANUFACTURER_INFO[mfr];
        if (mfrLicenses.length === 0) return null;

        return (
          <div key={mfr} className="space-y-2">
            <div className="flex items-center gap-2">
              <span className={`font-semibold ${info.color}`}>{info.name}</span>
              <span className="text-xs text-muted-foreground">
                {info.description}
              </span>
            </div>
            <div className="grid gap-2">
              {mfrLicenses.map((lic) => {
                const rank = getAllocation(lic.id);
                const canIncrement = rank < 3 && remainingPoints > 0;

                return (
                  <div
                    key={lic.id}
                    className={`flex items-center justify-between p-3 rounded-lg border transition-colors ${
                      rank > 0
                        ? "border-primary/50 bg-primary/10"
                        : "border-border hover:border-primary/30"
                    }`}
                  >
                    <div>
                      <div className="font-medium">{lic.name}</div>
                      <div className="text-xs text-muted-foreground">
                        {rank >= 3 ? "Frame unlocked" : `Rank ${rank}/3`}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => decrementRank(lic.id)}
                        disabled={rank === 0}
                        className="w-8 h-8 flex items-center justify-center rounded bg-background border border-border hover:bg-primary/10 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        -
                      </button>
                      <div className="w-8 text-center font-semibold">{rank}</div>
                      <button
                        type="button"
                        onClick={() => incrementRank(lic.id)}
                        disabled={!canIncrement}
                        className="w-8 h-8 flex items-center justify-center rounded bg-background border border-border hover:bg-primary/10 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        +
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}

      {allocations.length > 0 && (
        <div className="p-3 rounded-lg border border-border bg-muted/30">
          <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">
            Selected Licenses
          </div>
          <div className="flex flex-wrap gap-2">
            {allocations.map((alloc) => {
              const lic = licenses.find((l) => l.id === alloc.licenseId);
              if (!lic) return null;
              return (
                <span
                  key={alloc.licenseId}
                  className="px-2 py-1 text-xs rounded-full bg-primary/20 text-primary"
                >
                  {lic.name} {alloc.rank}
                </span>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
