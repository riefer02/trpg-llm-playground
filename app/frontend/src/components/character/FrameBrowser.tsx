/**
 * Frame browser for character creation.
 *
 * Shows available frames filtered by unlocked licenses.
 * GMS frames are always available, manufacturer frames require licenses.
 */

import { useState } from "react";
import type { MechFrameDefinition } from "../../lib/types/lancer";
import type { CompendiumLicense } from "../../lib/api";
import type { LicenseAllocation } from "./LicenseSelector";
import { FramePreview } from "./FramePreview";
import { LicenseBadge } from "../ui/LicenseBadge";

interface MechSelection {
  frameId: string;
  name: string;
}

interface FrameBrowserProps {
  frames: MechFrameDefinition[];
  licenses: CompendiumLicense[];
  allocations: LicenseAllocation[];
  level: number;
  selectedMechs: MechSelection[];
  onAddMech: (frameId: string, name: string) => void;
  onRemoveMech: (index: number) => void;
  onUpdateMechName: (index: number, name: string) => void;
}

const MANUFACTURER_COLORS: Record<string, string> = {
  GMS: "text-gray-500",
  "IPS-N": "text-orange-500",
  SSC: "text-blue-500",
  HORUS: "text-green-500",
  HA: "text-red-500",
};

export function FrameBrowser({
  frames,
  licenses,
  allocations,
  level,
  selectedMechs,
  onAddMech,
  onRemoveMech,
  onUpdateMechName,
}: FrameBrowserProps) {
  const [previewFrame, setPreviewFrame] = useState<MechFrameDefinition | null>(
    null
  );
  const [newMechName, setNewMechName] = useState("");

  // Create a map of license_id to allocated rank
  const licenseRankMap = new Map(
    allocations.map((a) => {
      const lic = licenses.find((l) => l.id === a.licenseId);
      return [lic?.frame_id, a.rank];
    })
  );

  // Check if a frame is available based on licenses
  const isFrameAvailable = (frame: MechFrameDefinition): boolean => {
    // GMS frames are always available
    if (frame.manufacturer === "GMS") return true;

    // Check if pilot has sufficient license rank
    const requiredRank = frame.license_rank ?? 3;
    const allocatedRank = licenseRankMap.get(frame.id) ?? 0;
    return allocatedRank >= requiredRank;
  };

  // Group frames by manufacturer
  const framesByManufacturer = frames.reduce(
    (acc, frame) => {
      const mfr = frame.manufacturer;
      if (!acc[mfr]) acc[mfr] = [];
      acc[mfr].push(frame);
      return acc;
    },
    {} as Record<string, MechFrameDefinition[]>
  );

  // Max mechs allowed is 1 at LL0, increases by 1 for each license point
  const maxMechs = level === 0 ? 1 : level + 1;

  const handleAddMech = (frameId: string) => {
    if (selectedMechs.length >= maxMechs) return;
    const frame = frames.find((f) => f.id === frameId);
    if (!frame) return;
    onAddMech(frameId, newMechName || frame.name);
    setNewMechName("");
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_300px]">
      <div className="space-y-6">
        {/* Selected mechs */}
        {selectedMechs.length > 0 && (
          <div className="space-y-3">
            <div className="text-sm font-medium">
              Your Mechs ({selectedMechs.length}/{maxMechs})
            </div>
            {selectedMechs.map((mech, i) => {
              const frame = frames.find((f) => f.id === mech.frameId);
              return (
                <div
                  key={i}
                  className="flex items-center gap-3 p-3 rounded-lg border border-primary/50 bg-primary/10"
                >
                  <div className="flex-1">
                    <input
                      type="text"
                      value={mech.name}
                      onChange={(e) => onUpdateMechName(i, e.target.value)}
                      className="w-full px-2 py-1 bg-background border border-border rounded text-sm"
                      placeholder={frame?.name ?? "Mech name"}
                    />
                    <div className="text-xs text-muted-foreground mt-1">
                      {frame?.name} ({frame?.manufacturer})
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => onRemoveMech(i)}
                    className="text-destructive hover:text-destructive/80 text-sm"
                  >
                    Remove
                  </button>
                </div>
              );
            })}
          </div>
        )}

        {/* Frame browser */}
        {selectedMechs.length < maxMechs && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">Available Frames</div>
              <input
                type="text"
                value={newMechName}
                onChange={(e) => setNewMechName(e.target.value)}
                className="px-3 py-1 bg-background border border-border rounded text-sm w-48"
                placeholder="Custom mech name"
              />
            </div>

            {(["GMS", "IPS-N", "SSC", "HORUS", "HA"] as const).map((mfr) => {
              const mfrFrames = framesByManufacturer[mfr] ?? [];
              if (mfrFrames.length === 0) return null;

              return (
                <div key={mfr}>
                  <div
                    className={`text-sm font-semibold mb-2 ${MANUFACTURER_COLORS[mfr] ?? ""}`}
                  >
                    {mfr}
                  </div>
                  <div className="grid gap-2">
                    {mfrFrames.map((frame) => {
                      const available = isFrameAvailable(frame);
                      const alreadySelected = selectedMechs.some(
                        (m) => m.frameId === frame.id
                      );

                      return (
                        <div
                          key={frame.id}
                          className={`flex items-center justify-between p-3 rounded-lg border transition-colors ${
                            !available
                              ? "border-border bg-muted/30 opacity-60"
                              : alreadySelected
                                ? "border-primary/30 bg-primary/5"
                                : "border-border hover:border-primary/50 cursor-pointer"
                          }`}
                          onMouseEnter={() => setPreviewFrame(frame)}
                          onMouseLeave={() => setPreviewFrame(null)}
                        >
                          <div className="flex items-center gap-2">
                            <span className="font-medium">{frame.name}</span>
                            <LicenseBadge licenseId={frame.license_id ?? null} />
                            {!available && (
                              <span className="text-xs text-muted-foreground">
                                (Requires {frame.license_id} {frame.license_rank})
                              </span>
                            )}
                          </div>
                          {available && !alreadySelected && (
                            <button
                              type="button"
                              onClick={() => handleAddMech(frame.id)}
                              className="px-3 py-1 text-xs bg-primary text-primary-foreground rounded hover:bg-primary/90"
                            >
                              Add
                            </button>
                          )}
                          {alreadySelected && (
                            <span className="text-xs text-primary">Selected</span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {selectedMechs.length >= maxMechs && (
          <div className="p-4 rounded-lg border border-border bg-muted/30 text-sm text-muted-foreground">
            Maximum mechs reached for LL{level}. Remove a mech to select a
            different one.
          </div>
        )}
      </div>

      {/* Preview panel */}
      <div className="lg:sticky lg:top-6 h-fit">
        {previewFrame ? (
          <div className="p-4 rounded-lg border border-border bg-card">
            <FramePreview frame={previewFrame} />
          </div>
        ) : (
          <div className="p-4 rounded-lg border border-border bg-muted/30 text-sm text-muted-foreground">
            Hover over a frame to preview its details.
          </div>
        )}
      </div>
    </div>
  );
}
