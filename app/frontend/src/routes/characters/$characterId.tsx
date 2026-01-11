/**
 * Character detail page.
 *
 * View and edit a character with their pilot stats and mech configurations.
 */

import { useEffect, useMemo, useState } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import {
  useCharacter,
  useCharacterValidation,
  usePilotGear,
  useWeapons,
  useSystems,
  useFrames,
  useUpdatePilotGear,
  useUpdateMechBuild,
  type CharacterResponse,
} from "../../lib/api";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  Button,
} from "../../components/ui";
import type { PilotLoadout, MechBuild } from "../../lib/types/lancer";

export const Route = createFileRoute("/characters/$characterId" as const)({
  component: CharacterDetailPage,
});

function CharacterDetailPage() {
  const { characterId } = Route.useParams();
  const { data: character, isLoading, error } = useCharacter(characterId);
  const { data: validation } = useCharacterValidation(characterId);

  if (isLoading) {
    return (
      <div className="p-6 max-w-4xl mx-auto">
        <div className="text-center py-8 text-muted-foreground">
          Loading character...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 max-w-4xl mx-auto">
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <p className="text-destructive">
              Error loading character: {error.message}
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!character) {
    return (
      <div className="p-6 max-w-4xl mx-auto">
        <Card>
          <CardContent className="pt-6 text-center">
            <p className="text-muted-foreground">Character not found</p>
            <Link to="/characters" className="text-primary hover:underline">
              Back to Characters
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="mb-6">
        <Link
          to="/characters"
          className="text-primary hover:underline text-sm"
        >
          ← Back to Characters
        </Link>
        <div className="flex justify-between items-start mt-2">
          <div>
            <h1 className="text-3xl font-bold text-foreground">
              {character.callsign}
            </h1>
            <p className="text-muted-foreground">
              {character.name || "Unnamed"} • License Level {character.level}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              onClick={() => {
                window.location.href = `/api/characters/${characterId}/export.pdf`;
              }}
            >
              Download PDF
            </Button>
            <Link
              to="/characters/$characterId/export"
              params={{ characterId }}
            >
              <Button variant="outline">Export PDF</Button>
            </Link>
            <ValidationBadge validation={validation} />
          </div>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <PilotSection character={character} />
        <MechSection character={character} />
      </div>

      <LoadoutSection character={character} />

      {validation && !validation.valid && (
        <Card className="mt-6 border-destructive">
          <CardHeader>
            <CardTitle className="text-destructive">
              Validation Issues
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {validation.issues.map((issue, i) => (
                <li
                  key={i}
                  className={
                    issue.severity === "warning"
                      ? "text-accent"
                      : "text-destructive"
                  }
                >
                  <span className="font-mono text-xs mr-2">[{issue.code}]</span>
                  {issue.message}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {character.triggers.length > 0 && (
        <Card className="mt-6">
          <CardHeader>
            <CardTitle>Triggers</CardTitle>
            <CardDescription>
              Bonuses for narrative skill checks
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {character.triggers.map((trigger, i) => (
                <div
                  key={i}
                  className="p-3 bg-card-foreground/5 rounded-md text-sm"
                >
                  <div className="font-medium capitalize">
                    {trigger.trigger_id.replace(/_/g, " ")}
                  </div>
                  <div className="text-primary">+{trigger.rank}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {character.talents.length > 0 && (
        <Card className="mt-6">
          <CardHeader>
            <CardTitle>Talents</CardTitle>
            <CardDescription>Combat abilities and specializations</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-3">
              {character.talents.map((talent, i) => (
                <div
                  key={i}
                  className="p-3 bg-card-foreground/5 rounded-md text-sm"
                >
                  <div className="font-medium capitalize">
                    {talent.talent_id.replace(/_/g, " ")}
                  </div>
                  <div className="text-muted-foreground">
                    Rank {talent.rank}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {character.notes && (
        <Card className="mt-6">
          <CardHeader>
            <CardTitle>Notes</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="whitespace-pre-wrap text-muted-foreground">
              {character.notes}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function ValidationBadge({
  validation,
}: {
  validation?: { valid: boolean; issues: Array<{ severity: string }> };
}) {
  if (!validation) return null;

  const warnings = validation.issues.filter(
    (i) => i.severity === "warning"
  ).length;
  const errors = validation.issues.filter(
    (i) => i.severity !== "warning"
  ).length;

  if (validation.valid && warnings === 0) {
    return (
      <div className="px-3 py-1 bg-horus/20 text-horus rounded-full text-sm font-medium">
        Valid
      </div>
    );
  }

  if (errors > 0) {
    return (
      <div className="px-3 py-1 bg-destructive/20 text-destructive rounded-full text-sm font-medium">
        {errors} Error{errors > 1 ? "s" : ""}
      </div>
    );
  }

  return (
    <div className="px-3 py-1 bg-accent/20 text-accent rounded-full text-sm font-medium">
      {warnings} Warning{warnings > 1 ? "s" : ""}
    </div>
  );
}

function PilotSection({ character }: { character: CharacterResponse }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Pilot</CardTitle>
        <CardDescription>Personal stats and abilities</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-4 mb-4">
          <StatBlock label="Grit" value={`+${character.grit}`} />
          <StatBlock label="HP" value={character.pilot_hp} />
          <StatBlock
            label="Background"
            value={character.background?.name || "None"}
          />
        </div>

        <div className="text-xs text-muted-foreground uppercase tracking-wide mb-2">
          Mech Skills
        </div>
        <div className="grid grid-cols-4 gap-2">
          <SkillBlock label="HULL" value={character.skills.hull ?? 0} />
          <SkillBlock label="AGI" value={character.skills.agility ?? 0} />
          <SkillBlock label="SYS" value={character.skills.systems ?? 0} />
          <SkillBlock label="ENG" value={character.skills.engineering ?? 0} />
        </div>
      </CardContent>
    </Card>
  );
}

function MechSection({ character }: { character: CharacterResponse }) {
  const activeMech = character.mechs.find(
    (m) => m.id === character.active_mech_id
  );
  const stats = character.active_mech_stats;

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          {activeMech?.name || "No Mech"}
          {activeMech && (
            <span className="text-sm font-normal text-muted-foreground ml-2">
              ({activeMech.frame_id.replace(/^gms_/, "GMS ").replace(/_/g, " ")})
            </span>
          )}
        </CardTitle>
        <CardDescription>
          {character.mechs.length} mech{character.mechs.length !== 1 ? "s" : ""}{" "}
          configured
        </CardDescription>
      </CardHeader>
      <CardContent>
        {stats ? (
          <>
            <div className="grid grid-cols-3 gap-4 mb-4">
              <StatBlock label="HP" value={stats.hp} />
              <StatBlock label="Armor" value={stats.armor} />
              <StatBlock label="Size" value={stats.size.replace("size_", "")} />
            </div>
            <div className="grid grid-cols-3 gap-4 mb-4">
              <StatBlock label="Evasion" value={stats.evasion} />
              <StatBlock label="E-Defense" value={stats.e_defense} />
              <StatBlock label="Speed" value={stats.speed} />
            </div>
            <div className="grid grid-cols-3 gap-4">
              <StatBlock label="Heat Cap" value={stats.heat_cap} />
              <StatBlock label="Repair Cap" value={stats.repair_cap} />
              <StatBlock label="SP" value={stats.system_points} />
            </div>
            <div className="mt-4 pt-4 border-t border-border grid grid-cols-3 gap-4">
              <StatBlock label="Tech Attack" value={`+${stats.tech_attack}`} />
              <StatBlock label="Sensor Range" value={stats.sensor_range} />
              <StatBlock label="Save Target" value={stats.save_target} />
            </div>
          </>
        ) : (
          <p className="text-muted-foreground">No active mech selected</p>
        )}
      </CardContent>
    </Card>
  );
}

function LoadoutSection({ character }: { character: CharacterResponse }) {
  return (
    <div className="mt-6 grid gap-6 md:grid-cols-2">
      <PilotGearSection character={character} />
      <MechBuildSection character={character} />
    </div>
  );
}

function PilotGearSection({ character }: { character: CharacterResponse }) {
  const { data: pilotGear } = usePilotGear();
  const updatePilotGear = useUpdatePilotGear();
  const [isEditing, setIsEditing] = useState(false);

  const currentLoadout = useMemo<PilotLoadout>(
    () => ({
      clothing: character.pilot_gear?.clothing ?? null,
      armor: character.pilot_gear?.armor ?? null,
      weapons: character.pilot_gear?.weapons ?? [],
      gear: character.pilot_gear?.gear ?? [],
    }),
    [character.pilot_gear]
  );

  const [draft, setDraft] = useState<PilotLoadout>(currentLoadout);

  useEffect(() => {
    if (!isEditing) {
      setDraft(currentLoadout);
    }
  }, [currentLoadout, isEditing]);

  const pilotGearMap = useMemo(
    () => new Map(pilotGear?.map((item) => [item.id, item]) ?? []),
    [pilotGear]
  );

  const clothingOptions =
    pilotGear?.filter((item) => item.category === "clothing") ?? [];
  const armorOptions =
    pilotGear?.filter((item) => item.category === "armor") ?? [];
  const weaponOptions =
    pilotGear?.filter((item) => item.category === "weapon") ?? [];
  const gearOptions =
    pilotGear?.filter((item) => item.category === "gear") ?? [];

  const toggleList = (key: "weapons" | "gear", id: string, max: number) => {
    setDraft((prev) => {
      const current = prev[key];
      const isSelected = current.includes(id);
      if (isSelected) {
        return { ...prev, [key]: current.filter((itemId) => itemId !== id) };
      }
      if (current.length >= max) {
        return prev;
      }
      return { ...prev, [key]: [...current, id] };
    });
  };

  const canSave = draft.clothing !== null && !updatePilotGear.isPending;
  const pilotGearError =
    updatePilotGear.error instanceof Error
      ? updatePilotGear.error.message
      : "Failed to update pilot gear.";

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Pilot Gear</CardTitle>
            <CardDescription>Mission-specific pilot loadout</CardDescription>
          </div>
          {!isEditing ? (
            <Button type="button" variant="outline" onClick={() => setIsEditing(true)}>
              Edit
            </Button>
          ) : (
            <div className="flex gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  setIsEditing(false);
                  setDraft(currentLoadout);
                }}
              >
                Cancel
              </Button>
              <Button
                type="button"
                onClick={() =>
                  updatePilotGear.mutate({
                    id: character.id,
                    data: { pilot_gear: draft },
                  })
                }
                disabled={!canSave}
              >
                {updatePilotGear.isPending ? "Saving..." : "Save"}
              </Button>
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {!isEditing ? (
          <div className="space-y-3 text-sm">
            <div>
              <div className="text-xs text-muted-foreground uppercase">Clothing</div>
              <div className="font-medium">
                {draft.clothing
                  ? pilotGearMap.get(draft.clothing)?.name ?? draft.clothing
                  : "None"}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground uppercase">Armor</div>
              <div className="font-medium">
                {draft.armor
                  ? pilotGearMap.get(draft.armor)?.name ?? draft.armor
                  : "None"}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground uppercase">Weapons</div>
              <div className="font-medium">
                {draft.weapons.length
                  ? draft.weapons
                      .map((id) => pilotGearMap.get(id)?.name ?? id)
                      .join(", ")
                  : "None"}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground uppercase">Gear</div>
              <div className="font-medium">
                {draft.gear.length
                  ? draft.gear
                      .map((id) => pilotGearMap.get(id)?.name ?? id)
                      .join(", ")
                  : "None"}
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            <div>
              <div className="text-xs text-muted-foreground uppercase tracking-wide mb-2">
                Clothing (Required)
              </div>
              <div className="grid gap-2">
                {clothingOptions.map((item) => {
                  const isSelected = draft.clothing === item.id;
                  return (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => setDraft((prev) => ({ ...prev, clothing: item.id }))}
                      className={`p-2 text-left border rounded-md transition-colors ${
                        isSelected ? "bg-primary/20 border-primary" : "hover:bg-primary/10"
                      }`}
                    >
                      <div className="font-medium text-sm">{item.name}</div>
                    </button>
                  );
                })}
              </div>
            </div>

            <div>
              <div className="text-xs text-muted-foreground uppercase tracking-wide mb-2">
                Armor (Optional)
              </div>
              <div className="grid gap-2">
                <button
                  type="button"
                  onClick={() => setDraft((prev) => ({ ...prev, armor: null }))}
                  className={`p-2 text-left border rounded-md transition-colors ${
                    draft.armor === null
                      ? "bg-primary/20 border-primary"
                      : "hover:bg-primary/10"
                  }`}
                >
                  <div className="font-medium text-sm">No armor</div>
                </button>
                {armorOptions.map((item) => {
                  const isSelected = draft.armor === item.id;
                  return (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => setDraft((prev) => ({ ...prev, armor: item.id }))}
                      className={`p-2 text-left border rounded-md transition-colors ${
                        isSelected ? "bg-primary/20 border-primary" : "hover:bg-primary/10"
                      }`}
                    >
                      <div className="font-medium text-sm">{item.name}</div>
                    </button>
                  );
                })}
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="text-xs text-muted-foreground uppercase tracking-wide">
                  Weapons (Up to 2)
                </div>
                <div className="text-xs text-muted-foreground">
                  {draft.weapons.length} / 2 selected
                </div>
              </div>
              <div className="grid gap-2">
                {weaponOptions.map((item) => {
                  const isSelected = draft.weapons.includes(item.id);
                  const isFull = draft.weapons.length >= 2 && !isSelected;
                  return (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => toggleList("weapons", item.id, 2)}
                      disabled={isFull}
                      className={`p-2 text-left border rounded-md transition-colors ${
                        isSelected
                          ? "bg-primary/20 border-primary"
                          : isFull
                          ? "opacity-50"
                          : "hover:bg-primary/10"
                      }`}
                    >
                      <div className="font-medium text-sm">{item.name}</div>
                    </button>
                  );
                })}
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="text-xs text-muted-foreground uppercase tracking-wide">
                  Gear (Up to 3)
                </div>
                <div className="text-xs text-muted-foreground">
                  {draft.gear.length} / 3 selected
                </div>
              </div>
              <div className="grid gap-2">
                {gearOptions.map((item) => {
                  const isSelected = draft.gear.includes(item.id);
                  const isFull = draft.gear.length >= 3 && !isSelected;
                  return (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => toggleList("gear", item.id, 3)}
                      disabled={isFull}
                      className={`p-2 text-left border rounded-md transition-colors ${
                        isSelected
                          ? "bg-primary/20 border-primary"
                          : isFull
                          ? "opacity-50"
                          : "hover:bg-primary/10"
                      }`}
                    >
                      <div className="font-medium text-sm">{item.name}</div>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {updatePilotGear.error && (
          <div className="mt-4 text-sm text-destructive">{pilotGearError}</div>
        )}
      </CardContent>
    </Card>
  );
}

type BuildDraft = Omit<MechBuild, "frame_id">;

function MechBuildSection({ character }: { character: CharacterResponse }) {
  const { data: frames } = useFrames();
  const { data: weapons } = useWeapons();
  const { data: systems } = useSystems();
  const updateMechBuild = useUpdateMechBuild();
  const [isEditing, setIsEditing] = useState(false);

  const activeMech = character.mechs.find(
    (mech) => mech.id === character.active_mech_id
  );

  const frame = frames?.find((item) => item.id === activeMech?.frame_id) ?? null;

  const licenseMap = useMemo(
    () => new Map(character.licenses.map((lic) => [lic.license_id, lic.rank])),
    [character.licenses]
  );

  const isLicenseAllowed = (licenseId: string | null | undefined, rank?: number | null) => {
    if (!licenseId) return true;
    const ownedRank = licenseMap.get(licenseId) ?? 0;
    return ownedRank >= (rank ?? 1);
  };

  const weaponMap = useMemo(
    () => new Map(weapons?.map((weapon) => [weapon.id, weapon]) ?? []),
    [weapons]
  );

  const systemMap = useMemo(
    () => new Map(systems?.map((system) => [system.id, system]) ?? []),
    [systems]
  );

  const currentDraft = useMemo<BuildDraft>(
    () => ({
      weapons: activeMech?.build?.weapons ?? [],
      systems: activeMech?.build?.systems ?? [],
    }),
    [activeMech?.build?.systems, activeMech?.build?.weapons]
  );

  const [draft, setDraft] = useState<BuildDraft>(currentDraft);

  useEffect(() => {
    if (!isEditing) {
      setDraft(currentDraft);
    }
  }, [currentDraft, isEditing]);

  const allowedSizesForSlot = (slotType: string) => {
    switch (slotType) {
      case "main":
        return ["main", "aux"];
      case "heavy":
        return ["superheavy", "heavy", "main", "aux"];
      case "aux_aux":
        return ["aux"];
      case "main_aux":
        return ["main", "aux"];
      case "flexible":
        return ["main", "aux"];
      case "integrated":
        return ["aux", "main", "heavy", "superheavy"];
      default:
        return ["main", "aux"];
    }
  };

  const slotCapacity = (slotType: string) => {
    if (slotType === "main" || slotType === "heavy" || slotType === "integrated") {
      return 1;
    }
    return 2;
  };

  const selectWeaponOptions = (
    slotType: string,
    selectedSizes: string[],
    includeId?: string
  ) => {
    const allowedSizes = allowedSizesForSlot(slotType);
    const hasMain = selectedSizes.includes("main");
    return (
      weapons?.filter((weapon) => {
        if (!allowedSizes.includes(weapon.size)) return false;
        if (!isLicenseAllowed(weapon.license_id, weapon.license_rank)) return false;
        if (weapon.integrated_only && weapon.integrated_frame_id !== activeMech?.frame_id) {
          return false;
        }
        if ((slotType === "flexible" || slotType === "main_aux") && hasMain) {
          if (includeId && weapon.id === includeId) {
            return true;
          }
          return weapon.size !== "main";
        }
        return true;
      }) ?? []
    );
  };

  const updateWeapon = (draftIndex: number, weaponId: string) => {
    const definition = weaponMap.get(weaponId);
    const weaponSize = definition?.size ?? "main";
    setDraft((prev) => ({
      ...prev,
      weapons: prev.weapons.map((weapon, index) =>
        index === draftIndex
          ? { ...weapon, weapon_id: weaponId, weapon_size: weaponSize }
          : weapon
      ),
    }));
  };

  const removeWeapon = (draftIndex: number) => {
    setDraft((prev) => ({
      ...prev,
      weapons: prev.weapons.filter((_, index) => index !== draftIndex),
    }));
  };

  const addWeapon = (
    mountIndex: number,
    options: Array<{ id: string; size: string }>
  ) => {
    const choice = options?.[0];
    if (!choice) return;
    setDraft((prev) => ({
      ...prev,
      weapons: [
        ...prev.weapons,
        { mount_index: mountIndex, weapon_id: choice.id, weapon_size: choice.size },
      ],
    }));
  };

  const toggleSystem = (systemId: string) => {
    setDraft((prev) => {
      const existingIndex = prev.systems.findIndex(
        (system) => system.system_id === systemId
      );
      if (existingIndex >= 0) {
        return {
          ...prev,
          systems: prev.systems.filter((_, index) => index !== existingIndex),
        };
      }
      return {
        ...prev,
        systems: [...prev.systems, { system_id: systemId }],
      };
    });
  };

  const spSpent = draft.systems.reduce((total, system) => {
    const definition = systemMap.get(system.system_id);
    return total + (system.sp_cost ?? definition?.sp_cost ?? 0);
  }, 0);

  const spLimit = character.active_mech_stats?.system_points ?? 0;
  const spOver = spSpent > spLimit;

  const canSave = Boolean(activeMech) && !updateMechBuild.isPending;
  const mechBuildError =
    updateMechBuild.error instanceof Error
      ? updateMechBuild.error.message
      : "Failed to update mech build.";

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Mech Loadout</CardTitle>
            <CardDescription>
              {activeMech?.name ?? "No active mech selected"}
            </CardDescription>
          </div>
          {!isEditing ? (
            <Button
              type="button"
              variant="outline"
              onClick={() => setIsEditing(true)}
              disabled={!activeMech}
            >
              Edit
            </Button>
          ) : (
            <div className="flex gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  setIsEditing(false);
                  setDraft(currentDraft);
                }}
              >
                Cancel
              </Button>
              <Button
                type="button"
                onClick={() => {
                  if (!activeMech) return;
                  updateMechBuild.mutate({
                    characterId: character.id,
                    mechId: activeMech.id,
                    data: { build: draft },
                  });
                }}
                disabled={!canSave}
              >
                {updateMechBuild.isPending ? "Saving..." : "Save"}
              </Button>
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {!activeMech ? (
          <p className="text-muted-foreground">Select an active mech to edit loadout.</p>
        ) : !isEditing ? (
          <div className="space-y-4 text-sm">
            <div>
              <div className="text-xs text-muted-foreground uppercase">Weapons</div>
              <div className="font-medium">
                {draft.weapons.length
                  ? draft.weapons
                      .map((weapon) => weaponMap.get(weapon.weapon_id)?.name ?? weapon.weapon_id)
                      .join(", ")
                  : "None"}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground uppercase">Systems</div>
              <div className="font-medium">
                {draft.systems.length
                  ? draft.systems
                      .map((system) => systemMap.get(system.system_id)?.name ?? system.system_id)
                      .join(", ")
                  : "None"}
              </div>
            </div>
            <div className="text-xs text-muted-foreground">
              SP: {spSpent} / {spLimit}
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            <div>
              <div className="text-xs text-muted-foreground uppercase tracking-wide mb-2">
                Mounts
              </div>
              <div className="space-y-3">
                {frame?.mounts?.map((mount, index) => {
                  const mountWeapons = draft.weapons
                    .map((weapon, weaponIndex) => ({
                      weapon,
                      weaponIndex,
                    }))
                    .filter((entry) => entry.weapon.mount_index === index);

                  const selectedSizes = mountWeapons.map(
                    (entry) =>
                      weaponMap.get(entry.weapon.weapon_id)?.size ??
                      entry.weapon.weapon_size
                  );

                  const baseOptions = selectWeaponOptions(
                    mount.slot_type,
                    selectedSizes
                  );
                  const maxWeapons = slotCapacity(mount.slot_type);
                  const canAdd =
                    mount.slot_type !== "integrated" &&
                    mountWeapons.length < maxWeapons &&
                    baseOptions.length > 0;

                  return (
                    <div key={`mount-${index}`} className="p-3 border rounded-md">
                      <div className="flex items-center justify-between mb-2">
                        <div className="text-sm font-medium capitalize">
                          Mount {index + 1}: {mount.slot_type.replace("_", " ")}
                        </div>
                        {canAdd && (
                          <Button
                            type="button"
                            variant="outline"
                            onClick={() => addWeapon(index, baseOptions)}
                          >
                            Add Weapon
                          </Button>
                        )}
                      </div>

                      {mount.slot_type === "integrated" && mount.integrated_weapon_id && (
                        <div className="text-sm text-muted-foreground">
                          Integrated: {weaponMap.get(mount.integrated_weapon_id)?.name ?? mount.integrated_weapon_id}
                        </div>
                      )}

                      {mountWeapons.length === 0 && mount.slot_type !== "integrated" && (
                        <div className="text-sm text-muted-foreground">
                          No weapons mounted.
                        </div>
                      )}

                      <div className="space-y-2">
                        {mountWeapons.map((entry) => (
                          <div key={`weapon-${entry.weaponIndex}`} className="flex items-center gap-2">
                            <select
                              value={entry.weapon.weapon_id}
                              onChange={(event) => updateWeapon(entry.weaponIndex, event.target.value)}
                              className="flex-1 px-3 py-2 bg-background border border-border rounded-md"
                            >
                              {selectWeaponOptions(
                                mount.slot_type,
                                selectedSizes,
                                entry.weapon.weapon_id
                              ).map((weapon) => (
                                <option key={weapon.id} value={weapon.id}>
                                  {weapon.name}
                                </option>
                              ))}
                            </select>
                            <Button
                              type="button"
                              variant="outline"
                              onClick={() => removeWeapon(entry.weaponIndex)}
                            >
                              Remove
                            </Button>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="text-xs text-muted-foreground uppercase tracking-wide">
                  Systems
                </div>
                <div className={spOver ? "text-xs text-destructive" : "text-xs text-muted-foreground"}>
                  SP: {spSpent} / {spLimit}
                </div>
              </div>
              <div className="grid gap-2 max-h-56 overflow-y-auto">
                {(systems ?? [])
                  .filter((system) => isLicenseAllowed(system.license_id, system.license_rank))
                  .map((system) => {
                    const isSelected = draft.systems.some(
                      (entry) => entry.system_id === system.id
                    );
                    return (
                      <button
                        key={system.id}
                        type="button"
                        onClick={() => toggleSystem(system.id)}
                        className={`p-2 text-left border rounded-md transition-colors ${
                          isSelected
                            ? "bg-primary/20 border-primary"
                            : "hover:bg-primary/10"
                        }`}
                      >
                        <div className="flex items-center justify-between text-sm">
                          <span className="font-medium">{system.name}</span>
                          <span className="text-xs text-muted-foreground">
                            SP {system.sp_cost}
                          </span>
                        </div>
                      </button>
                    );
                  })}
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                Showing systems available at your current license level.
              </p>
            </div>
          </div>
        )}

        {updateMechBuild.error && (
          <div className="mt-4 text-sm text-destructive">{mechBuildError}</div>
        )}
      </CardContent>
    </Card>
  );
}

function StatBlock({
  label,
  value,
}: {
  label: string;
  value: number | string;
}) {
  return (
    <div>
      <div className="text-muted-foreground text-xs uppercase">{label}</div>
      <div className="font-semibold text-lg">{value}</div>
    </div>
  );
}

function SkillBlock({ label, value }: { label: string; value: number }) {
  return (
    <div className="p-2 bg-card-foreground/5 rounded text-center">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-semibold">+{value}</div>
    </div>
  );
}
