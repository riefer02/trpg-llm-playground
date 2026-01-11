/**
 * Character export page.
 *
 * Print-friendly view for saving a character sheet to PDF.
 */

import { useMemo } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import {
  useCharacter,
  usePilotGear,
  useWeapons,
  useSystems,
  useFrames,
} from "../../lib/api";
import {
  Button,
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "../../components/ui";

export const Route = createFileRoute("/characters/$characterId/export" as const)({
  component: CharacterExportPage,
});

function CharacterExportPage() {
  const { characterId } = Route.useParams();
  const { data: character, isLoading, error } = useCharacter(characterId);
  const { data: pilotGear } = usePilotGear();
  const { data: weapons } = useWeapons();
  const { data: systems } = useSystems();
  const { data: frames } = useFrames();

  const pilotGearMap = useMemo(
    () => new Map(pilotGear?.map((item) => [item.id, item]) ?? []),
    [pilotGear]
  );
  const weaponMap = useMemo(
    () => new Map(weapons?.map((item) => [item.id, item]) ?? []),
    [weapons]
  );
  const systemMap = useMemo(
    () => new Map(systems?.map((item) => [item.id, item]) ?? []),
    [systems]
  );

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
        <Card className="border-destructive print-card">
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
        <Card className="print-card">
          <CardContent className="pt-6 text-center">
            <p className="text-muted-foreground">Character not found</p>
            <Link
              to="/characters"
              className="text-primary hover:underline no-print"
            >
              Back to Characters
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  const activeMech = character.mechs.find(
    (mech) => mech.id === character.active_mech_id
  );
  const stats = character.active_mech_stats;
  const frame = frames?.find((item) => item.id === activeMech?.frame_id) ?? null;

  const gear = character.pilot_gear;
  const gearName = (id: string | null | undefined) => {
    if (!id) return "None";
    return pilotGearMap.get(id)?.name ?? formatLabel(id);
  };

  const listNames = (ids: string[] | undefined) => {
    if (!ids?.length) return "None";
    return ids.map((id) => pilotGearMap.get(id)?.name ?? formatLabel(id)).join(", ");
  };

  const buildWeapons = activeMech?.build?.weapons ?? [];
  const buildSystems = activeMech?.build?.systems ?? [];

  const spSpent = buildSystems.reduce((total, system) => {
    const definition = systemMap.get(system.system_id);
    return total + (system.sp_cost ?? definition?.sp_cost ?? 0);
  }, 0);
  const spLimit = stats?.system_points ?? 0;

  return (
    <div className="min-h-screen bg-background text-foreground print-sheet">
      <div className="no-print max-w-5xl mx-auto px-6 pt-6">
        <Link to="/characters/$characterId" params={{ characterId }}>
          <Button variant="outline">Back to Character</Button>
        </Link>
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between mt-4">
          <div>
            <h1 className="text-3xl font-bold">Export Character</h1>
            <p className="text-muted-foreground">
              Print or save this page as a PDF.
            </p>
          </div>
          <Button type="button" onClick={() => window.print()}>
            Print / Save PDF
          </Button>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-6 pb-16 pt-8 space-y-6">
        <section className="print-grid">
          <Card className="print-card">
            <CardHeader>
              <CardTitle className="text-2xl">{character.callsign}</CardTitle>
              <CardDescription className="print-muted">
                {character.name || "Unnamed"} - License Level {character.level}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-6 md:grid-cols-2">
                <div className="space-y-3">
                  <DetailRow label="Background" value={character.background?.name || "None"} />
                  <DetailRow label="Pilot HP" value={character.pilot_hp} />
                  <DetailRow label="Grit" value={`+${character.grit}`} />
                  <DetailRow label="Notes" value={character.notes || ""} emptyLabel="None" />
                </div>
                <div className="space-y-3">
                  <div className="text-xs text-muted-foreground uppercase tracking-wide">
                    Pilot Skills
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <StatLine label="HULL" value={`+${character.skills.hull ?? 0}`} />
                    <StatLine label="AGI" value={`+${character.skills.agility ?? 0}`} />
                    <StatLine label="SYS" value={`+${character.skills.systems ?? 0}`} />
                    <StatLine label="ENG" value={`+${character.skills.engineering ?? 0}`} />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-6 md:grid-cols-2 print-grid">
          <Card className="print-card">
            <CardHeader>
              <CardTitle>Active Mech</CardTitle>
              <CardDescription className="print-muted">
                {activeMech
                  ? `${activeMech.name} - ${frame?.name ?? formatLabel(activeMech.frame_id)}`
                  : "No active mech"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {stats ? (
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <StatLine label="HP" value={stats.hp} />
                  <StatLine label="Armor" value={stats.armor} />
                  <StatLine label="Evasion" value={stats.evasion} />
                  <StatLine label="E-Defense" value={stats.e_defense} />
                  <StatLine label="Speed" value={stats.speed} />
                  <StatLine label="Heat Cap" value={stats.heat_cap} />
                  <StatLine label="SP" value={stats.system_points} />
                  <StatLine label="Save Target" value={stats.save_target} />
                </div>
              ) : (
                <p className="text-muted-foreground text-sm">No stats available.</p>
              )}
            </CardContent>
          </Card>

          <Card className="print-card">
            <CardHeader>
              <CardTitle>Pilot Gear</CardTitle>
              <CardDescription className="print-muted">
                Mission loadout snapshot
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <DetailRow label="Clothing" value={gearName(gear?.clothing)} />
              <DetailRow label="Armor" value={gearName(gear?.armor)} />
              <DetailRow label="Weapons" value={listNames(gear?.weapons)} />
              <DetailRow label="Gear" value={listNames(gear?.gear)} />
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-6 md:grid-cols-2 print-grid">
          <Card className="print-card">
            <CardHeader>
              <CardTitle>Mech Loadout</CardTitle>
              <CardDescription className="print-muted">
                Weapons and systems equipped
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <div>
                <div className="text-xs text-muted-foreground uppercase tracking-wide mb-2">
                  Weapons
                </div>
                {buildWeapons.length ? (
                  <ul className="space-y-1">
                    {buildWeapons.map((weapon, index) => (
                      <li key={`${weapon.weapon_id}-${index}`}>
                        Mount {weapon.mount_index + 1}: {weaponMap.get(weapon.weapon_id)?.name ?? formatLabel(weapon.weapon_id)} ({weapon.weapon_size})
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="text-muted-foreground">None</div>
                )}
              </div>
              <div>
                <div className="text-xs text-muted-foreground uppercase tracking-wide mb-2">
                  Systems
                </div>
                {buildSystems.length ? (
                  <ul className="space-y-1">
                    {buildSystems.map((system, index) => (
                      <li key={`${system.system_id}-${index}`}>
                        {systemMap.get(system.system_id)?.name ?? formatLabel(system.system_id)} (SP {system.sp_cost ?? systemMap.get(system.system_id)?.sp_cost ?? 0})
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="text-muted-foreground">None</div>
                )}
              </div>
              <div className="text-xs text-muted-foreground">
                SP Usage: {spSpent} / {spLimit}
              </div>
            </CardContent>
          </Card>

          <Card className="print-card">
            <CardHeader>
              <CardTitle>Licenses & Core Bonuses</CardTitle>
              <CardDescription className="print-muted">
                Access to frames and gear
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <div>
                <div className="text-xs text-muted-foreground uppercase tracking-wide mb-2">
                  Licenses
                </div>
                {character.licenses.length ? (
                  <ul className="space-y-1">
                    {character.licenses.map((license) => (
                      <li key={license.license_id}>
                        {formatLicense(license.license_id)} - Rank {license.rank}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="text-muted-foreground">None</div>
                )}
              </div>
              <div>
                <div className="text-xs text-muted-foreground uppercase tracking-wide mb-2">
                  Core Bonuses
                </div>
                {character.core_bonuses.length ? (
                  <ul className="space-y-1">
                    {character.core_bonuses.map((bonus) => (
                      <li key={bonus.core_bonus_id}>
                        {formatLabel(bonus.core_bonus_id)}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="text-muted-foreground">None</div>
                )}
              </div>
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-6 md:grid-cols-2 print-grid">
          <Card className="print-card">
            <CardHeader>
              <CardTitle>Talents</CardTitle>
            </CardHeader>
            <CardContent className="text-sm">
              {character.talents.length ? (
                <ul className="space-y-1">
                  {character.talents.map((talent) => (
                    <li key={talent.talent_id}>
                      {formatLabel(talent.talent_id)} - Rank {talent.rank}
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="text-muted-foreground">None</div>
              )}
            </CardContent>
          </Card>

          <Card className="print-card">
            <CardHeader>
              <CardTitle>Triggers</CardTitle>
            </CardHeader>
            <CardContent className="text-sm">
              {character.triggers.length ? (
                <ul className="space-y-1">
                  {character.triggers.map((trigger) => (
                    <li key={`${trigger.trigger_id}-${trigger.rank}`}>
                      {formatLabel(trigger.trigger_id)} - +{trigger.rank}
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="text-muted-foreground">None</div>
              )}
            </CardContent>
          </Card>
        </section>
      </div>
    </div>
  );
}

function DetailRow({
  label,
  value,
  emptyLabel,
}: {
  label: string;
  value: string | number | null | undefined;
  emptyLabel?: string;
}) {
  const displayValue = value === null || value === undefined || value === "" ? emptyLabel ?? "" : value;
  return (
    <div className="space-y-1">
      <div className="text-xs text-muted-foreground uppercase tracking-wide">
        {label}
      </div>
      <div className="text-sm font-medium">
        {displayValue || emptyLabel || "None"}
      </div>
    </div>
  );
}

function StatLine({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-center justify-between border-b border-border pb-2 print-border">
      <span className="text-xs text-muted-foreground uppercase">{label}</span>
      <span className="font-semibold">{value}</span>
    </div>
  );
}

function formatLabel(value: string) {
  return value.replace(/^gms_/, "GMS ").replace(/_/g, " ");
}

function formatLicense(value: string) {
  return value.replace(/_/g, " ").toUpperCase();
}
