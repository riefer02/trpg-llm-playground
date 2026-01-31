/**
 * Read-only mech display component for quarters.
 * Shows mech frame, combat stats, weapons, and systems.
 */

import { useMemo } from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "../ui";
import { StatBlock } from "../ui/stat-blocks";
import { LicenseBadge } from "../ui/LicenseBadge";
import {
  useFrames,
  useWeapons,
  useSystems,
  useSpendSalvageForRepair,
} from "../../lib/api";
import type { CharacterResponse } from "../../lib/api";
import { Button } from "../ui/button";


interface MechDisplayProps {
  character: CharacterResponse;
}

export function MechDisplay({ character }: MechDisplayProps) {
  const { data: frames } = useFrames();
  const { data: weapons } = useWeapons();
  const { data: systems } = useSystems();

  const activeMech = character.mechs.find(
    (m) => m.id === character.active_mech_id
  );
  const stats = character.active_mech_stats;

  const frame = frames?.find((item) => item.id === activeMech?.frame_id) ?? null;

  const weaponMap = useMemo(
    () => new Map(weapons?.map((weapon) => [weapon.id, weapon]) ?? []),
    [weapons]
  );

  const systemMap = useMemo(
    () => new Map(systems?.map((system) => [system.id, system]) ?? []),
    [systems]
  );



  const mountedWeapons = activeMech?.build?.weapons ?? [];
  const installedSystems = activeMech?.build?.systems ?? [];

  const spSpent = installedSystems.reduce((total, system) => {
    const definition = systemMap.get(system.system_id);
    return total + (system.sp_cost ?? definition?.sp_cost ?? 0);
  }, 0);

  const spLimit = stats?.system_points ?? 0;

  if (!activeMech) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>No Active Mech</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            Select an active mech in the character detail page.
          </p>
        </CardContent>
      </Card>
    );
  }

  const { mutate: spendSalvage, isPending } = useSpendSalvageForRepair();

  const handleRepair = (repairType: string, weaponId?: string, systemId?: string) => {
    spendSalvage({
      characterId: character.id,
      data: {
        repair_type: repairType,
        mech_id: activeMech.id,
        weapon_id: weaponId,
        system_id: systemId,
      },
    }, {
      onSuccess: () => {
        toast({
          title: "Repair successful",
          description: `Spent salvage to repair ${repairType}.`,
        });
      },
      onError: (error) => {
        toast({
          title: "Repair failed",
          description: error.message,
          variant: "destructive",
        });
      },
    });
  };

  const repairCosts = {
    hp: 1,
    structure: 2,
    stress: 2,
    destroyed_weapon: 1,
    destroyed_system: 1,
    destroyed_mech: 4,
  };

  // Determine if repairs are needed
  const damageState = activeMech.damage_state;
  const maxHp = stats?.hp ?? 0;
  const currentHp = damageState?.hp_current ?? maxHp;
  const currentStructure = damageState?.structure_current ?? 4;
  const currentStress = damageState?.stress_current ?? 0;
  const hasDestroyedWeapons = damageState?.destroyed_weapons && damageState.destroyed_weapons.length > 0;
  const hasDestroyedSystems = damageState?.destroyed_systems && damageState.destroyed_systems.length > 0;
  const isDestroyed = damageState?.is_destroyed ?? false;

  return (
    <div className="space-y-6">
      {/* Salvage & Repairs */}
      <Card>
        <CardHeader>
          <CardTitle>Salvage & Repairs</CardTitle>
          <CardDescription>
            Spend salvage to repair your mech. Current salvage: {character.salvage}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-2">
            <Button
              disabled={isPending || currentHp >= maxHp || character.salvage < repairCosts.hp}
              onClick={() => handleRepair('hp')}
            >
              Repair HP ({repairCosts.hp} salvage)
            </Button>
            <Button
              disabled={isPending || currentStructure >= 4 || character.salvage < repairCosts.structure}
              onClick={() => handleRepair('structure')}
            >
              Repair Structure ({repairCosts.structure} salvage)
            </Button>
            <Button
              disabled={isPending || currentStress <= 0 || character.salvage < repairCosts.stress}
              onClick={() => handleRepair('stress')}
            >
              Repair Stress ({repairCosts.stress} salvage)
            </Button>
            <Button
              disabled={isPending || !hasDestroyedWeapons || character.salvage < repairCosts.destroyed_weapon}
              onClick={() => handleRepair('destroyed_weapon')}
            >
              Repair Destroyed Weapon ({repairCosts.destroyed_weapon} salvage)
            </Button>
            <Button
              disabled={isPending || !hasDestroyedSystems || character.salvage < repairCosts.destroyed_system}
              onClick={() => handleRepair('destroyed_system')}
            >
              Repair Destroyed System ({repairCosts.destroyed_system} salvage)
            </Button>
            <Button
              disabled={isPending || !isDestroyed || character.salvage < repairCosts.destroyed_mech}
              onClick={() => handleRepair('destroyed_mech')}
            >
              Restore Destroyed Mech ({repairCosts.destroyed_mech} salvage)
            </Button>
          </div>
          <p className="text-xs text-muted-foreground mt-4">
            Repair costs match standard repair costs: 1 salvage for HP/weapon/system, 2 for structure/stress, 4 for destroyed mech.
          </p>
        </CardContent>
      </Card>

      {/* Mech Stats */}
      <Card>
        <CardHeader>
          <CardTitle>
            {activeMech.name}
            {frame && (
              <span className="text-sm font-normal text-muted-foreground ml-2">
                ({frame.id.replace(/^gms_/, "GMS ").replace(/_/g, " ")})
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
                <StatBlock label="HP" value={`${currentHp}/${stats.hp}`} />
                <StatBlock label="Armor" value={stats.armor} />
                <StatBlock label="Size" value={stats.size.replace("size_", "")} />
              </div>
              <div className="grid grid-cols-3 gap-4 mb-4">
                <StatBlock label="Structure" value={`${currentStructure}/4`} />
                <StatBlock label="Stress" value={currentStress} />
                <StatBlock label="Status" value={isDestroyed ? "DESTROYED" : (hasDestroyedWeapons || hasDestroyedSystems ? "DAMAGED" : "OK")} />
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
            <p className="text-muted-foreground">No combat stats available</p>
          )}
        </CardContent>
      </Card>

      {/* Weapons */}
      <Card>
        <CardHeader>
          <CardTitle>Weapons</CardTitle>
          <CardDescription>Mounted weapons</CardDescription>
        </CardHeader>
        <CardContent>
          {mountedWeapons.length === 0 ? (
            <p className="text-muted-foreground">No weapons mounted.</p>
          ) : (
            <div className="space-y-3">
              {mountedWeapons.map((weapon, index) => {
                const weaponDef = weaponMap.get(weapon.weapon_id);
                return (
                  <div
                    key={index}
                    className="p-3 bg-muted/50 border border-border rounded-md"
                  >
                    <div className="flex items-center justify-between">
                      <div className="font-medium">
                        {weaponDef?.name ?? weapon.weapon_id}
                      </div>
                      {weaponDef && (
                        <LicenseBadge
                          licenseId={weaponDef.license_id}
                          licenseRank={weaponDef.license_rank}
                        />
                      )}
                    </div>
                    {weaponDef && (
                      <div className="text-sm text-muted-foreground mt-1">
                        {weaponDef.damage} {weaponDef.damage_type} damage •{" "}
                        {weaponDef.range} range • Size: {weaponDef.size}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Systems */}
      <Card>
        <CardHeader>
          <CardTitle>Systems</CardTitle>
          <CardDescription>Installed systems and SP usage</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between mb-4">
            <div className="text-sm font-medium">System Points</div>
            <div className="text-sm">
              {spSpent} / {spLimit} SP
            </div>
          </div>
          {installedSystems.length === 0 ? (
            <p className="text-muted-foreground">No systems installed.</p>
          ) : (
            <div className="space-y-3">
              {installedSystems.map((system, index) => {
                const systemDef = systemMap.get(system.system_id);
                const spCost = system.sp_cost ?? systemDef?.sp_cost ?? 0;
                return (
                  <div
                    key={index}
                    className="p-3 bg-muted/50 border border-border rounded-md"
                  >
                    <div className="flex items-center justify-between">
                      <div className="font-medium">
                        {systemDef?.name ?? system.system_id}
                      </div>
                      <div className="flex items-center gap-2">
                        <LicenseBadge
                          licenseId={systemDef?.license_id}
                          licenseRank={systemDef?.license_rank}
                        />
                        <span className="text-sm text-muted-foreground">
                          SP {spCost}
                        </span>
                      </div>
                    </div>
                    {systemDef?.description && (
                      <p className="text-sm text-muted-foreground mt-1">
                        {systemDef.description}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          )}
          <p className="text-xs text-muted-foreground mt-4">
            Showing systems available at your current license level.
          </p>
        </CardContent>
      </Card>

      {/* Frame Traits (if frame exists) */}
      {frame && frame.traits && frame.traits.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Frame Traits</CardTitle>
            <CardDescription>Unique frame characteristics</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {frame.traits.map((trait, index) => (
                <div
                  key={index}
                  className="p-3 bg-muted/50 border border-border rounded-md"
                >
                  <div className="font-medium capitalize">
                    {trait.name?.replace(/_/g, " ")}
                  </div>
                  {trait.description && (
                    <p className="text-sm text-muted-foreground mt-1">
                      {trait.description}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}