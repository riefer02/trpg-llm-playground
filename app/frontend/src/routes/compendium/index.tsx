/**
 * Compendium page.
 *
 * Read-only reference for frames, weapons, systems, and pilot gear.
 */

import { useMemo, useState } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import {
  useFrames,
  useWeapons,
  useSystems,
  usePilotGear,
} from "../../lib/api";
import type {
  MechFrameDefinition,
  MechWeaponDefinition,
  MechSystemDefinition,
  PilotGearItemDefinition,
} from "../../lib/types/lancer";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  Button,
} from "../../components/ui";

export const Route = createFileRoute("/compendium/" as const)({
  component: CompendiumPage,
});

type CompendiumCategory = "frames" | "weapons" | "systems" | "pilot-gear";
type AvailabilityFilter = "all" | "gms" | "licensed";
type RankFilter = "all" | "1" | "2" | "3";
type ManufacturerFilter = "all" | "GMS" | "IPS-N" | "SSC" | "HORUS" | "HA";
type WeaponSizeFilter = "all" | string;
type WeaponTypeFilter = "all" | string;
type SystemTypeFilter = "all" | string;
type GearCategoryFilter = "all" | "clothing" | "armor" | "weapon" | "gear";

const manufacturerColors: Record<string, string> = {
  GMS: "text-gms",
  "IPS-N": "text-ipsn",
  SSC: "text-ssc",
  HORUS: "text-horus",
  HA: "text-ha",
};

const categoryLabels: Record<CompendiumCategory, string> = {
  frames: "Frames",
  weapons: "Weapons",
  systems: "Systems",
  "pilot-gear": "Pilot Gear",
};

function CompendiumPage() {
  const framesQuery = useFrames();
  const weaponsQuery = useWeapons();
  const systemsQuery = useSystems();
  const gearQuery = usePilotGear();

  const frames = framesQuery.data ?? [];
  const weapons = weaponsQuery.data ?? [];
  const systems = systemsQuery.data ?? [];
  const pilotGear = gearQuery.data ?? [];

  const [category, setCategory] = useState<CompendiumCategory>("frames");
  const [search, setSearch] = useState("");
  const [availability, setAvailability] =
    useState<AvailabilityFilter>("all");
  const [licenseRank, setLicenseRank] = useState<RankFilter>("all");
  const [manufacturer, setManufacturer] =
    useState<ManufacturerFilter>("all");
  const [weaponSize, setWeaponSize] = useState<WeaponSizeFilter>("all");
  const [weaponType, setWeaponType] = useState<WeaponTypeFilter>("all");
  const [systemType, setSystemType] = useState<SystemTypeFilter>("all");
  const [gearCategory, setGearCategory] =
    useState<GearCategoryFilter>("all");

  const normalizedSearch = search.trim().toLowerCase();

  const weaponSizeOptions = useMemo(() => {
    const sizes = new Set<string>();
    weapons.forEach((weapon) => sizes.add(weapon.size));
    return ["all", ...Array.from(sizes).sort()];
  }, [weapons]);

  const weaponTypeOptions = useMemo(() => {
    const types = new Set<string>();
    weapons.forEach((weapon) => types.add(weapon.weapon_type));
    return ["all", ...Array.from(types).sort()];
  }, [weapons]);

  const systemTypeOptions = useMemo(() => {
    const types = new Set<string>();
    systems.forEach((system) => types.add(system.system_type));
    return ["all", ...Array.from(types).sort()];
  }, [systems]);

  const gearCategoryOptions = useMemo(() => {
    const categories = new Set<string>();
    pilotGear.forEach((item) => categories.add(item.category));
    return ["all", ...Array.from(categories).sort()];
  }, [pilotGear]);

  const filteredFrames = useMemo(
    () =>
      frames.filter((frame) => {
        if (!matchesSearch(normalizedSearch, frame.name, frame.id, frame.license_id)) {
          return false;
        }
        if (!matchesAvailability(availability, frame.license_id)) {
          return false;
        }
        if (!matchesRank(licenseRank, frame.license_rank)) {
          return false;
        }
        if (manufacturer !== "all" && frame.manufacturer !== manufacturer) {
          return false;
        }
        return true;
      }),
    [frames, normalizedSearch, availability, licenseRank, manufacturer]
  );

  const filteredWeapons = useMemo(
    () =>
      weapons.filter((weapon) => {
        if (!matchesSearch(normalizedSearch, weapon.name, weapon.id, weapon.license_id)) {
          return false;
        }
        if (!matchesAvailability(availability, weapon.license_id)) {
          return false;
        }
        if (!matchesRank(licenseRank, weapon.license_rank)) {
          return false;
        }
        if (weaponSize !== "all" && weapon.size !== weaponSize) {
          return false;
        }
        if (weaponType !== "all" && weapon.weapon_type !== weaponType) {
          return false;
        }
        return true;
      }),
    [
      weapons,
      normalizedSearch,
      availability,
      licenseRank,
      weaponSize,
      weaponType,
    ]
  );

  const filteredSystems = useMemo(
    () =>
      systems.filter((system) => {
        if (!matchesSearch(normalizedSearch, system.name, system.id, system.license_id)) {
          return false;
        }
        if (!matchesAvailability(availability, system.license_id)) {
          return false;
        }
        if (!matchesRank(licenseRank, system.license_rank)) {
          return false;
        }
        if (systemType !== "all" && system.system_type !== systemType) {
          return false;
        }
        return true;
      }),
    [
      systems,
      normalizedSearch,
      availability,
      licenseRank,
      systemType,
    ]
  );

  const filteredPilotGear = useMemo(
    () =>
      pilotGear.filter((item) => {
        if (!matchesSearch(normalizedSearch, item.name, item.id)) {
          return false;
        }
        if (gearCategory !== "all" && item.category !== gearCategory) {
          return false;
        }
        return true;
      }),
    [pilotGear, normalizedSearch, gearCategory]
  );

  const categoryCounts: Record<CompendiumCategory, number> = {
    frames: frames.length,
    weapons: weapons.length,
    systems: systems.length,
    "pilot-gear": pilotGear.length,
  };

  const filteredCounts: Record<CompendiumCategory, number> = {
    frames: filteredFrames.length,
    weapons: filteredWeapons.length,
    systems: filteredSystems.length,
    "pilot-gear": filteredPilotGear.length,
  };

  const isLoading =
    framesQuery.isLoading ||
    weaponsQuery.isLoading ||
    systemsQuery.isLoading ||
    gearQuery.isLoading;

  const activeError =
    (category === "frames" && framesQuery.error) ||
    (category === "weapons" && weaponsQuery.error) ||
    (category === "systems" && systemsQuery.error) ||
    (category === "pilot-gear" && gearQuery.error);

  const clearFilters = () => {
    setSearch("");
    setAvailability("all");
    setLicenseRank("all");
    setManufacturer("all");
    setWeaponSize("all");
    setWeaponType("all");
    setSystemType("all");
    setGearCategory("all");
  };

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div className="flex flex-col gap-2">
        <Link to="/" className="text-primary hover:underline text-sm">
          ← Back to Home
        </Link>
        <h1 className="text-3xl font-bold text-foreground">Compendium</h1>
        <p className="text-muted-foreground">
          Reference data for frames, weapons, systems, and pilot gear. License
          requirements are shown for quick gating checks.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Browse</CardTitle>
          <CardDescription>Search and filter the compendium.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-2 md:grid-cols-2">
            <input
              type="text"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search by name or ID..."
              className="h-10 rounded-md border border-border bg-background px-3 text-sm text-foreground"
            />
            <div className="flex flex-wrap gap-2">
              {(
                [
                  "frames",
                  "weapons",
                  "systems",
                  "pilot-gear",
                ] as CompendiumCategory[]
              ).map((item) => (
                <Button
                  key={item}
                  type="button"
                  variant={category === item ? "primary" : "outline"}
                  size="sm"
                  onClick={() => setCategory(item)}
                >
                  {categoryLabels[item]} ({categoryCounts[item]})
                </Button>
              ))}
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-4">
            <FilterSelect
              label="Availability"
              value={availability}
              onChange={(value) => setAvailability(value as AvailabilityFilter)}
              options={[
                { value: "all", label: "All gear" },
                { value: "gms", label: "GMS only" },
                { value: "licensed", label: "Licensed only" },
              ]}
            />
            <FilterSelect
              label="License rank"
              value={licenseRank}
              onChange={(value) => setLicenseRank(value as RankFilter)}
              options={[
                { value: "all", label: "Any rank" },
                { value: "1", label: "Rank I" },
                { value: "2", label: "Rank II" },
                { value: "3", label: "Rank III" },
              ]}
            />
            {category === "frames" && (
              <FilterSelect
                label="Manufacturer"
                value={manufacturer}
                onChange={(value) =>
                  setManufacturer(value as ManufacturerFilter)
                }
                options={[
                  { value: "all", label: "All manufacturers" },
                  { value: "GMS", label: "GMS" },
                  { value: "IPS-N", label: "IPS-N" },
                  { value: "SSC", label: "SSC" },
                  { value: "HORUS", label: "HORUS" },
                  { value: "HA", label: "HA" },
                ]}
              />
            )}
            {category === "weapons" && (
              <FilterSelect
                label="Weapon size"
                value={weaponSize}
                onChange={(value) => setWeaponSize(value)}
                options={weaponSizeOptions.map((value) => ({
                  value,
                  label: value === "all" ? "All sizes" : formatEnum(value),
                }))}
              />
            )}
            {category === "weapons" && (
              <FilterSelect
                label="Weapon type"
                value={weaponType}
                onChange={(value) => setWeaponType(value)}
                options={weaponTypeOptions.map((value) => ({
                  value,
                  label: value === "all" ? "All types" : formatEnum(value),
                }))}
              />
            )}
            {category === "systems" && (
              <FilterSelect
                label="System type"
                value={systemType}
                onChange={(value) => setSystemType(value)}
                options={systemTypeOptions.map((value) => ({
                  value,
                  label: value === "all" ? "All types" : formatEnum(value),
                }))}
              />
            )}
            {category === "pilot-gear" && (
              <FilterSelect
                label="Gear category"
                value={gearCategory}
                onChange={(value) => setGearCategory(value as GearCategoryFilter)}
                options={gearCategoryOptions.map((value) => ({
                  value,
                  label: value === "all" ? "All categories" : formatEnum(value),
                }))}
              />
            )}
            <div className="flex items-end">
              <Button type="button" variant="ghost" onClick={clearFilters}>
                Reset filters
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {isLoading && (
        <div className="text-muted-foreground">Loading compendium data...</div>
      )}

      {activeError && (
        <Card className="border-destructive">
          <CardContent className="pt-6 text-destructive">
            Failed to load compendium data.
          </CardContent>
        </Card>
      )}

      {!isLoading && !activeError && (
        <div className="space-y-4">
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>
              Showing {filteredCounts[category]} of {categoryCounts[category]}
            </span>
            <span>Search and filters apply to the active category.</span>
          </div>

          {category === "frames" && (
            <FrameGrid frames={filteredFrames} />
          )}
          {category === "weapons" && (
            <WeaponGrid weapons={filteredWeapons} />
          )}
          {category === "systems" && (
            <SystemGrid systems={filteredSystems} />
          )}
          {category === "pilot-gear" && (
            <PilotGearGrid items={filteredPilotGear} />
          )}

          {filteredCounts[category] === 0 && (
            <Card>
              <CardContent className="pt-6 text-muted-foreground">
                No results found. Try clearing filters or using a broader search.
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}

function FilterSelect({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: Array<{ value: string; label: string }>;
}) {
  return (
    <label className="flex flex-col gap-1 text-sm text-muted-foreground">
      <span>{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="h-10 rounded-md border border-border bg-background px-3 text-sm text-foreground"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function FrameGrid({ frames }: { frames: MechFrameDefinition[] }) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {frames.map((frame) => (
        <Card key={frame.id}>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>{frame.name}</span>
              <span
                className={`text-sm font-semibold ${
                  manufacturerColors[frame.manufacturer] || "text-muted-foreground"
                }`}
              >
                {frame.manufacturer}
              </span>
            </CardTitle>
            <CardDescription>
              {formatLicense(frame.license_id, frame.license_rank)}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="grid grid-cols-2 gap-2 text-muted-foreground">
              <span>Size: {formatSize(frame.base_stats.size)}</span>
              <span>Armor: {frame.base_stats.armor}</span>
              <span>HP: {frame.base_stats.hp}</span>
              <span>Evasion: {frame.base_stats.evasion}</span>
              <span>E-Defense: {frame.base_stats.e_defense}</span>
              <span>Speed: {frame.base_stats.speed}</span>
              <span>SP: {frame.system_points}</span>
              <span>Save: {frame.base_stats.save_target}</span>
            </div>
            <div className="text-muted-foreground">
              Mounts: {formatMounts(frame.mounts)}
            </div>
            {frame.core_system?.name && (
              <div className="text-muted-foreground">
                Core: {frame.core_system.name}
              </div>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function WeaponGrid({ weapons }: { weapons: MechWeaponDefinition[] }) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {weapons.map((weapon) => (
        <Card key={weapon.id}>
          <CardHeader>
            <CardTitle>{weapon.name}</CardTitle>
            <CardDescription>
              {formatLicense(weapon.license_id, weapon.license_rank)}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex flex-wrap gap-2 text-muted-foreground">
              <span>{formatEnum(weapon.size)}</span>
              <span>{formatEnum(weapon.weapon_type)}</span>
              {weapon.damage_type && (
                <span>{formatEnum(weapon.damage_type)}</span>
              )}
              {weapon.unique && <span>Unique</span>}
              {weapon.integrated_only && <span>Integrated</span>}
            </div>
            <TagRow tags={weapon.tags} />
            {weapon.limited_uses != null && (
              <div className="text-muted-foreground">
                Limited uses: {weapon.limited_uses}
              </div>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function SystemGrid({ systems }: { systems: MechSystemDefinition[] }) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {systems.map((system) => (
        <Card key={system.id}>
          <CardHeader>
            <CardTitle>{system.name}</CardTitle>
            <CardDescription>
              {formatLicense(system.license_id, system.license_rank)}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex flex-wrap gap-2 text-muted-foreground">
              <span>{formatEnum(system.system_type)}</span>
              <span>SP {system.sp_cost}</span>
              {system.unique && <span>Unique</span>}
            </div>
            <TagRow tags={system.tags} />
            {system.limited_uses != null && (
              <div className="text-muted-foreground">
                Limited uses: {system.limited_uses}
              </div>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function PilotGearGrid({ items }: { items: PilotGearItemDefinition[] }) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {items.map((item) => (
        <Card key={item.id}>
          <CardHeader>
            <CardTitle>{item.name}</CardTitle>
            <CardDescription>{formatEnum(item.category)}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <TagRow tags={item.tags} />
            {item.limited_uses != null && (
              <div className="text-muted-foreground">
                Limited uses: {item.limited_uses}
              </div>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function TagRow({ tags }: { tags: Array<{ tag: string; value?: number | null }> }) {
  if (!tags.length) {
    return (
      <div className="text-muted-foreground">No tags</div>
    );
  }
  return (
    <div className="flex flex-wrap gap-2">
      {tags.map((tag, index) => (
        <span
          key={`${tag.tag}-${index}`}
          className="rounded-full bg-secondary px-2 py-1 text-xs text-secondary-foreground"
        >
          {formatTag(tag)}
        </span>
      ))}
    </div>
  );
}

function formatTag(tag: { tag: string; value?: number | null }) {
  if (tag.value === null || tag.value === undefined) {
    return formatEnum(tag.tag);
  }
  return `${formatEnum(tag.tag)} ${tag.value}`;
}

function matchesSearch(
  query: string,
  name: string,
  id: string,
  licenseId?: string | null
) {
  if (!query) {
    return true;
  }
  const lowerQuery = query.toLowerCase();
  return (
    name.toLowerCase().includes(lowerQuery) ||
    id.toLowerCase().includes(lowerQuery) ||
    (licenseId ?? "").toLowerCase().includes(lowerQuery)
  );
}

function matchesAvailability(
  availability: AvailabilityFilter,
  licenseId?: string | null
) {
  if (availability === "all") {
    return true;
  }
  const isGms = !licenseId;
  return availability === "gms" ? isGms : !isGms;
}

function matchesRank(rankFilter: RankFilter, rank?: number | null) {
  if (rankFilter === "all") {
    return true;
  }
  return rank === Number(rankFilter);
}

function formatLicense(licenseId?: string | null, rank?: number | null) {
  if (!licenseId) {
    return "License: GMS";
  }
  if (rank) {
    return `License: ${formatEnum(licenseId)} R${rank}`;
  }
  return `License: ${formatEnum(licenseId)}`;
}

function formatEnum(value: string) {
  return value.replace(/_/g, " ").toUpperCase();
}

function formatSize(value: string) {
  return value.replace("size_", "").replace("_", "/").toUpperCase();
}

function formatMounts(mounts: Array<{ slot_type: string }>) {
  if (!mounts.length) {
    return "None";
  }
  const counts = mounts.reduce<Record<string, number>>((acc, mount) => {
    const key = mount.slot_type;
    acc[key] = (acc[key] ?? 0) + 1;
    return acc;
  }, {});
  return Object.entries(counts)
    .map(([slot, count]) => `${formatEnum(slot)} x${count}`)
    .join(", ");
}
