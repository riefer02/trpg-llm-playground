/**
 * Characters list page.
 *
 * Displays all characters (unified pilot + mech) with options to create, view, and delete.
 */

import { createFileRoute, Link } from "@tanstack/react-router";
import {
  useCharacters,
  useDeleteCharacter,
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

export const Route = createFileRoute("/characters/" as const)({
  component: CharactersPage,
});

function CharactersPage() {
  const { data, isLoading, error } = useCharacters();
  const totalCharacters = data?.items.length ?? 0;
  const activeMechs =
    data?.items.filter((character) => character.active_mech_id).length ?? 0;
  const averageLevel =
    totalCharacters > 0
      ? Math.round(
          data!.items.reduce((sum, character) => sum + character.level, 0) /
            totalCharacters
        )
      : 0;

  return (
    <div className="px-6 py-8 max-w-7xl mx-auto space-y-6">
      <section className="dashboard-surface p-6 animate-rise">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h1 className="text-3xl font-heading font-semibold text-foreground">
              Characters
            </h1>
            <p className="text-muted-foreground">
              Monitor pilots, mechs, and current loadouts in one view.
            </p>
          </div>
          <Link to="/characters/new">
            <Button>Create Character</Button>
          </Link>
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <div className="rounded-lg border border-border bg-muted/40 p-3">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Total characters
            </div>
            <div className="text-lg font-semibold">{totalCharacters}</div>
          </div>
          <div className="rounded-lg border border-border bg-muted/40 p-3">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Active mechs
            </div>
            <div className="text-lg font-semibold">{activeMechs}</div>
          </div>
          <div className="rounded-lg border border-border bg-muted/40 p-3">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Average license level
            </div>
            <div className="text-lg font-semibold">LL{averageLevel}</div>
          </div>
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="space-y-4">
          {isLoading && (
            <div className="text-center py-8 text-muted-foreground">
              Loading characters...
            </div>
          )}

          {error && (
            <Card className="border-destructive/40">
              <CardContent className="pt-6">
                <p className="text-destructive">
                  Error loading characters: {error.message}
                </p>
              </CardContent>
            </Card>
          )}

          {data && data.items.length === 0 && (
            <Card>
              <CardContent className="pt-6 text-center">
                <p className="text-muted-foreground mb-4">
                  No characters yet. Create your first character to get started.
                </p>
                <Link to="/characters/new">
                  <Button>Create Character</Button>
                </Link>
              </CardContent>
            </Card>
          )}

          {data && data.items.length > 0 && (
            <div className="grid gap-4">
              {data.items.map((character) => (
                <CharacterCard key={character.id} character={character} />
              ))}
            </div>
          )}
        </div>

        <aside className="space-y-4 lg:sticky lg:top-6 h-fit">
          <Card>
            <CardHeader>
              <CardTitle>Ops Brief</CardTitle>
              <CardDescription>Character management tips</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-muted-foreground">
              <p>Open a character to edit mech loadout or pilot gear.</p>
              <p>Validation warnings come from core rules checks.</p>
              <p>Export PDFs for quick table references.</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Quick Links</CardTitle>
              <CardDescription>Reference and creation tools</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <Link to="/compendium" className="text-primary hover:underline">
                Browse Compendium
              </Link>
              <Link to="/characters/new" className="text-primary hover:underline">
                Start New Character
              </Link>
            </CardContent>
          </Card>
        </aside>
      </div>
    </div>
  );
}

function CharacterCard({ character }: { character: CharacterResponse }) {
  const deleteMutation = useDeleteCharacter();

  const handleDelete = () => {
    if (confirm(`Delete character "${character.callsign}"?`)) {
      deleteMutation.mutate(character.id);
    }
  };

  const activeMech = character.mechs.find(
    (m) => m.id === character.active_mech_id
  );
  const mechStats = character.active_mech_stats;

  return (
    <Card className="hover:border-primary/50 transition-colors">
      <CardHeader>
        <div className="flex justify-between items-start">
          <div>
            <CardTitle className="text-xl">{character.callsign}</CardTitle>
            <CardDescription>
              {character.name || "Unnamed"} - LL{character.level}
              {activeMech && ` - ${activeMech.name}`}
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Link
              to="/characters/$characterId"
              params={{ characterId: character.id }}
            >
              <Button variant="outline" size="sm">
                View
              </Button>
            </Link>
            <Button
              variant="outline"
              size="sm"
              onClick={handleDelete}
              disabled={deleteMutation.isPending}
              className="text-destructive hover:bg-destructive/10"
            >
              {deleteMutation.isPending ? "..." : "Delete"}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {/* Pilot Stats */}
        <div className="mb-3 text-xs text-muted-foreground uppercase tracking-wide">
          Pilot
        </div>
        <div className="grid grid-cols-4 gap-4 text-sm mb-4">
          <StatBlock label="Grit" value={`+${character.grit}`} />
          <StatBlock label="HP" value={character.pilot_hp} />
          <div className="col-span-2 grid grid-cols-4 gap-2 text-muted-foreground">
            <div>HULL +{character.skills.hull ?? 0}</div>
            <div>AGI +{character.skills.agility ?? 0}</div>
            <div>SYS +{character.skills.systems ?? 0}</div>
            <div>ENG +{character.skills.engineering ?? 0}</div>
          </div>
        </div>

        {/* Mech Stats */}
        {mechStats && (
          <>
            <div className="mb-3 text-xs text-muted-foreground uppercase tracking-wide">
              Mech ({activeMech?.name})
            </div>
            <div className="grid grid-cols-5 gap-4 text-sm">
              <StatBlock label="HP" value={mechStats.hp} />
              <StatBlock label="Evasion" value={mechStats.evasion} />
              <StatBlock label="E-Def" value={mechStats.e_defense} />
              <StatBlock label="Speed" value={mechStats.speed} />
              <StatBlock label="Heat Cap" value={mechStats.heat_cap} />
            </div>
          </>
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
      <div className="font-semibold">{value}</div>
    </div>
  );
}
