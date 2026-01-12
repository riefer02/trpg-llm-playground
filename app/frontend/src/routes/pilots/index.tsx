/**
 * Pilots list page.
 *
 * Displays all pilots with options to create, edit, and delete.
 */

import { createFileRoute, Link } from "@tanstack/react-router";
import { usePilots, useDeletePilot, type PilotResponse } from "../../lib/api";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  Button,
} from "../../components/ui";

export const Route = createFileRoute("/pilots/" as const)({
  component: PilotsPage,
});

function PilotsPage() {
  const { data, isLoading, error } = usePilots();
  const totalPilots = data?.items.length ?? 0;
  const averageLevel =
    totalPilots > 0
      ? Math.round(
          data!.items.reduce((sum, pilot) => sum + pilot.level, 0) / totalPilots
        )
      : 0;

  return (
    <div className="px-6 py-8 max-w-7xl mx-auto space-y-6">
      <section className="dashboard-surface p-6 animate-rise">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h1 className="text-3xl font-heading font-semibold text-foreground">
              Pilots
            </h1>
            <p className="text-muted-foreground">
              Manage pilot progression, licenses, and narrative stats.
            </p>
          </div>
          <Link to="/pilots/new">
            <Button>Create Pilot</Button>
          </Link>
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <div className="rounded-lg border border-border bg-muted/40 p-3">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Total pilots
            </div>
            <div className="text-lg font-semibold">{totalPilots}</div>
          </div>
          <div className="rounded-lg border border-border bg-muted/40 p-3">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Average license level
            </div>
            <div className="text-lg font-semibold">LL{averageLevel}</div>
          </div>
          <div className="rounded-lg border border-border bg-muted/40 p-3">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Active focus
            </div>
            <div className="text-lg font-semibold">Progression</div>
          </div>
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="space-y-4">
          {isLoading && (
            <div className="text-center py-8 text-muted-foreground">
              Loading pilots...
            </div>
          )}

          {error && (
            <Card className="border-destructive/40">
              <CardContent className="pt-6">
                <p className="text-destructive">
                  Error loading pilots: {error.message}
                </p>
              </CardContent>
            </Card>
          )}

          {data && data.items.length === 0 && (
            <Card>
              <CardContent className="pt-6 text-center">
                <p className="text-muted-foreground mb-4">
                  No pilots yet. Create your first pilot to get started.
                </p>
                <Link to="/pilots/new">
                  <Button>Create Pilot</Button>
                </Link>
              </CardContent>
            </Card>
          )}

          {data && data.items.length > 0 && (
            <div className="grid gap-4">
              {data.items.map((pilot) => (
                <PilotCard key={pilot.id} pilot={pilot} />
              ))}
            </div>
          )}
        </div>

        <aside className="space-y-4 lg:sticky lg:top-6 h-fit">
          <Card>
            <CardHeader>
              <CardTitle>Pilot Guidance</CardTitle>
              <CardDescription>Progression rules at a glance</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-muted-foreground">
              <p>LL0 pilots start with 2 HASE points and 3 talents.</p>
              <p>License ranks unlock mech gear in the compendium.</p>
              <p>Use pilot validation to catch missing points.</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Quick Links</CardTitle>
              <CardDescription>Jump to related tools</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <Link to="/compendium" className="text-primary hover:underline">
                Browse Compendium
              </Link>
              <Link to="/characters" className="text-primary hover:underline">
                View Characters
              </Link>
            </CardContent>
          </Card>
        </aside>
      </div>
    </div>
  );
}

function PilotCard({ pilot }: { pilot: PilotResponse }) {
  const deleteMutation = useDeletePilot();

  const handleDelete = () => {
    if (confirm(`Delete pilot "${pilot.callsign}"?`)) {
      deleteMutation.mutate(pilot.id);
    }
  };

  return (
    <Card className="hover:border-primary/50 transition-colors">
      <CardHeader>
        <div className="flex justify-between items-start">
          <div>
            <CardTitle className="text-xl">{pilot.callsign}</CardTitle>
            <CardDescription>
              {pilot.name || "Unnamed"} - LL{pilot.level}
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Link to="/pilots/$pilotId" params={{ pilotId: pilot.id }}>
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
        <div className="grid grid-cols-4 gap-4 text-sm">
          <StatBlock label="Grit" value={`+${pilot.grit}`} />
          <StatBlock label="HP" value={pilot.hp} />
          <StatBlock label="Evasion" value={pilot.evasion} />
          <StatBlock label="E-Defense" value={pilot.e_defense} />
        </div>
        <div className="mt-3 grid grid-cols-4 gap-4 text-sm text-muted-foreground">
          <div>HULL +{pilot.skills.hull ?? 0}</div>
          <div>AGI +{pilot.skills.agility ?? 0}</div>
          <div>SYS +{pilot.skills.systems ?? 0}</div>
          <div>ENG +{pilot.skills.engineering ?? 0}</div>
        </div>
      </CardContent>
    </Card>
  );
}

function StatBlock({ label, value }: { label: string; value: number | string }) {
  return (
    <div>
      <div className="text-muted-foreground text-xs uppercase">{label}</div>
      <div className="font-semibold">{value}</div>
    </div>
  );
}
