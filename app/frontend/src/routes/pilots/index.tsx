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

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Pilots</h1>
          <p className="text-muted-foreground">
            Manage your pilot characters
          </p>
        </div>
        <Link to="/pilots/new">
          <Button>Create Pilot</Button>
        </Link>
      </div>

      {isLoading && (
        <div className="text-center py-8 text-muted-foreground">
          Loading pilots...
        </div>
      )}

      {error && (
        <Card className="border-destructive">
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
              {pilot.name || "Unnamed"} • LL{pilot.level}
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
