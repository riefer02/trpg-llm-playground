/**
 * Home page with system health status.
 *
 * Demonstrates:
 * - React Query for data fetching
 * - Custom UI components
 * - API integration
 */

import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { useHealth, useDatabaseHealth } from "../lib/api";
import { CombatCanvas } from "../components/combat/CombatCanvas";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "../components/ui";
import { adaptCombatScenario } from "../lib/combat-render/adapter";
import { createHexLayout } from "../lib/combat-render/hex";
import type {
  ActionUse,
  CombatRound,
  CombatTurn,
  CombatantState,
  HexCoord,
  HexPosition,
  MechCombatScenario,
} from "../lib/types/lancer";

export const Route = createFileRoute("/" as const)({
  component: HomePage,
});

function HomePage() {
  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2 text-foreground">
          Welcome to Lancer Combat
        </h1>
        <p className="text-muted-foreground">
          A web application for running Lancer TTRPG tactical combat encounters.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <SystemHealthCard />
        <DatabaseHealthCard />
      </div>

      <div className="mt-8">
        <h2 className="text-xl font-semibold mb-4 text-foreground">
          Quick Actions
        </h2>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <QuickActionCard
            title="Create Character"
            description="Build a new pilot and mech"
            href="/characters/new"
          />
          <QuickActionCard
            title="My Characters"
            description="View and manage characters"
            href="/characters"
          />
          <QuickActionCard
            title="Compendium"
            description="Browse frames, weapons, systems, and gear"
            href="/compendium"
          />
          <QuickActionCard
            title="Campaigns"
            description="Create lobbies and invite players"
            href="/campaigns"
          />
        </div>
      </div>

      <div className="mt-10">
        <h2 className="text-xl font-semibold mb-4 text-foreground">
          Combat Canvas Preview
        </h2>
        <CombatPreview />
      </div>
    </div>
  );
}

function SystemHealthCard() {
  const { data, isLoading, error } = useHealth();

  return (
    <Card>
      <CardHeader>
        <CardTitle>System Status</CardTitle>
        <CardDescription>API server health</CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading && <div className="text-muted-foreground">Checking...</div>}
        {error && (
          <div className="flex items-center gap-2">
            <StatusDot status="error" />
            <span className="text-destructive">Offline - {error.message}</span>
          </div>
        )}
        {data && (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <StatusDot
                status={data.status === "healthy" ? "success" : "error"}
              />
              <span>API: {data.status}</span>
            </div>
            <div className="text-muted-foreground">Version: {data.version}</div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function DatabaseHealthCard() {
  const { data, isLoading, error } = useDatabaseHealth();

  return (
    <Card>
      <CardHeader>
        <CardTitle>Database Status</CardTitle>
        <CardDescription>PostgreSQL connection</CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading && <div className="text-muted-foreground">Checking...</div>}
        {error && (
          <div className="flex items-center gap-2">
            <StatusDot status="error" />
            <span className="text-destructive">Connection failed</span>
          </div>
        )}
        {data && (
          <div className="flex items-center gap-2">
            <StatusDot
              status={data.status === "healthy" ? "success" : "error"}
            />
            <span>
              {data.status === "healthy" ? "Connected" : data.database}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function StatusDot({ status }: { status: "success" | "error" | "warning" }) {
  const colorClass = {
    success: "bg-horus",
    error: "bg-destructive",
    warning: "bg-accent",
  }[status];

  return <div className={`w-3 h-3 rounded-full ${colorClass}`} />;
}

function QuickActionCard({
  title,
  description,
  href,
}: {
  title: string;
  description: string;
  href: string;
}) {
  return (
    <a href={href} className="block">
      <Card className="h-full transition-colors hover:border-primary cursor-pointer">
        <CardHeader>
          <CardTitle className="text-lg">{title}</CardTitle>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
      </Card>
    </a>
  );
}

function CombatPreview() {
  const [hovered, setHovered] = useState<HexCoord | null>(null);
  const [selected, setSelected] = useState<HexCoord | null>(null);
  const [targeted, setTargeted] = useState<HexCoord | null>(null);

  const previewData = useMemo(() => buildPreviewScenario(), []);
  const { scenario, turn, action } = previewData;

  const { state } = useMemo(
    () =>
      adaptCombatScenario({
        scenario,
        turn,
        action,
        hover: hovered,
        gridRadius: 4,
      }),
    [scenario, turn, action, hovered],
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle>Smoke Test</CardTitle>
        <CardDescription>
          Hover for hex highlight, left click to select, right click to target.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="rounded-md border border-border bg-muted/30 p-3">
          <div className="h-[420px] w-full">
            <CombatCanvas
              width={640}
              height={420}
              resizeToParent
              layout={(size) =>
                createHexLayout(30, {
                  x: size.width / 2,
                  y: size.height / 2,
                })
              }
              state={state}
              styles={{
                grid: { strokeStyle: "rgba(148, 163, 184, 0.5)" },
                tokens: { strokeStyle: "#0f172a", lineWidth: 2 },
                overlays: { fillStyle: "rgba(59, 130, 246, 0.12)" },
                hover: {
                  fillStyle: "rgba(59, 130, 246, 0.2)",
                  strokeStyle: "rgba(59, 130, 246, 0.7)",
                  lineWidth: 2,
                },
              }}
              onHover={(coord) => setHovered(coord)}
              onSelect={(coord) => setSelected(coord)}
              onTarget={(coord) => setTargeted(coord)}
              className="h-full w-full"
            />
          </div>
          <div className="mt-3 flex flex-wrap gap-4 text-sm text-muted-foreground">
            <div>Hover: {formatCoord(hovered)}</div>
            <div>Selected: {formatCoord(selected)}</div>
            <div>Targeted: {formatCoord(targeted)}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function formatCoord(coord: HexCoord | null): string {
  if (!coord) {
    return "--";
  }
  return `${coord.q},${coord.r}`;
}

function buildPreviewScenario(): {
  scenario: MechCombatScenario;
  round: CombatRound;
  turn: CombatTurn;
  action: ActionUse;
} {
  const combatants: CombatantState[] = [
    makeCombatant("alpha", "Atlas", "players", 0, 0),
    makeCombatant("bravo", "Banshee", "hostiles", 2, -1),
    makeCombatant("charlie", "Comet", "neutral", -2, 1),
  ];

  const action: ActionUse = {
    action_id: "preview-cone",
    action_type: "quick",
    area_pattern: { pattern: "cone", size: 2, cone_mode: "axis" },
    area_direction: makeCoord(1, 0),
    target_id: "bravo",
  };

  const turn: CombatTurn = {
    actor_id: "alpha",
    actions: [action],
  };

  const round: CombatRound = {
    round_index: 1,
    turns: [turn],
  };

  const scenario: MechCombatScenario = {
    combatants,
    rounds: [round],
  };

  return { scenario, round, turn, action };
}

function makeCombatant(
  id: string,
  name: string,
  side: CombatantState["side"],
  q: number,
  r: number,
): CombatantState {
  return {
    id,
    name,
    side,
    kind: "mech",
    stats: {
      size: "size_1",
      hp_max: 10,
      evasion: 10,
      e_defense: 10,
    },
    resources: {
      hp_current: 10,
    },
    position: makePosition(q, r),
  };
}

function makeCoord(q: number, r: number): HexCoord {
  return { q, r, s: -q - r };
}

function makePosition(q: number, r: number): HexPosition {
  return { coord: makeCoord(q, r) };
}
