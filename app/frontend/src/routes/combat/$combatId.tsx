import { useEffect, useMemo, useState } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";

import { useCombatSession } from "../../lib/api";
import { CombatCanvas } from "../../components/combat/CombatCanvas";
import {
  adaptCombatScenario,
  type CombatRenderAdapterOutput,
} from "../../lib/combat-render/adapter";
import { createHexLayout } from "../../lib/combat-render/hex";
import type {
  ActionUse,
  CombatRound,
  CombatTurn,
  HexCoord,
} from "../../lib/types/lancer";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../../components/ui";

export const Route = createFileRoute("/combat/$combatId")({
  component: CombatSessionPage,
});

function CombatSessionPage() {
  const { combatId } = Route.useParams();
  const { data, isLoading, error } = useCombatSession(combatId);

  const [hovered, setHovered] = useState<HexCoord | null>(null);
  const [selected, setSelected] = useState<HexCoord | null>(null);
  const [targeted, setTargeted] = useState<HexCoord | null>(null);

  const scenario = data?.scenario;
  const rounds = scenario?.rounds ?? [];
  const defaultRoundIndex = clampIndex((data?.current_round ?? 1) - 1, rounds);

  const [selectedRoundIndex, setSelectedRoundIndex] = useState<number | null>(null);
  const [selectedTurnIndex, setSelectedTurnIndex] = useState<number | null>(null);
  const [selectedActionIndex, setSelectedActionIndex] = useState<number | null>(
    null,
  );

  const activeRoundIndex = selectedRoundIndex ?? defaultRoundIndex;
  const round = rounds[activeRoundIndex] ?? null;
  const turns = round?.turns ?? [];
  const defaultTurnIndex = clampIndex(data?.current_turn_index ?? 0, turns);
  const activeTurnIndex = selectedTurnIndex ?? defaultTurnIndex;
  const turn = turns[activeTurnIndex] ?? null;
  const actions = turn?.actions ?? [];
  const defaultActionIndex = clampIndex(0, actions);
  const activeActionIndex = selectedActionIndex ?? defaultActionIndex;
  const action = actions[activeActionIndex] ?? null;

  useEffect(() => {
    setSelectedRoundIndex((prev) => clampIndex(prev ?? defaultRoundIndex, rounds));
  }, [defaultRoundIndex, rounds]);

  useEffect(() => {
    setSelectedTurnIndex((prev) => clampIndex(prev ?? defaultTurnIndex, turns));
  }, [defaultTurnIndex, turns]);

  useEffect(() => {
    setSelectedActionIndex((prev) => clampIndex(prev ?? defaultActionIndex, actions));
  }, [defaultActionIndex, actions]);

  const combatantNameById = useMemo(
    () => new Map((scenario?.combatants ?? []).map((c) => [c.id, c.name])),
    [scenario?.combatants],
  );

  const renderOutput: CombatRenderAdapterOutput | null = useMemo(() => {
    if (!scenario) {
      return null;
    }
    return adaptCombatScenario({
      scenario,
      round,
      turn,
      action,
      hover: hovered,
    });
  }, [action, hovered, round, scenario, turn]);

  if (isLoading) {
    return (
      <div className="p-6 max-w-6xl mx-auto">
        <div className="text-center py-8 text-muted-foreground">
          Loading combat session...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 max-w-6xl mx-auto">
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <p className="text-destructive">
              Error loading combat session: {error.message}
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!data || !scenario) {
    return (
      <div className="p-6 max-w-6xl mx-auto">
        <Card>
          <CardContent className="pt-6 text-center">
            <p className="text-muted-foreground">Combat session not found</p>
            <Link to="/" className="text-primary hover:underline">
              Back to dashboard
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="px-6 py-8 max-w-7xl mx-auto space-y-6">
      <section className="dashboard-surface p-6 animate-rise">
        <Link to="/" className="text-primary hover:underline text-sm">
          ← Back to Dashboard
        </Link>
        <div className="mt-3 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h1 className="text-3xl font-heading font-semibold text-foreground">
              {data.name}
            </h1>
            <p className="text-muted-foreground">
              Status: {data.status} · Round {data.current_round}
            </p>
          </div>
          <div className="text-xs text-muted-foreground">
            Session ID: {data.id}
          </div>
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <Card className="h-full">
          <CardHeader>
            <CardTitle>Combat Canvas</CardTitle>
            <CardDescription>
              Hover for hex highlight, left click to select, right click to target.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="rounded-md border border-border bg-muted/30 p-3">
              <div className="h-[520px] w-full">
                {renderOutput ? (
                  <CombatCanvas
                    width={720}
                    height={520}
                    resizeToParent
                    layout={(size) =>
                      createHexLayout(30, {
                        x: size.width / 2,
                        y: size.height / 2,
                      })
                    }
                    state={renderOutput.state}
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
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                    No scenario data available yet.
                  </div>
                )}
              </div>
              <div className="mt-3 flex flex-wrap gap-4 text-sm text-muted-foreground">
                <div>Hover: {formatCoord(hovered)}</div>
                <div>Selected: {formatCoord(selected)}</div>
                <div>Targeted: {formatCoord(targeted)}</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Action Selection</CardTitle>
              <CardDescription>
                Choose a recorded turn/action to preview area overlays.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <SelectionField
                label="Round"
                value={activeRoundIndex}
                onChange={(value) => {
                  setSelectedRoundIndex(value);
                  setSelectedTurnIndex(null);
                  setSelectedActionIndex(null);
                }}
                options={roundOptions(rounds)}
                disabled={!rounds.length}
                emptyLabel="No rounds recorded"
              />
              <SelectionField
                label="Turn"
                value={activeTurnIndex}
                onChange={(value) => {
                  setSelectedTurnIndex(value);
                  setSelectedActionIndex(null);
                }}
                options={turnOptions(turns, combatantNameById)}
                disabled={!turns.length}
                emptyLabel="No turns recorded"
              />
              <SelectionField
                label="Action"
                value={activeActionIndex}
                onChange={(value) => setSelectedActionIndex(value)}
                options={actionOptions(actions)}
                disabled={!actions.length}
                emptyLabel="No actions recorded"
              />
              {renderOutput?.overlayMetadata.length ? (
                <div className="rounded-md border border-border bg-muted/40 p-3 text-xs text-muted-foreground space-y-2">
                  {renderOutput.overlayMetadata.map((meta) => (
                    <div key={meta.id}>
                      <div className="font-medium text-foreground">
                        {meta.pattern.toUpperCase()} size {meta.size}
                      </div>
                      <div>Origin: {formatPosition(meta.origin)}</div>
                      <div>
                        Direction:{" "}
                        {meta.direction
                          ? `${meta.direction.q},${meta.direction.r}`
                          : "--"}
                      </div>
                      <div>Coords: {meta.coordCount}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">
                  No area overlays available for this action.
                </p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Combatants</CardTitle>
              <CardDescription>
                Positions mapped from the current scenario snapshot.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              {(scenario.combatants ?? []).map((combatant) => (
                <div
                  key={combatant.id}
                  className="flex items-center justify-between rounded-md border border-border bg-muted/40 px-3 py-2"
                >
                  <div>
                    <div className="font-medium">{combatant.name}</div>
                    <div className="text-xs text-muted-foreground">
                      {combatant.side} · {combatant.kind}
                    </div>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {combatant.position?.coord
                      ? `${combatant.position.coord.q},${combatant.position.coord.r}`
                      : "--"}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

function clampIndex<T>(value: number, list: T[]): number {
  if (!list.length) {
    return 0;
  }
  if (value < 0) {
    return 0;
  }
  if (value >= list.length) {
    return list.length - 1;
  }
  return value;
}

function formatCoord(coord: HexCoord | null): string {
  if (!coord) {
    return "--";
  }
  return `${coord.q},${coord.r}`;
}

function formatPosition(position?: { coord?: HexCoord | null } | null): string {
  if (!position?.coord) {
    return "--";
  }
  return `${position.coord.q},${position.coord.r}`;
}

function roundOptions(rounds: CombatRound[]) {
  return rounds.map((round, index) => ({
    label: `Round ${round.round_index ?? index + 1}`,
    value: index,
  }));
}

function turnOptions(
  turns: CombatTurn[],
  names: Map<string, string>,
): Array<{ label: string; value: number }> {
  return turns.map((turn, index) => ({
    label: `Turn ${index + 1} · ${names.get(turn.actor_id) ?? turn.actor_id}`,
    value: index,
  }));
}

function actionOptions(
  actions: ActionUse[],
): Array<{ label: string; value: number }> {
  return actions.map((action, index) => ({
    label: `${action.action_id} (${action.action_type})`,
    value: index,
  }));
}

function SelectionField({
  label,
  value,
  onChange,
  options,
  disabled,
  emptyLabel,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  options: Array<{ label: string; value: number }>;
  disabled?: boolean;
  emptyLabel: string;
}) {
  return (
    <div className="space-y-1">
      <label className="text-sm font-medium text-foreground">{label}</label>
      <select
        className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        disabled={disabled}
      >
        {options.length === 0 ? (
          <option value={0}>{emptyLabel}</option>
        ) : (
          options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))
        )}
      </select>
    </div>
  );
}
