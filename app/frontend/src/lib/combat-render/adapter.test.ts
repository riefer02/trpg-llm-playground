import { describe, expect, it } from "vitest";

import type {
  ActionUse,
  AttackPatternDefinition,
  CombatantState,
  HexCoord,
  HexPosition,
  MechCombatScenario,
} from "../types/lancer";

import {
  adaptCombatScenario,
  type CombatRenderAdapterInput,
} from "./adapter";
import {
  hexCone,
  hexConeCentered,
  hexesInRadius,
  hexLineFromDirection,
} from "./aoe";

function coord(q: number, r: number): HexCoord {
  return { q, r, s: -q - r };
}

function position(q: number, r: number): HexPosition {
  return { coord: coord(q, r) };
}

function combatant(
  id: string,
  name: string,
  q: number,
  r: number,
): CombatantState {
  return {
    id,
    name,
    side: "players",
    kind: "mech",
    stats: {
      size: "size_1",
      hp_max: 10,
      evasion: 8,
      e_defense: 8,
    },
    resources: {
      hp_current: 10,
    },
    position: position(q, r),
  };
}

function scenarioWith(actor: CombatantState): MechCombatScenario {
  return { combatants: [actor] };
}

function makeInput(
  action: ActionUse,
  actor: CombatantState,
  overrides: Partial<CombatRenderAdapterInput> = {},
): CombatRenderAdapterInput {
  return {
    scenario: scenarioWith(actor),
    action,
    actorId: actor.id,
    ...overrides,
  };
}

describe("combat-render adapter", () => {
  it("maps line overlays from action patterns", () => {
    const actor = combatant("alpha", "Alpha", 0, 0);
    const action: ActionUse = {
      action_id: "line-shot",
      action_type: "quick",
      area_pattern: { pattern: "line", size: 3 },
      area_direction: coord(1, 0),
    };

    const output = adaptCombatScenario(makeInput(action, actor));
    const overlay = output.state.overlays?.[0];

    expect(overlay?.coords).toEqual(
      hexLineFromDirection(coord(0, 0), coord(1, 0), 3),
    );
    expect(output.overlayMetadata[0]?.pattern).toBe("line");
  });

  it("uses wedge and axis cone modes", () => {
    const actor = combatant("alpha", "Alpha", 0, 0);
    const basePattern: AttackPatternDefinition = { pattern: "cone", size: 2 };

    const wedgeAction: ActionUse = {
      action_id: "cone-wedge",
      action_type: "quick",
      area_pattern: basePattern,
      area_direction: coord(1, 0),
    };
    const axisAction: ActionUse = {
      action_id: "cone-axis",
      action_type: "quick",
      area_pattern: { ...basePattern, cone_mode: "axis" },
      area_direction: coord(1, 0),
    };

    const wedge = adaptCombatScenario(makeInput(wedgeAction, actor));
    const axis = adaptCombatScenario(makeInput(axisAction, actor));

    expect(wedge.state.overlays?.[0]?.coords).toEqual(
      hexCone(coord(0, 0), coord(1, 0), 2),
    );
    expect(axis.state.overlays?.[0]?.coords).toEqual(
      hexConeCentered(coord(0, 0), coord(1, 0), 2),
    );
  });

  it("uses target origin for blast and actor origin for burst", () => {
    const actor = combatant("alpha", "Alpha", 0, 0);

    const blastAction: ActionUse = {
      action_id: "blast",
      action_type: "quick",
      area_pattern: { pattern: "blast", size: 1 },
      target_position: position(2, -1),
    };
    const burstAction: ActionUse = {
      action_id: "burst",
      action_type: "quick",
      area_pattern: { pattern: "burst", size: 1 },
    };

    const blast = adaptCombatScenario(makeInput(blastAction, actor));
    const burst = adaptCombatScenario(makeInput(burstAction, actor));

    expect(blast.state.overlays?.[0]?.coords).toEqual(
      hexesInRadius(coord(2, -1), 1),
    );
    expect(burst.state.overlays?.[0]?.coords).toEqual(
      hexesInRadius(coord(0, 0), 1),
    );
  });

  it("skips overlays when direction is invalid", () => {
    const actor = combatant("alpha", "Alpha", 0, 0);
    const action: ActionUse = {
      action_id: "bad-line",
      action_type: "quick",
      area_pattern: { pattern: "line", size: 3 },
      area_direction: coord(1, 1),
    };

    const output = adaptCombatScenario(makeInput(action, actor));
    expect(output.state.overlays ?? []).toHaveLength(0);
  });

  it("skips overlays when origin is missing", () => {
    const actor = combatant("alpha", "Alpha", 0, 0);
    actor.position = null;
    const action: ActionUse = {
      action_id: "burst",
      action_type: "quick",
      area_pattern: { pattern: "burst", size: 1 },
    };

    const output = adaptCombatScenario(makeInput(action, actor));
    expect(output.state.overlays ?? []).toHaveLength(0);
  });
});
