import type { ActionLogEffect, StatusType } from "./types/lancer";

export type IconName =
  | "send"
  | "undo2"
  | "crosshair"
  | "alertTriangle"
  | "scissors"
  | "ban"
  | "sun"
  | "anchor"
  | "snail"
  | "zap"
  | "eyeOff"
  | "ghost"
  | "arrowDown"
  | "shield"
  | "swords"
  | "power"
  | "circleDot"
  | "bomb"
  | "bot"
  | "package";

export type EffectIconConfig = {
  key: string;
  label: string;
  glyph: string;
  color: string;
  icon: IconName;
};

export const EFFECT_ICON_BY_TYPE: Record<ActionLogEffect["type"], EffectIconConfig> =
  {
    weapon_thrown: {
      key: "weapon_thrown",
      label: "Thrown",
      glyph: "T",
      color: "#f97316",
      icon: "send",
    },
    retrieve_thrown_weapon: {
      key: "retrieve_thrown_weapon",
      label: "Retrieved",
      glyph: "R",
      color: "#22c55e",
      icon: "undo2",
    },
    status_applied: {
      key: "status_applied",
      label: "Status",
      glyph: "S",
      color: "#64748b",
      icon: "circleDot",
    },
  };

export const STATUS_ICON_BY_ID: Partial<Record<StatusType, EffectIconConfig>> = {
  lock_on: {
    key: "status_lock_on",
    label: "Lock On",
    glyph: "L",
    color: "#38bdf8",
    icon: "crosshair",
  },
  impaired: {
    key: "status_impaired",
    label: "Impaired",
    glyph: "I",
    color: "#f59e0b",
    icon: "alertTriangle",
  },
  shredded: {
    key: "status_shredded",
    label: "Shredded",
    glyph: "X",
    color: "#ef4444",
    icon: "scissors",
  },
  jammed: {
    key: "status_jammed",
    label: "Jammed",
    glyph: "J",
    color: "#f43f5e",
    icon: "ban",
  },
  exposed: {
    key: "status_exposed",
    label: "Exposed",
    glyph: "E",
    color: "#fb7185",
    icon: "sun",
  },
  immobilized: {
    key: "status_immobilized",
    label: "Immobilized",
    glyph: "M",
    color: "#0ea5e9",
    icon: "anchor",
  },
  slowed: {
    key: "status_slowed",
    label: "Slowed",
    glyph: "S",
    color: "#94a3b8",
    icon: "snail",
  },
  stunned: {
    key: "status_stunned",
    label: "Stunned",
    glyph: "Z",
    color: "#facc15",
    icon: "zap",
  },
  hidden: {
    key: "status_hidden",
    label: "Hidden",
    glyph: "H",
    color: "#94a3b8",
    icon: "eyeOff",
  },
  invisible: {
    key: "status_invisible",
    label: "Invisible",
    glyph: "V",
    color: "#a78bfa",
    icon: "ghost",
  },
  prone: {
    key: "status_prone",
    label: "Prone",
    glyph: "P",
    color: "#fb923c",
    icon: "arrowDown",
  },
  braced: {
    key: "status_braced",
    label: "Braced",
    glyph: "B",
    color: "#10b981",
    icon: "shield",
  },
  engaged: {
    key: "status_engaged",
    label: "Engaged",
    glyph: "G",
    color: "#f97316",
    icon: "swords",
  },
  shutdown: {
    key: "status_shutdown",
    label: "Shutdown",
    glyph: "D",
    color: "#ef4444",
    icon: "power",
  },
};

export const DEFAULT_STATUS_ICON: EffectIconConfig = {
  key: "status_default",
  label: "Status",
  glyph: "S",
  color: "#94a3b8",
  icon: "circleDot",
};

export function resolveActionLogEffectIcon(
  effect: ActionLogEffect,
): EffectIconConfig {
  if (effect.type === "status_applied") {
    if (effect.status && STATUS_ICON_BY_ID[effect.status]) {
      return STATUS_ICON_BY_ID[effect.status]!;
    }
    return DEFAULT_STATUS_ICON;
  }
  return EFFECT_ICON_BY_TYPE[effect.type];
}

export type DeployableIconKey =
  | "mine"
  | "mine_armed"
  | "drone"
  | "deployable"
  | "other";

export const DEPLOYABLE_ICON_CONFIG: Record<DeployableIconKey, EffectIconConfig> =
  {
    mine: {
      key: "mine",
      label: "Mine",
      glyph: "M",
      color: "#ef4444",
      icon: "bomb",
    },
    mine_armed: {
      key: "mine_armed",
      label: "Armed Mine",
      glyph: "!",
      color: "#dc2626",
      icon: "alertTriangle",
    },
    drone: {
      key: "drone",
      label: "Drone",
      glyph: "D",
      color: "#3b82f6",
      icon: "bot",
    },
    deployable: {
      key: "deployable",
      label: "Deployable",
      glyph: "X",
      color: "#8b5cf6",
      icon: "package",
    },
    other: {
      key: "other",
      label: "Object",
      glyph: "O",
      color: "#64748b",
      icon: "circleDot",
    },
  };

const FALLBACK_ICON: EffectIconConfig = {
  key: "unknown",
  label: "Unknown",
  glyph: "?",
  color: "#64748b",
  icon: "circleDot",
};

export function getMarkerIconConfig(
  kind: string,
  armed?: boolean,
): EffectIconConfig {
  if (kind === "mine" && armed) {
    return DEPLOYABLE_ICON_CONFIG.mine_armed;
  }
  if (kind in DEPLOYABLE_ICON_CONFIG) {
    return DEPLOYABLE_ICON_CONFIG[kind as DeployableIconKey];
  }
  if (kind in EFFECT_ICON_BY_TYPE) {
    return EFFECT_ICON_BY_TYPE[kind as ActionLogEffect["type"]];
  }
  return FALLBACK_ICON;
}
