/**
 * Client-side loadout validation for real-time feedback.
 *
 * Mirrors validation rules from core/mech/build_validation.py for
 * immediate user feedback during editing. Server remains authoritative.
 */

import type {
  MechWeaponDefinition,
  MechSystemDefinition,
  MechFrameDefinition,
} from "../types/lancer";

// =============================================================================
// Types
// =============================================================================

export interface LoadoutValidationIssue {
  code: string;
  message: string;
  severity: "error" | "warning";
}

export interface LoadoutValidationResult {
  valid: boolean;
  issues: LoadoutValidationIssue[];
}

export interface MountedWeaponDraft {
  mount_index: number;
  weapon_id: string;
  weapon_size: string;
}

export interface InstalledSystemDraft {
  system_id: string;
  sp_cost?: number;
}

export interface BuildDraft {
  weapons: MountedWeaponDraft[];
  systems: InstalledSystemDraft[];
}

export interface LoadoutValidationContext {
  draft: BuildDraft;
  frame: MechFrameDefinition | null;
  weapons: Map<string, MechWeaponDefinition>;
  systems: Map<string, MechSystemDefinition>;
  licenses: Map<string, number>;
  spLimit: number;
  pilotLevel: number;
}

// =============================================================================
// Validation Functions
// =============================================================================

/**
 * Compute total SP spent from systems in the draft.
 */
function computeSPSpent(
  draft: BuildDraft,
  systems: Map<string, MechSystemDefinition>
): number {
  return draft.systems.reduce((total, system) => {
    const definition = systems.get(system.system_id);
    return total + (system.sp_cost ?? definition?.sp_cost ?? 0);
  }, 0);
}

/**
 * Validate that unique weapons are not duplicated.
 */
function validateUniqueWeapons(
  draft: BuildDraft,
  weapons: Map<string, MechWeaponDefinition>
): LoadoutValidationIssue[] {
  const issues: LoadoutValidationIssue[] = [];
  const weaponCounts = new Map<string, number>();

  for (const mounted of draft.weapons) {
    const count = weaponCounts.get(mounted.weapon_id) ?? 0;
    weaponCounts.set(mounted.weapon_id, count + 1);
  }

  for (const [weaponId, count] of weaponCounts) {
    if (count <= 1) continue;
    const definition = weapons.get(weaponId);
    if (definition?.unique) {
      issues.push({
        code: "unique_weapon_duplicate",
        message: `Weapon '${definition.name}' is unique and cannot be duplicated.`,
        severity: "error",
      });
    }
  }

  return issues;
}

/**
 * Validate that unique systems are not duplicated.
 */
function validateUniqueSystems(
  draft: BuildDraft,
  systems: Map<string, MechSystemDefinition>
): LoadoutValidationIssue[] {
  const issues: LoadoutValidationIssue[] = [];
  const systemCounts = new Map<string, number>();

  for (const installed of draft.systems) {
    const count = systemCounts.get(installed.system_id) ?? 0;
    systemCounts.set(installed.system_id, count + 1);
  }

  for (const [systemId, count] of systemCounts) {
    if (count <= 1) continue;
    const definition = systems.get(systemId);
    if (definition?.unique) {
      issues.push({
        code: "unique_system_duplicate",
        message: `System '${definition.name}' is unique and cannot be duplicated.`,
        severity: "error",
      });
    }
  }

  return issues;
}

/**
 * Validate AI system limit (default: 1).
 */
function validateAISystemLimit(
  draft: BuildDraft,
  systems: Map<string, MechSystemDefinition>,
  aiLimit: number = 1
): LoadoutValidationIssue[] {
  const issues: LoadoutValidationIssue[] = [];

  let aiCount = 0;
  for (const installed of draft.systems) {
    const definition = systems.get(installed.system_id);
    if (!definition?.tags) continue;
    if (definition.tags.some((tag) => tag.tag === "ai")) {
      aiCount++;
    }
  }

  if (aiCount > aiLimit) {
    issues.push({
      code: "ai_system_limit_exceeded",
      message: `AI systems installed (${aiCount}) exceed allowed limit (${aiLimit}).`,
      severity: "error",
    });
  }

  return issues;
}

/**
 * Validate LL0 GMS-only restrictions.
 * At License Level 0, only GMS (General Massive Systems) equipment is allowed.
 */
function validateLL0Restrictions(
  draft: BuildDraft,
  weapons: Map<string, MechWeaponDefinition>,
  systems: Map<string, MechSystemDefinition>
): LoadoutValidationIssue[] {
  const issues: LoadoutValidationIssue[] = [];

  for (const mounted of draft.weapons) {
    const definition = weapons.get(mounted.weapon_id);
    if (!definition) continue;
    // GMS weapons have no license_id or license_id is null/empty
    if (definition.license_id && !definition.license_id.startsWith("gms")) {
      issues.push({
        code: "ll0_non_gms_weapon",
        message: `Weapon '${definition.name}' requires a license. At LL0, only GMS weapons are allowed.`,
        severity: "error",
      });
    }
  }

  for (const installed of draft.systems) {
    const definition = systems.get(installed.system_id);
    if (!definition) continue;
    if (definition.license_id && !definition.license_id.startsWith("gms")) {
      issues.push({
        code: "ll0_non_gms_system",
        message: `System '${definition.name}' requires a license. At LL0, only GMS systems are allowed.`,
        severity: "error",
      });
    }
  }

  return issues;
}

/**
 * Validate mount allocation rules.
 */
function validateMountAllocation(
  draft: BuildDraft,
  frame: MechFrameDefinition | null
): LoadoutValidationIssue[] {
  const issues: LoadoutValidationIssue[] = [];
  if (!frame?.mounts) return issues;

  const mounts = frame.mounts;
  const weaponsByMount = new Map<number, MountedWeaponDraft[]>();

  for (const mounted of draft.weapons) {
    if (mounted.mount_index < 0 || mounted.mount_index >= mounts.length) {
      issues.push({
        code: "mount_index_out_of_range",
        message: `Mount index ${mounted.mount_index} is out of range.`,
        severity: "error",
      });
      continue;
    }
    const existing = weaponsByMount.get(mounted.mount_index) ?? [];
    existing.push(mounted);
    weaponsByMount.set(mounted.mount_index, existing);
  }

  for (const [index, mountedWeapons] of weaponsByMount) {
    const slot = mounts[index];
    const slotType = slot.slot_type;

    // Single-weapon mounts
    if (
      (slotType === "main" || slotType === "heavy" || slotType === "integrated") &&
      mountedWeapons.length > 1
    ) {
      issues.push({
        code: "too_many_weapons_on_mount",
        message: `Mount ${index + 1} allows only one weapon.`,
        severity: "error",
      });
    }

    // Aux/aux mounts
    if (slotType === "aux_aux" && mountedWeapons.length > 2) {
      issues.push({
        code: "aux_aux_capacity",
        message: `Mount ${index + 1} allows at most 2 aux weapons.`,
        severity: "error",
      });
    }

    // Main/aux mounts
    if (slotType === "main_aux" && mountedWeapons.length > 2) {
      issues.push({
        code: "main_aux_capacity",
        message: `Mount ${index + 1} allows at most 2 weapons.`,
        severity: "error",
      });
    }

    // Flexible mounts
    if (slotType === "flexible") {
      const hasMain = mountedWeapons.some((w) => w.weapon_size === "main");
      if (hasMain && mountedWeapons.length > 1) {
        issues.push({
          code: "flexible_main_capacity",
          message: `Mount ${index + 1} allows only 1 main weapon (or 2 aux).`,
          severity: "error",
        });
      }
      if (!hasMain && mountedWeapons.length > 2) {
        issues.push({
          code: "flexible_aux_capacity",
          message: `Mount ${index + 1} allows at most 2 aux weapons.`,
          severity: "error",
        });
      }
    }
  }

  return issues;
}

// =============================================================================
// Main Validation Entry Point
// =============================================================================

/**
 * Validate a mech loadout draft and return all issues.
 *
 * This is advisory validation for real-time feedback - the server
 * remains the authoritative source for final validation.
 */
export function validateLoadout(
  ctx: LoadoutValidationContext
): LoadoutValidationResult {
  const issues: LoadoutValidationIssue[] = [];

  // 1. SP Budget
  const spSpent = computeSPSpent(ctx.draft, ctx.systems);
  if (spSpent > ctx.spLimit) {
    issues.push({
      code: "system_points_exceeded",
      message: `SP ${spSpent}/${ctx.spLimit} - over budget by ${spSpent - ctx.spLimit}`,
      severity: "error",
    });
  }

  // 2. Unique weapon duplicates
  issues.push(...validateUniqueWeapons(ctx.draft, ctx.weapons));

  // 3. Unique system duplicates
  issues.push(...validateUniqueSystems(ctx.draft, ctx.systems));

  // 4. AI system limit
  issues.push(...validateAISystemLimit(ctx.draft, ctx.systems));

  // 5. LL0 GMS-only restrictions
  if (ctx.pilotLevel === 0) {
    issues.push(
      ...validateLL0Restrictions(ctx.draft, ctx.weapons, ctx.systems)
    );
  }

  // 6. Mount allocation
  issues.push(...validateMountAllocation(ctx.draft, ctx.frame));

  return {
    valid: !issues.some((i) => i.severity === "error"),
    issues,
  };
}
