/**
 * Frame preview panel for character creation.
 *
 * Shows detailed frame info including stats, traits, and core system.
 */

import type { MechFrameDefinition } from "../../lib/types/lancer";
import { LicenseBadge } from "../ui/LicenseBadge";

interface FramePreviewProps {
  frame: MechFrameDefinition;
}

export function FramePreview({ frame }: FramePreviewProps) {
  const stats = frame.base_stats;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <span className="text-lg font-semibold">{frame.name}</span>
        <LicenseBadge licenseId={frame.license_id ?? null} />
      </div>

      <div className="text-sm text-muted-foreground">
        {frame.manufacturer}
        {frame.license_id && frame.license_rank && (
          <span> - {frame.license_id} {frame.license_rank}</span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-2 text-sm">
        <StatRow label="Size" value={stats.size} />
        <StatRow label="HP" value={stats.hp ?? 10} />
        <StatRow label="Armor" value={stats.armor ?? 0} />
        <StatRow label="Evasion" value={stats.evasion ?? 8} />
        <StatRow label="E-Defense" value={stats.e_defense ?? 8} />
        <StatRow label="Speed" value={stats.speed ?? 4} />
        <StatRow label="Sensor Range" value={stats.sensor_range ?? 10} />
        <StatRow label="Tech Attack" value={stats.tech_attack ?? 0} signed />
        <StatRow label="Heat Cap" value={stats.heat_cap ?? 6} />
        <StatRow label="Repair Cap" value={stats.repair_cap ?? 4} />
        <StatRow label="Save Target" value={stats.save_target ?? 10} />
        <StatRow label="SP" value={frame.system_points ?? 6} />
      </div>

      {frame.mounts && frame.mounts.length > 0 && (
        <div>
          <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">
            Mounts
          </div>
          <div className="flex flex-wrap gap-1">
            {frame.mounts.map((mount, i) => (
              <span
                key={i}
                className="px-2 py-0.5 text-xs rounded bg-muted border border-border"
              >
                {String(mount.mount_type)}
              </span>
            ))}
          </div>
        </div>
      )}

      {frame.traits && frame.traits.length > 0 && (
        <div>
          <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">
            Frame Traits
          </div>
          <div className="space-y-2">
            {frame.traits.map((trait, i) => (
              <div key={i} className="p-2 rounded bg-muted/50 border border-border">
                <div className="font-medium text-sm">{trait.name}</div>
                {trait.effects && (
                  <div className="text-xs text-muted-foreground mt-1">
                    {formatEffects(trait.effects)}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {frame.core_system && (
        <div>
          <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">
            CORE System
          </div>
          <div className="p-3 rounded bg-primary/10 border border-primary/30">
            <div className="font-semibold text-sm">
              {String(frame.core_system.name)}
            </div>
            {Boolean(
              "passive" in frame.core_system && frame.core_system.passive
            ) && (
              <div className="mt-2">
                <span className="text-xs text-primary font-medium">Passive: </span>
                <span className="text-xs text-muted-foreground">
                  {String(formatEffects(frame.core_system.passive))}
                </span>
              </div>
            )}
            {Boolean(
              "active" in frame.core_system && frame.core_system.active
            ) && (
              <div className="mt-2">
                <span className="text-xs text-primary font-medium">Active: </span>
                <span className="text-xs text-muted-foreground">
                  {String(formatEffects(frame.core_system.active))}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function StatRow({
  label,
  value,
  signed = false,
}: {
  label: string;
  value: number | string;
  signed?: boolean;
}) {
  const displayValue =
    signed && typeof value === "number" && value >= 0 ? `+${value}` : value;

  return (
    <div className="flex justify-between">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium">{displayValue}</span>
    </div>
  );
}

function formatEffects(effects: unknown): string {
  if (typeof effects === "string") return effects;
  if (Array.isArray(effects)) return effects.join(", ");
  if (typeof effects === "object" && effects !== null) {
    const e = effects as Record<string, unknown>;
    const parts: string[] = [];
    if (e.description) parts.push(String(e.description));
    if (e.stat_modifiers) {
      const mods = e.stat_modifiers as Record<string, number>;
      for (const [stat, val] of Object.entries(mods)) {
        parts.push(`${stat}: ${val >= 0 ? "+" : ""}${val}`);
      }
    }
    return parts.join("; ") || "See rules";
  }
  return "See rules";
}
