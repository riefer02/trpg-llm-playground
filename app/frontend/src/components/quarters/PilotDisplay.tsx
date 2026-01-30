/**
 * Read-only pilot display component for quarters.
 * Shows pilot stats, skills, talents, triggers, licenses, and core bonuses.
 */

import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "../ui";
import { StatBlock, SkillBlock } from "../ui/stat-blocks";
import type { CharacterResponse } from "../../lib/api";

interface PilotDisplayProps {
  character: CharacterResponse;
}

export function PilotDisplay({ character }: PilotDisplayProps) {
  return (
    <div className="space-y-6">
      {/* Pilot Stats */}
      <Card>
        <CardHeader>
          <CardTitle>Pilot Stats</CardTitle>
          <CardDescription>Personal stats and abilities</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4 mb-4">
            <StatBlock label="Grit" value={`+${character.grit}`} />
            <StatBlock label="HP" value={character.pilot_hp} />
            <StatBlock
              label="Background"
              value={character.background?.name || "None"}
            />
          </div>

          <div className="text-xs text-muted-foreground uppercase tracking-wide mb-2">
            Mech Skills
          </div>
          <div className="grid grid-cols-4 gap-2">
            <SkillBlock label="HULL" value={character.skills.hull ?? 0} />
            <SkillBlock label="AGI" value={character.skills.agility ?? 0} />
            <SkillBlock label="SYS" value={character.skills.systems ?? 0} />
            <SkillBlock label="ENG" value={character.skills.engineering ?? 0} />
          </div>
        </CardContent>
      </Card>

      {/* Talents, Triggers, Licenses, Core Bonuses */}
      <div className="grid gap-6 md:grid-cols-2">
        {character.triggers.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Triggers</CardTitle>
              <CardDescription>
                Bonuses for narrative skill checks
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-3">
                {character.triggers.map((trigger, i) => (
                  <div
                    key={i}
                    className="p-3 bg-muted/50 border border-border rounded-md text-sm"
                  >
                    <div className="font-medium capitalize">
                      {trigger.trigger_id.replace(/_/g, " ")}
                    </div>
                    <div className="text-primary">+{trigger.rank}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {character.talents.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Talents</CardTitle>
              <CardDescription>
                Combat abilities and specializations
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-3">
                {character.talents.map((talent, i) => (
                  <div
                    key={i}
                    className="p-3 bg-muted/50 border border-border rounded-md text-sm"
                  >
                    <div className="font-medium capitalize">
                      {talent.talent_id.replace(/_/g, " ")}
                    </div>
                    <div className="text-muted-foreground">
                      Rank {talent.rank}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {character.licenses.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Licenses</CardTitle>
              <CardDescription>Manufacturer licenses</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {character.licenses.map((license, i) => (
                  <div
                    key={i}
                    className="flex justify-between items-center p-2 bg-muted/50 border border-border rounded"
                  >
                    <span className="capitalize">
                      {license.license_id.replace(/_/g, " ")}
                    </span>
                    <span className="text-muted-foreground">
                      Level {license.rank}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {character.core_bonuses.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Core Bonuses</CardTitle>
              <CardDescription>Manufacturer rewards</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {character.core_bonuses.map((cb, i) => (
                  <div
                    key={i}
                    className="p-2 bg-muted/50 border border-border rounded capitalize"
                  >
                    {cb.core_bonus_id.replace(/_/g, " ")}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {character.notes && (
        <Card>
          <CardHeader>
            <CardTitle>Notes</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="whitespace-pre-wrap text-muted-foreground">
              {character.notes}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}