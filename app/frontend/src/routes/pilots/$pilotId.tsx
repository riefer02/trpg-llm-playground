/**
 * Pilot detail page.
 *
 * Displays full pilot information with computed stats.
 */

import { createFileRoute, Link } from "@tanstack/react-router";
import { usePilot, usePilotValidation, type PilotValidationResponse } from "../../lib/api";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  Button,
} from "../../components/ui";

export const Route = createFileRoute("/pilots/$pilotId" as const)({
  component: PilotDetailPage,
});

function PilotDetailPage() {
  const { pilotId } = Route.useParams();
  const { data: pilot, isLoading, error } = usePilot(pilotId);
  const { data: validation } = usePilotValidation(pilotId);

  if (isLoading) {
    return (
      <div className="p-6 max-w-4xl mx-auto">
        <div className="text-center py-8 text-muted-foreground">
          Loading pilot...
        </div>
      </div>
    );
  }

  if (error || !pilot) {
    return (
      <div className="p-6 max-w-4xl mx-auto">
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <p className="text-destructive">
              {error?.message || "Pilot not found"}
            </p>
            <Link to="/pilots" className="mt-4 inline-block">
              <Button variant="outline">Back to Pilots</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="mb-6">
        <Link to="/pilots" className="text-primary hover:underline text-sm">
          ← Back to Pilots
        </Link>
      </div>

      <div className="flex justify-between items-start mb-6">
        <div>
          <h1 className="text-3xl font-bold text-foreground">{pilot.callsign}</h1>
          <p className="text-muted-foreground">
            {pilot.name || "Unnamed Pilot"} • License Level {pilot.level}
          </p>
        </div>
        {validation && !validation.valid && (
          <div className="px-3 py-1 bg-accent/20 border border-accent rounded-md text-accent text-sm">
            {validation.issues.length} validation issue(s)
          </div>
        )}
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Combat Stats</CardTitle>
            <CardDescription>Derived from level and skills</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <StatBlock label="Grit" value={`+${pilot.grit}`} />
              <StatBlock label="HP" value={pilot.hp} />
              <StatBlock label="Armor" value={pilot.armor} />
              <StatBlock label="Speed" value={pilot.speed} />
              <StatBlock label="Evasion" value={pilot.evasion} />
              <StatBlock label="E-Defense" value={pilot.e_defense} />
              <StatBlock label="Save Target" value={pilot.save_target} />
              <StatBlock label="Attack Bonus" value={`+${pilot.attack_bonus}`} />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Mech Skills</CardTitle>
            <CardDescription>Base stats for mech combat</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <SkillBlock label="HULL" value={pilot.skills.hull ?? 0} />
              <SkillBlock label="AGILITY" value={pilot.skills.agility ?? 0} />
              <SkillBlock label="SYSTEMS" value={pilot.skills.systems ?? 0} />
              <SkillBlock label="ENGINEERING" value={pilot.skills.engineering ?? 0} />
            </div>
          </CardContent>
        </Card>

        {pilot.triggers.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Triggers</CardTitle>
              <CardDescription>Pilot skill check bonuses</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {pilot.triggers.map((trigger, i) => (
                  <div
                    key={i}
                    className="flex justify-between items-center p-2 bg-card-foreground/5 rounded"
                  >
                    <span className="capitalize">
                      {trigger.trigger_id.replace(/_/g, " ")}
                    </span>
                    <span className="font-semibold">+{trigger.rank}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {pilot.talents.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Talents</CardTitle>
              <CardDescription>Special abilities</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {pilot.talents.map((talent, i) => (
                  <div
                    key={i}
                    className="flex justify-between items-center p-2 bg-card-foreground/5 rounded"
                  >
                    <span className="capitalize">
                      {talent.talent_id.replace(/_/g, " ")}
                    </span>
                    <span className="text-muted-foreground">Rank {talent.rank}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {pilot.licenses.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Licenses</CardTitle>
              <CardDescription>Manufacturer licenses</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {pilot.licenses.map((license, i) => (
                  <div
                    key={i}
                    className="flex justify-between items-center p-2 bg-card-foreground/5 rounded"
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

        {pilot.core_bonuses.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Core Bonuses</CardTitle>
              <CardDescription>Manufacturer rewards</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {pilot.core_bonuses.map((cb, i) => (
                  <div
                    key={i}
                    className="p-2 bg-card-foreground/5 rounded capitalize"
                  >
                    {cb.core_bonus_id.replace(/_/g, " ")}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {pilot.notes && (
          <Card className="md:col-span-2">
            <CardHeader>
              <CardTitle>Notes</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="whitespace-pre-wrap text-muted-foreground">
                {pilot.notes}
              </p>
            </CardContent>
          </Card>
        )}
      </div>

      {validation && validation.issues.length > 0 && (
        <Card className="mt-6 border-accent">
          <CardHeader>
            <CardTitle className="text-accent">Validation Issues</CardTitle>
            <CardDescription>
              This pilot has progression issues that should be resolved
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {validation.issues.map(
                (
                  issue: PilotValidationResponse["issues"][number],
                  i: number
                ) => (
                  <li
                    key={i}
                    className="flex items-start gap-2 text-sm text-muted-foreground"
                  >
                    <span className="text-accent">•</span>
                    <span>
                      <strong className="text-foreground">{issue.field}:</strong>{" "}
                      {issue.message}
                    </span>
                  </li>
                )
              )}
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function StatBlock({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="p-3 bg-card-foreground/5 rounded">
      <div className="text-muted-foreground text-xs uppercase">{label}</div>
      <div className="text-xl font-semibold">{value}</div>
    </div>
  );
}

function SkillBlock({ label, value }: { label: string; value: number }) {
  return (
    <div className="p-3 bg-card-foreground/5 rounded">
      <div className="text-muted-foreground text-xs">{label}</div>
      <div className="text-xl font-semibold">+{value}</div>
    </div>
  );
}
