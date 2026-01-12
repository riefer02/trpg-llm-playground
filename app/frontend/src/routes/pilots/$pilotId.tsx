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

  const validationIssues = validation?.issues ?? [];
  const issueCount = validationIssues.length;

  return (
    <div className="px-6 py-8 max-w-7xl mx-auto space-y-6">
      <section className="dashboard-surface p-6 animate-rise">
        <Link to="/pilots" className="text-primary hover:underline text-sm">
          ← Back to Pilots
        </Link>
        <div className="mt-3 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h1 className="text-3xl font-heading font-semibold text-foreground">
              {pilot.callsign}
            </h1>
            <p className="text-muted-foreground">
              {pilot.name || "Unnamed Pilot"} - License Level {pilot.level}
            </p>
          </div>
          <div
            className={`rounded-full border px-4 py-1 text-xs ${
              issueCount > 0
                ? "border-accent/40 bg-accent/10 text-accent"
                : "border-border text-muted-foreground"
            }`}
          >
            {issueCount > 0 ? `${issueCount} validation issue(s)` : "Validated"}
          </div>
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <div className="rounded-lg border border-border bg-muted/40 p-3">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Grit / HP
            </div>
            <div className="text-lg font-semibold">
              +{pilot.grit} / {pilot.hp} HP
            </div>
          </div>
          <div className="rounded-lg border border-border bg-muted/40 p-3">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Evasion / E-Def
            </div>
            <div className="text-lg font-semibold">
              {pilot.evasion} / {pilot.e_defense}
            </div>
          </div>
          <div className="rounded-lg border border-border bg-muted/40 p-3">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Attack / Save
            </div>
            <div className="text-lg font-semibold">
              +{pilot.attack_bonus} / {pilot.save_target}
            </div>
          </div>
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="space-y-6">
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
                  <SkillBlock
                    label="ENGINEERING"
                    value={pilot.skills.engineering ?? 0}
                  />
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
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
                        className="flex justify-between items-center p-2 bg-muted/50 border border-border rounded"
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
                        className="flex justify-between items-center p-2 bg-muted/50 border border-border rounded"
                      >
                        <span className="capitalize">
                          {talent.talent_id.replace(/_/g, " ")}
                        </span>
                        <span className="text-muted-foreground">
                          Rank {talent.rank}
                        </span>
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

          {pilot.notes && (
            <Card>
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

        <aside className="space-y-4 lg:sticky lg:top-6 h-fit">
          <Card>
            <CardHeader>
              <CardTitle>Validation Feed</CardTitle>
              <CardDescription>Core progression checks</CardDescription>
            </CardHeader>
            <CardContent>
              {validationIssues.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No validation issues detected.
                </p>
              ) : (
                <ul className="space-y-2">
                  {validationIssues.map(
                    (
                      issue: PilotValidationResponse["issues"][number],
                      i: number
                    ) => (
                      <li
                        key={i}
                        className="rounded-md border border-accent/40 bg-accent/10 px-3 py-2 text-sm text-accent"
                      >
                        <strong className="text-foreground">{issue.field}:</strong>{" "}
                        {issue.message}
                      </li>
                    )
                  )}
                </ul>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Field Guide</CardTitle>
              <CardDescription>Progression reminders</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-muted-foreground">
              <p>License ranks unlock frames, weapons, and systems.</p>
              <p>Core bonuses arrive every 3 license levels per manufacturer.</p>
              <p>Use the compendium to plan loadouts.</p>
            </CardContent>
          </Card>
        </aside>
      </div>
    </div>
  );
}

function StatBlock({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="p-3 bg-muted/50 border border-border rounded">
      <div className="text-muted-foreground text-xs uppercase">{label}</div>
      <div className="text-xl font-semibold">{value}</div>
    </div>
  );
}

function SkillBlock({ label, value }: { label: string; value: number }) {
  return (
    <div className="p-3 bg-muted/50 border border-border rounded">
      <div className="text-muted-foreground text-xs">{label}</div>
      <div className="text-xl font-semibold">+{value}</div>
    </div>
  );
}
