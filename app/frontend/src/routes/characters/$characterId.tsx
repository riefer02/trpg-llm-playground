/**
 * Character detail page.
 *
 * View and edit a character with their pilot stats and mech configurations.
 */

import { createFileRoute, Link } from "@tanstack/react-router";
import {
  useCharacter,
  useCharacterValidation,
  type CharacterResponse,
} from "../../lib/api";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  Button,
} from "../../components/ui";

export const Route = createFileRoute("/characters/$characterId" as const)({
  component: CharacterDetailPage,
});

function CharacterDetailPage() {
  const { characterId } = Route.useParams();
  const { data: character, isLoading, error } = useCharacter(characterId);
  const { data: validation } = useCharacterValidation(characterId);

  if (isLoading) {
    return (
      <div className="p-6 max-w-4xl mx-auto">
        <div className="text-center py-8 text-muted-foreground">
          Loading character...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 max-w-4xl mx-auto">
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <p className="text-destructive">
              Error loading character: {error.message}
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!character) {
    return (
      <div className="p-6 max-w-4xl mx-auto">
        <Card>
          <CardContent className="pt-6 text-center">
            <p className="text-muted-foreground">Character not found</p>
            <Link to="/characters" className="text-primary hover:underline">
              Back to Characters
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="mb-6">
        <Link
          to="/characters"
          className="text-primary hover:underline text-sm"
        >
          ← Back to Characters
        </Link>
        <div className="flex justify-between items-start mt-2">
          <div>
            <h1 className="text-3xl font-bold text-foreground">
              {character.callsign}
            </h1>
            <p className="text-muted-foreground">
              {character.name || "Unnamed"} • License Level {character.level}
            </p>
          </div>
          <ValidationBadge validation={validation} />
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <PilotSection character={character} />
        <MechSection character={character} />
      </div>

      {validation && !validation.valid && (
        <Card className="mt-6 border-destructive">
          <CardHeader>
            <CardTitle className="text-destructive">
              Validation Issues
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {validation.issues.map((issue, i) => (
                <li
                  key={i}
                  className={
                    issue.severity === "warning"
                      ? "text-accent"
                      : "text-destructive"
                  }
                >
                  <span className="font-mono text-xs mr-2">[{issue.code}]</span>
                  {issue.message}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {character.triggers.length > 0 && (
        <Card className="mt-6">
          <CardHeader>
            <CardTitle>Triggers</CardTitle>
            <CardDescription>
              Bonuses for narrative skill checks
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {character.triggers.map((trigger, i) => (
                <div
                  key={i}
                  className="p-3 bg-card-foreground/5 rounded-md text-sm"
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
        <Card className="mt-6">
          <CardHeader>
            <CardTitle>Talents</CardTitle>
            <CardDescription>Combat abilities and specializations</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-3">
              {character.talents.map((talent, i) => (
                <div
                  key={i}
                  className="p-3 bg-card-foreground/5 rounded-md text-sm"
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

      {character.notes && (
        <Card className="mt-6">
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

function ValidationBadge({
  validation,
}: {
  validation?: { valid: boolean; issues: Array<{ severity: string }> };
}) {
  if (!validation) return null;

  const warnings = validation.issues.filter(
    (i) => i.severity === "warning"
  ).length;
  const errors = validation.issues.filter(
    (i) => i.severity !== "warning"
  ).length;

  if (validation.valid && warnings === 0) {
    return (
      <div className="px-3 py-1 bg-horus/20 text-horus rounded-full text-sm font-medium">
        Valid
      </div>
    );
  }

  if (errors > 0) {
    return (
      <div className="px-3 py-1 bg-destructive/20 text-destructive rounded-full text-sm font-medium">
        {errors} Error{errors > 1 ? "s" : ""}
      </div>
    );
  }

  return (
    <div className="px-3 py-1 bg-accent/20 text-accent rounded-full text-sm font-medium">
      {warnings} Warning{warnings > 1 ? "s" : ""}
    </div>
  );
}

function PilotSection({ character }: { character: CharacterResponse }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Pilot</CardTitle>
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
  );
}

function MechSection({ character }: { character: CharacterResponse }) {
  const activeMech = character.mechs.find(
    (m) => m.id === character.active_mech_id
  );
  const stats = character.active_mech_stats;

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          {activeMech?.name || "No Mech"}
          {activeMech && (
            <span className="text-sm font-normal text-muted-foreground ml-2">
              ({activeMech.frame_id.replace(/^gms_/, "GMS ").replace(/_/g, " ")})
            </span>
          )}
        </CardTitle>
        <CardDescription>
          {character.mechs.length} mech{character.mechs.length !== 1 ? "s" : ""}{" "}
          configured
        </CardDescription>
      </CardHeader>
      <CardContent>
        {stats ? (
          <>
            <div className="grid grid-cols-3 gap-4 mb-4">
              <StatBlock label="HP" value={stats.hp} />
              <StatBlock label="Armor" value={stats.armor} />
              <StatBlock label="Size" value={stats.size.replace("size_", "")} />
            </div>
            <div className="grid grid-cols-3 gap-4 mb-4">
              <StatBlock label="Evasion" value={stats.evasion} />
              <StatBlock label="E-Defense" value={stats.e_defense} />
              <StatBlock label="Speed" value={stats.speed} />
            </div>
            <div className="grid grid-cols-3 gap-4">
              <StatBlock label="Heat Cap" value={stats.heat_cap} />
              <StatBlock label="Repair Cap" value={stats.repair_cap} />
              <StatBlock label="SP" value={stats.system_points} />
            </div>
            <div className="mt-4 pt-4 border-t border-border grid grid-cols-3 gap-4">
              <StatBlock label="Tech Attack" value={`+${stats.tech_attack}`} />
              <StatBlock label="Sensor Range" value={stats.sensor_range} />
              <StatBlock label="Save Target" value={stats.save_target} />
            </div>
          </>
        ) : (
          <p className="text-muted-foreground">No active mech selected</p>
        )}
      </CardContent>
    </Card>
  );
}

function StatBlock({
  label,
  value,
}: {
  label: string;
  value: number | string;
}) {
  return (
    <div>
      <div className="text-muted-foreground text-xs uppercase">{label}</div>
      <div className="font-semibold text-lg">{value}</div>
    </div>
  );
}

function SkillBlock({ label, value }: { label: string; value: number }) {
  return (
    <div className="p-2 bg-card-foreground/5 rounded text-center">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-semibold">+{value}</div>
    </div>
  );
}
