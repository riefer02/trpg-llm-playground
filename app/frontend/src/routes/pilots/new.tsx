/**
 * Create new pilot page.
 *
 * Form for creating a new pilot with validation.
 */

import { useState } from "react";
import { createFileRoute, useNavigate, Link } from "@tanstack/react-router";
import { useCreatePilot } from "../../lib/api";
import type { PilotCreateRequest } from "../../lib/api";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  Button,
} from "../../components/ui";

export const Route = createFileRoute("/pilots/new" as const)({
  component: NewPilotPage,
});

const defaultFormData: PilotCreateRequest = {
  callsign: "",
  name: "",
  level: 0,
  skills: { hull: 0, agility: 0, systems: 0, engineering: 0 },
  triggers: [],
  talents: [],
  licenses: [],
  core_bonuses: [],
  notes: "",
};

function NewPilotPage() {
  const navigate = useNavigate();
  const createMutation = useCreatePilot();

  const [formData, setFormData] = useState<PilotCreateRequest>(defaultFormData);
  const [error, setError] = useState<string | null>(null);

  const totalSkillPoints =
    (formData.skills?.hull ?? 0) +
    (formData.skills?.agility ?? 0) +
    (formData.skills?.systems ?? 0) +
    (formData.skills?.engineering ?? 0);
  const skillLimit = 2 + (formData.level ?? 0);
  const callsignReady = formData.callsign.trim().length > 0;
  const skillsReady = totalSkillPoints <= skillLimit;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!formData.callsign.trim()) {
      setError("Callsign is required");
      return;
    }

    try {
      const result = await createMutation.mutateAsync(formData);
      navigate({ to: "/pilots/$pilotId", params: { pilotId: result.id } });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create pilot");
    }
  };

  const updateField = <K extends keyof PilotCreateRequest>(
    field: K,
    value: PilotCreateRequest[K]
  ) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const updateSkill = (
    skill: "hull" | "agility" | "systems" | "engineering",
    value: number
  ) => {
    setFormData((prev) => ({
      ...prev,
      skills: {
        ...prev.skills,
        [skill]: Math.max(0, Math.min(6, value)),
      },
    }));
  };

  return (
    <div className="px-6 py-8 max-w-7xl mx-auto space-y-6">
      <section className="dashboard-surface p-6 animate-rise">
        <Link to="/pilots" className="text-primary hover:underline text-sm">
          ← Back to Pilots
        </Link>
        <div className="mt-3 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h1 className="text-3xl font-heading font-semibold text-foreground">
              Create New Pilot
            </h1>
            <p className="text-muted-foreground">
              Build pilot progression data with live validation prompts.
            </p>
          </div>
          <div className="rounded-full border border-border px-4 py-1 text-xs text-muted-foreground">
            LL{formData.level ?? 0} template
          </div>
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <form onSubmit={handleSubmit} className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Identity</CardTitle>
              <CardDescription>Basic pilot information</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {!callsignReady && (
                <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                  Callsign is required before saving.
                </div>
              )}
              <div>
                <label className="block text-sm font-medium mb-1">
                  Callsign <span className="text-destructive">*</span>
                </label>
                <input
                  type="text"
                  value={formData.callsign}
                  onChange={(e) => updateField("callsign", e.target.value)}
                  placeholder="NOVA"
                  className="w-full px-3 py-2 bg-background border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  Real Name
                </label>
                <input
                  type="text"
                  value={formData.name ?? ""}
                  onChange={(e) => updateField("name", e.target.value)}
                  placeholder="Nova Chen"
                  className="w-full px-3 py-2 bg-background border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  License Level
                </label>
                <select
                  value={formData.level ?? 0}
                  onChange={(e) => updateField("level", parseInt(e.target.value))}
                  className="w-full px-3 py-2 bg-background border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
                >
                  {Array.from({ length: 13 }, (_, i) => (
                    <option key={i} value={i}>
                      LL{i}
                    </option>
                  ))}
                </select>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Mech Skills</CardTitle>
              <CardDescription>
                Allocate skill points (LL0 = 2 points, +1 per level)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <SkillInput
                  label="HULL"
                  value={formData.skills?.hull ?? 0}
                  onChange={(v) => updateSkill("hull", v)}
                />
                <SkillInput
                  label="AGILITY"
                  value={formData.skills?.agility ?? 0}
                  onChange={(v) => updateSkill("agility", v)}
                />
                <SkillInput
                  label="SYSTEMS"
                  value={formData.skills?.systems ?? 0}
                  onChange={(v) => updateSkill("systems", v)}
                />
                <SkillInput
                  label="ENGINEERING"
                  value={formData.skills?.engineering ?? 0}
                  onChange={(v) => updateSkill("engineering", v)}
                />
              </div>
              <div className="mt-4 text-sm text-muted-foreground">
                Total: {totalSkillPoints} / {skillLimit} points
                {totalSkillPoints > skillLimit && (
                  <span className="text-destructive ml-2">(Over limit!)</span>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Notes</CardTitle>
              <CardDescription>
                Any additional notes about your pilot
              </CardDescription>
            </CardHeader>
            <CardContent>
              <textarea
                value={formData.notes ?? ""}
                onChange={(e) => updateField("notes", e.target.value)}
                placeholder="Character backstory, personality, goals..."
                rows={4}
                className="w-full px-3 py-2 bg-background border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-primary resize-none"
              />
            </CardContent>
          </Card>

          {error && (
            <Card className="border-destructive/40">
              <CardContent className="pt-6 text-destructive" role="alert">
                {error}
              </CardContent>
            </Card>
          )}

          <div className="flex gap-3">
            <Button type="submit" disabled={createMutation.isPending}>
              {createMutation.isPending ? "Creating..." : "Create Pilot"}
            </Button>
            <Link to="/pilots">
              <Button type="button" variant="outline">
                Cancel
              </Button>
            </Link>
          </div>
        </form>

        <aside className="space-y-4 lg:sticky lg:top-6 h-fit">
          <Card>
            <CardHeader>
              <CardTitle>Readiness Checklist</CardTitle>
              <CardDescription>Before you save this pilot</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <ChecklistItem
                label="Callsign set"
                detail={callsignReady ? "Locked" : "Required"}
                ok={callsignReady}
              />
              <ChecklistItem
                label="Skill points"
                detail={`${totalSkillPoints} / ${skillLimit}`}
                ok={skillsReady}
              />
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Progression Notes</CardTitle>
              <CardDescription>How this data is used</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-muted-foreground">
              <p>License levels unlock gear on the compendium.</p>
              <p>Skill totals scale by level and are validated on save.</p>
              <p>Notes are for narrative context and quick reference.</p>
            </CardContent>
          </Card>
        </aside>
      </div>
    </div>
  );
}

function ChecklistItem({
  label,
  detail,
  ok,
}: {
  label: string;
  detail: string;
  ok: boolean;
}) {
  return (
    <div className="flex items-start justify-between gap-3">
      <div>
        <div className="text-sm font-medium">{label}</div>
        <div className="text-xs text-muted-foreground">{detail}</div>
      </div>
      <span
        className={`mt-1 rounded-full border px-2 py-0.5 text-xs ${
          ok
            ? "border-primary/40 bg-primary/10 text-primary"
            : "border-border text-muted-foreground"
        }`}
      >
        {ok ? "Ready" : "Pending"}
      </span>
    </div>
  );
}

function SkillInput({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="flex items-center justify-between p-3 bg-muted/50 border border-border rounded-md">
      <span className="font-medium">{label}</span>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => onChange(value - 1)}
          className="w-8 h-8 flex items-center justify-center rounded bg-background border border-border hover:bg-primary/10"
        >
          -
        </button>
        <span className="w-8 text-center font-semibold">+{value}</span>
        <button
          type="button"
          onClick={() => onChange(value + 1)}
          className="w-8 h-8 flex items-center justify-center rounded bg-background border border-border hover:bg-primary/10"
        >
          +
        </button>
      </div>
    </div>
  );
}
