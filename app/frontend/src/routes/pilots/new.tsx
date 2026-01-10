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
    <div className="p-6 max-w-2xl mx-auto">
      <div className="mb-6">
        <Link to="/pilots" className="text-primary hover:underline text-sm">
          ← Back to Pilots
        </Link>
        <h1 className="text-3xl font-bold text-foreground mt-2">
          Create New Pilot
        </h1>
        <p className="text-muted-foreground">
          Build a new pilot character for your Lancer campaign
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Identity</CardTitle>
            <CardDescription>Basic pilot information</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
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

        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Mech Skills</CardTitle>
            <CardDescription>
              Allocate skill points (LL0 = 2 points, +1 per level)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
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
              Total: {totalSkillPoints} / {2 + (formData.level ?? 0)} points
              {totalSkillPoints > 2 + (formData.level ?? 0) && (
                <span className="text-destructive ml-2">(Over limit!)</span>
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="mb-6">
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
          <div className="mb-4 p-3 bg-destructive/10 border border-destructive rounded-md text-destructive text-sm">
            {error}
          </div>
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
    <div className="flex items-center justify-between p-3 bg-card-foreground/5 rounded-md">
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
