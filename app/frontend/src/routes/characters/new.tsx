/**
 * Create new character page.
 *
 * Simplified form that uses LL0 defaults for game-accurate character creation.
 * The backend's create_ll0_character factory handles triggers, talents, etc.
 */

import { useState } from "react";
import { createFileRoute, useNavigate, Link } from "@tanstack/react-router";
import { useCreateCharacter } from "../../lib/api";
import type { CharacterCreateRequest } from "../../lib/api";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  Button,
} from "../../components/ui";

export const Route = createFileRoute("/characters/new" as const)({
  component: NewCharacterPage,
});

interface FormData {
  callsign: string;
  name: string;
  mechName: string;
  skills: {
    hull: number;
    agility: number;
    systems: number;
    engineering: number;
  };
  notes: string;
}

const defaultFormData: FormData = {
  callsign: "",
  name: "",
  mechName: "",
  skills: { hull: 2, agility: 0, systems: 0, engineering: 0 },
  notes: "",
};

function NewCharacterPage() {
  const navigate = useNavigate();
  const createMutation = useCreateCharacter();

  const [formData, setFormData] = useState<FormData>(defaultFormData);
  const [error, setError] = useState<string | null>(null);

  const totalSkillPoints =
    formData.skills.hull +
    formData.skills.agility +
    formData.skills.systems +
    formData.skills.engineering;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!formData.callsign.trim()) {
      setError("Callsign is required");
      return;
    }

    if (totalSkillPoints !== 2) {
      setError("LL0 characters must have exactly 2 mech skill points");
      return;
    }

    // Build request - let backend use LL0 defaults
    const request: CharacterCreateRequest = {
      callsign: formData.callsign,
      name: formData.name || undefined,
      use_ll0_defaults: true,
      skills: formData.skills,
      mech_name: formData.mechName || undefined,
      notes: formData.notes || undefined,
    };

    try {
      const result = await createMutation.mutateAsync(request);
      navigate({
        to: "/characters/$characterId",
        params: { characterId: result.id },
      });
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to create character"
      );
    }
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
        <Link
          to="/characters"
          className="text-primary hover:underline text-sm"
        >
          ← Back to Characters
        </Link>
        <h1 className="text-3xl font-bold text-foreground mt-2">
          Create New Character
        </h1>
        <p className="text-muted-foreground">
          Build a License Level 0 character with pilot and mech
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Pilot Identity</CardTitle>
            <CardDescription>Who pilots this mech?</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">
                Callsign <span className="text-destructive">*</span>
              </label>
              <input
                type="text"
                value={formData.callsign}
                onChange={(e) =>
                  setFormData((prev) => ({ ...prev, callsign: e.target.value }))
                }
                placeholder="NOVA"
                className="w-full px-3 py-2 bg-background border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Your pilot's combat handle
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Real Name
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) =>
                  setFormData((prev) => ({ ...prev, name: e.target.value }))
                }
                placeholder="Nova Chen"
                className="w-full px-3 py-2 bg-background border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
          </CardContent>
        </Card>

        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Mech Skills</CardTitle>
            <CardDescription>
              Allocate 2 points across HASE. These affect your mech's stats.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <SkillInput
                label="HULL"
                description="+2 HP per point"
                value={formData.skills.hull}
                onChange={(v) => updateSkill("hull", v)}
              />
              <SkillInput
                label="AGILITY"
                description="+1 Evasion per point"
                value={formData.skills.agility}
                onChange={(v) => updateSkill("agility", v)}
              />
              <SkillInput
                label="SYSTEMS"
                description="+1 Tech Attack & E-Def"
                value={formData.skills.systems}
                onChange={(v) => updateSkill("systems", v)}
              />
              <SkillInput
                label="ENGINEERING"
                description="+1 Heat Cap per point"
                value={formData.skills.engineering}
                onChange={(v) => updateSkill("engineering", v)}
              />
            </div>
            <div className="mt-4 text-sm">
              <span
                className={
                  totalSkillPoints === 2
                    ? "text-horus"
                    : "text-destructive"
                }
              >
                Total: {totalSkillPoints} / 2 points
              </span>
              {totalSkillPoints !== 2 && (
                <span className="text-destructive ml-2">
                  (Must be exactly 2 for LL0)
                </span>
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Mech</CardTitle>
            <CardDescription>
              LL0 pilots start with a GMS Everest frame
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div>
              <label className="block text-sm font-medium mb-1">
                Mech Name
              </label>
              <input
                type="text"
                value={formData.mechName}
                onChange={(e) =>
                  setFormData((prev) => ({
                    ...prev,
                    mechName: e.target.value,
                  }))
                }
                placeholder={formData.callsign || "RAIJIN"}
                className="w-full px-3 py-2 bg-background border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Optional custom name for your mech (defaults to callsign)
              </p>
            </div>

            <div className="mt-4 p-4 bg-card-foreground/5 rounded-md">
              <div className="font-semibold mb-2">GMS Everest</div>
              <div className="grid grid-cols-5 gap-2 text-sm text-muted-foreground">
                <div>HP 10</div>
                <div>Evasion 8</div>
                <div>Speed 4</div>
                <div>Heat Cap 6</div>
                <div>SP 6</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Notes</CardTitle>
            <CardDescription>
              Background, personality, goals
            </CardDescription>
          </CardHeader>
          <CardContent>
            <textarea
              value={formData.notes}
              onChange={(e) =>
                setFormData((prev) => ({ ...prev, notes: e.target.value }))
              }
              placeholder="Character backstory..."
              rows={4}
              className="w-full px-3 py-2 bg-background border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-primary resize-none"
            />
          </CardContent>
        </Card>

        <div className="p-4 mb-6 bg-primary/10 border border-primary/20 rounded-md text-sm">
          <strong>LL0 Defaults Applied:</strong>
          <ul className="mt-2 text-muted-foreground list-disc list-inside">
            <li>4 triggers at +2 each (8 points)</li>
            <li>3 talents at rank I (3 points)</li>
            <li>No licenses or core bonuses</li>
            <li>GMS-only gear access</li>
          </ul>
        </div>

        {error && (
          <div className="mb-4 p-3 bg-destructive/10 border border-destructive rounded-md text-destructive text-sm">
            {error}
          </div>
        )}

        <div className="flex gap-3">
          <Button
            type="submit"
            disabled={createMutation.isPending || totalSkillPoints !== 2}
          >
            {createMutation.isPending ? "Creating..." : "Create Character"}
          </Button>
          <Link to="/characters">
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
  description,
  value,
  onChange,
}: {
  label: string;
  description: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="p-3 bg-card-foreground/5 rounded-md">
      <div className="flex items-center justify-between mb-1">
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
      <p className="text-xs text-muted-foreground">{description}</p>
    </div>
  );
}
