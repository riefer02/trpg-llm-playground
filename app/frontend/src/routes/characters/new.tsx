/**
 * Create new character page.
 *
 * Multi-step form following Lancer character creation:
 * 1. Background - suggests triggers
 * 2. Triggers - 4 at +2 each (can customize from background suggestions)
 * 3. HASE Skills - 2 points total
 * 4. Talents - 3 at rank I
 * 5. Mech - name + GMS Everest (LL0)
 */

import { useState } from "react";
import { createFileRoute, useNavigate, Link } from "@tanstack/react-router";
import {
  useCreateCharacter,
  useBackgrounds,
  useTriggers,
  useTalents,
  usePilotGear,
} from "../../lib/api";
import type {
  CharacterCreateRequest,
  Background,
} from "../../lib/api";
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

// Form steps
type Step = "background" | "triggers" | "skills" | "talents" | "gear" | "mech";

interface FormData {
  callsign: string;
  name: string;
  backgroundId: string | null;
  backgroundName: string;
  triggers: string[]; // 4 trigger IDs
  skills: {
    hull: number;
    agility: number;
    systems: number;
    engineering: number;
  };
  talents: string[]; // 3 talent IDs
  pilotGear: {
    clothing: string | null;
    armor: string | null;
    weapons: string[];
    gear: string[];
  };
  mechName: string;
  notes: string;
}

const defaultFormData: FormData = {
  callsign: "",
  name: "",
  backgroundId: null,
  backgroundName: "",
  triggers: [],
  skills: { hull: 0, agility: 0, systems: 0, engineering: 0 },
  talents: [],
  pilotGear: {
    clothing: null,
    armor: null,
    weapons: [],
    gear: [],
  },
  mechName: "",
  notes: "",
};

function NewCharacterPage() {
  const navigate = useNavigate();
  const createMutation = useCreateCharacter();

  // Reference data
  const { data: backgrounds, isLoading: loadingBackgrounds } = useBackgrounds();
  const { data: allTriggers, isLoading: loadingTriggers } = useTriggers();
  const { data: allTalents, isLoading: loadingTalents } = useTalents();
  const { data: pilotGear, isLoading: loadingPilotGear } = usePilotGear();

  const [step, setStep] = useState<Step>("background");
  const [formData, setFormData] = useState<FormData>(defaultFormData);
  const [error, setError] = useState<string | null>(null);

  // Create lookup maps
  const triggerMap = new Map(allTriggers?.map((t) => [t.id, t]) ?? []);
  const talentMap = new Map(allTalents?.map((t) => [t.id, t]) ?? []);
  const pilotGearMap = new Map(pilotGear?.map((item) => [item.id, item]) ?? []);

  const clothingOptions =
    pilotGear?.filter((item) => item.category === "clothing") ?? [];
  const armorOptions =
    pilotGear?.filter((item) => item.category === "armor") ?? [];
  const weaponOptions =
    pilotGear?.filter((item) => item.category === "weapon") ?? [];
  const gearOptions =
    pilotGear?.filter((item) => item.category === "gear") ?? [];

  // When background is selected, pre-populate triggers
  const handleBackgroundSelect = (bg: Background) => {
    setFormData((prev) => ({
      ...prev,
      backgroundId: bg.id,
      backgroundName: bg.name,
      triggers: [...bg.triggers], // Copy suggested triggers
    }));
    setStep("triggers");
  };

  const handleSubmit = async () => {
    setError(null);

    if (!formData.callsign.trim()) {
      setError("Callsign is required");
      return;
    }

    if (!formData.pilotGear.clothing) {
      setError("Pilot gear requires a clothing selection");
      return;
    }

    // Build request with all the selected data
    const request: CharacterCreateRequest = {
      callsign: formData.callsign,
      name: formData.name || undefined,
      use_ll0_defaults: false, // We're providing custom data
      skills: formData.skills,
      triggers: formData.triggers.map((id) => ({ trigger_id: id, rank: 2 })),
      talents: formData.talents.map((id) => ({ talent_id: id, rank: 1 })),
      background: formData.backgroundId
        ? {
            id: formData.backgroundId,
            name: formData.backgroundName,
            triggers: formData.triggers,
          }
        : undefined,
      pilot_gear: {
        clothing: formData.pilotGear.clothing,
        armor: formData.pilotGear.armor || null,
        weapons: formData.pilotGear.weapons,
        gear: formData.pilotGear.gear,
      },
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

  const togglePilotGearList = (
    key: "weapons" | "gear",
    id: string,
    max: number
  ) => {
    setFormData((prev) => {
      const current = prev.pilotGear[key];
      const isSelected = current.includes(id);
      if (isSelected) {
        return {
          ...prev,
          pilotGear: {
            ...prev.pilotGear,
            [key]: current.filter((itemId) => itemId !== id),
          },
        };
      }
      if (current.length >= max) {
        return prev;
      }
      return {
        ...prev,
        pilotGear: {
          ...prev.pilotGear,
          [key]: [...current, id],
        },
      };
    });
  };

  // Validation helpers
  const totalSkillPoints =
    formData.skills.hull +
    formData.skills.agility +
    formData.skills.systems +
    formData.skills.engineering;

  const canProceedFromTriggers = formData.triggers.length === 4;
  const canProceedFromSkills = totalSkillPoints === 2;
  const canProceedFromTalents = formData.talents.length === 3;
  const canProceedFromGear = formData.pilotGear.clothing !== null;

  if (loadingBackgrounds || loadingTriggers || loadingTalents || loadingPilotGear) {
    return (
      <div className="p-6 max-w-2xl mx-auto">
        <p className="text-muted-foreground">Loading reference data...</p>
      </div>
    );
  }

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
          Build a License Level 0 character
        </p>
      </div>

      {/* Progress indicator */}
      <div className="mb-6 flex gap-2">
        {([
          "background",
          "triggers",
          "skills",
          "talents",
          "gear",
          "mech",
        ] as Step[]).map(
          (s, i) => (
            <div
              key={s}
              className={`flex-1 h-2 rounded ${
                step === s
                  ? "bg-primary"
                  : ([
                      "background",
                      "triggers",
                      "skills",
                      "talents",
                      "gear",
                      "mech",
                    ] as Step[]).indexOf(step) > i
                  ? "bg-primary/50"
                  : "bg-border"
              }`}
            />
          )
        )}
      </div>

      {/* Step 1: Background */}
      {step === "background" && (
        <Card>
          <CardHeader>
            <CardTitle>Step 1: Background</CardTitle>
            <CardDescription>
              Choose your pilot's background. This suggests 4 starting triggers.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Callsign input first */}
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

              <div>
                <label className="block text-sm font-medium mb-2">
                  Select Background
                </label>
                <div className="grid gap-2 max-h-80 overflow-y-auto">
                  {backgrounds?.map((bg) => (
                    <button
                      key={bg.id}
                      type="button"
                      onClick={() => handleBackgroundSelect(bg)}
                      disabled={!formData.callsign.trim()}
                      className={`p-3 text-left border rounded-md transition-colors ${
                        !formData.callsign.trim()
                          ? "opacity-50 cursor-not-allowed"
                          : "hover:bg-primary/10 hover:border-primary"
                      }`}
                    >
                      <div className="font-medium">{bg.name}</div>
                      <div className="text-xs text-muted-foreground mt-1">
                        Triggers:{" "}
                        {bg.triggers
                          .map((id) => triggerMap.get(id)?.name ?? id)
                          .join(", ")}
                      </div>
                    </button>
                  ))}
                </div>
                {!formData.callsign.trim() && (
                  <p className="text-xs text-muted-foreground mt-2">
                    Enter a callsign first to select a background
                  </p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 2: Triggers */}
      {step === "triggers" && (
        <Card>
          <CardHeader>
            <CardTitle>Step 2: Triggers</CardTitle>
            <CardDescription>
              Select 4 triggers at +2 each. Your background suggested these, but
              you can customize.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="mb-4 p-3 bg-primary/10 rounded-md">
              <div className="text-sm font-medium">
                Background: {formData.backgroundName}
              </div>
              <div className="text-xs text-muted-foreground">
                Selected: {formData.triggers.length} / 4 triggers
              </div>
            </div>

            <div className="grid gap-2 max-h-96 overflow-y-auto">
              {allTriggers?.map((trigger) => {
                const isSelected = formData.triggers.includes(trigger.id);
                const isFull = formData.triggers.length >= 4;
                return (
                  <button
                    key={trigger.id}
                    type="button"
                    onClick={() => {
                      if (isSelected) {
                        setFormData((prev) => ({
                          ...prev,
                          triggers: prev.triggers.filter(
                            (id) => id !== trigger.id
                          ),
                        }));
                      } else if (!isFull) {
                        setFormData((prev) => ({
                          ...prev,
                          triggers: [...prev.triggers, trigger.id],
                        }));
                      }
                    }}
                    disabled={!isSelected && isFull}
                    className={`p-3 text-left border rounded-md transition-colors ${
                      isSelected
                        ? "bg-primary/20 border-primary"
                        : isFull
                        ? "opacity-50"
                        : "hover:bg-primary/10"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{trigger.name}</span>
                      {isSelected && (
                        <span className="text-xs text-primary font-medium">
                          +2
                        </span>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="flex gap-3 mt-4">
              <Button
                type="button"
                variant="outline"
                onClick={() => setStep("background")}
              >
                Back
              </Button>
              <Button
                type="button"
                onClick={() => setStep("skills")}
                disabled={!canProceedFromTriggers}
              >
                Next: Skills
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 3: Skills */}
      {step === "skills" && (
        <Card>
          <CardHeader>
            <CardTitle>Step 3: Mech Skills</CardTitle>
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
                onChange={(v) =>
                  setFormData((prev) => ({
                    ...prev,
                    skills: { ...prev.skills, hull: Math.max(0, Math.min(6, v)) },
                  }))
                }
              />
              <SkillInput
                label="AGILITY"
                description="+1 Evasion per point"
                value={formData.skills.agility}
                onChange={(v) =>
                  setFormData((prev) => ({
                    ...prev,
                    skills: {
                      ...prev.skills,
                      agility: Math.max(0, Math.min(6, v)),
                    },
                  }))
                }
              />
              <SkillInput
                label="SYSTEMS"
                description="+1 Tech Attack & E-Def"
                value={formData.skills.systems}
                onChange={(v) =>
                  setFormData((prev) => ({
                    ...prev,
                    skills: {
                      ...prev.skills,
                      systems: Math.max(0, Math.min(6, v)),
                    },
                  }))
                }
              />
              <SkillInput
                label="ENGINEERING"
                description="+1 Heat Cap per point"
                value={formData.skills.engineering}
                onChange={(v) =>
                  setFormData((prev) => ({
                    ...prev,
                    skills: {
                      ...prev.skills,
                      engineering: Math.max(0, Math.min(6, v)),
                    },
                  }))
                }
              />
            </div>

            <div className="mt-4 text-sm">
              <span
                className={
                  totalSkillPoints === 2 ? "text-horus" : "text-destructive"
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

            <div className="flex gap-3 mt-4">
              <Button
                type="button"
                variant="outline"
                onClick={() => setStep("triggers")}
              >
                Back
              </Button>
              <Button
                type="button"
                onClick={() => setStep("talents")}
                disabled={!canProceedFromSkills}
              >
                Next: Talents
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 4: Talents */}
      {step === "talents" && (
        <Card>
          <CardHeader>
            <CardTitle>Step 4: Talents</CardTitle>
            <CardDescription>
              Choose 3 talents at rank I. These provide combat abilities for
              your mech.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="mb-4 p-3 bg-primary/10 rounded-md">
              <div className="text-xs text-muted-foreground">
                Selected: {formData.talents.length} / 3 talents
              </div>
            </div>

            <div className="grid gap-2 max-h-96 overflow-y-auto">
              {allTalents?.map((talent) => {
                const isSelected = formData.talents.includes(talent.id);
                const isFull = formData.talents.length >= 3;
                return (
                  <button
                    key={talent.id}
                    type="button"
                    onClick={() => {
                      if (isSelected) {
                        setFormData((prev) => ({
                          ...prev,
                          talents: prev.talents.filter((id) => id !== talent.id),
                        }));
                      } else if (!isFull) {
                        setFormData((prev) => ({
                          ...prev,
                          talents: [...prev.talents, talent.id],
                        }));
                      }
                    }}
                    disabled={!isSelected && isFull}
                    className={`p-3 text-left border rounded-md transition-colors ${
                      isSelected
                        ? "bg-primary/20 border-primary"
                        : isFull
                        ? "opacity-50"
                        : "hover:bg-primary/10"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{talent.name}</span>
                      {isSelected && (
                        <span className="text-xs text-primary font-medium">
                          Rank I
                        </span>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="flex gap-3 mt-4">
              <Button
                type="button"
                variant="outline"
                onClick={() => setStep("skills")}
              >
                Back
              </Button>
              <Button
                type="button"
                onClick={() => setStep("gear")}
                disabled={!canProceedFromTalents}
              >
                Next: Pilot Gear
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 5: Pilot Gear */}
      {step === "gear" && (
        <Card>
          <CardHeader>
            <CardTitle>Step 5: Pilot Gear</CardTitle>
            <CardDescription>
              Choose your mission loadout: clothing, optional armor, up to 2
              weapons, and up to 3 gear items.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              <div>
                <div className="text-xs text-muted-foreground uppercase tracking-wide mb-2">
                  Clothing (Required)
                </div>
                <div className="grid gap-2">
                  {clothingOptions.map((item) => {
                    const isSelected = formData.pilotGear.clothing === item.id;
                    return (
                      <button
                        key={item.id}
                        type="button"
                        onClick={() =>
                          setFormData((prev) => ({
                            ...prev,
                            pilotGear: {
                              ...prev.pilotGear,
                              clothing: item.id,
                            },
                          }))
                        }
                        className={`p-3 text-left border rounded-md transition-colors ${
                          isSelected
                            ? "bg-primary/20 border-primary"
                            : "hover:bg-primary/10"
                        }`}
                      >
                        <div className="font-medium">{item.name}</div>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div>
                <div className="text-xs text-muted-foreground uppercase tracking-wide mb-2">
                  Armor (Optional)
                </div>
                <div className="grid gap-2">
                  <button
                    type="button"
                    onClick={() =>
                      setFormData((prev) => ({
                        ...prev,
                        pilotGear: { ...prev.pilotGear, armor: null },
                      }))
                    }
                    className={`p-3 text-left border rounded-md transition-colors ${
                      formData.pilotGear.armor === null
                        ? "bg-primary/20 border-primary"
                        : "hover:bg-primary/10"
                    }`}
                  >
                    <div className="font-medium">No armor</div>
                  </button>
                  {armorOptions.map((item) => {
                    const isSelected = formData.pilotGear.armor === item.id;
                    return (
                      <button
                        key={item.id}
                        type="button"
                        onClick={() =>
                          setFormData((prev) => ({
                            ...prev,
                            pilotGear: {
                              ...prev.pilotGear,
                              armor: item.id,
                            },
                          }))
                        }
                        className={`p-3 text-left border rounded-md transition-colors ${
                          isSelected
                            ? "bg-primary/20 border-primary"
                            : "hover:bg-primary/10"
                        }`}
                      >
                        <div className="font-medium">{item.name}</div>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="text-xs text-muted-foreground uppercase tracking-wide">
                    Weapons (Up to 2)
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Selected: {formData.pilotGear.weapons.length} / 2
                  </div>
                </div>
                <div className="grid gap-2">
                  {weaponOptions.map((item) => {
                    const isSelected = formData.pilotGear.weapons.includes(item.id);
                    const isFull =
                      formData.pilotGear.weapons.length >= 2 && !isSelected;
                    return (
                      <button
                        key={item.id}
                        type="button"
                        onClick={() =>
                          togglePilotGearList("weapons", item.id, 2)
                        }
                        disabled={isFull}
                        className={`p-3 text-left border rounded-md transition-colors ${
                          isSelected
                            ? "bg-primary/20 border-primary"
                            : isFull
                            ? "opacity-50"
                            : "hover:bg-primary/10"
                        }`}
                      >
                        <div className="font-medium">{item.name}</div>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="text-xs text-muted-foreground uppercase tracking-wide">
                    Gear (Up to 3)
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Selected: {formData.pilotGear.gear.length} / 3
                  </div>
                </div>
                <div className="grid gap-2">
                  {gearOptions.map((item) => {
                    const isSelected = formData.pilotGear.gear.includes(item.id);
                    const isFull =
                      formData.pilotGear.gear.length >= 3 && !isSelected;
                    return (
                      <button
                        key={item.id}
                        type="button"
                        onClick={() => togglePilotGearList("gear", item.id, 3)}
                        disabled={isFull}
                        className={`p-3 text-left border rounded-md transition-colors ${
                          isSelected
                            ? "bg-primary/20 border-primary"
                            : isFull
                            ? "opacity-50"
                            : "hover:bg-primary/10"
                        }`}
                      >
                        <div className="font-medium">{item.name}</div>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <Button
                type="button"
                variant="outline"
                onClick={() => setStep("talents")}
              >
                Back
              </Button>
              <Button
                type="button"
                onClick={() => setStep("mech")}
                disabled={!canProceedFromGear}
              >
                Next: Mech
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 6: Mech */}
      {step === "mech" && (
        <Card>
          <CardHeader>
            <CardTitle>Step 6: Your Mech</CardTitle>
            <CardDescription>
              LL0 pilots start with a GMS Everest frame
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div>
              <label className="block text-sm font-medium mb-1">Mech Name</label>
              <input
                type="text"
                value={formData.mechName}
                onChange={(e) =>
                  setFormData((prev) => ({ ...prev, mechName: e.target.value }))
                }
                placeholder={formData.callsign || "RAIJIN"}
                className="w-full px-3 py-2 bg-background border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Optional custom name (defaults to callsign)
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

            <div className="mt-4">
              <label className="block text-sm font-medium mb-1">Notes</label>
              <textarea
                value={formData.notes}
                onChange={(e) =>
                  setFormData((prev) => ({ ...prev, notes: e.target.value }))
                }
                placeholder="Character backstory..."
                rows={3}
                className="w-full px-3 py-2 bg-background border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-primary resize-none"
              />
            </div>

            {/* Summary */}
            <div className="mt-6 p-4 bg-primary/10 border border-primary/20 rounded-md text-sm">
              <strong>Character Summary:</strong>
              <ul className="mt-2 space-y-1 text-muted-foreground">
                <li>
                  <strong>Callsign:</strong> {formData.callsign}
                </li>
                <li>
                  <strong>Background:</strong> {formData.backgroundName}
                </li>
                <li>
                  <strong>Triggers:</strong>{" "}
                  {formData.triggers
                    .map((id) => triggerMap.get(id)?.name ?? id)
                    .join(", ")}
                </li>
                <li>
                  <strong>Skills:</strong> Hull +{formData.skills.hull}, Agi +
                  {formData.skills.agility}, Sys +{formData.skills.systems}, Eng
                  +{formData.skills.engineering}
                </li>
                <li>
                  <strong>Talents:</strong>{" "}
                  {formData.talents
                    .map((id) => talentMap.get(id)?.name ?? id)
                    .join(", ")}
                </li>
                <li>
                  <strong>Pilot Gear:</strong>{" "}
                  {[
                    formData.pilotGear.clothing
                      ? pilotGearMap.get(formData.pilotGear.clothing)?.name ??
                        formData.pilotGear.clothing
                      : "No clothing",
                    formData.pilotGear.armor
                      ? pilotGearMap.get(formData.pilotGear.armor)?.name ??
                        formData.pilotGear.armor
                      : "No armor",
                  ]
                    .concat(
                      formData.pilotGear.weapons.map(
                        (id) => pilotGearMap.get(id)?.name ?? id
                      )
                    )
                    .concat(
                      formData.pilotGear.gear.map(
                        (id) => pilotGearMap.get(id)?.name ?? id
                      )
                    )
                    .join(", ")}
                </li>
              </ul>
            </div>

            {error && (
              <div className="mt-4 p-3 bg-destructive/10 border border-destructive rounded-md text-destructive text-sm">
                {error}
              </div>
            )}

            <div className="flex gap-3 mt-4">
              <Button
                type="button"
                variant="outline"
                onClick={() => setStep("talents")}
              >
                Back
              </Button>
              <Button
                type="button"
                onClick={handleSubmit}
                disabled={createMutation.isPending}
              >
                {createMutation.isPending ? "Creating..." : "Create Character"}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
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
