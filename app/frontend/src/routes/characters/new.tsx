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
import { LicenseBadge } from "../../components/ui/LicenseBadge";

export const Route = createFileRoute("/characters/new" as const)({
  component: NewCharacterPage,
});

// Form steps
type Step = "background" | "triggers" | "skills" | "talents" | "gear" | "mech";

const STEP_ORDER: Step[] = [
  "background",
  "triggers",
  "skills",
  "talents",
  "gear",
  "mech",
];

const STEP_META: Record<
  Step,
  { title: string; summary: string; description: string }
> = {
  background: {
    title: "Background",
    summary: "Identity + origin",
    description: "Callsign and background set the tone for your pilot.",
  },
  triggers: {
    title: "Triggers",
    summary: "4 narrative skills",
    description: "Pick four triggers to anchor your LL0 playstyle.",
  },
  skills: {
    title: "Mech Skills",
    summary: "2 HASE points",
    description: "Allocate HASE points to define mech performance.",
  },
  talents: {
    title: "Talents",
    summary: "3 rank I picks",
    description: "Choose three talents that shape your combat plan.",
  },
  gear: {
    title: "Pilot Gear",
    summary: "Mission loadout",
    description: "Clothing required, armor optional, weapons + gear limited.",
  },
  mech: {
    title: "Mech",
    summary: "GMS Everest",
    description: "Name your starting frame and review the summary.",
  },
};

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
  const canProceedFromBackground =
    formData.callsign.trim().length > 0 && Boolean(formData.backgroundId);

  const stepStatus: Record<Step, boolean> = {
    background: canProceedFromBackground,
    triggers: canProceedFromTriggers,
    skills: canProceedFromSkills,
    talents: canProceedFromTalents,
    gear: canProceedFromGear,
    mech: true,
  };

  const completedSteps = STEP_ORDER.filter((id) => stepStatus[id]).length;
  const progressPercent = Math.round(
    (completedSteps / STEP_ORDER.length) * 100
  );
  const currentStepMeta = STEP_META[step];

  if (loadingBackgrounds || loadingTriggers || loadingTalents || loadingPilotGear) {
    return (
      <div className="p-6 max-w-2xl mx-auto">
        <p className="text-muted-foreground">Loading reference data...</p>
      </div>
    );
  }

  return (
    <div className="px-6 py-8 max-w-7xl mx-auto space-y-6">
      <section className="dashboard-surface p-6 animate-rise">
        <Link to="/characters" className="text-primary hover:underline text-sm">
          ← Back to Characters
        </Link>
        <div className="mt-3 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h1 className="text-3xl font-heading font-semibold text-foreground">
              Create New Character
            </h1>
            <p className="text-muted-foreground">
              LL0 build with guided validation and live readiness checks
            </p>
          </div>
          <div className="rounded-full border border-border px-4 py-1 text-xs text-muted-foreground">
            {completedSteps} / {STEP_ORDER.length} steps ready
          </div>
        </div>
        <div className="mt-4 grid gap-4 md:grid-cols-3">
          <div className="p-3 rounded-lg border border-border bg-muted/40">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Current Step
            </div>
            <div className="text-lg font-semibold">{currentStepMeta.title}</div>
            <div className="text-xs text-muted-foreground">
              {currentStepMeta.summary}
            </div>
          </div>
          <div className="p-3 rounded-lg border border-border bg-muted/40">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Callsign
            </div>
            <div className="text-lg font-semibold">
              {formData.callsign || "Not set"}
            </div>
          </div>
          <div className="p-3 rounded-lg border border-border bg-muted/40">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Background
            </div>
            <div className="text-lg font-semibold">
              {formData.backgroundName || "Pending selection"}
            </div>
          </div>
        </div>
        <div className="mt-4 h-2 w-full rounded-full bg-border overflow-hidden">
          <div
            className="h-full bg-primary transition-all"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-[240px_minmax(0,1fr)_320px]">
        <nav className="space-y-3">
          <div className="text-xs uppercase tracking-wide text-muted-foreground">
            Mission Steps
          </div>
          {STEP_ORDER.map((stepId) => {
            const meta = STEP_META[stepId];
            const isActive = step === stepId;
            const isComplete = stepStatus[stepId];
            return (
              <button
                key={stepId}
                type="button"
                onClick={() => setStep(stepId)}
                aria-current={isActive ? "step" : undefined}
                className={`w-full text-left px-3 py-2 rounded-lg border transition-colors ${
                  isActive
                    ? "border-primary/50 bg-primary/10"
                    : "border-border hover:bg-muted/60"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{meta.title}</span>
                  <span
                    className={`text-xs ${
                      isComplete ? "text-primary" : "text-muted-foreground"
                    }`}
                  >
                    {isComplete ? "Ready" : "Pending"}
                  </span>
                </div>
                <div className="text-xs text-muted-foreground">
                  {meta.summary}
                </div>
              </button>
            );
          })}
        </nav>

        <div className="space-y-6">
          {/* Step 1: Background */}
          {step === "background" && (
            <Card>
              <CardHeader>
                <CardTitle>Step 1: Background</CardTitle>
                <CardDescription>{STEP_META.background.description}</CardDescription>
              </CardHeader>
              <CardContent>
                {!formData.callsign.trim() && (
                  <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                    Callsign is required before you can select a background.
                  </div>
                )}
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-1">
                      Callsign <span className="text-destructive">*</span>
                    </label>
                    <input
                      type="text"
                      value={formData.callsign}
                      onChange={(e) =>
                        setFormData((prev) => ({
                          ...prev,
                          callsign: e.target.value,
                        }))
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
                        Enter a callsign first to select a background.
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
                <CardDescription>{STEP_META.triggers.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="mb-4 rounded-md border border-border bg-muted/50 px-3 py-2">
                  <div className="text-sm font-medium">
                    Background: {formData.backgroundName}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Selected: {formData.triggers.length} / 4 triggers
                  </div>
                </div>
                {!canProceedFromTriggers && (
                  <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                    Select exactly 4 triggers to continue.
                  </div>
                )}

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
                <CardDescription>{STEP_META.skills.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
                      totalSkillPoints === 2 ? "text-primary" : "text-destructive"
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
                <CardDescription>{STEP_META.talents.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="mb-4 rounded-md border border-border bg-muted/50 px-3 py-2">
                  <div className="text-xs text-muted-foreground">
                    Selected: {formData.talents.length} / 3 talents
                  </div>
                </div>
                {!canProceedFromTalents && (
                  <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                    Select exactly 3 talents to continue.
                  </div>
                )}

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
                <CardDescription>{STEP_META.gear.description}</CardDescription>
              </CardHeader>
              <CardContent>
                {!canProceedFromGear && (
                  <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                    Clothing is required before moving on.
                  </div>
                )}
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
                <CardDescription>{STEP_META.mech.description}</CardDescription>
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

                <div className="mt-4 p-4 bg-muted/50 border border-border rounded-md">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="font-semibold">GMS Everest</div>
                    <LicenseBadge licenseId={null} />
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-sm text-muted-foreground">
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

                <div className="mt-6 p-4 bg-primary/10 border border-primary/20 rounded-md text-sm">
                  <strong>Mission Summary</strong>
                  <ul className="mt-2 space-y-1 text-muted-foreground">
                    <li>
                      <strong>Callsign:</strong> {formData.callsign || "None"}
                    </li>
                    <li>
                      <strong>Background:</strong> {formData.backgroundName || "None"}
                    </li>
                    <li>
                      <strong>Triggers:</strong>{" "}
                      {formData.triggers.length
                        ? formData.triggers
                            .map((id) => triggerMap.get(id)?.name ?? id)
                            .join(", ")
                        : "None"}
                    </li>
                    <li>
                      <strong>Skills:</strong> Hull +{formData.skills.hull}, Agi +
                      {formData.skills.agility}, Sys +{formData.skills.systems}, Eng
                      +{formData.skills.engineering}
                    </li>
                    <li>
                      <strong>Talents:</strong>{" "}
                      {formData.talents.length
                        ? formData.talents
                            .map((id) => talentMap.get(id)?.name ?? id)
                            .join(", ")
                        : "None"}
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

        <aside className="space-y-4 lg:sticky lg:top-6 h-fit">
          <Card>
            <CardHeader>
              <CardTitle>Readiness Checklist</CardTitle>
              <CardDescription>LL0 requirements before deployment</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <ChecklistItem
                label="Callsign + background"
                detail={formData.callsign && formData.backgroundName ? "Locked" : "Required"}
                ok={canProceedFromBackground}
              />
              <ChecklistItem
                label="4 triggers"
                detail={`${formData.triggers.length} selected`}
                ok={canProceedFromTriggers}
              />
              <ChecklistItem
                label="2 HASE points"
                detail={`${totalSkillPoints} / 2`}
                ok={canProceedFromSkills}
              />
              <ChecklistItem
                label="3 talents"
                detail={`${formData.talents.length} selected`}
                ok={canProceedFromTalents}
              />
              <ChecklistItem
                label="Pilot clothing"
                detail={formData.pilotGear.clothing ? "Selected" : "Missing"}
                ok={canProceedFromGear}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Field Guide</CardTitle>
              <CardDescription>Fast guidance for new pilots</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>LL0 requires 4 triggers at +2 and exactly 2 HASE points.</li>
                <li>Talents start at Rank I; you can respec later.</li>
                <li>Clothing is mandatory; armor is optional.</li>
                <li>Weapons and gear are capped to keep loadouts lean.</li>
              </ul>
            </CardContent>
          </Card>

          {error && (
            <Card className="border-destructive/40">
              <CardHeader>
                <CardTitle className="text-destructive">Submission Error</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-destructive" role="alert">
                  {error}
                </p>
              </CardContent>
            </Card>
          )}
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
    <div className="p-3 bg-muted/50 border border-border rounded-md">
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
