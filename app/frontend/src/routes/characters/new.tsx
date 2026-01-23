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

import { useState, useMemo } from "react";
import { createFileRoute, useNavigate, Link } from "@tanstack/react-router";
import { toast } from "sonner";
import {
  useCreateCharacter,
  useBackgrounds,
  useTriggers,
  useTalents,
  usePilotGear,
  useLicenses,
  useFrames,
  useAddMech,
} from "../../lib/api";
import type {
  CharacterCreateRequest,
  CompendiumBackground,
} from "../../lib/api";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  Button,
} from "../../components/ui";
import { CharacterFormSkeleton } from "../../components/skeletons";
import {
  LevelSelector,
  LEVEL_PROGRESSION,
} from "../../components/character/LevelSelector";
import {
  LicenseSelector,
  type LicenseAllocation,
} from "../../components/character/LicenseSelector";
import { FrameBrowser } from "../../components/character/FrameBrowser";

export const Route = createFileRoute("/characters/new" as const)({
  component: NewCharacterPage,
});

// Form steps
type Step =
  | "background"
  | "level"
  | "licenses"
  | "triggers"
  | "skills"
  | "talents"
  | "gear"
  | "mech";

// Base step order - licenses step is conditionally included
const BASE_STEP_ORDER: Step[] = [
  "background",
  "level",
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
  level: {
    title: "License Level",
    summary: "LL0-3 progression",
    description: "Choose your pilot's experience level and unlocks.",
  },
  licenses: {
    title: "Licenses",
    summary: "Manufacturer access",
    description: "Allocate license points to unlock manufacturer frames and equipment.",
  },
  triggers: {
    title: "Triggers",
    summary: "4 narrative skills",
    description: "Pick four triggers to anchor your playstyle.",
  },
  skills: {
    title: "Mech Skills",
    summary: "HASE points",
    description: "Allocate HASE points to define mech performance.",
  },
  talents: {
    title: "Talents",
    summary: "Rank I picks",
    description: "Choose talents that shape your combat plan.",
  },
  gear: {
    title: "Pilot Gear",
    summary: "Mission loadout",
    description: "Clothing required, armor optional, weapons + gear limited.",
  },
  mech: {
    title: "Mech",
    summary: "Your frame(s)",
    description: "Select and name your mech(s) based on available licenses.",
  },
};

interface MechSelection {
  frameId: string;
  name: string;
}

interface FormData {
  callsign: string;
  name: string;
  backgroundId: string | null;
  backgroundName: string;
  level: number; // LL0-3
  licenseAllocations: LicenseAllocation[];
  triggers: string[]; // 4 trigger IDs
  skills: {
    hull: number;
    agility: number;
    systems: number;
    engineering: number;
  };
  talents: string[]; // talent IDs (3 at LL0, increases with level)
  pilotGear: {
    clothing: string | null;
    armor: string | null;
    weapons: string[];
    gear: string[];
  };
  mechs: MechSelection[]; // Support multiple mechs at higher levels
  notes: string;
}

const defaultFormData: FormData = {
  callsign: "",
  name: "",
  backgroundId: null,
  backgroundName: "",
  level: 0,
  licenseAllocations: [],
  triggers: [],
  skills: { hull: 0, agility: 0, systems: 0, engineering: 0 },
  talents: [],
  pilotGear: {
    clothing: null,
    armor: null,
    weapons: [],
    gear: [],
  },
  mechs: [],
  notes: "",
};

// Track which fields have been interacted with for validation feedback
interface TouchedFields {
  callsign: boolean;
  background: boolean;
  triggers: boolean;
  skills: boolean;
  talents: boolean;
  clothing: boolean;
  mech: boolean;
  licenses: boolean;
}

const defaultTouched: TouchedFields = {
  callsign: false,
  background: false,
  triggers: false,
  skills: false,
  talents: false,
  clothing: false,
  mech: false,
  licenses: false,
};

function NewCharacterPage() {
  const navigate = useNavigate();
  const createMutation = useCreateCharacter();
  const addMechMutation = useAddMech();

  // Reference data
  const { data: backgrounds, isLoading: loadingBackgrounds } = useBackgrounds();
  const { data: allTriggers, isLoading: loadingTriggers } = useTriggers();
  const { data: allTalents, isLoading: loadingTalents } = useTalents();
  const { data: pilotGear, isLoading: loadingPilotGear } = usePilotGear();
  const { data: licenses, isLoading: loadingLicenses } = useLicenses();
  const { data: frames, isLoading: loadingFrames } = useFrames();

  const [step, setStep] = useState<Step>("background");
  const [formData, setFormData] = useState<FormData>(defaultFormData);
  const [touched, setTouched] = useState<TouchedFields>(defaultTouched);
  const [error, setError] = useState<string | null>(null);

  // Mark a field as touched for validation feedback
  const markTouched = (field: keyof TouchedFields) => {
    setTouched((prev) => ({ ...prev, [field]: true }));
  };

  // Dynamic step order based on level
  const STEP_ORDER = useMemo(() => {
    if (formData.level >= 1) {
      // Insert licenses step after level for LL1+
      return [
        "background",
        "level",
        "licenses",
        "triggers",
        "skills",
        "talents",
        "gear",
        "mech",
      ] as Step[];
    }
    return BASE_STEP_ORDER;
  }, [formData.level]);

  // Progression values based on level
  const progression =
    LEVEL_PROGRESSION[formData.level as keyof typeof LEVEL_PROGRESSION] ??
    LEVEL_PROGRESSION[0];

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
  const handleBackgroundSelect = (bg: CompendiumBackground) => {
    setFormData((prev) => ({
      ...prev,
      backgroundId: bg.id,
      backgroundName: bg.name,
      triggers: [...bg.triggers], // Copy suggested triggers
    }));
    setStep("level");
  };

  // Handle level change - reset allocations when level changes
  const handleLevelChange = (newLevel: number) => {
    setFormData((prev) => ({
      ...prev,
      level: newLevel,
      licenseAllocations: [],
      // Reset mechs when changing level
      mechs: [],
      // Reset skills to 0 when level changes to let user reallocate
      skills: { hull: 0, agility: 0, systems: 0, engineering: 0 },
      // Reset talents when level changes
      talents: [],
    }));
  };

  const handleSubmit = async () => {
    setError(null);

    if (!formData.callsign.trim()) {
      toast.error("Callsign is required");
      setError("Callsign is required");
      return;
    }

    if (!formData.pilotGear.clothing) {
      toast.error("Pilot gear requires a clothing selection");
      setError("Pilot gear requires a clothing selection");
      return;
    }

    if (formData.mechs.length === 0) {
      toast.error("At least one mech is required");
      setError("At least one mech is required");
      return;
    }

    // Get primary mech (first selected)
    const primaryMech = formData.mechs[0];

    // Build request with all the selected data
    // Type assertions needed because TypeScript generates strict tuple types
    const request: CharacterCreateRequest = {
      callsign: formData.callsign,
      name: formData.name || undefined,
      level: formData.level,
      use_ll0_defaults: false, // We're providing custom data
      skills: formData.skills,
      triggers: formData.triggers.map((id) => ({ trigger_id: id, rank: 2 })),
      talents: formData.talents.map((id) => ({ talent_id: id, rank: 1 })),
      licenses:
        formData.level > 0
          ? formData.licenseAllocations.map((a) => ({
              license_id: a.licenseId,
              rank: a.rank,
            }))
          : undefined,
      background: formData.backgroundId
        ? {
            id: formData.backgroundId,
            name: formData.backgroundName,
            triggers: formData.triggers as [string, string, string, string],
          }
        : undefined,
      pilot_gear: {
        clothing: formData.pilotGear.clothing,
        armor: formData.pilotGear.armor || null,
        weapons: formData.pilotGear.weapons as [] | [string] | [string, string],
        gear: formData.pilotGear.gear as
          | []
          | [string]
          | [string, string]
          | [string, string, string],
      },
      mech_frame_id: primaryMech.frameId,
      mech_name: primaryMech.name || undefined,
      notes: formData.notes || undefined,
    };

    try {
      const result = await createMutation.mutateAsync(request);

      // Add additional mechs for LL1+ if there are more than one
      if (formData.mechs.length > 1) {
        for (let i = 1; i < formData.mechs.length; i++) {
          const mech = formData.mechs[i];
          await addMechMutation.mutateAsync({
            characterId: result.id,
            data: {
              name: mech.name || mech.frameId,
              frame_id: mech.frameId,
            },
          });
        }
      }

      toast.success(`Character "${formData.callsign}" created`);
      navigate({
        to: "/characters/$characterId",
        params: { characterId: result.id },
      });
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Failed to create character";
      toast.error(errorMessage);
      setError(errorMessage);
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

  const totalLicensePoints = formData.licenseAllocations.reduce(
    (sum, a) => sum + a.rank,
    0
  );

  const canProceedFromBackground =
    formData.callsign.trim().length > 0 && Boolean(formData.backgroundId);
  const canProceedFromLevel = true; // Always valid since default is LL0
  const canProceedFromLicenses =
    formData.level === 0 || totalLicensePoints === progression.licensePoints;
  const canProceedFromTriggers = formData.triggers.length === 4;
  const canProceedFromSkills = totalSkillPoints === progression.skillPoints;
  const canProceedFromTalents =
    formData.talents.length === progression.talentPoints;
  const canProceedFromGear = formData.pilotGear.clothing !== null;
  const canProceedFromMech = formData.mechs.length >= 1;

  const stepStatus: Record<Step, boolean> = {
    background: canProceedFromBackground,
    level: canProceedFromLevel,
    licenses: canProceedFromLicenses,
    triggers: canProceedFromTriggers,
    skills: canProceedFromSkills,
    talents: canProceedFromTalents,
    gear: canProceedFromGear,
    mech: canProceedFromMech,
  };

  const completedSteps = STEP_ORDER.filter((id) => stepStatus[id]).length;
  const progressPercent = Math.round(
    (completedSteps / STEP_ORDER.length) * 100
  );
  const currentStepMeta = STEP_META[step];

  if (
    loadingBackgrounds ||
    loadingTriggers ||
    loadingTalents ||
    loadingPilotGear ||
    loadingLicenses ||
    loadingFrames
  ) {
    return <CharacterFormSkeleton />;
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
              LL{formData.level} build with guided validation and live readiness checks
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
          {/* Background */}
          {step === "background" && (
            <Card>
              <CardHeader>
                <CardTitle>Background</CardTitle>
                <CardDescription>{STEP_META.background.description}</CardDescription>
              </CardHeader>
              <CardContent>
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
                      onBlur={() => markTouched("callsign")}
                      placeholder="NOVA"
                      className={`w-full px-3 py-2 bg-background border rounded-md focus:outline-none focus:ring-2 focus:ring-primary ${
                        touched.callsign && !formData.callsign.trim()
                          ? "border-destructive"
                          : "border-border"
                      }`}
                    />
                    {touched.callsign && !formData.callsign.trim() && (
                      <p className="mt-1 text-xs text-destructive">Callsign is required</p>
                    )}
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
                      Select Background <span className="text-destructive">*</span>
                    </label>
                    <div
                      className="grid gap-2 max-h-80 overflow-y-auto"
                      onMouseLeave={() => formData.callsign.trim() && markTouched("background")}
                    >
                      {backgrounds?.map((bg) => {
                        const isSelected = formData.backgroundId === bg.id;
                        return (
                          <button
                            key={bg.id}
                            type="button"
                            onClick={() => handleBackgroundSelect(bg)}
                            disabled={!formData.callsign.trim()}
                            className={`p-3 text-left border rounded-md transition-colors ${
                              isSelected
                                ? "bg-primary/20 border-primary"
                                : !formData.callsign.trim()
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
                        );
                      })}
                    </div>
                    {!formData.callsign.trim() && (
                      <p className="text-xs text-muted-foreground mt-2">
                        Enter a callsign first to select a background.
                      </p>
                    )}
                    {touched.background && formData.callsign.trim() && !formData.backgroundId && (
                      <p className="mt-2 text-xs text-destructive">Background selection is required</p>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Step 2: Level */}
          {step === "level" && (
            <Card>
              <CardHeader>
                <CardTitle>Step 2: License Level</CardTitle>
                <CardDescription>
                  {STEP_META.level.description}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <LevelSelector
                  value={formData.level}
                  onChange={handleLevelChange}
                />

                <div className="flex gap-3 mt-6">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setStep("background")}
                  >
                    Back
                  </Button>
                  <Button
                    type="button"
                    onClick={() =>
                      setStep(formData.level >= 1 ? "licenses" : "triggers")
                    }
                  >
                    Next: {formData.level >= 1 ? "Licenses" : "Triggers"}
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Step 3: Licenses (LL1+ only) */}
          {step === "licenses" && formData.level >= 1 && licenses && (
            <Card>
              <CardHeader>
                <CardTitle>Step 3: Licenses</CardTitle>
                <CardDescription>
                  {STEP_META.licenses.description}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {touched.licenses && !canProceedFromLicenses && (
                  <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                    Allocate exactly {progression.licensePoints} license point{progression.licensePoints > 1 ? "s" : ""} to continue.
                  </div>
                )}

                <div onMouseLeave={() => markTouched("licenses")}>
                  <LicenseSelector
                    licenses={licenses}
                    allocations={formData.licenseAllocations}
                    availablePoints={progression.licensePoints}
                    onChange={(allocations) =>
                      setFormData((prev) => ({
                        ...prev,
                        licenseAllocations: allocations,
                      }))
                    }
                  />
                </div>

                <div className="flex gap-3 mt-6">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setStep("level")}
                  >
                    Back
                  </Button>
                  <Button
                    type="button"
                    onClick={() => setStep("triggers")}
                    disabled={!canProceedFromLicenses}
                  >
                    Next: Triggers
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Triggers */}
          {step === "triggers" && (
            <Card>
              <CardHeader>
                <CardTitle>Triggers</CardTitle>
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
                {touched.triggers && !canProceedFromTriggers && (
                  <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                    Select exactly 4 triggers to continue.
                  </div>
                )}

                <div
                  className="grid gap-2 max-h-96 overflow-y-auto"
                  onMouseLeave={() => markTouched("triggers")}
                >
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
                    onClick={() =>
                      setStep(formData.level >= 1 ? "licenses" : "level")
                    }
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

          {/* Skills */}
          {step === "skills" && (
            <Card>
              <CardHeader>
                <CardTitle>Mech Skills</CardTitle>
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

                <div
                  className="mt-4 text-sm"
                  onMouseLeave={() => markTouched("skills")}
                >
                  <span
                    className={
                      canProceedFromSkills
                        ? "text-primary"
                        : touched.skills
                        ? "text-destructive"
                        : "text-muted-foreground"
                    }
                  >
                    Total: {totalSkillPoints} / {progression.skillPoints} points
                  </span>
                  {touched.skills && !canProceedFromSkills && (
                    <span className="text-destructive ml-2">
                      (Must be exactly {progression.skillPoints} for LL
                      {formData.level})
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

          {/* Talents */}
          {step === "talents" && (
            <Card>
              <CardHeader>
                <CardTitle>Talents</CardTitle>
                <CardDescription>{STEP_META.talents.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="mb-4 rounded-md border border-border bg-muted/50 px-3 py-2">
                  <div className="text-xs text-muted-foreground">
                    Selected: {formData.talents.length} / {progression.talentPoints}{" "}
                    talents
                  </div>
                </div>
                {touched.talents && !canProceedFromTalents && (
                  <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                    Select exactly {progression.talentPoints} talents to continue.
                  </div>
                )}

                <div
                  className="grid gap-2 max-h-96 overflow-y-auto"
                  onMouseLeave={() => markTouched("talents")}
                >
                  {allTalents?.map((talent) => {
                    const isSelected = formData.talents.includes(talent.id);
                    const isFull =
                      formData.talents.length >= progression.talentPoints;
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

          {/* Pilot Gear */}
          {step === "gear" && (
            <Card>
              <CardHeader>
                <CardTitle>Pilot Gear</CardTitle>
                <CardDescription>{STEP_META.gear.description}</CardDescription>
              </CardHeader>
              <CardContent>
                {touched.clothing && !canProceedFromGear && (
                  <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                    Clothing is required before moving on.
                  </div>
                )}
                <div className="space-y-6">
                  <div>
                    <div className="text-xs text-muted-foreground uppercase tracking-wide mb-2">
                      Clothing (Required) <span className="text-destructive">*</span>
                    </div>
                    <div
                      className="grid gap-2"
                      onMouseLeave={() => markTouched("clothing")}
                    >
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

          {/* Mech */}
          {step === "mech" && frames && licenses && (
            <Card>
              <CardHeader>
                <CardTitle>Your Mech{formData.level > 0 ? "s" : ""}</CardTitle>
                <CardDescription>{STEP_META.mech.description}</CardDescription>
              </CardHeader>
              <CardContent>
                {touched.mech && !canProceedFromMech && (
                  <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                    Select at least one mech to continue.
                  </div>
                )}

                <div onMouseLeave={() => markTouched("mech")}>
                <FrameBrowser
                  frames={frames}
                  licenses={licenses}
                  allocations={formData.licenseAllocations}
                  level={formData.level}
                  selectedMechs={formData.mechs}
                  onAddMech={(frameId, name) =>
                    setFormData((prev) => ({
                      ...prev,
                      mechs: [...prev.mechs, { frameId, name }],
                    }))
                  }
                  onRemoveMech={(index) =>
                    setFormData((prev) => ({
                      ...prev,
                      mechs: prev.mechs.filter((_, i) => i !== index),
                    }))
                  }
                  onUpdateMechName={(index, name) =>
                    setFormData((prev) => ({
                      ...prev,
                      mechs: prev.mechs.map((m, i) =>
                        i === index ? { ...m, name } : m
                      ),
                    }))
                  }
                />
                </div>

                <div className="mt-6">
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
                      <strong>Level:</strong> LL{formData.level}
                    </li>
                    <li>
                      <strong>Background:</strong> {formData.backgroundName || "None"}
                    </li>
                    {formData.licenseAllocations.length > 0 && (
                      <li>
                        <strong>Licenses:</strong>{" "}
                        {formData.licenseAllocations
                          .map((a) => {
                            const lic = licenses.find((l) => l.id === a.licenseId);
                            return `${lic?.name ?? a.licenseId} ${a.rank}`;
                          })
                          .join(", ")}
                      </li>
                    )}
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
                    <li>
                      <strong>Mechs:</strong>{" "}
                      {formData.mechs.length
                        ? formData.mechs
                            .map((m) => {
                              const frame = frames.find((f) => f.id === m.frameId);
                              return `${m.name || frame?.name || m.frameId}`;
                            })
                            .join(", ")
                        : "None"}
                    </li>
                  </ul>
                </div>

                <div className="flex gap-3 mt-4">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setStep("gear")}
                  >
                    Back
                  </Button>
                  <Button
                    type="button"
                    onClick={handleSubmit}
                    disabled={
                      createMutation.isPending ||
                      addMechMutation.isPending ||
                      !canProceedFromMech
                    }
                  >
                    {createMutation.isPending || addMechMutation.isPending
                      ? "Creating..."
                      : "Create Character"}
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
              <CardDescription>
                LL{formData.level} requirements before deployment
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <ChecklistItem
                label="Callsign + background"
                detail={
                  formData.callsign && formData.backgroundName
                    ? "Locked"
                    : "Required"
                }
                ok={canProceedFromBackground}
              />
              {formData.level >= 1 && (
                <ChecklistItem
                  label={`${progression.licensePoints} license points`}
                  detail={`${totalLicensePoints} / ${progression.licensePoints}`}
                  ok={canProceedFromLicenses}
                />
              )}
              <ChecklistItem
                label="4 triggers"
                detail={`${formData.triggers.length} selected`}
                ok={canProceedFromTriggers}
              />
              <ChecklistItem
                label={`${progression.skillPoints} HASE points`}
                detail={`${totalSkillPoints} / ${progression.skillPoints}`}
                ok={canProceedFromSkills}
              />
              <ChecklistItem
                label={`${progression.talentPoints} talents`}
                detail={`${formData.talents.length} selected`}
                ok={canProceedFromTalents}
              />
              <ChecklistItem
                label="Pilot clothing"
                detail={formData.pilotGear.clothing ? "Selected" : "Missing"}
                ok={canProceedFromGear}
              />
              <ChecklistItem
                label="Mech"
                detail={`${formData.mechs.length} selected`}
                ok={canProceedFromMech}
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
                <li>
                  LL{formData.level} requires 4 triggers at +2 and exactly{" "}
                  {progression.skillPoints} HASE points.
                </li>
                {formData.level >= 1 && (
                  <li>
                    Allocate {progression.licensePoints} license point
                    {progression.licensePoints > 1 ? "s" : ""} to unlock
                    manufacturer frames.
                  </li>
                )}
                <li>Talents start at Rank I; you can respec later.</li>
                <li>Clothing is mandatory; armor is optional.</li>
                <li>Weapons and gear are capped to keep loadouts lean.</li>
                {formData.level >= 1 && (
                  <li>
                    LL{formData.level} pilots can have up to{" "}
                    {formData.level + 1} mechs.
                  </li>
                )}
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
