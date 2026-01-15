import { useEffect, useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";

import {
  useCampaign,
  useCreateCampaignInvite,
  useAttachCampaignCharacter,
  useUpdateCampaignMemberSettings,
  useUpdateCampaignIdentity,
  useUpdateCampaignLobby,
  useLaunchCampaignMission,
  useUpdateSessionLifecycle,
  useRecordCampaignSessionOutcome,
  useRevokeCampaignInvite,
  useResendCampaignInvite,
} from "../../lib/api";
import type {
  Campaign as LancerCampaign,
  Session,
  MissionObjectiveBrief,
  ReservePlanEntry,
  MissionOutcomeReport,
} from "../../lib/types/lancer";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Button,
} from "../../components/ui";

export const Route = createFileRoute("/campaigns/$campaignId")({
  component: CampaignDetailPage,
});

function CampaignDetailPage() {
  const { campaignId } = Route.useParams();
  const { data, isLoading } = useCampaign(campaignId);
  const createInvite = useCreateCampaignInvite();
  const attachCharacter = useAttachCampaignCharacter();
  const updateMemberSettings = useUpdateCampaignMemberSettings();
  const updateIdentity = useUpdateCampaignIdentity();
  const updateLobby = useUpdateCampaignLobby();
  const launchMission = useLaunchCampaignMission();
  const updateLifecycle = useUpdateSessionLifecycle();
  const recordSessionOutcome = useRecordCampaignSessionOutcome();
  const revokeInvite = useRevokeCampaignInvite();
  const resendInvite = useResendCampaignInvite();

  const [characterId, setCharacterId] = useState("");
  const [inviteRole, setInviteRole] = useState<"player" | "co_gm">("player");
  const [inviteEmail, setInviteEmail] = useState("");
  const [identityForm, setIdentityForm] = useState({
    squad_name: "",
    patron: "",
    who_we_are: "",
  });
  const [lobbyForm, setLobbyForm] = useState({
    mission_name: "",
    briefing_notes: "",
    stakes_summary: "",
    stakes_type: "personal" as "personal" | "faction" | "immediate" | "gradual" | "custom",
    assigned_member_ids: [] as string[],
    preferred_pilot_count: 4,
    min_pilot_count: 3,
    objectives: [] as MissionObjectiveBrief[],
    support_assets: [] as string[],
    reserves: [] as ReservePlanEntry[],
  });
  const [launchNotes, setLaunchNotes] = useState("");
  const [outcomeDrafts, setOutcomeDrafts] = useState<
    Record<string, { outcome: MissionOutcomeReport["outcome"]; completion_score: number; debrief_notes: string }>
  >({});

  const sortedInvites = useMemo(
    () =>
      [...(data?.invites ?? [])].sort((a, b) =>
        b.created_at.localeCompare(a.created_at),
      ),
    [data?.invites],
  );

  if (isLoading || !data) {
    return (
      <div className="p-6">
        <p className="text-muted-foreground">Loading campaign...</p>
      </div>
    );
  }

  const campaignModel = data.data as LancerCampaign;
  const lobbyState = campaignModel.lobby_state;
  const readiness = data.readiness_summary;

  useEffect(() => {
    setIdentityForm({
      squad_name: campaignModel.identity?.squad_name ?? "",
      patron: campaignModel.identity?.patron ?? "",
      who_we_are: campaignModel.identity?.who_we_are ?? "",
    });
  }, [campaignModel.identity?.patron, campaignModel.identity?.squad_name, campaignModel.identity?.who_we_are]);

  useEffect(() => {
    setLobbyForm({
      mission_name: lobbyState?.mission_plan?.mission_name ?? "",
      briefing_notes: lobbyState?.mission_plan?.briefing_notes ?? "",
      stakes_summary: lobbyState?.mission_plan?.stakes?.summary ?? "",
      stakes_type: lobbyState?.mission_plan?.stakes?.stakes_type ?? "personal",
      assigned_member_ids: lobbyState?.assigned_member_ids ?? [],
      preferred_pilot_count: lobbyState?.preferred_pilot_count ?? readiness.preferred_pilots,
      min_pilot_count: lobbyState?.min_pilot_count ?? readiness.min_pilots,
      objectives: lobbyState?.mission_plan?.objectives?.map((objective) => ({
        ...objective,
      })) ?? [],
      support_assets: [...(lobbyState?.mission_plan?.support_assets ?? [])],
      reserves: lobbyState?.mission_plan?.reserves?.map((reserve) => ({
        ...reserve,
      })) ?? [],
    });
  }, [
    lobbyState?.assigned_member_ids,
    lobbyState?.mission_plan,
    lobbyState?.min_pilot_count,
    lobbyState?.preferred_pilot_count,
    readiness.min_pilots,
    readiness.preferred_pilots,
  ]);

  const handleInvite = (event: React.FormEvent) => {
    event.preventDefault();
    createInvite.mutate({
      campaignId,
      data: {
        role: inviteRole,
        invited_email: inviteEmail || undefined,
      },
    });
    setInviteEmail("");
  };

  const handleAttach = (event: React.FormEvent) => {
    event.preventDefault();
    if (!characterId.trim()) return;
    attachCharacter.mutate({
      campaignId,
      data: { character_id: characterId.trim() },
    });
    setCharacterId("");
  };

  const handleToggleReady = (memberId: string, readyState: string) => {
    updateMemberSettings.mutate({
      campaignId,
      memberId,
      data: { ready: readyState !== "ready" },
    });
  };

  const handleAssignCharacter = (memberId: string, assignedId: string) => {
    updateMemberSettings.mutate({
      campaignId,
      memberId,
      data: { assigned_character_id: assignedId || null },
    });
  };

  const handleIdentitySubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    updateIdentity.mutate({
      campaignId,
      data: {
        squad_name: identityForm.squad_name.trim() || undefined,
        patron: identityForm.patron.trim() || undefined,
        who_we_are: identityForm.who_we_are.trim() || undefined,
      },
    });
  };

  const addObjectiveField = () => {
    setLobbyForm((prev) => ({
      ...prev,
      objectives: [
        ...prev.objectives,
        {
          id: `obj-${Date.now()}`,
          title: "",
          success_condition: "",
          priority: "primary",
          related_objective_id: null,
        },
      ],
    }));
  };

  const updateObjectiveField = (
    index: number,
    field: keyof MissionObjectiveBrief,
    value: string | null,
  ) => {
    setLobbyForm((prev) => {
      const next = [...prev.objectives];
      next[index] = {
        ...next[index],
        [field]: value,
      } as MissionObjectiveBrief;
      return { ...prev, objectives: next };
    });
  };

  const removeObjectiveField = (index: number) => {
    setLobbyForm((prev) => {
      const next = prev.objectives.filter((_, idx) => idx !== index);
      return { ...prev, objectives: next };
    });
  };

  const addSupportAsset = () => {
    setLobbyForm((prev) => ({
      ...prev,
      support_assets: [...prev.support_assets, ""],
    }));
  };

  const updateSupportAsset = (index: number, value: string) => {
    setLobbyForm((prev) => {
      const next = [...prev.support_assets];
      next[index] = value;
      return { ...prev, support_assets: next };
    });
  };

  const removeSupportAsset = (index: number) => {
    setLobbyForm((prev) => ({
      ...prev,
      support_assets: prev.support_assets.filter((_, idx) => idx !== index),
    }));
  };

  const addReserveRow = () => {
    setLobbyForm((prev) => ({
      ...prev,
      reserves: [
        ...prev.reserves,
        {
          reserve_id: "",
          assigned_pilot_id: null,
          usage_notes: null,
          status: "planned",
        },
      ],
    }));
  };

  const updateReserveRow = (
    index: number,
    field: keyof ReservePlanEntry,
    value: string | null,
  ) => {
    setLobbyForm((prev) => {
      const next = [...prev.reserves];
      next[index] = {
        ...next[index],
        [field]: value,
      } as ReservePlanEntry;
      return { ...prev, reserves: next };
    });
  };

  const removeReserveRow = (index: number) => {
    setLobbyForm((prev) => ({
      ...prev,
      reserves: prev.reserves.filter((_, idx) => idx !== index),
    }));
  };

  const handleLobbySubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    updateLobby.mutate({
      campaignId,
      data: {
        mission_name: lobbyForm.mission_name || "Untitled Mission",
        briefing_notes: lobbyForm.briefing_notes || undefined,
        stakes: lobbyForm.stakes_summary
          ? {
              stakes_type: lobbyForm.stakes_type,
              summary: lobbyForm.stakes_summary,
            }
          : undefined,
        objectives: lobbyForm.objectives.map((objective, index) => ({
          id: objective.id || `obj-${index + 1}`,
          title: objective.title || `Objective ${index + 1}`,
          success_condition:
            objective.success_condition || objective.title || "Objective",
          priority: objective.priority ?? "primary",
          related_objective_id: objective.related_objective_id ?? null,
        })),
        support_assets: lobbyForm.support_assets.filter((asset) => asset.trim().length > 0),
        reserves: lobbyForm.reserves.filter((reserve) => reserve.reserve_id.trim().length > 0),
        assigned_member_ids: lobbyForm.assigned_member_ids,
        preferred_pilot_count: lobbyForm.preferred_pilot_count,
        min_pilot_count: lobbyForm.min_pilot_count,
      },
    });
  };

  const handleAssignedMemberToggle = (memberId: string) => {
    setLobbyForm((prev) => {
      const selected = prev.assigned_member_ids.includes(memberId)
        ? prev.assigned_member_ids.filter((id) => id !== memberId)
        : [...prev.assigned_member_ids, memberId];
      return { ...prev, assigned_member_ids: selected };
    });
  };

  const handleLaunchMission = () => {
    launchMission.mutate({
      campaignId,
      data: {
        environment: "standard",
        notes: launchNotes || undefined,
      },
    });
  };

  const handleLifecycleComplete = (sessionRecord: Session) => {
    updateLifecycle.mutate({
      campaignId,
      sessionId: sessionRecord.id,
      data: { phase: "mission", status: "complete" },
    });
  };

  const updateOutcomeDraft = (
    sessionId: string,
    field: "outcome" | "completion_score" | "debrief_notes",
    value: string,
  ) => {
    setOutcomeDrafts((prev) => {
      const draft = prev[sessionId] ?? {
        outcome: "success" as MissionOutcomeReport["outcome"],
        completion_score: 1,
        debrief_notes: "",
      };
      const nextValue =
        field === "completion_score" ? Number(value) : value;
      return {
        ...prev,
        [sessionId]: {
          ...draft,
          [field]: nextValue,
        },
      };
    });
  };

  const handleRecordOutcome = (sessionId: string) => {
    const draft = outcomeDrafts[sessionId] ?? {
      outcome: "success" as MissionOutcomeReport["outcome"],
      completion_score: 1,
      debrief_notes: "",
    };
    recordSessionOutcome.mutate({
      campaignId,
      sessionId,
      data: {
        outcome: draft.outcome,
        completion_score: draft.completion_score,
        debrief_notes: draft.debrief_notes || undefined,
      },
    });
  };

  const copyInvite = async (token: string) => {
    try {
      await navigator.clipboard.writeText(token);
    } catch (_) {
      // ignore clipboard errors
    }
  };

  const playerMembers = data.members.filter((member) => member.role === "player");

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-8">
      <div className="space-y-2">
        <p className="text-sm uppercase tracking-wide text-muted-foreground">
          Campaign
        </p>
        <h1 className="text-3xl font-heading font-semibold text-foreground">
          {data.name}
        </h1>
        <p className="text-muted-foreground max-w-2xl">{data.description}</p>
      </div>

      {data.seat_warning && (
        <div className="border border-destructive/40 bg-destructive/10 text-destructive rounded-lg px-3 py-2 text-sm">
          {data.seat_warning}
        </div>
      )}

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Squad Identity</CardTitle>
            <CardDescription>Keep onboarding prompts in sync.</CardDescription>
          </CardHeader>
          <CardContent>
            <form className="space-y-3" onSubmit={handleIdentitySubmit}>
              <div className="space-y-1">
                <label className="text-sm font-medium text-foreground">Squad name</label>
                <input
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                  value={identityForm.squad_name}
                  onChange={(event) =>
                    setIdentityForm((prev) => ({
                      ...prev,
                      squad_name: event.target.value,
                    }))
                  }
                  placeholder="ThirdComm Auxilia"
                />
              </div>
              <div className="space-y-1">
                <label className="text-sm font-medium text-foreground">Patron</label>
                <input
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                  value={identityForm.patron}
                  onChange={(event) =>
                    setIdentityForm((prev) => ({
                      ...prev,
                      patron: event.target.value,
                    }))
                  }
                  placeholder="Union Navy"
                />
              </div>
              <div className="space-y-1">
                <label className="text-sm font-medium text-foreground">Who we are</label>
                <textarea
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                  rows={3}
                  value={identityForm.who_we_are}
                  onChange={(event) =>
                    setIdentityForm((prev) => ({
                      ...prev,
                      who_we_are: event.target.value,
                    }))
                  }
                  placeholder="LL0 freelancers hunting interstellar salvage"
                />
              </div>
              <Button type="submit" disabled={updateIdentity.isPending}>
                {updateIdentity.isPending ? "Saving..." : "Save identity"}
              </Button>
            </form>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Ready Check</CardTitle>
            <CardDescription>
              {readiness.ready_players} ready pilots / {readiness.preferred_pilots} preferred
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-2 gap-3 text-sm">
              <StatTile label="Members" value={readiness.total_members.toString()} />
              <StatTile label="Ready" value={readiness.ready_members.toString()} />
              <StatTile
                label="Ready Pilots"
                value={`${readiness.ready_players}/${readiness.preferred_pilots}`}
              />
              <StatTile label="Assigned" value={readiness.assigned_ready_players.toString()} />
            </div>
            {readiness.issues.length > 0 && (
              <div className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-600 space-y-1">
                {readiness.issues.map((issue) => (
                  <div key={issue}>• {issue}</div>
                ))}
              </div>
            )}
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Launch notes</label>
              <textarea
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                rows={3}
                value={launchNotes}
                onChange={(event) => setLaunchNotes(event.target.value)}
                placeholder="Mission reminders, intel, or operator notes"
              />
            </div>
            <Button
              type="button"
              disabled={!readiness.can_launch || launchMission.isPending}
              onClick={handleLaunchMission}
            >
              {launchMission.isPending ? "Launching..." : "Launch mission"}
            </Button>
            {readiness.lobby_status && (
              <p className="text-xs text-muted-foreground">
                Lobby status: {readiness.lobby_status}
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Mission Lobby</CardTitle>
          <CardDescription>
            Capture briefs, stakes, and assignments before starting combat.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form className="space-y-4" onSubmit={handleLobbySubmit}>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Mission name</label>
                <input
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                  value={lobbyForm.mission_name}
                  onChange={(event) =>
                    setLobbyForm((prev) => ({
                      ...prev,
                      mission_name: event.target.value,
                    }))
                  }
                  placeholder="Operation Dawn"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Stakes summary</label>
                <input
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                  value={lobbyForm.stakes_summary}
                  onChange={(event) =>
                    setLobbyForm((prev) => ({
                      ...prev,
                      stakes_summary: event.target.value,
                    }))
                  }
                  placeholder="Keep the colony online"
                />
                <select
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                  value={lobbyForm.stakes_type}
                  onChange={(event) =>
                    setLobbyForm((prev) => ({
                      ...prev,
                      stakes_type: event.target.value as typeof prev.stakes_type,
                    }))
                  }
                >
                  <option value="personal">Personal</option>
                  <option value="faction">Faction</option>
                  <option value="immediate">Immediate</option>
                  <option value="gradual">Gradual</option>
                  <option value="custom">Custom</option>
                </select>
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Briefing notes</label>
              <textarea
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                rows={3}
                value={lobbyForm.briefing_notes}
                onChange={(event) =>
                  setLobbyForm((prev) => ({
                    ...prev,
                    briefing_notes: event.target.value,
                  }))
                }
                placeholder="Objectives, intel, or reserves"
              />
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Objectives</label>
                <div className="space-y-3">
                  {lobbyForm.objectives.length === 0 && (
                    <p className="text-sm text-muted-foreground">
                      No objectives yet. Add at least one so the table knows the plan.
                    </p>
                  )}
                  {lobbyForm.objectives.map((objective, index) => (
                    <div key={objective.id || `objective-${index}`} className="rounded-md border border-border/60 p-3 space-y-2">
                      <input
                        className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                        placeholder={`Objective ${index + 1}`}
                        value={objective.title}
                        onChange={(event) =>
                          updateObjectiveField(index, "title", event.target.value)
                        }
                      />
                      <input
                        className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                        placeholder="Success condition"
                        value={objective.success_condition}
                        onChange={(event) =>
                          updateObjectiveField(index, "success_condition", event.target.value)
                        }
                      />
                      <select
                        className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                        value={objective.priority ?? "primary"}
                        onChange={(event) =>
                          updateObjectiveField(index, "priority", event.target.value)
                        }
                      >
                        <option value="primary">Primary</option>
                        <option value="secondary">Secondary</option>
                        <option value="optional">Optional</option>
                      </select>
                      <div className="flex justify-end">
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={() => removeObjectiveField(index)}
                        >
                          Remove
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
                <Button type="button" variant="outline" size="sm" onClick={addObjectiveField}>
                  Add objective
                </Button>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Support assets</label>
                <div className="space-y-2">
                  {lobbyForm.support_assets.length === 0 && (
                    <p className="text-sm text-muted-foreground">List NPC squads, intel, or logistics support.</p>
                  )}
                  {lobbyForm.support_assets.map((asset, index) => (
                    <div key={`asset-${index}`} className="flex gap-2">
                      <input
                        className="flex-1 rounded-md border border-border bg-background px-3 py-2 text-sm"
                        placeholder="Talos flight"
                        value={asset}
                        onChange={(event) => updateSupportAsset(index, event.target.value)}
                      />
                      <Button type="button" variant="ghost" size="sm" onClick={() => removeSupportAsset(index)}>
                        Remove
                      </Button>
                    </div>
                  ))}
                </div>
                <Button type="button" variant="outline" size="sm" onClick={addSupportAsset}>
                  Add support asset
                </Button>
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Reserves & contingencies</label>
              <div className="space-y-2">
                {lobbyForm.reserves.length === 0 && (
                  <p className="text-sm text-muted-foreground">No reserves earmarked for this mission.</p>
                )}
                {lobbyForm.reserves.map((reserve, index) => (
                  <div key={`reserve-${index}`} className="grid gap-2 rounded-md border border-border/60 p-3 md:grid-cols-4">
                    <input
                      className="rounded-md border border-border bg-background px-3 py-2 text-sm"
                      placeholder="Reserve ID"
                      value={reserve.reserve_id}
                      onChange={(event) => updateReserveRow(index, "reserve_id", event.target.value)}
                    />
                    <input
                      className="rounded-md border border-border bg-background px-3 py-2 text-sm"
                      placeholder="Assigned pilot"
                      value={reserve.assigned_pilot_id ?? ""}
                      onChange={(event) => updateReserveRow(index, "assigned_pilot_id", event.target.value)}
                    />
                    <select
                      className="rounded-md border border-border bg-background px-3 py-2 text-sm"
                      value={reserve.status ?? "planned"}
                      onChange={(event) => updateReserveRow(index, "status", event.target.value)}
                    >
                      <option value="planned">Planned</option>
                      <option value="spent">Spent</option>
                      <option value="earned">Earned</option>
                    </select>
                    <div className="flex items-center gap-2">
                      <input
                        className="flex-1 rounded-md border border-border bg-background px-3 py-2 text-sm"
                        placeholder="Usage notes"
                        value={reserve.usage_notes ?? ""}
                        onChange={(event) => updateReserveRow(index, "usage_notes", event.target.value)}
                      />
                      <Button type="button" variant="ghost" size="sm" onClick={() => removeReserveRow(index)}>
                        Remove
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
              <Button type="button" variant="outline" size="sm" onClick={addReserveRow}>
                Add reserve plan
              </Button>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Assigned pilots</label>
                <div className="space-y-1 text-sm">
                  {playerMembers.length === 0 && (
                    <p className="text-muted-foreground text-sm">No player members yet.</p>
                  )}
                  {playerMembers.map((member) => (
                    <label key={member.id} className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        className="rounded border-border"
                        checked={lobbyForm.assigned_member_ids.includes(member.id)}
                        onChange={() => handleAssignedMemberToggle(member.id)}
                      />
                      <span>
                        {member.user_id} ({member.ready_state})
                      </span>
                    </label>
                  ))}
                </div>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Seat limits</label>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <span className="text-xs text-muted-foreground">Min pilots</span>
                    <input
                      type="number"
                      min={1}
                      max={6}
                      className="w-full rounded-md border border-border bg-background px-2 py-1 text-sm"
                      value={lobbyForm.min_pilot_count}
                      onChange={(event) =>
                        setLobbyForm((prev) => ({
                          ...prev,
                          min_pilot_count: Number(event.target.value),
                        }))
                      }
                    />
                  </div>
                  <div>
                    <span className="text-xs text-muted-foreground">Preferred</span>
                    <input
                      type="number"
                      min={1}
                      max={6}
                      className="w-full rounded-md border border-border bg-background px-2 py-1 text-sm"
                      value={lobbyForm.preferred_pilot_count}
                      onChange={(event) =>
                        setLobbyForm((prev) => ({
                          ...prev,
                          preferred_pilot_count: Number(event.target.value),
                        }))
                      }
                    />
                  </div>
                </div>
              </div>
            </div>
            <Button type="submit" disabled={updateLobby.isPending}>
              {updateLobby.isPending ? "Saving lobby..." : "Save lobby"}
            </Button>
            {lobbyState && (
              <p className="text-xs text-muted-foreground">
                Current status: {lobbyState.status}
              </p>
            )}
          </form>
        </CardContent>
      </Card>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Members</CardTitle>
            <CardDescription>
              {data.members.length} total • status: {data.status}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {data.members.map((member) => (
              <div
                key={member.id}
                className="flex flex-col gap-2 border border-border/50 rounded-lg px-3 py-2"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-foreground">{member.user_id}</div>
                    <div className="text-xs text-muted-foreground">
                      {member.role} • ready: {member.ready_state}
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleToggleReady(member.id, member.ready_state)}
                    disabled={updateMemberSettings.isPending}
                  >
                    Toggle ready
                  </Button>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  <label className="text-xs text-muted-foreground">Assigned</label>
                  <select
                    className="flex-1 rounded-md border border-border bg-background px-2 py-1 text-sm"
                    value={member.assigned_character_id ?? ""}
                    onChange={(event) =>
                      handleAssignCharacter(member.id, event.target.value)
                    }
                  >
                    <option value="">-- None --</option>
                    {data.characters
                      .filter((character) => character.user_id === member.user_id)
                      .map((character) => (
                        <option key={character.character_id} value={character.character_id}>
                          {character.callsign}
                        </option>
                      ))}
                  </select>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Attach Character</CardTitle>
            <CardDescription>Paste an ID to associate it with this campaign.</CardDescription>
          </CardHeader>
          <CardContent>
            <form className="space-y-3" onSubmit={handleAttach}>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Character ID</label>
                <input
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                  value={characterId}
                  onChange={(event) => setCharacterId(event.target.value)}
                  placeholder="char_xxxxx"
                />
              </div>
              <Button type="submit" disabled={attachCharacter.isPending}>
                {attachCharacter.isPending ? "Attaching..." : "Attach"}
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Characters</CardTitle>
            <CardDescription>
              {data.characters.length || "No"} linked characters
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {data.characters.length === 0 && (
              <p className="text-sm text-muted-foreground">Attach a character to see it here.</p>
            )}
            {data.characters.map((character) => (
              <div key={character.id} className="flex items-center justify-between rounded-lg border border-border/50 px-3 py-2">
                <div>
                  <div className="font-medium text-foreground">{character.callsign}</div>
                  <div className="text-xs text-muted-foreground">{character.character_id}</div>
                </div>
                <span className="px-2 py-0.5 text-xs rounded-full bg-muted text-muted-foreground">
                  {character.role}
                </span>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Invites</CardTitle>
            <CardDescription>Share tokens with other users.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <form className="space-y-3" onSubmit={handleInvite}>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Role</label>
                <select
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                  value={inviteRole}
                  onChange={(event) => setInviteRole(event.target.value as "player" | "co_gm")}
                >
                  <option value="player">Player</option>
                  <option value="co_gm">Co-GM</option>
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Email / memo (optional)</label>
                <input
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                  value={inviteEmail}
                  onChange={(event) => setInviteEmail(event.target.value)}
                  placeholder="friend@example.com"
                />
              </div>
              <Button type="submit" disabled={createInvite.isPending}>
                {createInvite.isPending ? "Creating invite..." : "Generate invite"}
              </Button>
            </form>

            <div className="space-y-2">
              {sortedInvites.length === 0 && (
                <p className="text-sm text-muted-foreground">No invites yet.</p>
              )}
              {sortedInvites.map((invite) => (
                <div key={invite.id} className="border border-border/60 rounded-lg px-3 py-2 space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="font-medium text-sm">{invite.role}</div>
                    <span
                      className={`px-2 py-0.5 text-xs rounded-full ${
                        invite.status === "pending"
                          ? "bg-primary/10 text-primary"
                          : "bg-muted text-muted-foreground"
                      }`}
                    >
                      {invite.status}
                    </span>
                  </div>
                  <div className="text-xs font-mono break-all">{invite.token}</div>
                  {invite.redeemed_by_user_id && (
                    <p className="text-xs text-muted-foreground">
                      Accepted by {invite.redeemed_by_user_id}
                    </p>
                  )}
                  <div className="flex flex-wrap gap-2">
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      onClick={() => copyInvite(invite.token)}
                    >
                      Copy token
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant="ghost"
                      disabled={invite.status !== "pending" || revokeInvite.isPending}
                      onClick={() =>
                        revokeInvite.mutate({
                          campaignId,
                          inviteId: invite.id,
                        })
                      }
                    >
                      Revoke
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant="secondary"
                      disabled={resendInvite.isPending}
                      onClick={() =>
                        resendInvite.mutate({
                          campaignId,
                          inviteId: invite.id,
                        })
                      }
                    >
                      Resend
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Session History</CardTitle>
          <CardDescription>Downtime → Brief → Prep → Mission → Debrief checkpoints.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {campaignModel.sessions.length === 0 && (
            <p className="text-sm text-muted-foreground">No sessions yet.</p>
          )}
          {campaignModel.sessions.map((sessionRecord) => (
            <div key={sessionRecord.id} className="border border-border/50 rounded-lg px-3 py-2 space-y-2">
              <div className="flex items-center justify-between text-sm">
                <div>
                  Session {sessionRecord.session_number}
                  {sessionRecord.mission_plan?.mission_name
                    ? ` • ${sessionRecord.mission_plan.mission_name}`
                    : ""}
                </div>
                {sessionRecord.lifecycle_checkpoints?.some(
                  (checkpoint) =>
                    checkpoint.phase === "mission" && checkpoint.status !== "complete",
                ) && (
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={() => handleLifecycleComplete(sessionRecord as Session)}
                    disabled={updateLifecycle.isPending}
                  >
                    Mark mission complete
                  </Button>
                )}
              </div>
              <div className="flex flex-wrap gap-2 text-xs">
                {sessionRecord.lifecycle_checkpoints?.map((checkpoint) => (
                  <span
                    key={`${sessionRecord.id}-${checkpoint.phase}`}
                    className={`px-2 py-1 rounded-full border text-xs ${
                      checkpoint.status === "complete"
                        ? "border-primary/40 text-primary"
                        : checkpoint.status === "in_progress"
                          ? "border-amber-500/40 text-amber-500"
                          : "border-border/60 text-muted-foreground"
                    }`}
                  >
                    {checkpoint.phase}: {checkpoint.status}
                  </span>
                ))}
              </div>
              {sessionRecord.mission_outcome ? (
                <div className="rounded-md border border-border/60 bg-muted/30 px-3 py-2 text-sm text-muted-foreground">
                  Outcome: {sessionRecord.mission_outcome.outcome} •{" "}
                  {sessionRecord.mission_outcome.debrief_notes || "No notes recorded"}
                </div>
              ) : (
                <form
                  className="space-y-2"
                  onSubmit={(event) => {
                    event.preventDefault();
                    handleRecordOutcome(sessionRecord.id);
                  }}
                >
                  <div className="grid gap-2 md:grid-cols-2">
                    <select
                      className="rounded-md border border-border bg-background px-3 py-2 text-sm"
                      value={outcomeDrafts[sessionRecord.id]?.outcome ?? "success"}
                      onChange={(event) =>
                        updateOutcomeDraft(sessionRecord.id, "outcome", event.target.value)
                      }
                    >
                      <option value="success">Success</option>
                      <option value="partial">Partial</option>
                      <option value="failure">Failure</option>
                      <option value="catastrophic">Catastrophic</option>
                    </select>
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.1}
                      className="rounded-md border border-border bg-background px-3 py-2 text-sm"
                      value={outcomeDrafts[sessionRecord.id]?.completion_score ?? 1}
                      onChange={(event) =>
                        updateOutcomeDraft(sessionRecord.id, "completion_score", event.target.value)
                      }
                      placeholder="Completion score (0-1)"
                    />
                  </div>
                  <textarea
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                    rows={2}
                    placeholder="Debrief notes"
                    value={outcomeDrafts[sessionRecord.id]?.debrief_notes ?? ""}
                    onChange={(event) =>
                      updateOutcomeDraft(sessionRecord.id, "debrief_notes", event.target.value)
                    }
                  />
                  <Button
                    type="submit"
                    size="sm"
                    disabled={recordSessionOutcome.isPending}
                  >
                    {recordSessionOutcome.isPending ? "Saving..." : "Record outcome"}
                  </Button>
                </form>
              )}
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-border/60 rounded-lg px-3 py-2">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-lg font-semibold text-foreground">{value}</div>
    </div>
  );
}
