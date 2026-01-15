import { useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";

import {
  useCampaign,
  useCreateCampaignInvite,
  useAttachCampaignCharacter,
  useUpdateCampaignMemberSettings,
} from "../../lib/api";
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

  const [characterId, setCharacterId] = useState("");
  const [inviteRole, setInviteRole] = useState<"player" | "co_gm">("player");
  const [inviteEmail, setInviteEmail] = useState("");

  const sortedInvites = useMemo(
    () => [...(data?.invites ?? [])].sort((a, b) => b.created_at.localeCompare(a.created_at)),
    [data?.invites],
  );

  if (isLoading || !data) {
    return (
      <div className="p-6">
        <p className="text-muted-foreground">Loading campaign...</p>
      </div>
    );
  }

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

  const copyInvite = async (token: string) => {
    try {
      await navigator.clipboard.writeText(token);
    } catch (_) {
      // ignore clipboard failures
    }
  };

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
                className="flex items-center justify-between border border-border/50 rounded-lg px-3 py-2"
              >
                <div>
                  <div className="font-medium text-foreground">
                    {member.user_id}
                  </div>
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
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Attach Character</CardTitle>
            <CardDescription>
              Paste a character ID to associate it with this campaign.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form className="space-y-3" onSubmit={handleAttach}>
              <div className="space-y-2">
                <label
                  htmlFor="character-id"
                  className="text-sm font-medium text-foreground"
                >
                  Character ID
                </label>
                <input
                  id="character-id"
                  value={characterId}
                  onChange={(event) => setCharacterId(event.target.value)}
                  placeholder="char_xxxxx"
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
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
              <p className="text-sm text-muted-foreground">
                Attach a character to see it here.
              </p>
            )}
            {data.characters.map((character) => (
              <div
                key={character.id}
                className="flex items-center justify-between rounded-lg border border-border/50 px-3 py-2"
              >
                <div>
                  <div className="font-medium text-foreground">
                    {character.callsign}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {character.character_id}
                  </div>
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
                <label
                  htmlFor="invite-role"
                  className="text-sm font-medium text-foreground"
                >
                  Role
                </label>
                <select
                  id="invite-role"
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                  value={inviteRole}
                  onChange={(event) =>
                    setInviteRole(event.target.value as "player" | "co_gm")
                  }
                >
                  <option value="player">Player</option>
                  <option value="co_gm">Co-GM</option>
                </select>
              </div>
              <div className="space-y-2">
                <label
                  htmlFor="invite-email"
                  className="text-sm font-medium text-foreground"
                >
                  Email / memo (optional)
                </label>
                <input
                  id="invite-email"
                  value={inviteEmail}
                  onChange={(event) => setInviteEmail(event.target.value)}
                  placeholder="friend@example.com"
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
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
                <div
                  key={invite.id}
                  className="border border-border/60 rounded-lg px-3 py-2 space-y-1"
                >
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
                  <div className="flex gap-2">
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      onClick={() => copyInvite(invite.token)}
                    >
                      Copy token
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
