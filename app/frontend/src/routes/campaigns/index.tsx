import { useState } from "react";
import { Link, createFileRoute, useNavigate } from "@tanstack/react-router";

import {
  useCampaigns,
  useCreateCampaign,
  CampaignSummary,
} from "../../lib/api";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Button,
} from "../../components/ui";

export const Route = createFileRoute("/campaigns/")({
  component: CampaignListPage,
});

function CampaignListPage() {
  const navigate = useNavigate();
  const { data, isLoading } = useCampaigns();
  const createMutation = useCreateCampaign();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!name.trim()) return;

    createMutation.mutate(
      { name: name.trim(), description: description.trim() || undefined },
      {
        onSuccess: (campaign) => {
          setName("");
          setDescription("");
          navigate({
            to: "/campaigns/$campaignId",
            params: { campaignId: campaign.id },
          });
        },
      },
    );
  };

  const campaigns = data?.items ?? [];

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-8">
      <div className="space-y-2">
        <h1 className="text-3xl font-heading font-semibold text-foreground">
          Campaigns
        </h1>
        <p className="text-muted-foreground">
          Organize parties, invite players, and prepare lobbies before missions.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Create Campaign</CardTitle>
          <CardDescription>Start a new squad or GM prep board.</CardDescription>
        </CardHeader>
        <CardContent>
          <form className="space-y-4" onSubmit={handleSubmit}>
            <div className="space-y-2">
              <label
                htmlFor="campaign-name"
                className="text-sm font-medium text-foreground"
              >
                Name
              </label>
              <input
                id="campaign-name"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="Operation Dawn"
                required
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              />
            </div>
            <div className="space-y-2">
              <label
                htmlFor="campaign-description"
                className="text-sm font-medium text-foreground"
              >
                Description
              </label>
              <input
                id="campaign-description"
                value={description}
                onChange={(event) => setDescription(event.target.value)}
                placeholder="Union Auxiliaries on Ras Shamra"
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              />
            </div>
            <Button type="submit" disabled={createMutation.isPending}>
              {createMutation.isPending ? "Creating..." : "Create Campaign"}
            </Button>
            {createMutation.isError && (
              <p className="text-sm text-destructive">
                {(createMutation.error as Error).message}
              </p>
            )}
          </form>
        </CardContent>
      </Card>

      <div className="space-y-4">
        <h2 className="text-xl font-semibold text-foreground">My Campaigns</h2>
        {isLoading && <p className="text-muted-foreground">Loading campaigns...</p>}
        {!isLoading && campaigns.length === 0 && (
          <p className="text-muted-foreground">
            You have no campaigns yet. Create one above to get started.
          </p>
        )}
        <div className="grid gap-4 md:grid-cols-2">
          {campaigns.map((campaign) => (
            <CampaignCard key={campaign.id} campaign={campaign} />
          ))}
        </div>
      </div>
    </div>
  );
}

function CampaignCard({ campaign }: { campaign: CampaignSummary }) {
  return (
    <Card className="h-full border-border/60">
      <CardHeader>
        <CardTitle className="text-lg text-foreground flex items-center justify-between">
          <span>{campaign.name}</span>
          <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
            {campaign.membership_role}
          </span>
        </CardTitle>
        <CardDescription>{campaign.description || "No description"}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="text-sm text-muted-foreground flex gap-4">
          <span>{campaign.member_count} members</span>
          <span>{campaign.character_count} characters</span>
        </div>
        <div className="text-xs text-muted-foreground space-y-1">
          <div>
            Outcomes: {campaign.mission_summary.successful_missions} success /{" "}
            {campaign.mission_summary.partial_missions} partial /{" "}
            {campaign.mission_summary.failed_missions} failure
          </div>
          <div>
            Last mission: {campaign.mission_summary.last_outcome ?? "None"}
          </div>
        </div>
        <Link
          to="/campaigns/$campaignId"
          params={{ campaignId: campaign.id }}
          className="inline-flex text-sm text-primary hover:underline"
        >
          View campaign
        </Link>
      </CardContent>
    </Card>
  );
}
