import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";

import {
  usePreviewCampaignInvite,
  useAcceptCampaignInvite,
} from "../../lib/api";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Button,
} from "../../components/ui";

export const Route = createFileRoute("/invites/$token")({
  component: InviteLandingPage,
});

function InviteLandingPage() {
  const { token } = Route.useParams();
  const previewQuery = usePreviewCampaignInvite(token);
  const acceptInvite = useAcceptCampaignInvite();
  const [accepted, setAccepted] = useState(false);

  const handleAccept = () => {
    if (!token) return;
    acceptInvite.mutate(token, {
      onSuccess: () => setAccepted(true),
    });
  };

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-6">
      <div className="space-y-1 text-center">
        <p className="text-sm uppercase tracking-wide text-muted-foreground">
          Campaign Invite
        </p>
        <h1 className="text-3xl font-heading font-semibold text-foreground">
          {previewQuery.data?.campaign_name ?? "Loading invite"}
        </h1>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Join the table</CardTitle>
          <CardDescription>
            {previewQuery.isLoading && "Checking invite details..."}
            {previewQuery.isError && "Unable to load invite. Double-check the link."}
            {previewQuery.data && previewQuery.data.patron
              ? `Sponsored by ${previewQuery.data.patron}`
              : undefined}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {previewQuery.data && (
            <div className="space-y-2 text-sm">
              <div>
                <span className="font-medium text-foreground">Squad:</span>{" "}
                {previewQuery.data.squad_name || previewQuery.data.campaign_name}
              </div>
              <div>
                <span className="font-medium text-foreground">Role:</span>{" "}
                {previewQuery.data.role}
              </div>
              <div>
                <span className="font-medium text-foreground">Seat guidance:</span>{" "}
                {previewQuery.data.ready_players}/{previewQuery.data.preferred_pilots} pilots ready
              </div>
              {previewQuery.data.seat_warning && (
                <div className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-amber-600">
                  {previewQuery.data.seat_warning}
                </div>
              )}
            </div>
          )}
          <div className="flex gap-2">
            <Button
              type="button"
              className="flex-1"
              disabled={accepted || acceptInvite.isPending || !previewQuery.data || !previewQuery.data.can_join}
              onClick={handleAccept}
            >
              {accepted ? "Joined" : acceptInvite.isPending ? "Joining..." : "Accept invite"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
