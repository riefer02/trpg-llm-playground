/**
 * Home page with system health status.
 *
 * Demonstrates:
 * - React Query for data fetching
 * - Custom UI components
 * - API integration
 */

import { createFileRoute } from "@tanstack/react-router";
import { useHealth, useDatabaseHealth } from "../lib/api";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "../components/ui";

export const Route = createFileRoute("/" as const)({
  component: HomePage,
});

function HomePage() {
  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2 text-foreground">
          Welcome to Lancer Combat
        </h1>
        <p className="text-muted-foreground">
          A web application for running Lancer TTRPG tactical combat encounters.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <SystemHealthCard />
        <DatabaseHealthCard />
      </div>

      <div className="mt-8">
        <h2 className="text-xl font-semibold mb-4 text-foreground">
          Quick Actions
        </h2>
        <div className="grid gap-4 md:grid-cols-3">
          <QuickActionCard
            title="Create Pilot"
            description="Build a new pilot character"
            href="/pilots/new"
          />
          <QuickActionCard
            title="Mech Loadout"
            description="Configure mech equipment"
            href="/mechs"
          />
          <QuickActionCard
            title="Start Combat"
            description="Begin a tactical encounter"
            href="/combat"
          />
        </div>
      </div>
    </div>
  );
}

function SystemHealthCard() {
  const { data, isLoading, error } = useHealth();

  return (
    <Card>
      <CardHeader>
        <CardTitle>System Status</CardTitle>
        <CardDescription>API server health</CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading && <div className="text-muted-foreground">Checking...</div>}
        {error && (
          <div className="flex items-center gap-2">
            <StatusDot status="error" />
            <span className="text-destructive">Offline - {error.message}</span>
          </div>
        )}
        {data && (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <StatusDot
                status={data.status === "healthy" ? "success" : "error"}
              />
              <span>API: {data.status}</span>
            </div>
            <div className="text-muted-foreground">Version: {data.version}</div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function DatabaseHealthCard() {
  const { data, isLoading, error } = useDatabaseHealth();

  return (
    <Card>
      <CardHeader>
        <CardTitle>Database Status</CardTitle>
        <CardDescription>PostgreSQL connection</CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading && <div className="text-muted-foreground">Checking...</div>}
        {error && (
          <div className="flex items-center gap-2">
            <StatusDot status="error" />
            <span className="text-destructive">Connection failed</span>
          </div>
        )}
        {data && (
          <div className="flex items-center gap-2">
            <StatusDot
              status={data.status === "healthy" ? "success" : "error"}
            />
            <span>
              {data.status === "healthy" ? "Connected" : data.database}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function StatusDot({ status }: { status: "success" | "error" | "warning" }) {
  const colorClass = {
    success: "bg-horus",
    error: "bg-destructive",
    warning: "bg-accent",
  }[status];

  return <div className={`w-3 h-3 rounded-full ${colorClass}`} />;
}

function QuickActionCard({
  title,
  description,
  href,
}: {
  title: string;
  description: string;
  href: string;
}) {
  return (
    <a href={href} className="block">
      <Card className="h-full transition-colors hover:border-primary cursor-pointer">
        <CardHeader>
          <CardTitle className="text-lg">{title}</CardTitle>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
      </Card>
    </a>
  );
}
