/**
 * Breadcrumbs navigation component.
 *
 * Automatically builds navigation trail from current route.
 * Shows on detail pages (paths with IDs or nested routes).
 */

import { useMatches, Link } from "@tanstack/react-router";

// Map route segments to human-readable labels
const ROUTE_LABELS: Record<string, string> = {
  characters: "Characters",
  campaigns: "Campaigns",
  combat: "Combat",
  compendium: "Compendium",
  pilots: "Pilots",
  new: "Create New",
  edit: "Edit",
  export: "Export",
};

// Patterns for dynamic route params to skip or format
const PARAM_PATTERNS = [
  /^\$characterId$/,
  /^\$campaignId$/,
  /^\$combatId$/,
  /^\$pilotId$/,
];

interface BreadcrumbItem {
  label: string;
  href: string;
  isCurrent: boolean;
}

export function Breadcrumbs() {
  const matches = useMatches();

  // Build breadcrumb items from route matches
  const breadcrumbs: BreadcrumbItem[] = [];

  // Skip root match, process remaining
  for (let i = 1; i < matches.length; i++) {
    const match = matches[i];
    const routeId = match.routeId;

    // Skip index routes and special patterns
    if (routeId === "/" || routeId.endsWith("/")) continue;

    // Extract path segments from route ID
    // Route IDs look like: "/characters/$characterId" or "/campaigns/$campaignId/edit"
    const segments = routeId.split("/").filter(Boolean);
    const lastSegment = segments[segments.length - 1];

    // Skip if this is just a param placeholder (we'll use context for labels)
    const isParamRoute = PARAM_PATTERNS.some((p) => p.test(lastSegment));

    // Get label - either from map or from the route context/loaderData
    let label: string;
    if (isParamRoute) {
      // Try to get a meaningful label from loader data or context
      const loaderData = match.loaderData as Record<string, unknown> | undefined;
      const contextData = match.context as Record<string, unknown> | undefined;

      // Look for common name fields
      label =
        (loaderData?.name as string) ||
        (loaderData?.callsign as string) ||
        (loaderData?.title as string) ||
        (contextData?.name as string) ||
        (contextData?.callsign as string) ||
        lastSegment.replace("$", "");
    } else {
      label = ROUTE_LABELS[lastSegment] || formatSegment(lastSegment);
    }

    // Build the href by joining segments up to this point
    const href = "/" + segments.join("/").replace(/\$\w+/g, (param) => {
      const paramName = param.slice(1); // Remove $
      return (match.params as Record<string, string>)[paramName] || param;
    });

    const isCurrent = i === matches.length - 1;

    breadcrumbs.push({ label, href, isCurrent });
  }

  // Don't show breadcrumbs on root pages (home, characters list, campaigns list)
  if (breadcrumbs.length <= 1) {
    return null;
  }

  return (
    <nav
      aria-label="Breadcrumb navigation"
      className="px-6 py-2 bg-muted/30 border-b border-border/50 text-sm"
    >
      <ol className="flex items-center gap-1 max-w-7xl mx-auto">
        <li>
          <Link
            to="/"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            Home
          </Link>
        </li>
        {breadcrumbs.map((crumb, index) => (
          <li key={crumb.href} className="flex items-center gap-1">
            <span className="text-muted-foreground/60 mx-1">/</span>
            {crumb.isCurrent ? (
              <span className="text-foreground font-medium" aria-current="page">
                {crumb.label}
              </span>
            ) : (
              <Link
                to={crumb.href}
                className="text-muted-foreground hover:text-foreground transition-colors"
              >
                {crumb.label}
              </Link>
            )}
          </li>
        ))}
      </ol>
    </nav>
  );
}

function formatSegment(segment: string): string {
  // Convert kebab-case or camelCase to Title Case
  return segment
    .replace(/[-_]/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
