import { Card, CardContent, CardHeader, Skeleton } from '../ui'

/**
 * Skeleton loader for the combat session page.
 * Matches the two-column layout with canvas area and sidebar.
 */
export function CombatSessionSkeleton() {
  return (
    <div className="px-6 py-8 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <section className="dashboard-surface p-6">
        <Skeleton className="h-4 w-32 mb-3" />
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <Skeleton className="h-8 w-48 mb-2" />
            <Skeleton className="h-4 w-36" />
          </div>
          <div className="flex items-center gap-4">
            <Skeleton className="h-10 w-28" />
            <div className="flex flex-col items-end gap-1">
              <Skeleton className="h-4 w-12" />
              <Skeleton className="h-3 w-36" />
            </div>
          </div>
        </div>
      </section>

      {/* Two-column layout */}
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
        {/* Combat Canvas */}
        <Card className="h-full">
          <CardHeader>
            <Skeleton className="h-6 w-32" />
            <Skeleton className="h-4 w-72" />
          </CardHeader>
          <CardContent>
            <div className="rounded-md border border-border bg-muted/30 p-3">
              {/* Canvas placeholder */}
              <Skeleton className="h-[520px] w-full" />
              {/* Coordinates display */}
              <div className="mt-3 flex flex-wrap gap-4">
                <Skeleton className="h-4 w-24" />
                <Skeleton className="h-4 w-28" />
                <Skeleton className="h-4 w-28" />
              </div>
              {/* Terrain legend */}
              <div className="mt-3 flex flex-wrap gap-2">
                {[1, 2, 3, 4, 5].map((i) => (
                  <Skeleton key={i} className="h-6 w-20" />
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Right sidebar */}
        <div className="space-y-4">
          {/* Turn Controls */}
          <Card>
            <CardHeader className="py-3">
              <Skeleton className="h-5 w-28" />
              <Skeleton className="h-4 w-36" />
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-center justify-between">
                <Skeleton className="h-5 w-24" />
                <Skeleton className="h-5 w-20" />
              </div>
              <div className="flex gap-2">
                <Skeleton className="h-10 w-28" />
                <Skeleton className="h-10 w-24" />
              </div>
            </CardContent>
          </Card>

          {/* Victory Conditions */}
          <Card>
            <CardHeader className="py-3">
              <Skeleton className="h-5 w-36" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-4 w-full mb-2" />
              <Skeleton className="h-4 w-3/4" />
            </CardContent>
          </Card>

          {/* Action Log */}
          <Card>
            <CardHeader className="py-3">
              <Skeleton className="h-5 w-24" />
              <Skeleton className="h-3 w-56" />
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="max-h-48 space-y-2">
                {[1, 2, 3, 4].map((i) => (
                  <div key={i} className="rounded-md border border-border px-3 py-2">
                    <Skeleton className="h-4 w-32 mb-1" />
                    <Skeleton className="h-3 w-48" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Combatants */}
          <Card>
            <CardHeader className="py-3">
              <Skeleton className="h-5 w-28" />
              <Skeleton className="h-3 w-52" />
            </CardHeader>
            <CardContent className="space-y-2">
              {[1, 2, 3, 4].map((i) => (
                <div
                  key={i}
                  className="flex items-center justify-between rounded-md border border-border px-3 py-2"
                >
                  <div>
                    <Skeleton className="h-5 w-24 mb-1" />
                    <Skeleton className="h-3 w-20" />
                  </div>
                  <Skeleton className="h-4 w-12" />
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
