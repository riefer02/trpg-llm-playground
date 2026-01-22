import { Card, CardContent, CardHeader, Skeleton } from '../ui'

/**
 * Skeleton loader for the compendium browser page.
 * Matches the two-column layout with sidebar filters and card grid.
 */
export function CompendiumSkeleton() {
  return (
    <div className="px-6 py-8 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <section className="dashboard-surface p-6">
        <Skeleton className="h-4 w-28 mb-3" />
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <Skeleton className="h-8 w-36 mb-2" />
            <Skeleton className="h-4 w-96 max-w-full" />
          </div>
          <Skeleton className="h-6 w-28 rounded-full" />
        </div>
        {/* Stats row */}
        <div className="mt-4 grid gap-3 md:grid-cols-4">
          {[1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="rounded-lg border border-border bg-muted/40 p-3"
            >
              <Skeleton className="h-3 w-16 mb-1" />
              <Skeleton className="h-6 w-12" />
            </div>
          ))}
        </div>
      </section>

      {/* Two-column layout */}
      <div className="grid gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
        {/* Left sidebar */}
        <aside className="space-y-4 lg:sticky lg:top-6 h-fit">
          <Card>
            <CardHeader>
              <Skeleton className="h-6 w-20" />
              <Skeleton className="h-4 w-44" />
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Search */}
              <Skeleton className="h-10 w-full" />
              {/* Category buttons */}
              <div className="flex flex-wrap gap-2">
                {[1, 2, 3, 4].map((i) => (
                  <Skeleton key={i} className="h-8 w-24" />
                ))}
              </div>
              {/* Filters */}
              <div className="space-y-3">
                {[1, 2, 3].map((i) => (
                  <div key={i}>
                    <Skeleton className="h-4 w-24 mb-1" />
                    <Skeleton className="h-10 w-full" />
                  </div>
                ))}
              </div>
              <Skeleton className="h-8 w-28" />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <Skeleton className="h-6 w-28" />
              <Skeleton className="h-4 w-36" />
            </CardHeader>
            <CardContent className="space-y-2">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-4/5" />
              <Skeleton className="h-4 w-5/6" />
            </CardContent>
          </Card>
        </aside>

        {/* Main content grid */}
        <section className="space-y-4">
          {/* Results count */}
          <div className="flex items-center justify-between">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-4 w-64" />
          </div>

          {/* Card grid */}
          <div className="grid gap-4 md:grid-cols-2">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <Card key={i}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <Skeleton className="h-6 w-32" />
                    <Skeleton className="h-5 w-16" />
                  </div>
                  <Skeleton className="h-4 w-40" />
                </CardHeader>
                <CardContent className="space-y-3">
                  {/* Stats row */}
                  <div className="grid grid-cols-2 gap-2">
                    {[1, 2, 3, 4].map((j) => (
                      <Skeleton key={j} className="h-4 w-20" />
                    ))}
                  </div>
                  {/* Tags */}
                  <div className="flex flex-wrap gap-2">
                    {[1, 2, 3].map((j) => (
                      <Skeleton key={j} className="h-6 w-16 rounded-full" />
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>
      </div>
    </div>
  )
}
