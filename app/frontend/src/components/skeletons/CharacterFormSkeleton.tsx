import { Card, CardContent, CardHeader, Skeleton } from '../ui'

/**
 * Skeleton loader for the character creation form.
 * Matches the three-column layout with nav, main content, and sidebar.
 */
export function CharacterFormSkeleton() {
  return (
    <div className="px-6 py-8 max-w-7xl mx-auto space-y-6">
      {/* Dashboard header */}
      <section className="dashboard-surface p-6">
        <Skeleton className="h-4 w-32 mb-3" />
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <Skeleton className="h-8 w-56 mb-2" />
            <Skeleton className="h-4 w-72" />
          </div>
          <Skeleton className="h-6 w-36 rounded-full" />
        </div>
        {/* Stats row */}
        <div className="mt-4 grid gap-4 md:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="p-3 rounded-lg border border-border bg-muted/40">
              <Skeleton className="h-3 w-20 mb-1" />
              <Skeleton className="h-6 w-28 mb-1" />
              <Skeleton className="h-3 w-24" />
            </div>
          ))}
        </div>
        {/* Progress bar */}
        <div className="mt-4 h-2 w-full rounded-full bg-border overflow-hidden">
          <Skeleton className="h-full w-1/4" />
        </div>
      </section>

      {/* Three-column layout */}
      <div className="grid gap-6 lg:grid-cols-[240px_minmax(0,1fr)_320px]">
        {/* Left nav */}
        <nav className="space-y-3">
          <Skeleton className="h-3 w-24 mb-2" />
          {[1, 2, 3, 4, 5, 6, 7].map((i) => (
            <div key={i} className="px-3 py-2 rounded-lg border border-border">
              <div className="flex items-center justify-between mb-1">
                <Skeleton className="h-4 w-20" />
                <Skeleton className="h-4 w-14" />
              </div>
              <Skeleton className="h-3 w-28" />
            </div>
          ))}
        </nav>

        {/* Main content */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <Skeleton className="h-6 w-32" />
              <Skeleton className="h-4 w-64" />
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Skeleton className="h-4 w-20 mb-2" />
                <Skeleton className="h-10 w-full" />
              </div>
              <div>
                <Skeleton className="h-4 w-24 mb-2" />
                <Skeleton className="h-10 w-full" />
              </div>
              <div>
                <Skeleton className="h-4 w-36 mb-2" />
                <div className="grid gap-2 max-h-80">
                  {[1, 2, 3, 4, 5, 6].map((i) => (
                    <div key={i} className="p-3 border border-border rounded-md">
                      <Skeleton className="h-5 w-36 mb-1" />
                      <Skeleton className="h-3 w-56" />
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right sidebar */}
        <aside className="space-y-4 lg:sticky lg:top-6 h-fit">
          <Card>
            <CardHeader>
              <Skeleton className="h-6 w-40" />
              <Skeleton className="h-4 w-52" />
            </CardHeader>
            <CardContent className="space-y-3">
              {[1, 2, 3, 4, 5, 6, 7].map((i) => (
                <div key={i} className="flex items-start justify-between gap-3">
                  <div>
                    <Skeleton className="h-4 w-32 mb-1" />
                    <Skeleton className="h-3 w-20" />
                  </div>
                  <Skeleton className="h-5 w-16 rounded-full" />
                </div>
              ))}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <Skeleton className="h-6 w-28" />
              <Skeleton className="h-4 w-44" />
            </CardHeader>
            <CardContent className="space-y-2">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-5/6" />
              <Skeleton className="h-4 w-4/5" />
            </CardContent>
          </Card>
        </aside>
      </div>
    </div>
  )
}
