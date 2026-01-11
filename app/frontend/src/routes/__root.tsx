/**
 * Root layout component.
 * 
 * Sets up:
 * - React Query provider for data fetching
 * - Global styles
 * - Header/navigation
 * - Dev tools (in development)
 */

import { HeadContent, Scripts, createRootRoute } from '@tanstack/react-router'
import { TanStackRouterDevtoolsPanel } from '@tanstack/react-router-devtools'
import { TanStackDevtools } from '@tanstack/react-devtools'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState } from 'react'

import appCss from '../styles.css?url'

// Create QueryClient with sensible defaults
function makeQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 60 * 1000, // 1 minute
        gcTime: 5 * 60 * 1000, // 5 minutes (previously cacheTime)
        refetchOnWindowFocus: false,
        retry: 2,
        retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      },
      mutations: {
        retry: 1,
      },
    },
  })
}

// Singleton for browser, new instance for SSR
let browserQueryClient: QueryClient | undefined

function getQueryClient() {
  if (typeof window === 'undefined') {
    // Server: always make a new client
    return makeQueryClient()
  }
  // Browser: reuse singleton
  if (!browserQueryClient) {
    browserQueryClient = makeQueryClient()
  }
  return browserQueryClient
}

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      { title: 'Lancer Combat' },
      { name: 'description', content: 'Lancer TTRPG web application' },
    ],
    links: [
      { rel: 'stylesheet', href: appCss },
      { rel: 'icon', href: '/favicon.ico' },
    ],
  }),
  shellComponent: RootDocument,
})

function RootDocument({ children }: { children: React.ReactNode }) {
  // Create query client once per component lifecycle
  const [queryClient] = useState(() => getQueryClient())

  return (
    <html lang="en">
      <head>
        <HeadContent />
      </head>
      <body>
        <QueryClientProvider client={queryClient}>
          <div className="min-h-screen flex flex-col">
            <Header />
            <main className="flex-1">
              {children}
            </main>
          </div>
          <TanStackDevtools
            config={{ position: 'bottom-right' }}
            plugins={[
              {
                name: 'TanStack Router',
                render: <TanStackRouterDevtoolsPanel />,
              },
            ]}
          />
        </QueryClientProvider>
        <Scripts />
      </body>
    </html>
  )
}

function Header() {
  return (
    <header className="border-b border-border bg-card px-6 py-4">
      <div className="flex items-center justify-between max-w-7xl mx-auto">
        <div className="flex items-center gap-2">
          <span className="text-xl font-bold text-primary">LANCER</span>
          <span className="text-sm text-muted-foreground">Combat</span>
        </div>
        <nav className="flex gap-4">
          <a href="/" className="text-foreground hover:underline">
            Home
          </a>
          <a href="/characters" className="text-muted-foreground hover:underline">
            Characters
          </a>
          <a href="/compendium" className="text-muted-foreground hover:underline">
            Compendium
          </a>
          <a href="/pilots" className="text-muted-foreground hover:underline">
            Pilots
          </a>
        </nav>
      </div>
    </header>
  )
}
