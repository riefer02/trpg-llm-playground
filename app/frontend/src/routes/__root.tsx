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
          <div className="app-shell flex flex-col">
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
    <header className="border-b border-border bg-card/80 backdrop-blur px-6 py-4">
      <div className="flex items-center justify-between max-w-7xl mx-auto">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-full bg-primary/10 border border-primary/30 flex items-center justify-center font-heading text-primary font-semibold">
            LC
          </div>
          <div>
            <div className="text-lg font-heading font-semibold text-foreground">
              Lancer Control
            </div>
            <div className="text-xs text-muted-foreground">
              Operations Console
            </div>
          </div>
        </div>
        <nav className="flex gap-2 text-sm" aria-label="Primary navigation">
          <a
            href="/"
            className="px-3 py-1.5 rounded-full border border-transparent text-foreground hover:bg-muted"
          >
            Home
          </a>
          <a
            href="/characters"
            className="px-3 py-1.5 rounded-full border border-transparent text-muted-foreground hover:bg-muted hover:text-foreground"
          >
            Characters
          </a>
          <a
            href="/compendium"
            className="px-3 py-1.5 rounded-full border border-transparent text-muted-foreground hover:bg-muted hover:text-foreground"
          >
            Compendium
          </a>
        </nav>
      </div>
    </header>
  )
}
