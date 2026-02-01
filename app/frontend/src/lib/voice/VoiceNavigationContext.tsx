/**
 * Context for voice navigation to allow screens to handle specific intents.
 */

import { createContext, useContext, useState, useEffect, useCallback, useMemo, ReactNode } from 'react'
import { NavigationIntent } from './navigation'

type IntentHandler = (intent: NavigationIntent) => boolean // return true if handled

interface VoiceNavigationContextValue {
  currentIntent: NavigationIntent | null
  registerHandler: (handler: IntentHandler) => () => void
  unregisterHandler: (handler: IntentHandler) => void
}

const VoiceNavigationContext = createContext<VoiceNavigationContextValue | undefined>(undefined)

interface VoiceNavigationProviderProps {
  children: ReactNode
  intent: NavigationIntent | null
  onIntent?: (intent: NavigationIntent) => void
}

export function VoiceNavigationProvider({
  children,
  intent,
  onIntent,
}: VoiceNavigationProviderProps) {
  const [handlers, setHandlers] = useState<IntentHandler[]>([])

  const unregisterHandler = useCallback((handler: IntentHandler) => {
    setHandlers(prev => prev.filter(h => h !== handler))
  }, [])

  const registerHandler = useCallback((handler: IntentHandler) => {
    setHandlers(prev => [...prev, handler])
    return () => unregisterHandler(handler)
  }, [unregisterHandler])
  
  // When intent changes, notify handlers
  useEffect(() => {
    if (!intent) return
    
    // Give screen-specific handlers first chance
    let handled = false
    for (const handler of handlers) {
      if (handler(intent)) {
        handled = true
        break
      }
    }
    
    // If no handler consumed the intent, call the default onIntent
    if (!handled && onIntent) {
      onIntent(intent)
    }
  }, [intent, handlers, onIntent])
  
  const value = useMemo<VoiceNavigationContextValue>(() => ({
    currentIntent: intent,
    registerHandler,
    unregisterHandler,
  }), [intent, registerHandler, unregisterHandler])

  return (
    <VoiceNavigationContext.Provider value={value}>
      {children}
    </VoiceNavigationContext.Provider>
  )
}

export function useVoiceNavigationContext() {
  const context = useContext(VoiceNavigationContext)
  if (context === undefined) {
    throw new Error('useVoiceNavigationContext must be used within a VoiceNavigationProvider')
  }
  return context
}