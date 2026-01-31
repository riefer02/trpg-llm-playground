/**
 * Manager component that provides voice navigation context and floating button.
 */

import { useLocation } from '@tanstack/react-router'
import { useSettings } from '../../lib/hooks/useSettings'
import { useVoiceNavigation } from '../../lib/voice/navigation'
import { VoiceNavigationProvider } from '../../lib/voice/VoiceNavigationContext'
import { VoiceNavigationFloatingButton } from './VoiceNavigationFloatingButton'
import { toast } from 'sonner'
import { useNavigate } from '@tanstack/react-router'

export function VoiceNavigationManager({ children }: { children: React.ReactNode }) {
  const location = useLocation()
  const navigate = useNavigate()
  const { settings } = useSettings()
  const isCombatRoute = location.pathname.startsWith('/combat/')
  
  // Use voice navigation hook (disabled in combat)
  const {
    intent,
    isListening,
    transcript,
    error,
    parseTranscript,
    reset,
  } = useVoiceNavigation({
    autoParse: true,
    enabled: !isCombatRoute && settings.enableVoiceInput,
    onIntentParsed: (intent) => {
      // Default handling for navigation intents
      switch (intent.type) {
        case 'navigate':
          if (intent.targetPath) {
            navigate({ to: intent.targetPath })
            toast.success(`Navigating to ${intent.target}`, {
              description: `Voice command: "${intent.rawCommand}"`,
            })
          }
          break
        case 'back':
          navigate({ to: '..' })
          toast.info(`Going back`, {
            description: `Voice command: "${intent.rawCommand}"`,
          })
          break
        case 'launch':
        case 'read_briefing':
        case 'select':
          // These are screen-specific - let screen handlers handle them
          break
        default:
          break
      }
    },
  })
  
  return (
    <VoiceNavigationProvider intent={intent}>
      {children}
      <VoiceNavigationFloatingButton
        isListening={isListening}
        transcript={transcript}
        error={error}
        reset={reset}
      />
    </VoiceNavigationProvider>
  )
}