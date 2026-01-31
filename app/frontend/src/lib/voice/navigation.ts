/**
 * Voice navigation parser and hook.
 * 
 * Handles voice commands for navigating game screens outside of combat.
 */

import { useNavigate, useLocation } from '@tanstack/react-router'
import { useSpeechRecognition } from './speech-to-text'
import { useSettings } from '../hooks/useSettings'
import { useCallback, useEffect, useState, useRef } from 'react'

// Screen name to path mapping
const SCREEN_PATHS: Record<string, string> = {
  'home': '/',
  'title': '/',
  'quarters': '/quarters',
  'missions': '/missions',
  'settings': '/settings',
  'pilot': '/quarters/pilot',
  'mech': '/quarters/mech',
  'compendium': '/compendium',
  'characters': '/characters',
  'campaigns': '/campaigns',
}

// Alias mapping for voice commands
const SCREEN_ALIASES: Record<string, string[]> = {
  'home': ['home', 'main', 'title'],
  'quarters': ['quarters', 'hub', 'menu', 'main menu'],
  'missions': ['missions', 'mission select', 'mission list'],
  'settings': ['settings', 'options', 'preferences'],
  'pilot': ['pilot', 'pilot details'],
  'mech': ['mech', 'mech loadout', 'loadout'],
  'compendium': ['compendium', 'reference', 'database'],
  'characters': ['characters', 'pilots', 'characters list'],
  'campaigns': ['campaigns', 'campaign list'],
}

// Reverse mapping from alias to screen name
const ALIAS_TO_SCREEN: Record<string, string> = {}
Object.entries(SCREEN_ALIASES).forEach(([screen, aliases]) => {
  aliases.forEach(alias => {
    ALIAS_TO_SCREEN[alias] = screen
  })
})

export interface NavigationIntent {
  type: 'navigate' | 'back' | 'launch' | 'read_briefing' | 'select' | 'unknown'
  target?: string
  targetPath?: string
  rawCommand: string
  confidence: number
}

/**
 * Parse a voice command into a navigation intent.
 * Returns null if command doesn't match navigation patterns.
 */
export function parseNavigationIntent(transcript: string): NavigationIntent | null {
  const lower = transcript.toLowerCase().trim()
  
  // 1. "go to [screen]" or "open [screen]" or "[screen]"
  const goToMatch = lower.match(/^(?:go to|open|navigate to|show)\s+(.+)$/) || 
    lower.match(/^(.+)$/)
  if (goToMatch) {
    const targetPhrase = goToMatch[1].trim()
    // Check if targetPhrase matches any alias
    const screen = ALIAS_TO_SCREEN[targetPhrase]
    if (screen && SCREEN_PATHS[screen]) {
      return {
        type: 'navigate',
        target: screen,
        targetPath: SCREEN_PATHS[screen],
        rawCommand: transcript,
        confidence: 0.9,
      }
    }
    // Try direct match with screen paths (e.g., "missions")
    if (SCREEN_PATHS[targetPhrase]) {
      return {
        type: 'navigate',
        target: targetPhrase,
        targetPath: SCREEN_PATHS[targetPhrase],
        rawCommand: transcript,
        confidence: 0.9,
      }
    }
  }
  
  // 2. "back" or "go back" or "previous"
  if (lower.match(/^(back|go back|previous|return)$/)) {
    return {
      type: 'back',
      rawCommand: transcript,
      confidence: 1.0,
    }
  }
  
  // 3. "launch" or "launch mission"
  if (lower.match(/^(launch|launch mission|start mission|deploy)$/)) {
    return {
      type: 'launch',
      rawCommand: transcript,
      confidence: 0.8,
    }
  }
  
  // 4. "read briefing" or "read mission briefing"
  if (lower.match(/^(read briefing|read mission briefing|read the briefing)$/)) {
    return {
      type: 'read_briefing',
      rawCommand: transcript,
      confidence: 0.9,
    }
  }
  
  // 5. "select [item]" - generic selection (will need context)
  const selectMatch = lower.match(/^select\s+(.+)$/)
  if (selectMatch) {
    const item = selectMatch[1].trim()
    return {
      type: 'select',
      target: item,
      rawCommand: transcript,
      confidence: 0.7,
    }
  }
  
  return null
}

export interface UseVoiceNavigationOptions {
  /** Whether to enable automatic parsing when transcript changes (default: true) */
  autoParse?: boolean
  /** Callback when a navigation intent is parsed */
  onIntentParsed?: (intent: NavigationIntent) => void
  /** Whether speech recognition is enabled (default: true) */
  enabled?: boolean
}

export interface UseVoiceNavigationReturn {
  /** Current navigation intent, if any */
  intent: NavigationIntent | null
  /** Whether voice navigation is currently active (listening) */
  isListening: boolean
  /** Latest transcript */
  transcript: string
  /** Speech recognition error, if any */
  error: string | null
  /** Manually parse a transcript string */
  parseTranscript: (text: string) => NavigationIntent | null
  /** Clear current intent and transcript */
  reset: () => void
}

/**
 * React hook for voice navigation across all screens.
 * Uses speech recognition and parses navigation intents.
 */
export function useVoiceNavigation(
  options: UseVoiceNavigationOptions = {}
): UseVoiceNavigationReturn {
  const {
    autoParse = true,
    onIntentParsed,
    enabled = true,
  } = options
  
  const navigate = useNavigate()
  const location = useLocation()
  const { settings } = useSettings()
  
  // Use speech recognition (same as combat)
  const speechRecognition = useSpeechRecognition({
    language: settings.voiceLanguage,
    continuous: false,
    interimResults: true,
    enabled,
  })
  
  const [intent, setIntent] = useState<NavigationIntent | null>(null)
  const lastParsedTranscriptRef = useRef('')
  
  // Parse transcript when it changes
  useEffect(() => {
    if (!autoParse) return
    if (!speechRecognition.isListening && 
        speechRecognition.transcript && 
        speechRecognition.transcript !== lastParsedTranscriptRef.current) {
      lastParsedTranscriptRef.current = speechRecognition.transcript
      const parsed = parseNavigationIntent(speechRecognition.transcript)
      if (parsed) {
        setIntent(parsed)
        onIntentParsed?.(parsed)
      }
    }
  }, [speechRecognition.isListening, speechRecognition.transcript, autoParse, onIntentParsed])
  
  // Execute navigation when intent changes
  useEffect(() => {
    if (!intent) return
    
    switch (intent.type) {
      case 'navigate':
        if (intent.targetPath) {
          navigate({ to: intent.targetPath })
        }
        break
      case 'back':
        navigate({ to: '..' })
        break
      case 'launch':
        // Launch is context-specific; we'll handle it in the component
        break
      case 'read_briefing':
        // TTS will be handled in the component
        break
      case 'select':
        // Selection is context-specific
        break
    }
    
    // Clear intent after handling (prevent infinite loops)
    setIntent(null)
  }, [intent, navigate])
  
  const parseTranscript = useCallback((text: string) => {
    return parseNavigationIntent(text)
  }, [])
  
  const reset = useCallback(() => {
    setIntent(null)
    speechRecognition.resetTranscript()
    lastParsedTranscriptRef.current = ''
  }, [speechRecognition])
  
  return {
    intent,
    isListening: speechRecognition.isListening,
    transcript: speechRecognition.transcript,
    error: speechRecognition.error,
    parseTranscript,
    reset,
  }
}