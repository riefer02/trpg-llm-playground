/**
 * Text-to-speech module using Web Speech API.
 * 
 * Features:
 * - Queue system to prevent overlapping speech
 * - Configurable voice, speed, and volume
 * - Interruption support (high-priority messages cancel queue)
 * - Graceful degradation if API unavailable
 * 
 * Usage:
 *   const { speak, cancel, isSpeaking, availableVoices } = useTextToSpeech()
 *   speak('Hello world')
 *   speak('Urgent message', { priority: 'high' }) // Interrupts current speech
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { useSettings } from '../hooks/useSettings'

export type TtsPriority = 'low' | 'normal' | 'high'

export interface TtsOptions {
  priority?: TtsPriority
  rate?: number // Override default voice speed
  volume?: number // Override default volume (0-1)
  voice?: SpeechSynthesisVoice // Override default voice
  onStart?: () => void
  onEnd?: () => void
  onError?: (error: string) => void
}

export interface QueuedUtterance {
  id: string
  text: string
  options: TtsOptions
  utterance: SpeechSynthesisUtterance
}

export interface UseTextToSpeechReturn {
  // State
  isSpeaking: boolean
  availableVoices: SpeechSynthesisVoice[]
  error: string | null
  ttsSupported: boolean
  
  // Controls
  speak: (text: string, options?: TtsOptions) => void
  cancel: () => void
  pause: () => void
  resume: () => void
  
  // Queue management
  clearQueue: () => void
  queueLength: number
}

// Get the speechSynthesis API
function getSpeechSynthesis(): SpeechSynthesis | null {
  if (typeof window === 'undefined') return null
  return window.speechSynthesis
}

/**
 * React hook for text-to-speech using Web Speech API
 */
export function useTextToSpeech(): UseTextToSpeechReturn {
  const { settings } = useSettings()
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [availableVoices, setAvailableVoices] = useState<SpeechSynthesisVoice[]>([])
  const [error, setError] = useState<string | null>(null)
  const [queueLength, setQueueLength] = useState(0)
  
  const synthesisRef = useRef<SpeechSynthesis | null>(null)
  const queueRef = useRef<QueuedUtterance[]>([])
  const currentUtteranceRef = useRef<QueuedUtterance | null>(null)
  
  // Check if speechSynthesis is supported
  const ttsSupported = typeof window !== 'undefined' && 'speechSynthesis' in window
  
  // Initialize speechSynthesis and load voices
  useEffect(() => {
    if (!ttsSupported) {
      setError('Text-to-speech is not supported in this browser.')
      return
    }
    
    const synthesis = getSpeechSynthesis()
    if (!synthesis) {
      setError('Speech synthesis is not available.')
      return
    }
    
    synthesisRef.current = synthesis
    
    // Load available voices
    const loadVoices = () => {
      const voices = synthesis.getVoices()
      setAvailableVoices(voices)
    }
    
    // Voices may load asynchronously
    loadVoices()
    synthesis.onvoiceschanged = loadVoices
    
    // Cleanup
    return () => {
      if (synthesisRef.current) {
        synthesisRef.current.cancel()
        synthesis.onvoiceschanged = null
      }
    }
  }, [ttsSupported])
  
  // Process the next item in queue
  const processQueue = useCallback(() => {
    const synthesis = synthesisRef.current
    if (!synthesis || queueRef.current.length === 0) {
      setIsSpeaking(false)
      return
    }
    
    // If already speaking and current utterance is high priority, don't interrupt
    if (isSpeaking && currentUtteranceRef.current?.options.priority === 'high') {
      return
    }
    
    // Cancel current speech if any (unless it's also high priority)
    if (isSpeaking && currentUtteranceRef.current?.options.priority !== 'high') {
      synthesis.cancel()
    }
    
    // Get next utterance from queue
    const nextUtterance = queueRef.current.shift()!
    setQueueLength(queueRef.current.length)
    currentUtteranceRef.current = nextUtterance
    
    // Configure utterance based on settings and options
    const utterance = nextUtterance.utterance
    utterance.lang = settings.voiceLanguage
    utterance.rate = nextUtterance.options.rate ?? settings.voiceSpeed
    utterance.volume = nextUtterance.options.volume ?? (settings.masterVolume / 100)
    if (nextUtterance.options.voice) {
      utterance.voice = nextUtterance.options.voice
    } else {
      // Try to find a voice matching the language
      const matchingVoice = availableVoices.find(
        voice => voice.lang.startsWith(settings.voiceLanguage)
      )
      if (matchingVoice) {
        utterance.voice = matchingVoice
      }
    }
    
    // Event handlers
    utterance.onstart = () => {
      setIsSpeaking(true)
      nextUtterance.options.onStart?.()
    }
    
    utterance.onend = () => {
      setIsSpeaking(false)
      nextUtterance.options.onEnd?.()
      currentUtteranceRef.current = null
      // Process next item after a short delay
      setTimeout(() => processQueue(), 100)
    }
    
    utterance.onerror = (event) => {
      setIsSpeaking(false)
      const errorMsg = `Speech synthesis error: ${event.error}`
      setError(errorMsg)
      nextUtterance.options.onError?.(errorMsg)
      currentUtteranceRef.current = null
      // Process next item after error
      setTimeout(() => processQueue(), 100)
    }
    
    // Start speaking
    synthesis.speak(utterance)
  }, [isSpeaking, settings.voiceLanguage, settings.voiceSpeed, settings.masterVolume, availableVoices])
  
  // Speak function - adds utterance to queue
  const speak = useCallback((text: string, options: TtsOptions = {}) => {
    if (!ttsSupported || !settings.enableTTS) {
      return // Silent failure if TTS disabled or not supported
    }
    
    const synthesis = synthesisRef.current
    if (!synthesis) {
      setError('Speech synthesis not initialized')
      return
    }
    
    const utterance = new SpeechSynthesisUtterance(text)
    const id = Math.random().toString(36).substring(2)
    const queuedItem: QueuedUtterance = { id, text, options, utterance }
    
    // Handle priority
    if (options.priority === 'high') {
      // Cancel current speech and clear queue
      synthesis.cancel()
      queueRef.current = []
      setQueueLength(0)
      queueRef.current.push(queuedItem)
      setQueueLength(1)
    } else {
      // Add to end of queue
      queueRef.current.push(queuedItem)
      setQueueLength(queueRef.current.length)
    }
    
    // If not currently speaking, start processing
    if (!isSpeaking) {
      processQueue()
    }
  }, [ttsSupported, settings.enableTTS, isSpeaking, processQueue])
  
  const cancel = useCallback(() => {
    const synthesis = synthesisRef.current
    if (!synthesis) return
    
    synthesis.cancel()
    queueRef.current = []
    setQueueLength(0)
    currentUtteranceRef.current = null
    setIsSpeaking(false)
  }, [])
  
  const pause = useCallback(() => {
    const synthesis = synthesisRef.current
    if (!synthesis) return
    
    synthesis.pause()
    setIsSpeaking(false)
  }, [])
  
  const resume = useCallback(() => {
    const synthesis = synthesisRef.current
    if (!synthesis) return
    
    synthesis.resume()
    setIsSpeaking(true)
  }, [])
  
  const clearQueue = useCallback(() => {
    queueRef.current = []
    setQueueLength(0)
  }, [])
  
  return {
    isSpeaking,
    availableVoices,
    error,
    ttsSupported,
    speak,
    cancel,
    pause,
    resume,
    clearQueue,
    queueLength,
  }
}

/**
 * Convenience hook for narrating combat events
 */
export function useCombatNarration() {
  const { speak } = useTextToSpeech()
  
  const narrateAction = useCallback((
    actorName: string,
    actionName: string,
    targetName?: string,
    damage?: number,
    status?: string
  ) => {
    let text = `${actorName} uses ${actionName}`
    if (targetName) {
      text += ` on ${targetName}`
    }
    if (damage !== undefined) {
      text += `, dealing ${damage} damage`
    }
    if (status) {
      text += `, applying ${status}`
    }
    text += '.'
    
    speak(text, { priority: 'normal' })
  }, [speak])
  
  const narrateTurnStart = useCallback((actorName: string, isPlayer: boolean) => {
    const text = isPlayer ? 'Your turn.' : `${actorName}'s turn.`
    speak(text, { priority: 'normal' })
  }, [speak])
  
  const narrateStatusChange = useCallback((
    targetName: string,
    status: string,
    applied: boolean
  ) => {
    const text = `${targetName} ${applied ? 'is now' : 'is no longer'} ${status}.`
    speak(text, { priority: 'low' })
  }, [speak])
  
  const narrateVictory = useCallback((victoriousSide: 'player' | 'enemy') => {
    const text = victoriousSide === 'player' 
      ? 'Mission accomplished. Victory!' 
      : 'Mission failed. Defeat.'
    speak(text, { priority: 'high' })
  }, [speak])
  
  return {
    narrateAction,
    narrateTurnStart,
    narrateStatusChange,
    narrateVictory,
  }
}