/**
 * Speech-to-text module using Web Speech API.
 * 
 * Features:
 * - Push-to-talk activation via spacebar (when not in input)
 * - Real-time transcription display
 * - Configurable language (default: en-US)
 * - Graceful degradation if API unavailable
 * 
 * Usage:
 *   const { isListening, transcript, error, startListening, stopListening } = useSpeechRecognition()
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { useSettings } from '../hooks/useSettings'

// Get the SpeechRecognition constructor with vendor prefixes
function getSpeechRecognition(): typeof SpeechRecognition | null {
  if (typeof window === 'undefined') return null
  
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
  return SpeechRecognition || null
}

export interface UseSpeechRecognitionOptions {
  language?: string
  continuous?: boolean
  interimResults?: boolean
  enabled?: boolean
}

export interface UseSpeechRecognitionReturn {
  // State
  isListening: boolean
  transcript: string
  error: string | null
  recognitionSupported: boolean
  
  // Controls
  startListening: () => void
  stopListening: () => void
  toggleListening: () => void
  resetTranscript: () => void
  
  // Configuration
  setLanguage: (lang: string) => void
}

/**
 * React hook for speech recognition using Web Speech API
 */
export function useSpeechRecognition(
  options: UseSpeechRecognitionOptions = {}
): UseSpeechRecognitionReturn {
  const {
    language = 'en-US',
    continuous = false,
    interimResults = true,
    enabled = true,
  } = options
  
  const { settings } = useSettings()
  const [isListening, setIsListening] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [currentLanguage, setCurrentLanguage] = useState(language)
  
  // Update currentLanguage when language prop changes
  useEffect(() => {
    setCurrentLanguage(language)
  }, [language])
  
  const recognitionRef = useRef<SpeechRecognition | null>(null)
  const transcriptRef = useRef('')
  
  // Check if SpeechRecognition is supported
  const recognitionSupported = typeof window !== 'undefined' && 
    (window.SpeechRecognition || window.webkitSpeechRecognition) !== undefined
  
  // Initialize SpeechRecognition instance
  useEffect(() => {
    if (!enabled) {
      // If disabled, clear any existing recognition instance
      if (recognitionRef.current) {
        recognitionRef.current.stop()
        recognitionRef.current = null
      }
      return
    }
    
    if (!recognitionSupported) {
      setError('Speech recognition is not supported in this browser.')
      return
    }
    
    const SpeechRecognition = getSpeechRecognition()
    if (!SpeechRecognition) {
      setError('Speech recognition is not available.')
      return
    }
    
    const recognition = new SpeechRecognition()
    recognition.lang = currentLanguage
    recognition.continuous = continuous
    recognition.interimResults = interimResults
    
    recognition.onstart = () => {
      setIsListening(true)
      setError(null)
    }
    
    recognition.onend = () => {
      setIsListening(false)
    }
    
    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      setIsListening(false)
      setError(`Speech recognition error: ${event.error}`)
    }
    
    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let interimTranscript = ''
      let finalTranscript = ''
      
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript
        if (event.results[i].isFinal) {
          finalTranscript += transcript + ' '
        } else {
          interimTranscript += transcript
        }
      }
      
      transcriptRef.current = finalTranscript || interimTranscript
      setTranscript(transcriptRef.current)
    }
    
    recognitionRef.current = recognition
    
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop()
      }
    }
  }, [enabled, recognitionSupported, currentLanguage, continuous, interimResults])
  
  // Update language when recognition instance changes
  useEffect(() => {
    if (recognitionRef.current && enabled) {
      recognitionRef.current.lang = currentLanguage
    }
  }, [currentLanguage, enabled])

  // Define control functions before the useEffect that uses them
  const startListening = useCallback(() => {
    if (!recognitionRef.current || isListening) return

    try {
      recognitionRef.current.start()
    } catch (err) {
      setError(`Failed to start listening: ${err}`)
    }
  }, [isListening])

  const stopListening = useCallback(() => {
    if (!recognitionRef.current || !isListening) return

    try {
      recognitionRef.current.stop()
    } catch (err) {
      setError(`Failed to stop listening: ${err}`)
    }
  }, [isListening])

  // Push-to-talk spacebar handler
  useEffect(() => {
    if (!settings.enableVoiceInput || !enabled) return

    const handleKeyDown = (event: KeyboardEvent) => {
      // Only activate spacebar when not in an input element
      if (event.code === 'Space' && !isInputElement(event.target)) {
        event.preventDefault() // Prevent spacebar from scrolling
        if (!isListening && recognitionRef.current) {
          startListening()
        }
      }
    }

    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.code === 'Space' && !isInputElement(event.target)) {
        if (isListening && recognitionRef.current) {
          stopListening()
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
    }
  }, [settings.enableVoiceInput, enabled, isListening, startListening, stopListening])
  
  const toggleListening = useCallback(() => {
    if (isListening) {
      stopListening()
    } else {
      startListening()
    }
  }, [isListening, startListening, stopListening])
  
  const resetTranscript = useCallback(() => {
    setTranscript('')
    transcriptRef.current = ''
  }, [])
  
  const setLanguage = useCallback((lang: string) => {
    setCurrentLanguage(lang)
  }, [])
  
  return {
    isListening,
    transcript,
    error,
    recognitionSupported,
    startListening,
    stopListening,
    toggleListening,
    resetTranscript,
    setLanguage,
  }
}

// Helper function to check if element is an input or textarea
function isInputElement(element: EventTarget | null): boolean {
  if (!element || !(element instanceof Element)) return false
  
  const tagName = element.tagName.toLowerCase()
  return tagName === 'input' || tagName === 'textarea' || element.hasAttribute('contenteditable')
}