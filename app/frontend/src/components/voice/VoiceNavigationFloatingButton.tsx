/**
 * Floating button for voice navigation.
 * Appears on all screens (including title) when voice input is enabled.
 */

import { useState, useEffect } from 'react'
import { useLocation } from '@tanstack/react-router'
import { useSettings } from '../../lib/hooks/useSettings'
import { toast } from 'sonner'

export interface VoiceNavigationFloatingButtonProps {
  isListening: boolean
  transcript: string
  error: string | null
  reset: () => void
}

export function VoiceNavigationFloatingButton({
  isListening,
  transcript,
  error,
  reset,
}: VoiceNavigationFloatingButtonProps) {
  const location = useLocation()
  const { settings } = useSettings()
  const [showTranscript, setShowTranscript] = useState(false)
  
  // Hide transcript after 3 seconds if not listening
  useEffect(() => {
    if (!isListening && transcript) {
      const timer = setTimeout(() => {
        setShowTranscript(false)
        reset()
      }, 3000)
      return () => clearTimeout(timer)
    }
  }, [isListening, transcript, reset])
  
  // Show transcript when new speech arrives
  useEffect(() => {
    if (transcript) {
      setShowTranscript(true)
    }
  }, [transcript])
  
  // Show error toast
  useEffect(() => {
    if (error) {
      toast.error(`Voice navigation error: ${error}`)
    }
  }, [error])
  
  // Don't render if voice input is disabled
  if (!settings.enableVoiceInput) {
    return null
  }
  
  // Don't render in combat route (combat has its own voice UI)
  if (location.pathname.startsWith('/combat/')) {
    return null
  }
  
  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-2">
      {/* Transcript panel */}
      {showTranscript && transcript && (
        <div className="max-w-sm p-3 rounded-lg bg-card border shadow-lg animate-in slide-in-from-bottom-2">
          <div className="text-xs uppercase text-muted-foreground mb-1">Voice command</div>
          <div className="text-sm font-medium">{transcript}</div>
        </div>
      )}
      
      {/* Floating button */}
      <button
        onClick={() => setShowTranscript(!showTranscript)}
        className={`w-14 h-14 rounded-full shadow-lg flex items-center justify-center transition-all ${
          isListening 
            ? 'bg-green-500 text-white animate-pulse' 
            : 'bg-primary text-primary-foreground hover:bg-primary/90'
        }`}
        aria-label={isListening ? 'Voice listening - click to show transcript' : 'Voice navigation - click to show transcript'}
        title="Voice navigation (hold spacebar)"
      >
        {isListening ? (
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
          </svg>
        ) : (
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
          </svg>
        )}
      </button>
      
      {/* Listening indicator */}
      {isListening && (
        <div className="text-xs text-muted-foreground bg-background/80 backdrop-blur px-2 py-1 rounded">
          Listening... (hold spacebar)
        </div>
      )}
    </div>
  )
}