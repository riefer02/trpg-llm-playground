/**
 * Voice indicator component showing listening state and real-time transcription.
 */

import { useSpeechRecognition } from './speech-to-text'
import { useSettings } from '../hooks/useSettings'

export function VoiceIndicator() {
  const { settings } = useSettings()
  const {
    isListening,
    transcript,
    error,
    recognitionSupported,
    toggleListening,
    resetTranscript,
  } = useSpeechRecognition({
    language: settings.voiceLanguage,
    continuous: false,
    interimResults: true,
  })
  
  if (!recognitionSupported) {
    return (
      <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20">
        <div className="font-semibold text-destructive">Speech recognition not supported</div>
        <p className="text-sm text-muted-foreground mt-1">
          Your browser does not support the Web Speech API. Voice input is unavailable.
        </p>
      </div>
    )
  }
  
  if (error) {
    return (
      <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20">
        <div className="font-semibold text-destructive">Voice input error</div>
        <p className="text-sm text-muted-foreground mt-1">{error}</p>
        <button
          onClick={toggleListening}
          className="mt-2 px-3 py-1 bg-primary text-primary-foreground rounded text-sm font-medium"
        >
          Retry
        </button>
      </div>
    )
  }
  
  return (
    <div className="p-4 rounded-lg bg-card border shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <div className="font-semibold">Voice Input</div>
        <div className="flex items-center gap-2">
          <button
            onClick={toggleListening}
            className={`px-3 py-1 rounded text-sm font-medium ${
              isListening
                ? 'bg-destructive text-destructive-foreground'
                : 'bg-primary text-primary-foreground'
            }`}
          >
            {isListening ? 'Stop Listening' : 'Start Listening'}
          </button>
          <button
            onClick={resetTranscript}
            className="px-3 py-1 bg-muted text-muted-foreground rounded text-sm font-medium"
          >
            Clear
          </button>
        </div>
      </div>
      
      {/* Listening indicator */}
      <div className="flex items-center gap-2 mb-3">
        <div className={`w-3 h-3 rounded-full ${isListening ? 'bg-green-500 animate-pulse' : 'bg-muted'}`} />
        <span className="text-sm text-muted-foreground">
          {isListening ? 'Listening... (hold spacebar)' : 'Ready'}
        </span>
      </div>
      
      {/* Transcription display */}
      <div className="mt-3">
        <div className="text-xs uppercase text-muted-foreground mb-1">Transcript</div>
        <div className="min-h-20 p-3 rounded bg-muted/20 border whitespace-pre-wrap break-words">
          {transcript || 'Speak to see transcription here...'}
        </div>
      </div>
      
      <div className="mt-3 text-xs text-muted-foreground">
        <p>Push-to-talk: Hold <kbd className="px-1 py-0.5 bg-muted rounded text-xs">Space</kbd> when not in an input field.</p>
      </div>
    </div>
  )
}