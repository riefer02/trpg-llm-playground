import { Mic, MicOff, Volume2 } from "lucide-react";

/**
 * Component for displaying real-time voice transcription during speech recognition.
 * Shows listening indicator, live transcript, and error states.
 */

export interface VoiceTranscriptDisplayProps {
  /** Whether speech recognition is currently active */
  isListening: boolean;
  /** Current transcript text (interim + final) */
  transcript: string;
  /** Error message if recognition failed */
  error: string | null;
  /** Whether speech recognition is supported by the browser */
  recognitionSupported: boolean;
  /** Whether voice input is enabled in settings */
  voiceEnabled?: boolean;
  /** Optional callback when user clicks to retry after error */
  onRetry?: () => void;
  /** Optional CSS class name */
  className?: string;
}

export function VoiceTranscriptDisplay({
  isListening,
  transcript,
  error,
  recognitionSupported,
  voiceEnabled = true,
  onRetry,
  className = "",
}: VoiceTranscriptDisplayProps) {
  // If voice is not supported, don't render anything
  if (!recognitionSupported) {
    return null;
  }

  // If voice is disabled in settings, don't render anything
  if (!voiceEnabled) {
    return null;
  }

  const hasTranscript = transcript.trim().length > 0;
  const showError = error !== null;

  return (
    <div className={`rounded-md border bg-card text-card-foreground shadow-sm overflow-hidden ${className}`}>
      {/* Header with status indicator */}
      <div className={`px-3 py-2 flex items-center justify-between ${isListening ? "bg-primary/10" : "bg-muted/30"}`}>
        <div className="flex items-center gap-2">
          {isListening ? (
            <>
              <div className="relative">
                <Mic className="w-4 h-4 text-primary animate-pulse" />
                <div className="absolute -top-1 -left-1 w-6 h-6 rounded-full border-2 border-primary/30 animate-ping"></div>
              </div>
              <span className="text-sm font-medium text-primary">Listening...</span>
            </>
          ) : hasTranscript ? (
            <>
              <Volume2 className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm font-medium text-muted-foreground">Voice Input</span>
            </>
          ) : (
            <>
              <MicOff className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm font-medium text-muted-foreground">Voice Ready</span>
            </>
          )}
        </div>
        {isListening && (
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse"></div>
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse" style={{ animationDelay: "0.2s" }}></div>
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse" style={{ animationDelay: "0.4s" }}></div>
          </div>
        )}
      </div>

      {/* Transcript content */}
      <div className="px-3 py-3">
        {showError ? (
          <div className="space-y-2">
            <div className="text-sm font-medium text-destructive">Voice Recognition Error</div>
            <div className="text-sm text-muted-foreground">{error}</div>
            {onRetry && (
              <button
                type="button"
                onClick={onRetry}
                className="text-xs px-2 py-1 rounded border border-input bg-background hover:bg-accent hover:text-accent-foreground"
              >
                Try Again
              </button>
            )}
          </div>
        ) : hasTranscript ? (
          <div className="space-y-1">
            <div className="text-xs font-medium text-muted-foreground">Transcript:</div>
            <div className="text-sm font-medium">{transcript}</div>
            {isListening && (
              <div className="text-xs text-muted-foreground italic">
                 Speak your command (e.g., "Move to B3", "Attack the Ronin", "Use Scan", "Overcharge")
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-1">
            <div className="text-xs font-medium text-muted-foreground">No voice input yet</div>
            <div className="text-xs text-muted-foreground italic">
              Click the microphone button or press Space to speak
            </div>
          </div>
        )}
      </div>

      {/* Footer with usage hints */}
      {!showError && (
        <div className="px-3 py-2 border-t bg-muted/20 text-xs text-muted-foreground">
          <div className="flex items-center justify-between">
             <span>Examples: "Move to B3", "Attack Ronin", "Use Scan on Scout", "Overcharge", "End turn"</span>
          </div>
        </div>
      )}
    </div>
  );
}