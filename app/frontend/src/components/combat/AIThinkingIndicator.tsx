import { Loader2, Clock } from "lucide-react";
import { useEffect, useState } from "react";

export interface AIThinkingIndicatorProps {
  /** Whether AI is currently thinking */
  isThinking: boolean;
  /** Optional custom message */
  message?: string;
  /** Timeout in seconds after which to show a timeout message */
  timeoutSeconds?: number;
  /** Whether reduced motion is enabled */
  reducedMotion?: boolean;
  /** Callback when timeout is reached */
  onTimeout?: () => void;
}

export function AIThinkingIndicator({
  isThinking,
  message = "Enemy is thinking...",
  timeoutSeconds = 10,
  reducedMotion = false,
  onTimeout,
}: AIThinkingIndicatorProps) {
  const [timeExceeded, setTimeExceeded] = useState(false);
  const [startTime, setStartTime] = useState<number | null>(null);

  // Reset timer when thinking starts/stops
  useEffect(() => {
    if (isThinking) {
      setStartTime(Date.now());
      setTimeExceeded(false);
      const timer = setTimeout(() => {
        setTimeExceeded(true);
        onTimeout?.();
      }, timeoutSeconds * 1000);
      return () => clearTimeout(timer);
    } else {
      setStartTime(null);
      setTimeExceeded(false);
    }
  }, [isThinking, timeoutSeconds, onTimeout]);

  if (!isThinking) {
    return null;
  }

  const elapsed = startTime ? (Date.now() - startTime) / 1000 : 0;

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/20 backdrop-blur-sm pointer-events-none">
      <div className="bg-background/90 border border-border rounded-lg shadow-xl p-6 max-w-sm mx-4 animate-in fade-in zoom-in-95 duration-200">
        <div className="flex items-center gap-4">
          {/* Spinner */}
          <div className="relative">
            {reducedMotion ? (
              <div className="w-8 h-8 rounded-full border-2 border-primary border-t-transparent" />
            ) : (
              <Loader2 className="w-8 h-8 text-primary animate-spin" />
            )}
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="font-medium text-foreground">{message}</div>
            <div className="text-sm text-muted-foreground mt-1">
              {timeExceeded ? (
                <div className="flex items-center gap-1.5 text-amber-500">
                  <Clock className="w-3.5 h-3.5" />
                  <span>Taking longer than expected... Still analyzing.</span>
                </div>
              ) : (
                <span>Analyzing tactical situation...</span>
              )}
            </div>
            {/* Optional progress bar */}
            {!reducedMotion && timeoutSeconds > 0 && (
              <div className="mt-2 h-1.5 w-full bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-300"
                  style={{ width: `${Math.min(100, (elapsed / timeoutSeconds) * 100)}%` }}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}