/**
 * Save Indicator Component
 * 
 * Shows a brief "Saving..." indicator when auto-save is in progress.
 * Automatically fades out after save completes.
 */

import { useEffect, useState } from 'react';
import { Save } from 'lucide-react';

interface SaveIndicatorProps {
  /** Whether a save is currently in progress */
  isSaving: boolean;
  /** Duration to show the indicator after save completes (ms) */
  displayDuration?: number;
  /** Optional custom message */
  message?: string;
}

export function SaveIndicator({ 
  isSaving, 
  displayDuration = 1500,
  message = "Saving..."
}: SaveIndicatorProps) {
  const [show, setShow] = useState(false);
  const [hasCompleted, setHasCompleted] = useState(false);

  useEffect(() => {
    if (isSaving) {
      setShow(true);
      setHasCompleted(false);
    } else if (show && !hasCompleted) {
      // Save just completed, keep showing briefly
      setHasCompleted(true);
      const timer = setTimeout(() => {
        setShow(false);
      }, displayDuration);
      return () => clearTimeout(timer);
    }
  }, [isSaving, show, hasCompleted, displayDuration]);

  if (!show) return null;

  return (
    <div 
      className={`
        fixed bottom-4 right-4 z-50
        flex items-center gap-2 px-4 py-2
        bg-primary text-primary-foreground
        rounded-lg shadow-lg
        transition-opacity duration-300
        ${isSaving ? 'opacity-100' : 'opacity-0'}
      `}
      role="status"
      aria-live="polite"
      aria-label={isSaving ? "Saving game" : "Save complete"}
    >
      <Save className={`w-4 h-4 ${isSaving ? 'animate-pulse' : ''}`} />
      <span className="text-sm font-medium">
        {isSaving ? message : "Saved!"}
      </span>
    </div>
  );
}
