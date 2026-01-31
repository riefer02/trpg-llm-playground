import { useEffect, useState } from "react";
import { useSettings } from "../../lib/hooks/useSettings";
import { Modal } from "../ui/modal";

export interface VictoryCelebrationProps {
  isOpen: boolean;
  outcome: "victory" | "defeat";
  onClose?: () => void;
}

/**
 * Victory celebration modal that shows after mission completion.
 * Respects reduced motion setting (static display if enabled).
 * Shows distinct visual treatment for victory vs defeat.
 */
export function VictoryCelebration({
  isOpen,
  outcome,
  onClose,
}: VictoryCelebrationProps) {
  const { settings } = useSettings();
  const [showAnimation, setShowAnimation] = useState(false);

  // Trigger animation after mount (unless reduced motion)
  useEffect(() => {
    if (!isOpen) return;
    const timer = setTimeout(() => {
      if (!settings.reducedMotion) {
        setShowAnimation(true);
      }
    }, 100);
    return () => clearTimeout(timer);
  }, [isOpen, settings.reducedMotion]);

  // Auto-close after delay (for victory celebration before navigation)
  // The parent component should handle navigation, not auto-close
  // This component just displays the celebration

  const isVictory = outcome === "victory";

  // Colors and text based on outcome
  const title = isVictory ? "VICTORY" : "DEFEAT";
  const subtitle = isVictory ? "Mission objectives secured" : "Mission failed";
  const bgColor = isVictory
    ? "bg-gradient-to-br from-green-900/90 to-emerald-800/90"
    : "bg-gradient-to-br from-red-900/90 to-rose-900/90";
  const borderColor = isVictory ? "border-green-500" : "border-red-500";
  const textColor = isVictory ? "text-green-300" : "text-red-300";
  const accentColor = isVictory ? "text-emerald-300" : "text-rose-300";

  // Animation classes
  const pulseAnimation = settings.reducedMotion ? "" : "animate-pulse";
  const scaleAnimation = settings.reducedMotion ? "" : "animate-scale-up";
  const glowAnimation = settings.reducedMotion ? "" : "animate-glow";
  const floatAnimation = settings.reducedMotion ? "" : "animate-float";

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      disableBackdropClose={true}
      ariaLabel={`${title} - ${subtitle}`}
    >
      <div
        className={`rounded-lg border-4 ${borderColor} ${bgColor} p-8 text-center shadow-2xl ${scaleAnimation}`}
      >
        {/* Main title with animation */}
        <h1
          className={`text-6xl font-bold mb-4 ${textColor} ${pulseAnimation}`}
          style={{
            animationDuration: "2s",
            animationIterationCount: "infinite",
          }}
        >
          {title}
        </h1>

        {/* Subtitle */}
        <p className={`text-xl ${accentColor} mb-6`}>{subtitle}</p>

        {/* Animated elements (only if not reduced motion) */}
        {!settings.reducedMotion && (
          <>
            {/* Confetti-like dots for victory */}
            {isVictory && (
              <div className="absolute inset-0 overflow-hidden pointer-events-none">
                {Array.from({ length: 30 }).map((_, i) => (
                  <div
                    key={i}
                    className="absolute w-2 h-2 bg-yellow-300 rounded-full opacity-70"
                    style={{
                      top: `${Math.random() * 100}%`,
                      left: `${Math.random() * 100}%`,
                      animation: `float ${2 + Math.random() * 3}s infinite ease-in-out`,
                      animationDelay: `${Math.random() * 1}s`,
                    }}
                  />
                ))}
              </div>
            )}

            {/* Cracked effect for defeat */}
            {!isVictory && (
              <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-1/2 left-1/2 w-full h-1 bg-red-800/50 transform -translate-x-1/2 -translate-y-1/2 rotate-45" />
                <div className="absolute top-1/2 left-1/2 w-full h-1 bg-red-800/50 transform -translate-x-1/2 -translate-y-1/2 -rotate-45" />
              </div>
            )}
          </>
        )}

        {/* Continue hint (static) */}
        <div className={`mt-8 text-sm ${textColor} opacity-80`}>
          Preparing mission debrief...
        </div>
      </div>
    </Modal>
  );
}
