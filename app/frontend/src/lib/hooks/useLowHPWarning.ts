/**
 * Hook to monitor player mech HP and structure for low HP warnings.
 * Provides warning state and triggers audio/visual cues.
 */

import { useEffect, useRef, useState } from 'react'
import { useSettings } from './useSettings'
import type { CombatantState } from '../types/lancer'

interface LowHPWarningState {
  /** Whether low HP warning is currently active */
  isWarningActive: boolean
  /** Reason for warning: 'hp' for low HP, 'structure' for structure damage */
  reason: 'hp' | 'structure' | null
  /** Player combatant ID that triggered warning */
  combatantId: string | null
}

/**
 * Check if a combatant is at low HP (≤25%) or has structure damage.
 */
function checkLowHPWarning(
  combatant: CombatantState,
  hpThresholdPercent = 0.25
): { active: boolean; reason: 'hp' | 'structure' | null } {
  // Check HP
  const hpMax = combatant.stats?.hp_max ?? 1
  const hpCurrent = combatant.resources?.hp_current ?? 0
  const hpPercent = hpMax > 0 ? hpCurrent / hpMax : 0

  // Check structure (armor break)
  const structureCurrent = combatant.resources?.structure_current
  // Structure damage is considered if current < max (default 4)
  // For simplicity, assume structure damage if structure_current is defined and < 4
  // In Lancer, max structure is typically 4, but can be increased by talents.
  // We'll treat any reduction as damage.
  const hasStructureDamage = structureCurrent !== undefined && structureCurrent < 4

  if (hasStructureDamage) {
    return { active: true, reason: 'structure' }
  }
  if (hpPercent <= hpThresholdPercent && hpCurrent > 0) {
    return { active: true, reason: 'hp' }
  }
  return { active: false, reason: null }
}

/**
 * Generate a warning beep using Web Audio API.
 * Respects master volume and SFX volume settings.
 */
function playWarningBeep(
  audioContext: AudioContext,
  masterVolume: number,
  sfxVolume: number
) {
  try {
    const effectiveVolume = (masterVolume / 100) * (sfxVolume / 100)
    const oscillator = audioContext.createOscillator()
    const gainNode = audioContext.createGain()
    
    oscillator.connect(gainNode)
    gainNode.connect(audioContext.destination)
    
    oscillator.type = 'sine'
    oscillator.frequency.setValueAtTime(800, audioContext.currentTime)
    oscillator.frequency.exponentialRampToValueAtTime(400, audioContext.currentTime + 0.3)
    
    gainNode.gain.setValueAtTime(0, audioContext.currentTime)
    gainNode.gain.linearRampToValueAtTime(effectiveVolume * 0.2, audioContext.currentTime + 0.05)
    gainNode.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + 0.5)
    
    oscillator.start(audioContext.currentTime)
    oscillator.stop(audioContext.currentTime + 0.5)
  } catch (error) {
    console.warn('Failed to play warning beep:', error)
  }
}

/**
 * Hook that monitors player combatants for low HP/structure damage.
 * Returns warning state and manages audio/visual cues.
 */
export function useLowHPWarning(
  combatants: CombatantState[],
  playerSide: 'players' = 'players'
): LowHPWarningState {
  const { settings } = useSettings()
  const [warningState, setWarningState] = useState<LowHPWarningState>({
    isWarningActive: false,
    reason: null,
    combatantId: null,
  })
  
  const audioContextRef = useRef<AudioContext | null>(null)
  const beepIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const lastWarningActiveRef = useRef(false)
  
  // Get player combatants
  const playerCombatants = combatants.filter(c => c.side === playerSide)
  
  // Check for low HP/structure damage
  const activeWarnings = playerCombatants
    .map(combatant => ({
      combatant,
      check: checkLowHPWarning(combatant),
    }))
    .filter(item => item.check.active)
  
  const isWarningActive = activeWarnings.length > 0
  const currentReason = activeWarnings[0]?.check.reason ?? null
  const currentCombatantId = activeWarnings[0]?.combatant.id ?? null
  
  // Initialize AudioContext on user interaction (lazy)
  const ensureAudioContext = () => {
    if (!audioContextRef.current && typeof window !== 'undefined' && 'AudioContext' in window) {
      audioContextRef.current = new AudioContext()
    }
    return audioContextRef.current
  }
  
  // Effect to update warning state and trigger audio
  useEffect(() => {
    // Skip if warnings disabled
    if (!settings.enableLowHPWarning) {
      if (beepIntervalRef.current) {
        clearInterval(beepIntervalRef.current)
        beepIntervalRef.current = null
      }
      setWarningState({
        isWarningActive: false,
        reason: null,
        combatantId: null,
      })
      lastWarningActiveRef.current = false
      return
    }
    
    const warningChanged = lastWarningActiveRef.current !== isWarningActive
    
    // Update state
    setWarningState({
      isWarningActive,
      reason: currentReason,
      combatantId: currentCombatantId,
    })
    
    // Handle audio cues
    if (isWarningActive && warningChanged) {
      // Warning just activated - play immediate beep
      const audioContext = ensureAudioContext()
      if (audioContext) {
        playWarningBeep(audioContext, settings.masterVolume, settings.sfxVolume)
      }
      
      // Start periodic beeps (subtle, every 15 seconds)
      if (beepIntervalRef.current) {
        clearInterval(beepIntervalRef.current)
      }
      beepIntervalRef.current = setInterval(() => {
        const audioContext = ensureAudioContext()
        if (audioContext) {
          playWarningBeep(audioContext, settings.masterVolume, settings.sfxVolume)
        }
      }, 15000) // 15 seconds
    } else if (!isWarningActive && warningChanged) {
      // Warning deactivated - stop periodic beeps
      if (beepIntervalRef.current) {
        clearInterval(beepIntervalRef.current)
        beepIntervalRef.current = null
      }
    }
    
    lastWarningActiveRef.current = isWarningActive
    
    // Cleanup
    return () => {
      if (beepIntervalRef.current) {
        clearInterval(beepIntervalRef.current)
        beepIntervalRef.current = null
      }
    }
  }, [
    isWarningActive,
    currentReason,
    currentCombatantId,
    settings.enableLowHPWarning,
    settings.masterVolume,
    settings.sfxVolume,
  ])
  
  // Cleanup AudioContext on unmount
  useEffect(() => {
    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close()
        audioContextRef.current = null
      }
    }
  }, [])
  
  return warningState
}