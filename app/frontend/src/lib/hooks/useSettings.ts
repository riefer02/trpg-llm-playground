/**
 * Settings hook with localStorage persistence.
 * 
 * Usage:
 *   const { settings, updateSettings } = useSettings()
 *   updateSettings({ masterVolume: 80 })
 */

import { useState, useEffect, useCallback } from 'react'

export interface Settings {
  // Audio
  masterVolume: number  // 0-100
  sfxVolume: number    // 0-100
  musicVolume: number  // 0-100
  
  // Voice
  enableVoiceInput: boolean
  enableTTS: boolean
  voiceSpeed: number  // 0.5-2.0
  voiceLanguage: string  // BCP 47 language tag
  
  // Display
  theme: 'light' | 'dark'
  
  // Accessibility
  reducedMotion: boolean
  highContrast: boolean
  
   // AI
  showAIReasoning: boolean
   
   // Combat
  confirmEndTurn: boolean
  enableLowHPWarning: boolean
}

const STORAGE_KEY = 'lancer_tactics_settings'

 const defaultSettings: Settings = {
  masterVolume: 80,
  sfxVolume: 80,
  musicVolume: 60,
  enableVoiceInput: false,
  enableTTS: false,
  voiceSpeed: 1.0,
  voiceLanguage: 'en-US',
  theme: 'light',
  reducedMotion: false,
  highContrast: false,
  showAIReasoning: true,
  confirmEndTurn: true,
  enableLowHPWarning: true,
}

function loadSettings(): Settings {
  if (typeof window === 'undefined') return defaultSettings
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      const parsed = JSON.parse(stored)
      // Merge with defaults to ensure new fields are added
      return { ...defaultSettings, ...parsed }
    }
  } catch {
    // Ignore parse errors
  }
  return defaultSettings
}

function saveSettings(settings: Settings): void {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
  } catch {
    // Ignore storage errors (quota exceeded, etc.)
  }
}

export function useSettings() {
  const [settings, setSettings] = useState<Settings>(loadSettings)

  // Persist changes to localStorage
  useEffect(() => {
    saveSettings(settings)
  }, [settings])

  const updateSettings = useCallback((updates: Partial<Settings>) => {
    setSettings(prev => ({ ...prev, ...updates }))
  }, [])

  const resetSettings = useCallback(() => {
    setSettings(defaultSettings)
  }, [])

  return {
    settings,
    updateSettings,
    resetSettings,
  }
}