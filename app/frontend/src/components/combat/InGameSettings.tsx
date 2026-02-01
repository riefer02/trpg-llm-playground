/**
 * In-game settings modal for pause menu.
 * Simplified settings focused on combat-relevant options.
 */

import { Volume2, Mic, Eye, Accessibility, X } from "lucide-react";
import { useSettings } from "../../lib/hooks/useSettings";
import { Modal, Button, Slider, Toggle } from "../ui";

export interface InGameSettingsProps {
  /** Whether the modal is open */
  isOpen: boolean;
  /** Callback when user closes the modal */
  onClose: () => void;
}

export function InGameSettings({ isOpen, onClose }: InGameSettingsProps) {
  const { settings, updateSettings } = useSettings();

  return (
    <Modal isOpen={isOpen} onClose={onClose} ariaLabel="Game settings">
      <div className="bg-background rounded-lg border shadow-xl w-full max-w-md overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
              <Volume2 className="w-4 h-4 text-primary" />
            </div>
            <div>
              <h3 className="font-semibold text-foreground">Settings</h3>
              <p className="text-sm text-muted-foreground">
                Audio, voice, display, and accessibility
              </p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            aria-label="Close settings"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>

        {/* Settings Content */}
        <div className="px-6 py-4 space-y-6 max-h-[60vh] overflow-y-auto">
          {/* Audio Section */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Volume2 className="w-4 h-4 text-muted-foreground" />
              <h4 className="text-sm font-medium">Audio</h4>
            </div>
            <div className="space-y-3 pl-6">
              <Slider
                value={settings.masterVolume}
                onChange={(value) => updateSettings({ masterVolume: value })}
                min={0}
                max={1}
                step={0.05}
                label="Master Volume"
                unit="%"
              />
              <Slider
                value={settings.sfxVolume}
                onChange={(value) => updateSettings({ sfxVolume: value })}
                min={0}
                max={1}
                step={0.05}
                label="SFX Volume"
                unit="%"
              />
            </div>
          </div>

          {/* Voice Section */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Mic className="w-4 h-4 text-muted-foreground" />
              <h4 className="text-sm font-medium">Voice</h4>
            </div>
            <div className="space-y-3 pl-6">
              <Toggle
                checked={settings.enableVoiceInput}
                onChange={(checked) =>
                  updateSettings({ enableVoiceInput: checked })
                }
                label="Enable Voice Input"
              />
              <Toggle
                checked={settings.enableTTS}
                onChange={(checked) => updateSettings({ enableTTS: checked })}
                label="Text-to-Speech"
              />
              {settings.enableTTS && (
                <Slider
                  value={settings.voiceSpeed}
                  onChange={(value) => updateSettings({ voiceSpeed: value })}
                  min={0.5}
                  max={2}
                  step={0.1}
                  label="Voice Speed"
                  unit="x"
                />
              )}
            </div>
          </div>

          {/* Display Section */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Eye className="w-4 h-4 text-muted-foreground" />
              <h4 className="text-sm font-medium">Display</h4>
            </div>
            <div className="space-y-3 pl-6">
              <div className="flex items-center justify-between">
                <label className="text-sm">Theme</label>
                <select
                  value={settings.theme}
                  onChange={(e) =>
                    updateSettings({
                      theme: e.target.value as "light" | "dark",
                    })
                  }
                  className="text-sm bg-muted rounded px-2 py-1 border"
                  aria-label="Theme selection"
                >
                  <option value="light">Light</option>
                  <option value="dark">Dark</option>
                </select>
              </div>
              <Toggle
                checked={settings.showAIReasoning}
                onChange={(checked) =>
                  updateSettings({ showAIReasoning: checked })
                }
                label="Show AI Reasoning"
              />
              <Toggle
                checked={settings.confirmEndTurn}
                onChange={(checked) =>
                  updateSettings({ confirmEndTurn: checked })
                }
                label="Confirm End Turn"
              />
            </div>
          </div>

          {/* Accessibility Section */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Accessibility className="w-4 h-4 text-muted-foreground" />
              <h4 className="text-sm font-medium">Accessibility</h4>
            </div>
            <div className="space-y-3 pl-6">
              <Toggle
                checked={settings.reducedMotion}
                onChange={(checked) =>
                  updateSettings({ reducedMotion: checked })
                }
                label="Reduced Motion"
              />
              <Toggle
                checked={settings.highContrast}
                onChange={(checked) =>
                  updateSettings({ highContrast: checked })
                }
                label="High Contrast"
              />
              <Toggle
                checked={settings.enableLowHPWarning}
                onChange={(checked) =>
                  updateSettings({ enableLowHPWarning: checked })
                }
                label="Low HP Warning"
              />
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t bg-muted/30">
          <Button onClick={onClose} className="w-full">
            Close
          </Button>
        </div>
      </div>
    </Modal>
  );
}
