/**
 * Settings screen for audio, voice, display, and accessibility preferences.
 */

import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, Slider, Toggle } from "../components/ui";
import { useSettings } from "../lib/hooks/useSettings";
import { useSaveSlots } from "../lib/save/useSaveSlots";
import { useActiveCharacter } from "../lib/api/quarters";

function SaveLoadManager() {
  const { slots, saveToSlot, deleteSlot, loadSlot, isLoading } = useSaveSlots();
  const { character } = useActiveCharacter();
  const [message, setMessage] = useState<string | null>(null);

  const handleSave = (slotIndex: number) => {
    if (!character) {
      setMessage("No active character to save.");
      return;
    }
    try {
      saveToSlot(slotIndex, character, `Manual save ${new Date().toLocaleDateString()}`);
      setMessage(`Saved to slot ${slotIndex + 1}`);
    } catch (error) {
      setMessage(`Error saving: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const handleLoad = (slotIndex: number) => {
    const slot = slots.find(s => s.slot === slotIndex);
    if (!slot) {
      setMessage(`No save in slot ${slotIndex + 1}`);
      return;
    }
    try {
      loadSlot(slotIndex);
      setMessage(`Loaded slot ${slotIndex + 1}: ${slot.character.callsign} is now active.`);
    } catch (error) {
      setMessage(`Error loading: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const handleDelete = (slotIndex: number) => {
    if (confirm(`Delete save slot ${slotIndex + 1}?`)) {
      deleteSlot(slotIndex);
      setMessage(`Deleted slot ${slotIndex + 1}`);
    }
  };

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-4">
        {[0, 1, 2].map(slotIndex => {
          const slot = slots.find(s => s.slot === slotIndex);
          return (
            <div key={slotIndex} className="border rounded-lg p-4 bg-muted/30">
              <h3 className="font-bold text-lg mb-2">Slot {slotIndex + 1}</h3>
              {slot ? (
                <>
                  <p className="text-sm text-muted-foreground">{slot.name || 'Unnamed save'}</p>
                  <p className="text-xs text-muted-foreground">{formatDate(slot.timestamp)}</p>
                  <p className="text-sm">Pilot: {slot.character.callsign}</p>
                  <div className="flex flex-col gap-2 mt-4">
                    <Button size="sm" onClick={() => handleSave(slotIndex)} disabled={isLoading}>
                      Overwrite Save
                    </Button>
                    <Button size="sm" variant="outline" onClick={() => handleLoad(slotIndex)}>
                      Load
                    </Button>
                    <Button size="sm" variant="destructive" onClick={() => handleDelete(slotIndex)}>
                      Delete
                    </Button>
                  </div>
                </>
              ) : (
                <>
                  <p className="text-sm text-muted-foreground italic">Empty</p>
                  <div className="mt-4">
                    <Button size="sm" onClick={() => handleSave(slotIndex)} disabled={isLoading || !character}>
                      Save
                    </Button>
                  </div>
                </>
              )}
            </div>
          );
        })}
      </div>
      {message && (
        <div className="p-3 rounded bg-muted/50 text-sm">
          {message}
        </div>
      )}
      <p className="text-sm text-muted-foreground">
        Save data is stored in your browser's local storage. Export/import coming soon.
      </p>
    </div>
  );
}

export const Route = createFileRoute("/settings" as const)({
  component: SettingsScreen,
});

function SettingsScreen() {
  const navigate = useNavigate();
  const { settings, updateSettings, resetSettings } = useSettings();

  const handleBack = () => {
    navigate({ to: "/" });
  };

  const handleReset = () => {
    if (confirm("Reset all settings to default values?")) {
      resetSettings();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/30 p-6">
      <div className="max-w-4xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold tracking-tight text-foreground">
              Settings
            </h1>
            <p className="text-xl text-muted-foreground mt-2">
              Configure audio, voice, display, and accessibility preferences
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={handleReset}>
              Reset to Defaults
            </Button>
            <Button variant="outline" onClick={handleBack}>
              Back to Title
            </Button>
          </div>
        </div>

        {/* Settings Sections */}
        <div className="space-y-8">
          {/* Audio Section */}
          <Card className="dashboard-surface">
            <CardHeader>
              <CardTitle>Audio</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <Slider
                label="Master Volume"
                value={settings.masterVolume}
                onChange={(value) => updateSettings({ masterVolume: value })}
                min={0}
                max={100}
                step={1}
                unit="%"
              />
              <Slider
                label="SFX Volume"
                value={settings.sfxVolume}
                onChange={(value) => updateSettings({ sfxVolume: value })}
                min={0}
                max={100}
                step={1}
                unit="%"
              />
              <Slider
                label="Music Volume"
                value={settings.musicVolume}
                onChange={(value) => updateSettings({ musicVolume: value })}
                min={0}
                max={100}
                step={1}
                unit="%"
              />
            </CardContent>
          </Card>

          {/* Voice Section */}
          <Card className="dashboard-surface">
            <CardHeader>
              <CardTitle>Voice</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <Toggle
                label="Enable Voice Input"
                checked={settings.enableVoiceInput}
                onChange={(checked) => updateSettings({ enableVoiceInput: checked })}
                description="Use speech recognition for commands"
              />
              <Toggle
                label="Enable Text-to-Speech"
                checked={settings.enableTTS}
                onChange={(checked) => updateSettings({ enableTTS: checked })}
                description="Narrate action results and AI turns"
              />
              <Slider
                label="Voice Speed"
                value={settings.voiceSpeed}
                onChange={(value) => updateSettings({ voiceSpeed: value })}
                min={0.5}
                max={2.0}
                step={0.1}
                unit="x"
              />
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">
                  Voice Language
                </label>
                <select
                  value={settings.voiceLanguage}
                  onChange={(e) => updateSettings({ voiceLanguage: e.target.value })}
                  className="w-full p-2 rounded border border-input bg-background text-foreground"
                >
                  <option value="en-US">English (US)</option>
                  <option value="en-GB">English (UK)</option>
                  <option value="es-ES">Spanish (Spain)</option>
                  <option value="fr-FR">French (France)</option>
                  <option value="de-DE">German (Germany)</option>
                  <option value="ja-JP">Japanese</option>
                  <option value="ko-KR">Korean</option>
                  <option value="zh-CN">Chinese (Simplified)</option>
                </select>
                <p className="text-sm text-muted-foreground">
                  Language for speech recognition and text-to-speech
                </p>
              </div>
            </CardContent>
          </Card>

          {/* AI Section */}
          <Card className="dashboard-surface">
            <CardHeader>
              <CardTitle>AI</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <Toggle
                label="Show AI Reasoning"
                checked={settings.showAIReasoning}
                onChange={(checked) => updateSettings({ showAIReasoning: checked })}
                description="Display AI tactical reasoning panel during NPC turns"
              />
            </CardContent>
          </Card>

          {/* Display Section */}
          <Card className="dashboard-surface">
            <CardHeader>
              <CardTitle>Display</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <Toggle
                label="Dark Theme"
                checked={settings.theme === 'dark'}
                onChange={(checked) => updateSettings({ theme: checked ? 'dark' : 'light' })}
                description="Switch between light and dark interface"
              />
            </CardContent>
          </Card>

          {/* Accessibility Section */}
          <Card className="dashboard-surface">
            <CardHeader>
              <CardTitle>Accessibility</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <Toggle
                label="Reduced Motion"
                checked={settings.reducedMotion}
                onChange={(checked) => updateSettings({ reducedMotion: checked })}
                description="Disable animations and transitions"
              />
              <Toggle
                label="High Contrast"
                checked={settings.highContrast}
                onChange={(checked) => updateSettings({ highContrast: checked })}
                description="Increase contrast for better visibility"
              />
             </CardContent>
           </Card>

           {/* Save/Load Section */}
           <Card className="dashboard-surface">
             <CardHeader>
               <CardTitle>Save/Load</CardTitle>
             </CardHeader>
             <CardContent className="space-y-6">
               <SaveLoadManager />
             </CardContent>
           </Card>
         </div>

        {/* Keyboard navigation note */}
        <div className="text-sm text-muted-foreground pt-8 border-t border-border">
          <p>
            Use <kbd>Tab</kbd> to navigate between settings, <kbd>Enter</kbd> to
            toggle switches, <kbd>Arrow keys</kbd> to adjust sliders.
          </p>
          <p className="mt-2">
            Settings are automatically saved to your browser's local storage.
          </p>
        </div>
      </div>
    </div>
  );
}