/**
 * Settings screen for audio, voice, display, and accessibility preferences.
 */

import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, Slider, Toggle } from "../components/ui";
import { useSettings } from "../lib/hooks/useSettings";

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