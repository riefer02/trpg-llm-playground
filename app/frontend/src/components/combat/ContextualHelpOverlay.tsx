import { useState, useEffect } from "react";
import { Modal } from "../ui/modal";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { 
  Lightbulb, 
  Mic, 
  Zap, 
  Swords, 
  Clock,
  Target,
  Footprints,
  Shield,
  Cpu,
  Heart,
  Flame,
  ChevronRight,
  ChevronLeft
} from "lucide-react";
import type { ActionEconomyState, AvailableActionsResponse } from "../../lib/api";

export interface ContextualHelpOverlayProps {
  isOpen: boolean;
  onClose: () => void;
  economy: ActionEconomyState | null;
  availableActions: AvailableActionsResponse | null;
  currentRound?: number;
}

interface ActionExplanation {
  id: string;
  name: string;
  icon: React.ReactNode;
  description: string;
  type: string;
  tips: string[];
}

interface TipItem {
  id: string;
  title: string;
  content: string;
}

const ACTION_EXPLANATIONS: ActionExplanation[] = [
  {
    id: "move",
    name: "Move",
    icon: <Footprints className="w-4 h-4" />,
    description: "Move your mech up to its speed in hexes.",
    type: "quick",
    tips: ["Click hexes on the canvas to plot a path", "You can undo by clicking the last hex again", "Boost gives you double movement"]
  },
  {
    id: "boost",
    name: "Boost",
    icon: <Zap className="w-4 h-4" />,
    description: "Move at double speed, but generates 2 heat.",
    type: "full",
    tips: ["Great for closing distance quickly", "Watch your heat buildup", "Can be combined with attacks in the same turn"]
  },
  {
    id: "skirmish",
    name: "Skirmish",
    icon: <Swords className="w-4 h-4" />,
    description: "Make a single attack with one weapon.",
    type: "full",
    tips: ["Choose the right weapon for the target", "Check range and accuracy", "Some weapons have special effects"]
  },
  {
    id: "barrage",
    name: "Barrage",
    icon: <Target className="w-4 h-4" />,
    description: "Attack with all weapons at a single target.",
    type: "full",
    tips: ["Best against tough targets", "Uses all weapons in range", "Great for finishing off enemies"]
  },
  {
    id: "quick_tech",
    name: "Quick Tech",
    icon: <Cpu className="w-4 h-4" />,
    description: "Perform a quick technical action.",
    type: "quick",
    tips: ["Scan reveals enemy information", "Lock On gives allies +1d6 to hit", "Use Bolster to help allies"]
  },
  {
    id: "full_tech",
    name: "Full Tech",
    icon: <Cpu className="w-4 h-4" />,
    description: "Perform two quick tech actions or one powerful invade.",
    type: "full",
    tips: ["Invade can inflict devastating effects", "Choose targets carefully", "Great for support builds"]
  },
  {
    id: "stabilize",
    name: "Stabilize",
    icon: <Heart className="w-4 h-4" />,
    description: "Clear heat and repair systems.",
    type: "quick",
    tips: ["Essential when heat is high", "Can prevent reactor meltdown", "Clears Burn condition"]
  },
  {
    id: "brace",
    name: "Brace",
    icon: <Shield className="w-4 h-4" />,
    description: "Gain resistance to all damage until next turn.",
    type: "quick",
    tips: ["Use when expecting heavy damage", "Reduces incoming damage by half", "Good defensive option"]
  },
  {
    id: "overcharge",
    name: "Overcharge",
    icon: <Flame className="w-4 h-4" />,
    description: "Generate heat to gain an extra quick action.",
    type: "free",
    tips: ["Risk vs reward - watch your heat!", "Each overcharge generates more heat", "Can save you in desperate situations"]
  }
];

const KEYBOARD_SHORTCUTS = [
  { key: "?", description: "Open this help overlay" },
  { key: "1-0", description: "Select action by number" },
  { key: "Escape", description: "Cancel targeting or close modal" },
  { key: "Space", description: "Push-to-talk voice commands" },
];

const VOICE_COMMANDS = [
  { command: "Attack [target] with [weapon]", example: "Attack striker with rifle" },
  { command: "Move to [position]", example: "Move forward" },
  { command: "Use [action] on [target]", example: "Use scan on tank" },
  { command: "End turn", example: "I'm done" },
  { command: "Yes / No", example: "Confirm or cancel actions" },
];

type HelpSection = "overview" | "actions" | "shortcuts" | "voice";

const SECTIONS: { id: HelpSection; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "actions", label: "Actions" },
  { id: "shortcuts", label: "Shortcuts" },
  { id: "voice", label: "Voice" },
];

export function ContextualHelpOverlay({
  isOpen,
  onClose,
  economy,
  availableActions,
  currentRound = 1
}: ContextualHelpOverlayProps) {
  const [activeSection, setActiveSection] = useState<HelpSection>("overview");
  const titleId = "contextual-help-title";

  // Reset to overview when opened
  useEffect(() => {
    if (isOpen) {
      setActiveSection("overview");
    }
  }, [isOpen]);

  // Get available action IDs for filtering explanations
  const availableActionIds = new Set([
    ...(availableActions?.full_actions?.map(a => a.action_id) || []),
    ...(availableActions?.quick_actions?.map(a => a.action_id) || []),
    ...(availableActions?.free_actions?.map(a => a.action_id) || []),
  ]);

  // Filter explanations to show only available actions
  const relevantExplanations = ACTION_EXPLANATIONS.filter(
    exp => availableActionIds.has(exp.id) || exp.id === "overcharge"
  );

  // Calculate remaining actions
  const fullRemaining = economy ? 1 - economy.full_actions_used : 1;
  const quickTotal = economy ? 2 + (economy.overcharge_used ? 1 : 0) : 2;
  const quickRemaining = economy ? quickTotal - economy.quick_actions_used : 2;
  const reactRemaining = economy ? 1 - economy.reactions_used_this_turn : 1;

  // Generate context-sensitive tips based on game state
  const getContextTips = (): TipItem[] => {
    const tips: TipItem[] = [];

    if (currentRound === 1) {
      tips.push({
        id: "first-turn",
        title: "First Turn",
        content: "Consider positioning carefully. You can use 1 Full action and 2 Quick actions this turn."
      });
    }

    if (quickRemaining === 0 && fullRemaining === 0) {
      tips.push({
        id: "no-actions",
        title: "Out of Actions",
        content: "You've used all your actions. You can Overcharge to gain +1 Quick action (generates heat), or end your turn."
      });
    }

    if (economy && economy.heat_current > economy.heat_max * 0.7) {
      tips.push({
        id: "high-heat",
        title: "High Heat Warning",
        content: "Your mech is running hot! Consider using Stabilize to clear heat, or you risk reactor stress."
      });
    }

    if (reactRemaining > 0) {
      tips.push({
        id: "reaction-available",
        title: "Reaction Available",
        content: "You have a reaction available. You can use it to interrupt enemy actions during their turn."
      });
    }

    if (availableActions?.can_overcharge && !economy?.overcharge_used) {
      tips.push({
        id: "overcharge-tip",
        title: "Overcharge Available",
        content: "Need more actions? Overcharge gives you +1 Quick action but generates heat."
      });
    }

    return tips;
  };

  const contextTips = getContextTips();

  const renderSectionContent = () => {
    switch (activeSection) {
      case "overview":
        return (
          <div className="space-y-4">
            {/* Turn Phase Explanation */}
            <div className="rounded-lg border border-border bg-muted/30 p-4">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Clock className="w-4 h-4" />
                Your Turn
              </h3>
              <p className="text-sm text-muted-foreground mb-3">
                Each turn, you can use:
              </p>
              <div className="grid grid-cols-3 gap-3">
                <div className={`p-3 rounded-md text-center ${fullRemaining > 0 ? 'bg-blue-500/20 border border-blue-500/50' : 'bg-muted border border-border'}`}>
                  <div className="text-lg font-bold text-blue-500">{fullRemaining}</div>
                  <div className="text-xs text-muted-foreground">Full Action</div>
                </div>
                <div className={`p-3 rounded-md text-center ${quickRemaining > 0 ? 'bg-green-500/20 border border-green-500/50' : 'bg-muted border border-border'}`}>
                  <div className="text-lg font-bold text-green-500">{quickRemaining}</div>
                  <div className="text-xs text-muted-foreground">Quick Actions</div>
                </div>
                <div className={`p-3 rounded-md text-center ${reactRemaining > 0 ? 'bg-amber-500/20 border border-amber-500/50' : 'bg-muted border border-border'}`}>
                  <div className="text-lg font-bold text-amber-500">{reactRemaining}</div>
                  <div className="text-xs text-muted-foreground">Reaction</div>
                </div>
              </div>
              {economy?.overcharge_used && (
                <p className="text-xs text-amber-500 mt-2">
                  Overcharged: +1 Quick action, generated heat
                </p>
              )}
            </div>

            {/* Context-Sensitive Tips */}
            {contextTips.length > 0 && (
              <div className="space-y-3">
                <h3 className="font-semibold flex items-center gap-2">
                  <Lightbulb className="w-4 h-4 text-yellow-500" />
                  Tips for Your Situation
                </h3>
                {contextTips.map((tip) => (
                  <div key={tip.id} className="rounded-lg border border-yellow-500/30 bg-yellow-500/10 p-3">
                    <div className="font-medium text-sm">{tip.title}</div>
                    <p className="text-xs text-muted-foreground mt-1">{tip.content}</p>
                  </div>
                ))}
              </div>
            )}

            {/* Quick Reference */}
            <div className="rounded-lg border border-border p-4">
              <h3 className="font-semibold mb-2">Quick Reference</h3>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Hover over actions to see previews</li>
                <li>• Press number keys (1-0) to quickly select actions</li>
                <li>• Use Spacebar for voice commands</li>
                <li>• Press Escape to cancel targeting</li>
                <li>• Press ? anytime to reopen this help</li>
              </ul>
            </div>
          </div>
        );

      case "actions":
        return (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Available actions for your current turn:
            </p>
            {relevantExplanations.length === 0 ? (
              <p className="text-sm text-muted-foreground italic">
                No actions available. End your turn to proceed.
              </p>
            ) : (
              <div className="grid gap-3">
                {relevantExplanations.map((action) => (
                  <div key={action.id} className="rounded-lg border border-border p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <div className={`w-6 h-6 rounded flex items-center justify-center ${
                        action.type === 'full' ? 'bg-blue-500/20 text-blue-500' :
                        action.type === 'quick' ? 'bg-green-500/20 text-green-500' :
                        'bg-gray-500/20 text-gray-500'
                      }`}>
                        {action.icon}
                      </div>
                      <span className="font-medium">{action.name}</span>
                      <span className={`text-xs px-1.5 py-0.5 rounded ${
                        action.type === 'full' ? 'bg-blue-500/20 text-blue-500' :
                        action.type === 'quick' ? 'bg-green-500/20 text-green-500' :
                        'bg-gray-500/20 text-gray-500'
                      }`}>
                        {action.type}
                      </span>
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">{action.description}</p>
                    <div className="space-y-1">
                      {action.tips.map((tip, idx) => (
                        <div key={idx} className="text-xs text-muted-foreground flex items-start gap-1">
                          <span className="text-primary">•</span>
                          {tip}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        );

      case "shortcuts":
        return (
          <div className="space-y-4">
            <div className="rounded-lg border border-border">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border bg-muted/50">
                    <th className="text-left p-3 text-sm font-medium">Key</th>
                    <th className="text-left p-3 text-sm font-medium">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {KEYBOARD_SHORTCUTS.map((shortcut, idx) => (
                    <tr key={idx} className="border-b border-border/50 last:border-0">
                      <td className="p-3">
                        <kbd className="inline-flex items-center justify-center min-w-[2rem] px-2 py-1 rounded-md bg-muted border border-border text-sm font-mono">
                          {shortcut.key}
                        </kbd>
                      </td>
                      <td className="p-3 text-sm text-muted-foreground">{shortcut.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-muted-foreground">
              Shortcuts are disabled when typing in input fields.
            </p>
          </div>
        );

      case "voice":
        return (
          <div className="space-y-4">
            <div className="rounded-lg border border-border bg-muted/30 p-4">
              <h3 className="font-semibold mb-2 flex items-center gap-2">
                <Mic className="w-4 h-4" />
                Voice Commands
              </h3>
              <p className="text-sm text-muted-foreground mb-4">
                Hold Spacebar and speak naturally. Release to submit.
              </p>
            </div>
            <div className="space-y-3">
              {VOICE_COMMANDS.map((cmd, idx) => (
                <div key={idx} className="rounded-lg border border-border p-3">
                  <div className="font-medium text-sm">{cmd.command}</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Example: "{cmd.example}"
                  </div>
                </div>
              ))}
            </div>
            <div className="rounded-lg border border-blue-500/30 bg-blue-500/10 p-3">
              <p className="text-xs text-blue-500">
                Tip: You can also say "yes" or "no" to confirm or cancel actions during voice confirmation.
              </p>
            </div>
          </div>
        );
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} ariaLabelledBy={titleId}>
      <Card className="max-w-2xl max-h-[80vh] flex flex-col">
        <CardHeader className="flex-none">
          <CardTitle id={titleId} className="flex items-center gap-2">
            <Lightbulb className="w-5 h-5 text-yellow-500" />
            Combat Help & Reference
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-1 overflow-hidden flex flex-col">
          {/* Navigation tabs */}
          <div className="flex gap-1 mb-4 flex-none border-b border-border pb-2">
            {SECTIONS.map((section) => (
              <button
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                  activeSection === section.id
                    ? 'bg-primary text-primary-foreground'
                    : 'hover:bg-muted text-muted-foreground'
                }`}
              >
                {section.label}
              </button>
            ))}
          </div>

          {/* Content area */}
          <div className="flex-1 overflow-y-auto">
            {renderSectionContent()}
          </div>

          <div className="flex justify-between items-center mt-4 pt-4 border-t border-border flex-none">
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  const currentIdx = SECTIONS.findIndex(s => s.id === activeSection);
                  if (currentIdx > 0) {
                    setActiveSection(SECTIONS[currentIdx - 1].id);
                  }
                }}
                disabled={activeSection === "overview"}
                className="p-1 rounded hover:bg-muted disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="text-sm text-muted-foreground">
                {SECTIONS.findIndex(s => s.id === activeSection) + 1} / {SECTIONS.length}
              </span>
              <button
                onClick={() => {
                  const currentIdx = SECTIONS.findIndex(s => s.id === activeSection);
                  if (currentIdx < SECTIONS.length - 1) {
                    setActiveSection(SECTIONS[currentIdx + 1].id);
                  }
                }}
                disabled={activeSection === "voice"}
                className="p-1 rounded hover:bg-muted disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
            <Button variant="outline" onClick={onClose}>
              Close
            </Button>
          </div>
        </CardContent>
      </Card>
    </Modal>
  );
}
