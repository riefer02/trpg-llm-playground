import { useLocation } from "@tanstack/react-router";
import { Modal } from "./modal";
import { Card, CardContent, CardHeader, CardTitle } from "./card";
import { Button } from "./button";

export interface KeyboardShortcutsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

type ShortcutCategory = "navigation" | "combat" | "voice" | "general";

interface ShortcutItem {
  key: string;
  description: string;
  category: ShortcutCategory;
  context?: "combat" | "hub" | "all";
}

const ALL_SHORTCUTS: ShortcutItem[] = [
  // Navigation (hub: quarters, missions, etc.)
  { key: "Arrow Keys", description: "Navigate between buttons, cards, or list items", category: "navigation", context: "hub" },
  { key: "Enter", description: "Activate selected item (e.g., open mission briefing)", category: "navigation", context: "hub" },
  { key: "Escape", description: "Go back / focus back button", category: "navigation", context: "hub" },
  // Combat
  { key: "1‑0 (Number keys)", description: "Select corresponding action (1‑9, then 0) when action bar is visible", category: "combat", context: "combat" },
  { key: "Escape", description: "Cancel targeting, close context menu, or close modal", category: "combat", context: "combat" },
  // Voice
  { key: "Space", description: "Push‑to‑talk for voice commands (when voice input enabled)", category: "voice", context: "all" },
  // General
  { key: "?", description: "Show this keyboard shortcuts reference", category: "general", context: "all" },
  { key: "Escape", description: "Close modal or cancel current operation", category: "general", context: "all" },
];

export function KeyboardShortcutsModal({ isOpen, onClose }: KeyboardShortcutsModalProps) {
  const location = useLocation();
  const isCombat = location.pathname.includes("/combat/");
  const titleId = "keyboard-shortcuts-title";
  
  const filteredShortcuts = ALL_SHORTCUTS.filter((item) => {
    if (item.context === "all") return true;
    if (isCombat) return item.context === "combat" || item.context === "all";
    return item.context === "hub" || item.context === "all";
  });
  
  const shortcutsByCategory = filteredShortcuts.reduce((acc, item) => {
    if (!acc[item.category]) acc[item.category] = [];
    acc[item.category].push(item);
    return acc;
  }, {} as Record<ShortcutCategory, ShortcutItem[]>);
  
  const categories: ShortcutCategory[] = ["navigation", "combat", "voice", "general"];
  
  return (
    <Modal isOpen={isOpen} onClose={onClose} ariaLabelledBy={titleId}>
      <Card className="max-w-2xl">
        <CardHeader>
          <CardTitle id={titleId} className="flex items-center justify-between">
            <span>Keyboard Shortcuts</span>
            <span className="text-sm font-normal text-muted-foreground">
              {isCombat ? "Combat Mode" : "Hub Mode"}
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {categories.map((category) => {
            const items = shortcutsByCategory[category];
            if (!items || items.length === 0) return null;
            
            return (
              <div key={category} className="space-y-2">
                <h3 className="text-sm font-semibold text-foreground uppercase tracking-wide">
                  {category === "navigation" && "Navigation"}
                  {category === "combat" && "Combat"}
                  {category === "voice" && "Voice"}
                  {category === "general" && "General"}
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="border-b border-border text-left text-xs text-muted-foreground">
                        <th className="py-2 px-3 w-1/3">Key</th>
                        <th className="py-2 px-3">Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      {items.map((item, idx) => (
                        <tr key={idx} className="border-b border-border/50 last:border-0">
                          <td className="py-3 px-3">
                            <kbd className="inline-flex items-center justify-center min-w-[2rem] px-2 py-1 rounded-md bg-muted border border-border text-sm font-mono font-medium">
                              {item.key}
                            </kbd>
                          </td>
                          <td className="py-3 px-3 text-sm">{item.description}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })}
          <div className="pt-4 border-t border-border text-sm text-muted-foreground">
            <p>Shortcuts are disabled when typing in input fields.</p>
            <p>Press <kbd className="inline-flex items-center justify-center min-w-[1.5rem] px-1 py-0.5 rounded-sm bg-muted border border-border text-xs font-mono">Escape</kbd> or click outside to close.</p>
          </div>
        </CardContent>
      </Card>
    </Modal>
  );
}