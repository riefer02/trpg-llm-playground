import { useEffect, useState } from "react";
import {
  Footprints,
  Zap,
  Swords,
  Crosshair,
  ArrowBigRight,
  Grip,
  Cpu,
  Radio,
  Scan,
  Lock,
  Heart,
  Flame,
  Eye,
  Shield,
  Hand,
  RotateCcw,
  Skull,
  Target,
  Mic,
  type LucideIcon,
} from "lucide-react";

import type {
  ActionEconomyState,
  AvailableActionItem,
  AvailableActionsResponse,
} from "../../lib/api/combat";
import { useActionPreview } from "../../lib/api/combat";
import type { CombatantState } from "../../lib/types/lancer";

/**
 * WoW-style action bar for combat actions.
 * Fixed at the bottom of the screen during active turn.
 */

// Icon mapping for each action
const ACTION_ICONS: Record<string, LucideIcon> = {
  // Movement
  move: Footprints,
  boost: Zap,
  // Attacks
  skirmish: Swords,
  barrage: Crosshair,
  fight: Swords,
  // Melee
  ram: ArrowBigRight,
  grapple: Grip,
  // Tech
  quick_tech: Cpu,
  full_tech: Radio,
  scan: Scan,
  lock_on: Lock,
  invade: Skull,
  bolster: Shield,
  // Utility
  stabilize: Heart,
  activate_system: Cpu,
  reload: RotateCcw,
  // Defensive
  overwatch: Eye,
  brace: Shield,
  // Special
  overcharge: Flame,
  dismount: Hand,
  eject: Hand,
  self_destruct: Skull,
  hide: Eye,
  search: Scan,
  prepare: Target,
  disengage: Footprints,
  improvised_attack: Swords,
};

// Default icon for unmapped actions
const DEFAULT_ICON = Target;

// Action type colors
const ACTION_TYPE_COLORS = {
  full: "border-blue-500",
  quick: "border-green-500",
  free: "border-gray-400",
  protocol: "border-purple-500",
  reaction: "border-amber-500",
};

interface ActionButtonProps {
  action: AvailableActionItem;
  disabled: boolean;
  onClick: () => void;
  shortcutKey?: string;
  sessionId?: string;
  actorId?: string | null;
  targetId?: string | null;
  weaponId?: string | null;
}

function ActionButton({ 
  action, 
  disabled, 
  onClick, 
  shortcutKey,
  sessionId,
  actorId = null,
  targetId = null,
  weaponId = null,
}: ActionButtonProps) {
  const [isHovered, setIsHovered] = useState(false);
  const { mutate: fetchPreview, data: previewData, isPending: isPreviewLoading, error: previewError } = useActionPreview(sessionId || '');

  // Fetch preview when hovered and we have required IDs
  useEffect(() => {
    if (!isHovered || !sessionId || !actorId || !targetId) {
      return;
    }
    // Debounce to avoid rapid calls
    const timer = setTimeout(() => {
      fetchPreview({
        sessionId,
        actionId: action.action_id,
        actorId,
        targetId,
        weaponId: weaponId ?? undefined,
      });
    }, 150);
    return () => clearTimeout(timer);
  }, [isHovered, sessionId, actorId, targetId, weaponId, action.action_id, fetchPreview]);

  const Icon = ACTION_ICONS[action.action_id] ?? DEFAULT_ICON;
  const isAvailable = action.is_available && !disabled;
  const typeColor = ACTION_TYPE_COLORS[action.action_type as keyof typeof ACTION_TYPE_COLORS] ?? "border-gray-500";

  return (
    <div className="group relative">
      <button
        type="button"
        onClick={onClick}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        disabled={!isAvailable}
        className={`
          relative w-12 h-12 rounded-lg border-2 transition-all
          flex items-center justify-center
          ${typeColor}
          ${isAvailable
            ? "bg-muted hover:bg-primary/20 hover:border-primary cursor-pointer shadow-md hover:shadow-lg hover:scale-105"
            : "bg-muted/30 opacity-40 cursor-not-allowed"
          }
          ${action.unavailable_reason ? "border-destructive/50" : ""}
        `}
        aria-label={action.action_name}
      >
        <Icon
          className={`w-6 h-6 ${
            isAvailable ? "text-foreground" : "text-muted-foreground/50"
          }`}
        />
        {shortcutKey && (
          <span className="absolute bottom-0.5 right-0.5 text-[10px] font-mono text-muted-foreground/70">
            {shortcutKey}
          </span>
        )}
      </button>

      {/* Tooltip */}
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-150 pointer-events-none z-50">
        <div className="bg-popover border border-border rounded-md shadow-lg px-3 py-2 min-w-[160px] max-w-[240px]">
          <div className="font-medium text-sm text-foreground">
            {action.action_name}
          </div>
          <div className="text-xs text-muted-foreground capitalize">
            {action.action_type} action
            {action.requires_weapon && " · weapon"}
            {action.requires_target && " · target"}
            {action.requires_path && " · path"}
            {action.requires_system && " · system"}
          </div>
          {action.unavailable_reason && (
            <div className="text-xs text-destructive mt-1">
              {action.unavailable_reason}
            </div>
          )}
          {shortcutKey && (
            <div className="text-xs text-muted-foreground/70 mt-1">
              Press {shortcutKey}
            </div>
          )}
          {/* Action Preview */}
          {isPreviewLoading && (
            <div className="text-xs text-muted-foreground mt-1">
              Calculating preview...
            </div>
          )}
          {previewError && (
            <div className="text-xs text-destructive mt-1">
              Preview failed: {previewError.message}
            </div>
          )}
          {previewData && !isPreviewLoading && !previewError && (
            <div className="mt-2 pt-2 border-t border-border space-y-1">
              {/* Hit Probability */}
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Hit Chance</span>
                <span className="font-mono text-foreground">
                  {(previewData.hit_probability * 100).toFixed(0)}%
                </span>
              </div>
              {/* Damage Range */}
              {previewData.damage_average > 0 && (
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Damage</span>
                  <span className="font-mono text-foreground">
                    {previewData.damage_min}-{previewData.damage_max} avg {previewData.damage_average.toFixed(1)}
                  </span>
                </div>
              )}
              {/* Damage Types */}
              {previewData.damage_types.length > 0 && (
                <div className="text-xs text-muted-foreground">
                  Damage types: {previewData.damage_types.join(', ')}
                </div>
              )}
              {/* Predicted Effects */}
              {previewData.predicted_effects.length > 0 && (
                <div className="text-xs text-muted-foreground">
                  Effects: {previewData.predicted_effects.length} predicted
                </div>
              )}
            </div>
          )}
        </div>
        {/* Tooltip arrow */}
        <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1">
          <div className="w-2 h-2 bg-popover border-r border-b border-border rotate-45" />
        </div>
      </div>
    </div>
  );
}

interface EconomyDisplayProps {
  economy: ActionEconomyState;
  canOvercharge: boolean;
  overchargeLevel: number;
}

function EconomyDisplay({ economy, canOvercharge, overchargeLevel }: EconomyDisplayProps) {
  const fullRemaining = 1 - economy.full_actions_used;
  const quickTotal = 2 + (economy.overcharge_used ? 1 : 0);
  const quickRemaining = quickTotal - economy.quick_actions_used;
  const reactRemaining = 1 - economy.reactions_used_this_turn;

  return (
    <div className="flex items-center gap-4 px-4 py-2 bg-background/80 rounded-lg border border-border">
      <div className="flex items-center gap-1.5">
        <span className="text-xs text-blue-400">Full</span>
        <div className="flex gap-0.5">
          {[0].map((i) => (
            <div
              key={i}
              className={`w-3 h-3 rounded-full ${
                i < fullRemaining ? "bg-blue-500" : "bg-muted-foreground/20"
              }`}
            />
          ))}
        </div>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-xs text-green-400">Quick</span>
        <div className="flex gap-0.5">
          {Array.from({ length: quickTotal }).map((_, i) => (
            <div
              key={i}
              className={`w-3 h-3 rounded-full ${
                i < quickRemaining ? "bg-green-500" : "bg-muted-foreground/20"
              }`}
            />
          ))}
        </div>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-xs text-amber-400">React</span>
        <div className="flex gap-0.5">
          {[0].map((i) => (
            <div
              key={i}
              className={`w-3 h-3 rounded-full ${
                i < reactRemaining ? "bg-amber-500" : "bg-muted-foreground/20"
              }`}
            />
          ))}
        </div>
      </div>
      {canOvercharge && !economy.overcharge_used && (
        <span className="text-xs px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 font-medium">
          OC
        </span>
      )}
      {economy.overcharge_used && (
        <span className="text-xs px-1.5 py-0.5 rounded bg-destructive/20 text-destructive font-medium">
          OC{overchargeLevel}
        </span>
          )}
        </div>
  );
}

export interface ActionBarProps {
  sessionId: string;
  availableActions: AvailableActionsResponse | null;
  economy: ActionEconomyState | null;
  onActionSelect: (action: AvailableActionItem) => void;
  onOvercharge: () => void;
  canOvercharge?: boolean;
  overchargeLevel?: number;
  isExecuting?: boolean;
  visible?: boolean;
  // Preview targeting
  previewTargetId?: string | null;
  currentActor?: CombatantState | null;
  // Voice control
  onVoiceToggle?: () => void;
  isVoiceListening?: boolean;
  voiceEnabled?: boolean;
  voiceSupported?: boolean;
}

export function ActionBar({
  sessionId,
  availableActions,
  economy,
  onActionSelect,
  onOvercharge,
  canOvercharge = false,
  overchargeLevel = 0,
  isExecuting = false,
  visible = true,
  // Preview targeting
  previewTargetId = null,
  currentActor = null,
  // Voice control
  onVoiceToggle = () => {},
  isVoiceListening = false,
  voiceEnabled = false,
  voiceSupported = false,
}: ActionBarProps) {
  // Helper to get first weapon ID for an action that requires a weapon
  const getWeaponIdForAction = (action: AvailableActionItem): string | null => {
    if (!action.requires_weapon || !currentActor?.inventory?.mounts) {
      return null;
    }
    // Find first weapon in inventory (any mount)
    for (const mount of currentActor.inventory.mounts) {
      if (mount.weapons && mount.weapons.length > 0) {
        // Return first weapon ID
        return mount.weapons[0].weapon_id;
      }
    }
    return null;
  };

  // Flatten actions into a single array with shortcuts
  const allActions = [
    ...(availableActions?.full_actions ?? []),
    ...(availableActions?.quick_actions ?? []),
    ...(availableActions?.free_actions ?? []),
    ...(availableActions?.protocols ?? []),
  ];

  // Assign keyboard shortcuts (1-9, then 0)
  const actionsWithShortcuts = allActions.slice(0, 10).map((action, i) => ({
    action,
    shortcut: i === 9 ? "0" : String(i + 1),
  }));

  // Check economy constraints
  const isFullDisabled = economy ? economy.full_actions_used > 0 : true;
  const isQuickDisabled = economy
    ? economy.quick_actions_used >= 2 + (economy.overcharge_used ? 1 : 0)
    : true;

  const getDisabledState = (action: AvailableActionItem) => {
    if (isExecuting) return true;
    if (!action.is_available) return true;
    if (action.action_type === "full" && isFullDisabled) return true;
    if (action.action_type === "quick" && isQuickDisabled) return true;
    return false;
  };

  // Keyboard shortcuts
  useEffect(() => {
    if (!visible) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      const key = e.key;
      const match = actionsWithShortcuts.find((a) => a.shortcut === key);
      if (match && !getDisabledState(match.action)) {
        e.preventDefault();
        onActionSelect(match.action);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [visible, actionsWithShortcuts, onActionSelect, isExecuting]);

  if (!visible || !availableActions || !economy) {
    return null;
  }

  // Group actions by type for visual separation
  const fullActions = availableActions.full_actions;
  const quickActions = availableActions.quick_actions;
  const freeActions = availableActions.free_actions;
  const protocols = availableActions.protocols;

  // Create shortcut map
  let shortcutIndex = 0;
  const getNextShortcut = () => {
    const idx = shortcutIndex++;
    return idx < 9 ? String(idx + 1) : idx === 9 ? "0" : undefined;
  };

  return (
    <div className="fixed bottom-0 left-0 right-0 z-30 pointer-events-none">
      <div className="flex justify-center pb-4">
        <div className="pointer-events-auto bg-background/95 backdrop-blur-sm border border-border rounded-xl shadow-xl px-4 py-3 flex flex-col gap-3">
          {/* Top Row: Economy + Full + Quick Actions */}
          <div className="flex items-center justify-center gap-6">
            {/* Economy Display */}
            <EconomyDisplay
              economy={economy}
              canOvercharge={canOvercharge}
              overchargeLevel={overchargeLevel}
            />

            {/* Divider */}
            {(fullActions.length > 0 || quickActions.length > 0) && (
              <div className="w-px h-10 bg-border" />
            )}

            {/* Full Actions */}
            {fullActions.length > 0 && (
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-blue-400 uppercase font-medium">Full</span>
                <div className="flex gap-1.5">
                  {fullActions.map((action) => (
                    <ActionButton
                      key={action.action_id}
                      action={action}
                      disabled={getDisabledState(action)}
                      onClick={() => onActionSelect(action)}
                      shortcutKey={getNextShortcut()}
                      sessionId={sessionId}
                      actorId={currentActor?.id ?? null}
                      targetId={previewTargetId}
                      weaponId={getWeaponIdForAction(action)}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Quick Actions */}
            {quickActions.length > 0 && (
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-green-400 uppercase font-medium">Quick</span>
                <div className="flex gap-1.5">
                  {quickActions.map((action) => (
                    <ActionButton
                      key={action.action_id}
                      action={action}
                      disabled={getDisabledState(action)}
                      onClick={() => onActionSelect(action)}
                      shortcutKey={getNextShortcut()}
                      sessionId={sessionId}
                      actorId={currentActor?.id ?? null}
                      targetId={previewTargetId}
                      weaponId={getWeaponIdForAction(action)}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Bottom Row: Free + Protocol + Overcharge + Voice */}
          {(freeActions.length > 0 || protocols.length > 0 || (canOvercharge && !economy.overcharge_used) || (voiceSupported && voiceEnabled)) && (
            <div className="flex items-center justify-center gap-6">
              {/* Free Actions */}
              {freeActions.length > 0 && (
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-muted-foreground uppercase font-medium">Free</span>
                  <div className="flex gap-1.5">
                    {freeActions.map((action) => (
                      <ActionButton
                        key={action.action_id}
                        action={action}
                        disabled={getDisabledState(action)}
                        onClick={() => onActionSelect(action)}
                        shortcutKey={getNextShortcut()}
                        sessionId={sessionId}
                        actorId={currentActor?.id ?? null}
                        targetId={previewTargetId}
                        weaponId={getWeaponIdForAction(action)}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Protocols */}
              {protocols.length > 0 && (
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-purple-400 uppercase font-medium">Protocol</span>
                  <div className="flex gap-1.5">
                    {protocols.map((action) => (
                      <ActionButton
                        key={action.action_id}
                        action={action}
                        disabled={getDisabledState(action)}
                        onClick={() => onActionSelect(action)}
                        shortcutKey={getNextShortcut()}
                        sessionId={sessionId}
                        actorId={currentActor?.id ?? null}
                        targetId={previewTargetId}
                        weaponId={getWeaponIdForAction(action)}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Overcharge Button */}
              {canOvercharge && !economy.overcharge_used && (
                <>
                  {(freeActions.length > 0 || protocols.length > 0) && (
                    <div className="w-px h-10 bg-border" />
                  )}
                  <div className="group relative">
                    <button
                      type="button"
                      onClick={onOvercharge}
                      disabled={isExecuting}
                      className={`
                        relative w-12 h-12 rounded-lg border-2 border-amber-500 transition-all
                        flex items-center justify-center
                        bg-amber-500/10 hover:bg-amber-500/30 hover:border-amber-400
                        cursor-pointer shadow-md hover:shadow-lg hover:scale-105
                        ${isExecuting ? "opacity-40 cursor-not-allowed" : ""}
                      `}
                      aria-label="Overcharge"
                    >
                      <Flame className="w-6 h-6 text-amber-500" />
                    </button>
                    {/* Tooltip */}
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-150 pointer-events-none z-50">
                      <div className="bg-popover border border-border rounded-md shadow-lg px-3 py-2 min-w-[160px]">
                        <div className="font-medium text-sm text-amber-500">Overcharge</div>
                        <div className="text-xs text-muted-foreground">
                          +1 Quick action, generates heat
                        </div>
                        <div className="text-xs text-amber-400/70 mt-1">
                          Level {overchargeLevel + 1} heat
                        </div>
                      </div>
                      <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1">
                        <div className="w-2 h-2 bg-popover border-r border-b border-border rotate-45" />
                      </div>
                    </div>
                  </div>
                </>
              )}

              {/* Voice Toggle Button */}
              {voiceSupported && voiceEnabled && (
                <>
                  {(freeActions.length > 0 || protocols.length > 0 || (canOvercharge && !economy.overcharge_used)) && (
                    <div className="w-px h-10 bg-border" />
                  )}
                  <div className="group relative">
                    {/* Pulsing ring when listening */}
                    {isVoiceListening && (
                      <div className="absolute inset-0 rounded-lg border-2 border-green-500 animate-ping opacity-60" />
                    )}
                    <button
                      type="button"
                      onClick={onVoiceToggle}
                      disabled={isExecuting}
                      className={`
                        relative w-12 h-12 rounded-lg border-2 transition-all
                        flex items-center justify-center
                        ${isVoiceListening
                          ? "border-green-500 bg-green-500/10 hover:bg-green-500/30 hover:border-green-400 animate-pulse"
                          : "border-blue-500 bg-blue-500/10 hover:bg-blue-500/30 hover:border-blue-400"
                        }
                        cursor-pointer shadow-md hover:shadow-lg hover:scale-105
                        ${isExecuting ? "opacity-40 cursor-not-allowed" : ""}
                      `}
                      aria-label="Voice control"
                    >
                      <Mic className={`w-6 h-6 ${isVoiceListening ? "text-green-500" : "text-blue-500"}`} />
                    </button>
                    {/* Tooltip */}
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-150 pointer-events-none z-50">
                      <div className="bg-popover border border-border rounded-md shadow-lg px-3 py-2 min-w-[160px]">
                        <div className="font-medium text-sm">
                          {isVoiceListening ? "Stop voice input" : "Start voice input"}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {isVoiceListening ? "Click or press space to stop" : "Click or press space to start"}
                        </div>
                      </div>
                      <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1">
                        <div className="w-2 h-2 bg-popover border-r border-b border-border rotate-45" />
                      </div>
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
