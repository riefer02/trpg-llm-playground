import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  createFileRoute,
  Link,
  useNavigate,
  useSearch,
  useBlocker,
} from "@tanstack/react-router";
import { toast } from "sonner";
import { useCanvasViewport } from "../../lib/hooks/useCanvasViewport";
import { useSettings } from "../../lib/hooks/useSettings";
import { useLowHPWarning } from "../../lib/hooks/useLowHPWarning";
import { useKeyboardShortcuts } from "../../lib/hooks/useKeyboardShortcuts";
import { useModalManager } from "../../hooks/useModalManager";
import { useCombatNarration } from "../../lib/voice/text-to-speech";
import { useSpeechRecognition } from "../../lib/voice/speech-to-text";

import {
  useCombatSession,
  useCombatWebSocket,
  useStartTurn,
  useEndTurn,
  useExecuteAction,
  useAvailableActions,
  useSubmitReaction,
  useReactionOpportunity,
  usePendingDecisions,
  useSubmitDecision,
  useCompleteCombat,
  useForfeitCombat,
  useSpendReserve,
  useWeapons,
  useAutoNpcTurn,
  type AutoNPCTurnResponse,
  type ActionRequest,
  type ActionEconomyState,
  type AvailableActionItem,
  type ReactionRequest,
  type DecisionSubmitRequest,
  type CombatCompleteRequest,
  type ActionPreviewResponse,
  useActionPreview,
} from "../../lib/api";
import { useParseVoiceIntent } from "../../lib/api/combat";
import {
  CombatCanvas,
  type TargetingMode,
  type ContextMenuInfo,
} from "../../components/combat/CombatCanvas";
import { ViewportControls } from "../../components/combat/ViewportControls";
import {
  ContextMenu,
  type ContextMenuTarget,
  type ContextMenuOption,
} from "../../components/combat/ContextMenu";
import {
  MapTooltip,
  type HoverTarget,
} from "../../components/combat/MapTooltip";
import { ActionPreviewPanel } from "../../components/combat/ActionPreviewPanel";
import { AIThinkingIndicator } from "../../components/combat/AIThinkingIndicator";
import { type SelectedAction } from "../../components/combat/ActionLog";
import { CurrentActorPanel } from "../../components/combat/CurrentActorPanel";
import { InitiativeStrip } from "../../components/combat/InitiativeStrip";
import { CollapsibleActionLog } from "../../components/combat/CollapsibleActionLog";
import { CombatantList } from "../../components/combat/CombatantList";
import {
  TurnControls,
  type TurnState,
} from "../../components/combat/TurnControls";
// TurnIndicator removed - replaced by CurrentActorPanel and InitiativeStrip (E9-US-006)
import {
  ActionPanel,
  type TargetMode,
  type ActionConfirmationReadyData,
  type ActionPanelHandle,
} from "../../components/combat/ActionPanel";
import { ActionBar } from "../../components/combat/ActionBar";
import {
  ActionConfirmationOverlay,
  type ActionConfirmationData,
} from "../../components/combat/ActionConfirmationOverlay";
import { TurnStateBanner, type TurnBannerState } from "../../components/combat/TurnStateBanner";
import { ActionFeed } from "../../components/combat/ActionFeed";
import { VoiceTranscriptDisplay } from "../../components/combat/VoiceTranscriptDisplay";
import { VoiceActionConfirmationDialog } from "../../components/combat/VoiceActionConfirmationDialog";
import { OverchargeConfirm } from "../../components/combat/OverchargeConfirm";
import { ReactionPrompt } from "../../components/combat/ReactionPrompt";
import { SaveCheckPrompt } from "../../components/combat/SaveCheckPrompt";
import { TraumaSelectionPrompt } from "../../components/combat/TraumaSelectionPrompt";
import { MissionCompleteModal } from "../../components/combat/MissionCompleteModal";
import { ForfeitConfirmationModal } from "../../components/combat/ForfeitConfirmationModal";
import { PauseMenu } from "../../components/combat/PauseMenu";
import { InGameSettings } from "../../components/combat/InGameSettings";
import { VictoryCelebration } from "../../components/combat/VictoryCelebration";
import { VictoryConditionPanel } from "../../components/combat/VictoryConditionPanel";
import { ObjectiveTracker } from "../../components/combat/ObjectiveTracker";
import { ReservesPanel } from "../../components/combat/ReservesPanel";
import { ContextualHelpOverlay } from "../../components/combat/ContextualHelpOverlay";
import { FirstCombatTutorial } from "../../components/combat/FirstCombatTutorial";
import {
  adaptCombatScenario,
  buildMovementRangeOverlays,
  type CombatRenderAdapterOutput,
} from "../../lib/combat-render/adapter";
import { createHexLayout } from "../../lib/combat-render/hex";
import type { HexCoord, AttackPatternDefinition } from "../../lib/types/lancer";
import { Button, Card, CardContent, Modal } from "../../components/ui";
import { CombatSessionSkeleton } from "../../components/skeletons";

interface CombatSearch {
  missionId?: string;
}

export const Route = createFileRoute("/combat/$combatId")({
  component: CombatSessionPage,
  validateSearch: (search: Record<string, unknown>): CombatSearch => ({
    missionId:
      typeof search.missionId === "string" ? search.missionId : undefined,
  }),
});

/** Polling interval when WebSocket is disconnected (5 seconds) */
const FALLBACK_POLLING_INTERVAL = 5000;

function CombatSessionPage() {
  const { combatId } = Route.useParams();
  const search = useSearch({ from: Route.fullPath });

  // WebSocket connection for real-time updates
  const { isConnected: wsConnected } = useCombatWebSocket(combatId);

  // Fallback to polling if WebSocket is disconnected
  const { data, isLoading, error } = useCombatSession(combatId, {
    pollingInterval: wsConnected ? undefined : FALLBACK_POLLING_INTERVAL,
  });

  // Turn management mutations
  const startTurn = useStartTurn(combatId);
  const endTurn = useEndTurn(combatId);
  const executeAction = useExecuteAction(combatId);
  const submitReaction = useSubmitReaction(combatId);
  const completeCombat = useCompleteCombat(combatId);
  const forfeitCombat = useForfeitCombat(combatId);
  const spendReserve = useSpendReserve(combatId);
  const autoNpcTurn = useAutoNpcTurn(combatId);
  const { settings } = useSettings();
  const {
    narrateAction,
    narrateTurnStart,
    narrateStatusChange,
    narrateVictory,
  } = useCombatNarration();
  // Speech recognition
  const speechRecognition = useSpeechRecognition({
    language: settings.voiceLanguage,
    continuous: false,
    interimResults: true,
  });
  const parseVoiceIntent = useParseVoiceIntent(combatId);
  // Voice control state
  const [voiceTranscript, setVoiceTranscript] = useState("");
  const [showVoiceConfirmation, setShowVoiceConfirmation] = useState(false);
  const [parsedAction, setParsedAction] = useState<Record<
    string,
    unknown
  > | null>(null);
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const navigate = useNavigate();

  // ActionPanel ref for cancel targeting
  const actionPanelRef = useRef<ActionPanelHandle>(null);

  // Extract UI state and scenario data
  // UI state provides pre-computed lookups, scenario provides full data when needed
  const ui = data?.ui ?? null;
  const scenario = useMemo(() => data?.scenario ?? null, [data?.scenario]);
  const rounds = scenario?.rounds ?? [];
  const currentRound = data?.current_round ?? 1;
  const currentTurnIndex = data?.current_turn_index ?? 0;
  // Prefer UI combatants when available (minimal data, already typed)
  const combatants = scenario?.combatants ?? [];
  const lowHPWarning = useLowHPWarning(combatants, 'players');

  // Determine current actor - use UI state when available (pre-computed)
  const currentActor = useMemo(() => {
    // If UI state has current_actor, use it to find the full combatant
    if (ui?.current_actor) {
      const actorId = ui.current_actor.actor_id;
      return combatants.find((c) => c.id === actorId) ?? null;
    }
    // Fall back to computing from scenario
    if (!scenario) return null;
    const round = scenario.rounds?.[currentRound - 1];
    const turn = round?.turns?.[currentTurnIndex];
    if (turn?.actor_id) {
      return combatants.find((c) => c.id === turn.actor_id) ?? null;
    }
    return combatants[0] ?? null;
  }, [ui?.current_actor, scenario, currentRound, currentTurnIndex, combatants]);

  // Determine if it's currently the player's turn - use UI state when available
  const isPlayerTurn = useMemo(() => {
    if (ui) return ui.is_player_turn;
    return currentActor?.side === "players";
  }, [ui, currentActor]);

  // Voice transcript parsing
  const lastParsedTranscriptRef = useRef("");
  useEffect(() => {
    // Skip parsing if confirmation dialog is open (waiting for yes/no)
    if (showVoiceConfirmation) return;
    if (
      !speechRecognition.isListening &&
      speechRecognition.transcript &&
      speechRecognition.transcript !== lastParsedTranscriptRef.current
    ) {
      lastParsedTranscriptRef.current = speechRecognition.transcript;
      setVoiceTranscript(speechRecognition.transcript);
      parseVoiceIntent.mutate(
        {
          transcript: speechRecognition.transcript,
          actor_id: currentActor?.id,
        },
        {
          onSuccess: (data) => {
            if (data.success && data.action) {
              setParsedAction(data.action);
              setShowVoiceConfirmation(true);
              setVoiceError(null);
            } else {
              setVoiceError(data.error || "Could not understand command");
              setParsedAction(null);
            }
          },
          onError: (error) => {
            setVoiceError(error.message || "Failed to parse voice command");
          },
        },
      );
    }
  }, [
    speechRecognition.isListening,
    speechRecognition.transcript,
    currentActor?.id,
    parseVoiceIntent,
    showVoiceConfirmation,
  ]);

  // Yes/No voice confirmation when dialog is open
  useEffect(() => {
    if (
      showVoiceConfirmation &&
      speechRecognition.transcript &&
      !speechRecognition.isListening
    ) {
      const transcript = speechRecognition.transcript.trim().toLowerCase();
      // Check for yes/no keywords
      if (
        transcript === "yes" ||
        transcript === "confirm" ||
        transcript === "okay" ||
        transcript === "ok"
      ) {
        // Trigger confirm
        if (parsedAction) {
          executeAction.mutate(parsedAction as ActionRequest, {
            onSuccess: () => {
              setShowVoiceConfirmation(false);
              setParsedAction(null);
              setVoiceTranscript("");
              speechRecognition.resetTranscript();
            },
            onError: (error) => {
              setVoiceError(error.message || "Action execution failed");
            },
          });
        }
      } else if (
        transcript === "no" ||
        transcript === "cancel" ||
        transcript === "dismiss"
      ) {
        // Trigger cancel
        setShowVoiceConfirmation(false);
        setVoiceError(null);
        setParsedAction(null);
        setVoiceTranscript("");
        speechRecognition.resetTranscript();
      }
      // Reset transcript after processing to avoid re-triggering
      lastParsedTranscriptRef.current = speechRecognition.transcript;
    }
  }, [
    showVoiceConfirmation,
    speechRecognition.transcript,
    speechRecognition.isListening,
    parsedAction,
    executeAction,
  ]);

  // Modal state - consolidated via discriminated union (E10-US-013)
  const { modal, openModal, closeModal, isOpen, isAnyModalOpen } = useModalManager();

  // Victory celebration state (separate from modal - displays as full-screen overlay)
  const [showVictoryCelebration, setShowVictoryCelebration] = useState(false);
  const [victoryOutcome, setVictoryOutcome] = useState<"victory" | "defeat">(
    "victory",
  );

  // Navigation protection state (E8-US-004)
  const [isForfeiting, setIsForfeiting] = useState(false);

  // Tutorial dismissal state (separate - tracks session/permanent dismissal)
  const [tutorialDismissedThisSession, setTutorialDismissedThisSession] = useState(false);
  const [hasSeenTutorial, setHasSeenTutorial] = useState(() => {
    // Check localStorage for whether user has seen tutorial
    if (typeof window !== 'undefined') {
      return localStorage.getItem('lancer-combat-tutorial-seen') === 'true';
    }
    return false;
  });

  // Keyboard shortcuts for help
  const keyboardShortcuts = useKeyboardShortcuts();

  // Global keyboard shortcut: Ctrl+Q to forfeit mission
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+Q (or Cmd+Q on Mac) - forfeit mission
      if ((e.ctrlKey || e.metaKey) && e.key === 'q' && !e.repeat) {
        // Only trigger if mission is active, it's player's turn, and no modal is open
        if (data?.status === 'active' && isPlayerTurn && !isAnyModalOpen) {
          e.preventDefault();
          openModal({ type: "forfeit" });
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [data?.status, isPlayerTurn, isAnyModalOpen, openModal]);

  // Global keyboard shortcut: ? to open help overlay
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // ? key (Shift+/) - open help overlay
      if (e.key === '?' && !e.ctrlKey && !e.metaKey && !e.altKey && !e.repeat) {
        // Don't trigger if user is typing in an input
        if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
          return;
        }
        // Don't trigger if other modals are open or showing voice/victory overlays
        if (showVoiceConfirmation || showVictoryCelebration || isAnyModalOpen) {
          return;
        }
        e.preventDefault();
        openModal({ type: "help" });
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showVoiceConfirmation, showVictoryCelebration, isAnyModalOpen, openModal]);

  // Show first combat tutorial when entering first turn of first combat
  useEffect(() => {
    if (data?.status === 'active' &&
        currentRound === 1 &&
        isPlayerTurn &&
        !hasSeenTutorial &&
        !tutorialDismissedThisSession &&
        !isOpen("tutorial")) {
      // Small delay to let the combat UI settle
      const timer = setTimeout(() => {
        openModal({ type: "tutorial" });
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [data?.status, currentRound, isPlayerTurn, hasSeenTutorial, tutorialDismissedThisSession, isOpen, openModal]);

  // Handle tutorial "don't show again" preference
  const handleTutorialDontShowAgain = () => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('lancer-combat-tutorial-seen', 'true');
      setHasSeenTutorial(true);
    }
  };

  // Navigation blocking (E8-US-004): Prevent accidental navigation during combat
  const blocker = useBlocker({
    shouldBlockFn: () => {
      // Only block if combat is active, not forfeiting, and not already showing confirmation
      return data?.status === 'active' && !isForfeiting && !isOpen("navigationConfirm");
    },
    enableBeforeUnload: true,
    withResolver: true,
  });

  // Handle blocker status changes to show custom modal
  useEffect(() => {
    if (blocker.status === 'blocked') {
      openModal({ type: "navigationConfirm" });
    }
  }, [blocker.status, openModal]);

  // Handle navigation confirmation response
  const handleNavigationConfirm = useCallback((shouldLeave: boolean) => {
    closeModal();
    if (shouldLeave && blocker.status === 'blocked') {
      blocker.proceed();
    } else {
      blocker.reset();
    }
  }, [blocker, closeModal]);

  // beforeunload event for browser tab/window closing (E8-US-004)
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      // Only show warning if combat is active and not forfeiting
      if (data?.status === 'active' && !isForfeiting) {
        // Standard beforeunload message (browser shows generic text for security)
        e.preventDefault();
        // Required for older browsers
        e.returnValue = '';
        return '';
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [data?.status, isForfeiting]);

  // Weapons data (needed for voice confirmation mapping)
  const weaponsQuery = useWeapons();

  // Mapping functions for friendly names in voice confirmation dialog
  const getCombatantName = useCallback(
    (id: string): string => {
      if (!data?.scenario?.combatants) return id;
      const combatant = data.scenario.combatants[id];
      return combatant?.name ?? id;
    },
    [data?.scenario?.combatants],
  );

  const getWeaponName = useCallback(
    (id: string): string => {
      // Map weapon ID to weapon name from compendium
      if (weaponsQuery.data) {
        const weapon = weaponsQuery.data.find((w) => w.id === id);
        if (weapon) return weapon.name;
      }
      // Fallback: humanized ID
      return id.replace(/_/g, " ");
    },
    [weaponsQuery.data],
  );

  // Helper to get first weapon ID for an action that requires a weapon
  const getWeaponIdForAction = useCallback(
    (action: AvailableActionItem): string | null => {
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
    },
    [currentActor],
  );

  // Show error toast when voiceError changes
  useEffect(() => {
    if (voiceError) {
      toast.error(`Voice command error: ${voiceError}`);
    }
  }, [voiceError]);

  // Turn state tracking (must be defined before useEffect that uses turnActive)
  const [turnActive, setTurnActive] = useState(false);
  const [economy, setEconomy] = useState<ActionEconomyState | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  // AI reasoning display
  const [aiReasoning, setAiReasoning] = useState<AutoNPCTurnResponse | null>(
    null,
  );
  const [showReasoningPanel, setShowReasoningPanel] = useState(false);

  // Context menu state (must be before Escape key useEffect)
  const [contextMenu, setContextMenu] = useState<{
    position: { x: number; y: number };
    target: ContextMenuTarget;
  } | null>(null);

  // Header expansion state (for compact header feature)
  const [isHeaderExpanded, setIsHeaderExpanded] = useState(false);
  const [isHeaderHovered, setIsHeaderHovered] = useState(false);

  // Action bar minimize state
  const [actionBarMinimized, setActionBarMinimized] = useState(false);

  // Tab key listener to toggle action bar minimize
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Only handle Tab key when turn is active
      if (event.key !== "Tab" || !turnActive) return;
      // Don't interfere with input fields
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLTextAreaElement
      ) {
        return;
      }
      // Don't interfere with dialogs or overlays
      if (showVoiceConfirmation || showVictoryCelebration || isAnyModalOpen) {
        return;
      }
      event.preventDefault();
      setActionBarMinimized((prev) => !prev);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [turnActive, showVoiceConfirmation, showVictoryCelebration, isAnyModalOpen]);

  // Escape key listener to cancel targeting or open pause menu
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Only handle Escape key
      if (event.key !== "Escape") return;
      // Don't interfere with other dialogs or overlays
      if (showVoiceConfirmation || contextMenu || showVictoryCelebration || isAnyModalOpen) {
        return;
      }
      // If turn is active, try to cancel targeting first
      if (turnActive && actionPanelRef.current?.cancel()) {
        event.preventDefault();
        return;
      }
      // Otherwise open pause menu
      event.preventDefault();
      openModal({ type: "pause" });
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [turnActive, showVoiceConfirmation, contextMenu, showVictoryCelebration, isAnyModalOpen, openModal]);

  // Available actions query (only when turn is active)
  const { data: availableActions } = useAvailableActions(combatId, {
    enabled: turnActive,
  });

  // Canvas interaction state
  const [hovered, setHovered] = useState<HexCoord | null>(null);
  const [selectedAction, setSelectedAction] = useState<SelectedAction | null>(
    null,
  );

  // Action preview state
  const [previewAction, setPreviewAction] =
    useState<AvailableActionItem | null>(null);

  // Targeting mode state
  const [targetMode, setTargetMode] = useState<TargetMode | null>(null);
  const [selectedTargetIds, setSelectedTargetIds] = useState<string[]>([]);
  const [maxTargets, setMaxTargets] = useState<number>(1);

  // Area targeting state for line/cone attacks
  const [areaPattern, setAreaPattern] =
    useState<AttackPatternDefinition | null>(null);
  const [areaDirection, setAreaDirection] = useState<HexCoord | null>(null);
  // Preview origin for blast patterns (follows cursor)
  const [previewOrigin, setPreviewOrigin] = useState<HexCoord | null>(null);

  // Movement path state
  const [isPathMode, setIsPathMode] = useState(false);
  const [movementPath, setMovementPath] = useState<HexCoord[]>([]);
  const [pathHexClick, setPathHexClick] = useState<HexCoord | null>(null);

  // Movement range preview state
  const [showMovementRange, setShowMovementRange] = useState(false);
  const [movementRangeSpeed, setMovementRangeSpeed] = useState(0);

  // Action confirmation overlay state (UI overhaul)
  const [confirmationData, setConfirmationData] = useState<ActionConfirmationData | null>(null);
  const [showConfirmationOverlay, setShowConfirmationOverlay] = useState(false);

  // Action feed expansion state
  const [actionFeedExpanded, setActionFeedExpanded] = useState(false);

  // Viewport pan/zoom state
  const {
    viewport,
    setPan,
    setZoom,
    resetViewport,
    centerOnCoord,
    zoomAtPoint,
    MIN_ZOOM,
    MAX_ZOOM,
  } = useCanvasViewport();

  // Canvas size ref for centering calculations
  const canvasSizeRef = useRef<{ width: number; height: number }>({
    width: 720,
    height: 520,
  });

  // Hover tooltip state
  const [hoverTooltip, setHoverTooltip] = useState<{
    target: HoverTarget;
    position: { x: number; y: number };
  } | null>(null);

  // Preview target ID for action preview (hovered or selected target)
  const previewTargetId = useMemo(() => {
    if (hoverTooltip?.target.type === "combatant") {
      const combatant = hoverTooltip.target.combatant;
      // Don't preview targeting self
      if (combatant.id === currentActor?.id) return null;
      return combatant.id;
    }
    if (selectedTargetIds.length > 0) {
      return selectedTargetIds[0];
    }
    return null;
  }, [hoverTooltip, selectedTargetIds, currentActor]);

  // Preview target combatant object
  const previewTargetCombatant = useMemo(() => {
    if (!previewTargetId) return null;
    return combatants.find((c) => c.id === previewTargetId) ?? null;
  }, [previewTargetId, combatants]);

  // Action preview mutation
  const {
    mutate: fetchPreview,
    data: previewResponse,
    isPending: isPreviewLoading,
    error: previewError,
    reset: resetPreview,
  } = useActionPreview(combatId);

  // Fetch action preview when preview action and target are available
  useEffect(() => {
    if (!previewAction || !previewTargetId || !currentActor?.id) {
      resetPreview();
      return;
    }

    // Debounce to avoid rapid calls
    const timer = setTimeout(() => {
      fetchPreview({
        action_id: previewAction.action_id,
        actor_id: currentActor.id,
        target_id: previewTargetId,
        weapon_id: getWeaponIdForAction(previewAction) ?? undefined,
      });
    }, 150);

    return () => clearTimeout(timer);
  }, [
    previewAction,
    previewTargetId,
    currentActor?.id,
    fetchPreview,
    getWeaponIdForAction,
    resetPreview,
  ]);

  // Get player combatants for reaction polling (when not our turn)
  const playerCombatants = useMemo(
    () => combatants.filter((c) => c.side === "players"),
    [combatants],
  );

  // Poll for reaction opportunities when it's not our turn
  const firstPlayerCombatant = playerCombatants[0];
  const playerCombatantName = firstPlayerCombatant?.name ?? "Player";
  const { data: reactionOpportunity } = useReactionOpportunity(
    combatId,
    firstPlayerCombatant?.id ?? null,
    {
      enabled: !turnActive && !!firstPlayerCombatant,
      pollingInterval: 3000, // Poll every 3 seconds
    },
  );

  // Poll for pending decisions (save prompts, system trauma)
  const { data: pendingDecisions } = usePendingDecisions(
    combatId,
    firstPlayerCombatant?.id ?? null,
    {
      enabled: !!firstPlayerCombatant,
      pollingInterval: 3000, // Poll every 3 seconds
    },
  );
  const submitDecision = useSubmitDecision(combatId);

  // Derive turn state
  const turnState: TurnState = useMemo(() => {
    if (startTurn.isPending) return "not_started";
    if (endTurn.isPending) return "ending";
    if (turnActive) return "active";
    return "not_started";
  }, [startTurn.isPending, endTurn.isPending, turnActive]);

  // Handle start turn
  const handleStartTurn = useCallback(() => {
    setActionError(null);
    startTurn.mutate(undefined, {
      onSuccess: (result) => {
        setTurnActive(true);
        setEconomy(result.economy);
        toast.success("Turn started");
        narrateTurnStart(playerCombatantName, true);
      },
      onError: (err) => {
        setActionError(err.message || "Failed to start turn");
        toast.error(err.message || "Failed to start turn");
      },
    });
  }, [startTurn, narrateTurnStart, playerCombatantName]);

  // Handle end turn
  const handleEndTurn = useCallback(() => {
    endTurn.mutate(undefined, {
      onSuccess: () => {
        setTurnActive(false);
        setEconomy(null);
        setTargetMode(null);
        setSelectedTargetIds([]);
        setMaxTargets(1);
        toast.success("Turn ended");
      },
      onError: (err) => toast.error(err.message || "Failed to end turn"),
    });
  }, [endTurn]);

  // Handle auto NPC turn
  const handleAutoNpcTurn = useCallback(() => {
    // Don't process AI turns while game is paused
    if (isOpen("pause")) {
      return;
    }
    // Clear previous reasoning
    setAiReasoning(null);
    autoNpcTurn.mutate(undefined, {
      onSuccess: (response) => {
        // Reset state synchronously
        setTurnActive(false);
        setEconomy(null);
        setTargetMode(null);
        setSelectedTargetIds([]);
        setMaxTargets(1);
        // Store AI reasoning for display
        setAiReasoning(response);
        if (settings.showAIReasoning) {
          setShowReasoningPanel(true);
        }
        // Narrate AI turn
        narrateTurnStart(response.actor_name, false);
        if (response.decision_action) {
          narrateAction(
            response.actor_name,
            response.decision_action,
            response.decision_target,
          );
        }
        // Defer toast to avoid DOM race condition with Sonner + WebSocket updates
        setTimeout(() => {
          toast.success("NPC turn completed");
        }, 0);
      },
      onError: (err) => {
        // Defer toast to avoid DOM race condition
        setTimeout(() => {
          toast.error(err.message || "NPC turn failed");
        }, 0);
      },
    });
  }, [autoNpcTurn, settings, narrateTurnStart, narrateAction, isOpen]);

  // Action triggered from ActionBar (to pass to ActionPanel)
  const [triggeredAction, setTriggeredAction] =
    useState<AvailableActionItem | null>(null);

  // Handle action selection from ActionPanel or ActionBar
  const handleActionSelect = useCallback((action: AvailableActionItem) => {
    // Reset targets when selecting new action and set max targets from action
    setSelectedTargetIds([]);
    setMaxTargets(action.max_targets);
    // Set triggered action for ActionPanel to process
    setTriggeredAction(action);
  }, []);

  // Handle action execution
  const handleExecuteAction = useCallback(
    (request: ActionRequest) => {
      // Intercept overcharge to show confirmation modal
      if (request.is_overcharge) {
        openModal({ type: "overcharge" });
        return;
      }

      // Check if any target is a deployable (Phase 60)
      const deployableIds = new Set(Object.keys(scenario?.deployables ?? {}));
      const deployableTargets = (request.target_ids ?? []).filter((id) =>
        deployableIds.has(id),
      );
      const combatantTargets = (request.target_ids ?? []).filter(
        (id) => !deployableIds.has(id),
      );

      // Build final request with deployable targeting if applicable
      const finalRequest: ActionRequest =
        deployableTargets.length > 0
          ? {
              ...request,
              target_ids: combatantTargets,
              target_deployable_id: deployableTargets[0], // Only one deployable target supported
            }
          : request;

      setActionError(null);
      executeAction.mutate(finalRequest, {
        onSuccess: (result) => {
          if (result.success) {
            const newEconomy = result.economy;
            setEconomy(newEconomy);
            setTargetMode(null);
            setSelectedTargetIds([]);
            setMaxTargets(1);
            setAreaPattern(null);
            setAreaDirection(null);
            setPreviewOrigin(null);
            setActionError(null);
            toast.success("Action executed");
            // Narrate action
            narrateAction(
              playerCombatantName,
              request.action_id,
              undefined,
              result.damage_dealt,
            );

            // Auto-end turn if all actions exhausted
            if (newEconomy) {
              const fullExhausted = newEconomy.full_actions_used >= 1;
              const quickTotal = 2 + (newEconomy.overcharge_used ? 1 : 0);
              const quickExhausted =
                newEconomy.quick_actions_used >= quickTotal;

              if (fullExhausted && quickExhausted) {
                // Brief delay before auto-ending to let user see the result
                setTimeout(() => {
                  toast.info("All actions used - ending turn automatically");
                  handleEndTurn();
                }, 1000);
              }
            }
          } else {
            const errorMsg = result.error || "Action failed";
            setActionError(errorMsg);
            toast.error(errorMsg);
          }
        },
        onError: (err) => {
          const errorMsg = err.message || "Action failed";
          setActionError(errorMsg);
          toast.error(errorMsg);
        },
      });
    },
    [
      executeAction,
      scenario?.deployables,
      handleEndTurn,
      narrateAction,
      playerCombatantName,
      openModal,
    ],
  );

  // Handle overcharge confirmation
  const handleOverchargeConfirm = useCallback(() => {
    closeModal();
    executeAction.mutate(
      {
        action_id: "overcharge",
        action_type: "free",
        is_overcharge: true,
      },
      {
        onSuccess: (result) => {
          if (result.success) {
            setEconomy(result.economy);
            toast.success("Overcharge activated");
          }
        },
        onError: (err) => toast.error(err.message || "Overcharge failed"),
      },
    );
  }, [executeAction, closeModal]);

  // Handle reaction submission
  const handleReactionSubmit = useCallback(
    (reaction: ReactionRequest) => {
      submitReaction.mutate(reaction, {
        onSuccess: () => toast.success("Reaction executed"),
        onError: (err) => toast.error(err.message || "Reaction failed"),
      });
    },
    [submitReaction],
  );

  // Handle decision submission (save prompts, system trauma)
  const handleDecisionSubmit = useCallback(
    (request: DecisionSubmitRequest) => {
      submitDecision.mutate(request, {
        onSuccess: () => toast.success("Decision submitted"),
        onError: (err) => toast.error(err.message || "Decision failed"),
      });
    },
    [submitDecision],
  );

  // Handle mission completion
  const handleMissionComplete = useCallback(
    (request: CombatCompleteRequest) => {
      completeCombat.mutate(request, {
        onSuccess: (result) => {
          closeModal();
          toast.success("Mission completed");
          // Narrate victory/defeat
          const victoriousSide =
            request.outcome === "success" || request.outcome === "partial"
              ? "player"
              : "enemy";
          narrateVictory(victoriousSide);

          // Determine victory/defeat for celebration
          const outcome =
            request.outcome === "success" || request.outcome === "partial"
              ? "victory"
              : "defeat";
          setVictoryOutcome(outcome);
          setShowVictoryCelebration(true);

          // Schedule navigation after celebration delay (3 seconds)
          const navigationTimer = setTimeout(() => {
            // Redirect to debrief if missionId is known, otherwise to campaign/dashboard
            if (search.missionId) {
              // Get statistics from API response
              const stats = result.statistics;
              const turnsTaken = stats?.total_turns ??
                (scenario?.rounds?.reduce(
                  (acc, round) => acc + (round.turns?.length || 0),
                  0,
                ) ?? 0);
              const damageDealt = stats?.total_damage_dealt_by_players ?? 0;
              const damageReceived = stats?.total_damage_received_by_players ?? 0;
              const enemiesDestroyed = stats?.total_enemies_destroyed ?? 0;
              const closestCall = stats?.closest_call_hp ?? 0;
              const closestCallCombatant = stats?.closest_call_combatant ?? "";
              const maxOverkill = stats?.max_overkill ?? 0;
              const attacks = stats?.action_totals?.attacks ?? 0;
              const moves = stats?.action_totals?.moves ?? 0;
              const techs = stats?.action_totals?.techs ?? 0;
              const xpEarned = result.xp_awarded ?? 0;
              const salvageEarned = result.salvage_awarded ?? 0;

              navigate({
                to: "/missions/$missionId/debrief",
                params: { missionId: search.missionId },
                search: {
                  outcome: outcome,
                  turns: turnsTaken,
                  damageDealt,
                  damageReceived,
                  enemiesDestroyed,
                  closestCall,
                  closestCallCombatant,
                  maxOverkill,
                  attacks,
                  moves,
                  techs,
                  xp: xpEarned,
                  salvage: salvageEarned,
                },
              });
            } else if (result.campaign_id) {
              navigate({
                to: "/campaigns/$campaignId",
                params: { campaignId: result.campaign_id },
              });
            } else {
              navigate({ to: "/" });
            }
          }, 3000); // 3 second pause for celebration

          // Cleanup timer if component unmounts (though navigation will cause unmount)
          return () => clearTimeout(navigationTimer);
        },
        onError: (err) =>
          toast.error(err.message || "Failed to complete mission"),
      });
    },
    [completeCombat, navigate, search, scenario, narrateVictory, closeModal],
  );

  // Handle mission forfeit
  const handleForfeitMission = useCallback(() => {
    // Set forfeiting flag to bypass navigation blocker (E8-US-004)
    setIsForfeiting(true);
    forfeitCombat.mutate(undefined, {
      onSuccess: (result) => {
        closeModal();
        toast.info("Mission forfeited - counted as defeat");
        // Narrate enemy victory
        narrateVictory("enemy");
        // Set outcome for celebration
        setVictoryOutcome("defeat");
        setShowVictoryCelebration(true);

        // Schedule navigation after celebration delay (3 seconds)
        const navigationTimer = setTimeout(() => {
          if (search.missionId) {
            // Compute statistics (partial salvage based on enemies defeated)
            const enemyCount =
              scenario?.combatants?.filter((c) => c.side !== "players").length || 0;
            const enemiesDefeated =
              scenario?.combatants?.filter(
                (c) => c.side !== "players" && c.status === "defeated"
              ).length || 0;
            const turnsTaken =
              scenario?.rounds?.reduce(
                (acc, round) => acc + (round.turns?.length || 0),
                0
              ) || 0;
            const damageDealt = enemiesDefeated * 300; // placeholder
            const damageReceived = 1200; // placeholder
            const xpEarned = 0; // No XP for forfeit
            const salvageEarned = result.salvage_awarded ?? 0;

            navigate({
              to: "/missions/$missionId/debrief",
              params: { missionId: search.missionId },
              search: {
                outcome: "defeat",
                turns: turnsTaken,
                damageDealt,
                damageReceived,
                xp: xpEarned,
                salvage: salvageEarned,
                forfeit: true,
              },
            });
          } else if (result.campaign_id) {
            navigate({
              to: "/campaigns/$campaignId",
              params: { campaignId: result.campaign_id },
            });
          } else {
            navigate({ to: "/" });
          }
        }, 3000); // 3 second pause for celebration

        // Cleanup timer if component unmounts
        return () => clearTimeout(navigationTimer);
      },
      onError: (err) =>
        toast.error(err.message || "Failed to forfeit mission"),
    });
  }, [forfeitCombat, navigate, search, scenario, narrateVictory, closeModal]);

  // Handle reserve spending
  const handleSpendReserve = useCallback(
    (reserveId: string) => {
      spendReserve.mutate(
        { reserve_id: reserveId },
        {
          onSuccess: () => toast.success("Reserve spent"),
          onError: (err) =>
            toast.error(err.message || "Failed to spend reserve"),
        },
      );
    },
    [spendReserve],
  );

  // Handle target mode changes from ActionPanel
  const handleTargetModeChange = useCallback((mode: TargetMode | null) => {
    setTargetMode(mode);
    setSelectedTargetIds([]);
    if (!mode) {
      setMaxTargets(1);
    }
  }, []);

  // Handle path mode changes from ActionPanel
  const handlePathModeChange = useCallback(
    (isActive: boolean, path: HexCoord[]) => {
      setIsPathMode(isActive);
      setMovementPath(path);
      if (!isActive) {
        setPathHexClick(null);
      }
    },
    [],
  );

  // Handle AoE preview changes from ActionPanel (weapon selection)
  const handleAreaPreviewChange = useCallback(
    (
      pattern: AttackPatternDefinition | null,
      origin: HexCoord | null,
      direction: HexCoord | null,
    ) => {
      setAreaPattern(pattern);
      setPreviewOrigin(origin);
      setAreaDirection(direction);
    },
    [],
  );

  // Handle hex click for path building (from CombatCanvas)
  const handlePathHexClick = useCallback(
    (coord: HexCoord) => {
      if (isPathMode) {
        setPathHexClick(coord);
      }
    },
    [isPathMode],
  );

  // Handle movement range preview changes from ActionPanel
  const handleMovementRangeChange = useCallback(
    (show: boolean, speed: number) => {
      setShowMovementRange(show);
      setMovementRangeSpeed(speed);
    },
    [],
  );

  // Handle confirmation ready from ActionPanel (for overlay)
  const handleConfirmationReady = useCallback(
    (data: ActionConfirmationReadyData | null) => {
      if (!data) {
        setShowConfirmationOverlay(false);
        setConfirmationData(null);
        return;
      }

      // Build target names from IDs
      const targetNames = data.targetIds?.map(
        (id) => combatants.find((c) => c.id === id)?.name ?? id
      );

      // Build weapon name
      const weaponName = data.weaponId
        ? weaponsQuery.data?.find((w) => w.id === data.weaponId)?.name ?? data.weaponId
        : undefined;

      // Build confirmation data for overlay
      const confirmData: ActionConfirmationData = {
        action: data.action,
        targetNames,
        targetIds: data.targetIds,
        weaponName,
        weaponId: data.weaponId ?? undefined,
        systemName: data.systemId ?? undefined,
        systemId: data.systemId ?? undefined,
        movementPath: data.movementPath,
        pathDistance: data.pathDistance,
        preview: previewResponse ?? null,
        isPreviewLoading,
      };

      setConfirmationData(confirmData);
      setShowConfirmationOverlay(true);
    },
    [combatants, weaponsQuery.data, previewResponse, isPreviewLoading]
  );

  // Handle confirmation overlay confirm action
  const handleOverlayConfirm = useCallback(() => {
    if (!confirmationData) return;

    const { action, targetIds, weaponId, systemId, movementPath } = confirmationData;

    // IMPORTANT: Reset ActionPanel state BEFORE executing to stop it from re-triggering the overlay
    actionPanelRef.current?.reset();
    setShowConfirmationOverlay(false);
    setConfirmationData(null);

    // Build the action request based on what data we have
    if (movementPath && movementPath.length >= 2) {
      // Movement action
      handleExecuteAction({
        action_id: action.action_id,
        action_type: action.action_type as ActionRequest["action_type"],
        movement_path: movementPath.map((c) => ({ coord: { q: c.q, r: c.r } })),
      });
    } else {
      // Target-based or simple action
      handleExecuteAction({
        action_id: action.action_id,
        action_type: action.action_type as ActionRequest["action_type"],
        target_ids: targetIds,
        weapon_id: weaponId ?? undefined,
        system_id: systemId ?? undefined,
      });
    }
  }, [confirmationData, handleExecuteAction]);

  // Handle confirmation overlay cancel
  const handleOverlayCancel = useCallback(() => {
    setShowConfirmationOverlay(false);
    setConfirmationData(null);
    // Fully reset ActionPanel state to stop it from re-triggering the overlay
    actionPanelRef.current?.reset();
  }, []);

  // Compute turn banner state for UI overhaul
  const turnBannerState: TurnBannerState = useMemo(() => {
    if (autoNpcTurn.isPending) return "ai_thinking";
    if (!turnActive && currentActor?.ai_controlled) return "enemy_turn";
    if (turnActive && targetMode?.requiresTarget) return "select_target";
    if (turnActive && isPathMode) return "select_destination";
    if (turnActive) return "your_turn";
    return "waiting";
  }, [turnActive, currentActor?.ai_controlled, targetMode, isPathMode, autoNpcTurn.isPending]);

  // Combatant sides lookup - use pre-computed from UI when available
  const combatantSides = useMemo(() => {
    if (ui?.combatant_sides) {
      // Convert from UI format to expected format (players/enemies)
      const map = new Map<string, "players" | "enemies">();
      for (const [id, side] of Object.entries(ui.combatant_sides)) {
        map.set(id, side === "players" ? "players" : "enemies");
      }
      return map;
    }
    const map = new Map<string, "players" | "enemies">();
    for (const c of combatants) {
      map.set(c.id, c.side === "players" ? "players" : "enemies");
    }
    return map;
  }, [ui?.combatant_sides, combatants]);

  // Viewport control handlers
  const handleZoomIn = useCallback(() => {
    setZoom(viewport.zoom + 0.2);
  }, [setZoom, viewport.zoom]);

  const handleZoomOut = useCallback(() => {
    setZoom(viewport.zoom - 0.2);
  }, [setZoom, viewport.zoom]);

  const handleZoomDelta = useCallback(
    (delta: number) => {
      setZoom(viewport.zoom + delta);
    },
    [setZoom, viewport.zoom],
  );

  const handleCenterOnActor = useCallback(() => {
    const actorCoord = currentActor?.position?.coord;
    if (!actorCoord) return;

    // Use stored canvas size for centering
    const baseLayout = createHexLayout(30 * viewport.zoom, {
      x: canvasSizeRef.current.width / 2,
      y: canvasSizeRef.current.height / 2,
    });
    centerOnCoord(actorCoord, baseLayout, canvasSizeRef.current);
  }, [currentActor?.position?.coord, viewport.zoom, centerOnCoord]);

  // Handle token click for targeting - toggle targets in array up to maxTargets
  const handleTokenClick = useCallback(
    (tokenId: string) => {
      if (!targetMode?.requiresTarget) return;

      setSelectedTargetIds((prev) => {
        // If already selected, remove it
        if (prev.includes(tokenId)) {
          return prev.filter((id) => id !== tokenId);
        }
        // If at max capacity, replace the last target
        if (prev.length >= maxTargets) {
          return [...prev.slice(0, -1), tokenId];
        }
        // Add to selection
        return [...prev, tokenId];
      });
    },
    [targetMode, maxTargets],
  );

  // Handle right-click context menu on canvas
  const handleContextMenu = useCallback(
    (info: ContextMenuInfo) => {
      // Right-click during path mode cancels path selection
      if (isPathMode) {
        actionPanelRef.current?.cancel();
        return;
      }

      // If in targeting mode, right-click cancels targeting (both empty hex and tokens)
      if (targetMode) {
        actionPanelRef.current?.cancel();
        return;
      }

      // Determine what was clicked
      let target: ContextMenuTarget;

      if (info.tokenId) {
        // Clicked on a combatant token
        const combatant = combatants.find((c) => c.id === info.tokenId);
        if (combatant) {
          const isEnemy = combatant.side !== "players";
          target = isEnemy
            ? {
                type: "enemy",
                combatantId: combatant.id,
                combatantName: combatant.name,
                coord: info.coord,
              }
            : {
                type: "friendly",
                combatantId: combatant.id,
                combatantName: combatant.name,
                coord: info.coord,
              };
        } else {
          // Token not found in combatants, treat as empty hex
          target = { type: "empty", coord: info.coord };
        }
      } else if (info.markerId?.startsWith("deployable:")) {
        // Clicked on a deployable
        const deployableId = info.markerId.replace("deployable:", "");
        const deployable = scenario?.deployables?.[deployableId];
        target = {
          type: "deployable",
          deployableId,
          deployableName: deployable?.name ?? "Deployable",
          coord: info.coord,
        };
      } else {
        // Clicked on empty hex
        target = { type: "empty", coord: info.coord };
      }

      setContextMenu({
        position: info.screenPosition,
        target,
      });
    },
    [combatants, scenario?.deployables, isPathMode, targetMode, actionPanelRef],
  );

  // Handle context menu option selection
  const handleContextMenuSelect = useCallback(
    (option: ContextMenuOption) => {
      // Close the menu
      setContextMenu(null);

      // Handle info/view options (non-action)
      if (option.id.startsWith("view_")) {
        // TODO: Show info panel/tooltip for the target
        toast.info(`View info: ${option.label}`);
        return;
      }

      // Handle action options
      if (option.action) {
        // Trigger the action through the action bar flow
        handleActionSelect(option.action);

        // If the action targets the right-clicked entity, pre-select it
        if (
          contextMenu?.target.type === "enemy" ||
          contextMenu?.target.type === "friendly"
        ) {
          const targetId =
            contextMenu.target.type === "enemy"
              ? contextMenu.target.combatantId
              : contextMenu.target.combatantId;
          setSelectedTargetIds([targetId]);
        } else if (contextMenu?.target.type === "deployable") {
          setSelectedTargetIds([contextMenu.target.deployableId]);
        }
      }
    },
    [handleActionSelect, contextMenu],
  );

  // Build targeting mode for canvas
  const canvasTargetingMode: TargetingMode = useMemo(() => {
    if (!targetMode?.requiresTarget) {
      return { active: false };
    }
    // All combatants except current actor are valid targets for attacks
    const combatantTargets = combatants
      .filter((c) => c.id !== currentActor?.id)
      .map((c) => c.id);

    // Include non-destroyed deployables as valid targets (Phase 60)
    const deployableTargets = Object.entries(scenario?.deployables ?? {})
      .filter(([_, d]) => !d.is_destroyed)
      .map(([id, _]) => id);

    return {
      active: true,
      validTargetIds: [...combatantTargets, ...deployableTargets],
      selectedTargetIds,
      maxTargets,
    };
  }, [
    targetMode,
    combatants,
    currentActor,
    selectedTargetIds,
    maxTargets,
    scenario?.deployables,
  ]);

  // Derive active indices from selectedAction or fall back to current position
  const activeRoundIndex =
    selectedAction?.roundIdx ?? clampIndex(currentRound - 1, rounds);
  const round = rounds[activeRoundIndex] ?? null;
  const turns = round?.turns ?? [];
  const activeTurnIndex =
    selectedAction?.turnIdx ?? clampIndex(currentTurnIndex, turns);
  const turn = turns[activeTurnIndex] ?? null;
  const actions = turn?.actions ?? [];
  const activeActionIndex = selectedAction?.actionIdx ?? 0;
  const action = actions[activeActionIndex] ?? null;

  // Combatant name lookup - use pre-computed from UI when available
  const combatantNameById = useMemo(() => {
    if (ui?.combatant_names) {
      return new Map(Object.entries(ui.combatant_names));
    }
    return new Map((scenario?.combatants ?? []).map((c) => [c.id, c.name]));
  }, [ui?.combatant_names, scenario?.combatants]);
  const weaponDefinitions = useMemo(
    () =>
      new Map((weaponsQuery.data ?? []).map((weapon) => [weapon.id, weapon])),
    [weaponsQuery.data],
  );

  // Check if there are recent actions to determine action log default state
  const hasRecentActions = useMemo(() => {
    const currentRoundTurns = rounds[currentRound - 1]?.turns ?? [];
    // Check the last 2 turns for any actions
    return currentRoundTurns
      .slice(-2)
      .some((t) => (t.actions ?? []).length > 0);
  }, [rounds, currentRound]);

  // Build movement range overlays when in path mode
  // Prefer backend-computed data when available for consistent terrain cost calculations
  const movementRangeOverlays = useMemo(() => {
    if (!showMovementRange || !currentActor?.position?.coord) {
      return [];
    }

    const speed = movementRangeSpeed;
    if (speed <= 0) return [];

    // Use backend-computed movement range when available (E10-US-015)
    // This eliminates expensive frontend BFS and ensures consistent terrain handling
    if (ui?.movement_range && ui.movement_range.actor_id === currentActor.id) {
      const backendData = ui.movement_range;
      const difficultSet = new Set(
        (backendData.difficult_hexes ?? []).map((h) => `${h.q},${h.r}`)
      );

      // Color code based on movement cost (50% threshold)
      const easyThreshold = Math.floor(speed / 2);
      const easy: HexCoord[] = [];
      const atMax: HexCoord[] = [];

      // Since backend returns reachable hexes but not cost, we estimate based on
      // distance and difficult terrain (this matches the visual intent)
      for (const hex of backendData.reachable_hexes ?? []) {
        const originCoord = currentActor.position.coord;
        const basicDistance = Math.max(
          Math.abs(hex.q - originCoord.q),
          Math.abs(hex.r - originCoord.r),
          Math.abs((-hex.q - hex.r) - (-originCoord.q - originCoord.r))
        );

        // Estimate if it's an "easy" move (within half speed threshold)
        // Account for difficult terrain making it harder to reach
        const isDifficult = difficultSet.has(`${hex.q},${hex.r}`);
        const estimatedCost = basicDistance + (isDifficult ? 1 : 0);

        if (estimatedCost <= easyThreshold) {
          easy.push(hex);
        } else {
          atMax.push(hex);
        }
      }

      // Return overlays with style matching frontend's MOVEMENT_OVERLAY_STYLES
      const overlays: typeof import("../../lib/combat-render/canvas").AreaOverlay[] = [];
      if (easy.length > 0) {
        overlays.push({
          coords: easy,
          style: {
            fillStyle: "rgba(34, 197, 94, 0.2)", // green-500
            strokeStyle: "rgba(34, 197, 94, 0.5)",
            lineWidth: 1,
          },
        });
      }
      if (atMax.length > 0) {
        overlays.push({
          coords: atMax,
          style: {
            fillStyle: "rgba(234, 179, 8, 0.2)", // yellow-500
            strokeStyle: "rgba(234, 179, 8, 0.5)",
            lineWidth: 1,
          },
        });
      }
      return overlays;
    }

    // Fall back to frontend calculation when backend data unavailable
    if (!scenario) return [];

    const origin = currentActor.position.coord;

    // Build valid hex set from grid (we need to compute the grid first)
    // Use a temporary grid calculation matching what adaptCombatScenario would use
    const combatants = scenario.combatants ?? [];
    let maxDistance = 0;
    for (const combatant of combatants) {
      if (combatant.position?.coord) {
        const dist =
          Math.abs(combatant.position.coord.q) +
          Math.abs(combatant.position.coord.r);
        maxDistance = Math.max(maxDistance, dist);
      }
    }
    for (const tile of scenario.terrain?.tiles ?? []) {
      const dist = Math.abs(tile.coord.q) + Math.abs(tile.coord.r);
      maxDistance = Math.max(maxDistance, dist);
    }
    const gridRadius = Math.max(4, maxDistance + 1);

    // Build valid hex set
    const validHexes = new Set<string>();
    for (let q = -gridRadius; q <= gridRadius; q++) {
      const rMin = Math.max(-gridRadius, -q - gridRadius);
      const rMax = Math.min(gridRadius, -q + gridRadius);
      for (let r = rMin; r <= rMax; r++) {
        validHexes.add(`${q},${r}`);
      }
    }

    // Build blocked hex set (other combatants' positions)
    const blockedHexes = new Set<string>();
    for (const combatant of combatants) {
      if (combatant.id !== currentActor.id && combatant.position?.coord) {
        blockedHexes.add(
          `${combatant.position.coord.q},${combatant.position.coord.r}`,
        );
      }
    }

    // Build difficult terrain set from scenario
    const difficultHexes = new Set<string>();
    for (const tile of scenario.terrain?.tiles ?? []) {
      if (tile.difficult) {
        difficultHexes.add(`${tile.coord.q},${tile.coord.r}`);
      }
    }

    return buildMovementRangeOverlays(
      origin,
      speed,
      validHexes,
      blockedHexes,
      difficultHexes,
    );
  }, [showMovementRange, currentActor, scenario, movementRangeSpeed, ui?.movement_range]);

  const renderOutput: CombatRenderAdapterOutput | null = useMemo(() => {
    if (!scenario) {
      return null;
    }
    // For blast patterns, use preview origin (follows cursor) instead of actor position
    const effectivePatternOrigin =
      areaPattern?.pattern === "blast" && previewOrigin
        ? { coord: previewOrigin }
        : currentActor?.position;

    const result = adaptCombatScenario({
      scenario,
      round,
      turn,
      action,
      hover: hovered,
      // Include area targeting preview
      attackPattern: areaPattern ?? undefined,
      patternOrigin: effectivePatternOrigin,
      patternDirection: areaDirection ?? undefined,
      actorId: currentActor?.id,
      // Include movement range overlays in grid calculation
      additionalOverlays: movementRangeOverlays.length > 0 ? movementRangeOverlays : undefined,
    });

    // Add movement range overlays to the display (before other overlays so they appear underneath)
    if (movementRangeOverlays.length > 0) {
      result.state.overlays = [
        ...movementRangeOverlays,
        ...(result.state.overlays ?? []),
      ];
    }

    return result;
  }, [
    action,
    hovered,
    round,
    scenario,
    turn,
    areaPattern,
    areaDirection,
    currentActor,
    previewOrigin,
    movementRangeOverlays,
  ]);

  // Canvas key for forced re-render when combatant positions change
  // This ensures tokens visually move after actions execute
  const canvasKey = useMemo(() => {
    if (!scenario?.combatants) return "no-scenario";
    const positions = scenario.combatants
      .map((c) => `${c.id}:${c.position?.coord?.q ?? "?"},${c.position?.coord?.r ?? "?"}`)
      .join("|");
    return `r${currentRound}-t${currentTurnIndex}-${positions}`;
  }, [scenario?.combatants, currentRound, currentTurnIndex]);

  if (isLoading) {
    return <CombatSessionSkeleton />;
  }

  if (error) {
    return (
      <div className="p-6 max-w-6xl mx-auto">
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <p className="text-destructive">
              Error loading combat session: {error.message}
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!data || !scenario) {
    return (
      <div className="p-6 max-w-6xl mx-auto">
        <Card>
          <CardContent className="pt-6 text-center">
            <p className="text-muted-foreground">Combat session not found</p>
            <Link to="/" className="text-primary hover:underline">
              Back to dashboard
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="px-4 py-3 max-w-7xl mx-auto space-y-3">
      {/* WebSocket disconnection banner */}
      {!wsConnected && (
        <div className="fixed top-0 left-0 right-0 z-40 bg-amber-500 text-amber-950 px-4 py-1.5 text-xs text-center font-medium shadow-lg animate-in slide-in-from-top duration-300">
          <div className="flex items-center justify-center gap-2">
            <svg
              className="w-3 h-3 animate-spin"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
            Reconnecting...
          </div>
        </div>
      )}

      {/* Compact Header (E9-US-005) */}
      {!settings.hideHeader && (
        <div
          className={`relative transition-all duration-300 ease-in-out overflow-hidden ${
            settings.compactHeader && turnActive && !isHeaderHovered && !isHeaderExpanded
              ? "h-8 py-1"
              : "h-auto py-3"
          }`}
          onMouseEnter={() => setIsHeaderHovered(true)}
          onMouseLeave={() => {
            setIsHeaderHovered(false);
            setIsHeaderExpanded(false);
          }}
          onClick={() => {
            if (settings.compactHeader && turnActive) {
              setIsHeaderExpanded(!isHeaderExpanded);
            }
          }}
        >
          {/* Thin strip mode (compact) */}
          {settings.compactHeader && turnActive && !isHeaderHovered && !isHeaderExpanded ? (
            <div className="flex items-center justify-between px-2">
              <div className="flex items-center gap-3">
                <Link
                  to="/"
                  className="text-primary hover:text-primary/80 transition-colors"
                  aria-label="Back to menu"
                  onClick={(e) => e.stopPropagation()}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clipRule="evenodd" />
                  </svg>
                </Link>
                <span className="text-xs font-medium text-foreground truncate max-w-[200px]">
                  {data.name}
                </span>
                <span className="text-[10px] text-muted-foreground">
                  R{data.current_round}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${wsConnected ? "bg-green-500" : "bg-amber-500"}`}
                  title={wsConnected ? "Live" : "Polling"}
                />
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setIsHeaderExpanded(true);
                  }}
                  className="text-[10px] text-muted-foreground hover:text-foreground transition-colors"
                  aria-label="Expand header"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
            </div>
          ) : (
            /* Full header mode */
            <div className="flex items-center justify-between px-2">
              <div className="flex items-center gap-4">
                <Link
                  to="/"
                  className="text-primary hover:text-primary/80 transition-colors"
                  aria-label="Back to menu"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clipRule="evenodd" />
                  </svg>
                </Link>
                <div>
                  <h1 className="text-lg font-heading font-semibold text-foreground">
                    {data.name}
                  </h1>
                  <p className="text-xs text-muted-foreground">
                    Round {data.current_round} · {data.status}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <span
                    className={`w-2 h-2 rounded-full ${wsConnected ? "bg-green-500" : "bg-amber-500"}`}
                  />
                  {wsConnected ? "Live" : "Polling"}
                </div>
                {settings.compactHeader && turnActive && (
                  <button
                    type="button"
                    onClick={() => setIsHeaderExpanded(false)}
                    className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                    aria-label="Collapse header"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 01-1.414-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z" clipRule="evenodd" />
                    </svg>
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Mission Complete Modal */}
      <MissionCompleteModal
        isOpen={isOpen("missionComplete")}
        onComplete={handleMissionComplete}
        onCancel={closeModal}
        isSubmitting={completeCombat.isPending}
        campaignId={data.campaign_id}
        missionReserves={scenario?.mission_reserves}
      />

      {/* Forfeit Confirmation Modal */}
      <ForfeitConfirmationModal
        isOpen={isOpen("forfeit")}
        onConfirm={handleForfeitMission}
        onCancel={closeModal}
        isSubmitting={forfeitCombat.isPending}
      />

      {/* Victory Celebration */}
      <VictoryCelebration
        isOpen={showVictoryCelebration}
        outcome={victoryOutcome}
        onClose={() => setShowVictoryCelebration(false)}
      />

      {/* Pause Menu */}
      <PauseMenu
        isOpen={isOpen("pause")}
        onResume={closeModal}
        onOpenSettings={() => openModal({ type: "settings" })}
        onOpenHelp={() => {
          closeModal();
          keyboardShortcuts.open();
        }}
        onOpenForfeit={() => openModal({ type: "forfeit" })}
        isPaused={isOpen("pause")}
      />

      {/* In-Game Settings */}
      <InGameSettings
        isOpen={isOpen("settings")}
        onClose={closeModal}
      />

      {/* Navigation Confirmation Modal (E8-US-004) */}
      <Modal
        isOpen={isOpen("navigationConfirm")}
        onClose={() => handleNavigationConfirm(false)}
        title="Leave Combat?"
        size="sm"
      >
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Progress will be lost. Are you sure you want to leave combat?
          </p>
          <div className="flex gap-3 justify-end">
            <Button
              variant="outline"
              onClick={() => handleNavigationConfirm(false)}
            >
              Stay
            </Button>
            <Button
              variant="destructive"
              onClick={() => handleNavigationConfirm(true)}
            >
              Leave
            </Button>
          </div>
        </div>
      </Modal>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
        {/* Canvas area - reduced height when action bar visible */}
        <div className="relative rounded-md border border-border bg-muted/30 p-2">
          <div
            className={`min-h-[400px] w-full ${turnActive ? "h-[calc(100vh-280px)]" : "h-[calc(100vh-180px)]"}`}
          >
            {renderOutput ? (
              <CombatCanvas
                key={canvasKey}
                width={720}
                height={520}
                resizeToParent
                layout={(size) => {
                  // Store canvas size for centering calculations
                  canvasSizeRef.current = size;
                  return createHexLayout(30, {
                    x: size.width / 2,
                    y: size.height / 2,
                  });
                }}
                state={renderOutput.state}
                styles={{
                  grid: { strokeStyle: "rgba(148, 163, 184, 0.5)" },
                  tokens: { strokeStyle: "#0f172a", lineWidth: 2 },
                  overlays: { fillStyle: "rgba(59, 130, 246, 0.12)" },
                  hover: {
                    fillStyle: "rgba(59, 130, 246, 0.2)",
                    strokeStyle: "rgba(59, 130, 246, 0.7)",
                    lineWidth: 2,
                  },
                }}
                targetingMode={canvasTargetingMode}
                movementPath={isPathMode ? movementPath : undefined}
                isPathMode={isPathMode}
                viewport={viewport}
                onZoomAtPoint={zoomAtPoint}
                onPan={setPan}
                onZoomDelta={handleZoomDelta}
                onCenterOnActor={handleCenterOnActor}
                onHover={(coord, point) => {
                  setHovered(coord);
                  // Update blast preview origin when hovering (blast follows cursor)
                  if (
                    areaPattern?.pattern === "blast" &&
                    targetMode?.requiresTarget
                  ) {
                    setPreviewOrigin(coord);
                  }

                  // Update hover tooltip
                  if (!coord) {
                    setHoverTooltip(null);
                    return;
                  }

                  // Determine what's at this hex
                  const combatant = combatants.find(
                    (c) =>
                      c.position?.coord?.q === coord.q &&
                      c.position?.coord?.r === coord.r,
                  );
                  const deployableEntry = Object.entries(
                    scenario?.deployables ?? {},
                  ).find(
                    ([_, d]) =>
                      d.position?.coord?.q === coord.q &&
                      d.position?.coord?.r === coord.r,
                  );

                  let target: HoverTarget;
                  if (combatant) {
                    target = {
                      type: "combatant",
                      combatant,
                      isEnemy: combatant.side !== "players",
                      coord,
                    };
                  } else if (deployableEntry) {
                    target = {
                      type: "deployable",
                      deployable: deployableEntry[1],
                      deployableId: deployableEntry[0],
                      coord,
                    };
                  } else {
                    target = { type: "empty", coord };
                  }

                  // Get screen position from canvas bounding rect + point offset
                  const canvasEl = document.querySelector("canvas");
                  if (canvasEl && point) {
                    const rect = canvasEl.getBoundingClientRect();
                    setHoverTooltip({
                      target,
                      position: {
                        x: rect.left + point.x,
                        y: rect.top + point.y,
                      },
                    });
                  }
                }}
                onTokenClick={handleTokenClick}
                onHexClick={handlePathHexClick}
                onContextMenu={handleContextMenu}
                className="h-full w-full"
              />
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                No scenario data available yet.
              </div>
            )}

            {/* Turn State Banner (UI Overhaul - overlay on canvas) */}
            <div className="absolute top-2 left-2 right-2 z-10">
              <TurnStateBanner
                state={turnBannerState}
                actorName={currentActor?.name}
                mechName={currentActor?.mech_name}
                economy={economy}
                contextHint={
                  turnBannerState === "select_destination"
                    ? `Click within ${movementRangeSpeed} spaces`
                    : turnBannerState === "select_target"
                    ? "Click an enemy to target"
                    : undefined
                }
                isAiControlled={currentActor?.ai_controlled}
              />
            </div>

            {/* Action Confirmation Overlay (UI Overhaul - centered on canvas) */}
            <ActionConfirmationOverlay
              isOpen={showConfirmationOverlay}
              data={confirmationData}
              onConfirm={handleOverlayConfirm}
              onCancel={handleOverlayCancel}
              isExecuting={executeAction.isPending}
            />

            {/* Action Feed (UI Overhaul - bottom left overlay) */}
            <ActionFeed
              recentActions={ui?.recent_actions}
              totalActionCount={ui?.total_action_count}
              rounds={rounds}
              currentRound={currentRound}
              combatantNames={combatantNameById}
              combatantSides={combatantSides}
              maxVisibleEntries={4}
              expanded={actionFeedExpanded}
              onToggleExpanded={() => setActionFeedExpanded(!actionFeedExpanded)}
            />
          </div>
          {/* Viewport Controls (pan/zoom) */}
          <ViewportControls
            zoom={viewport.zoom}
            minZoom={MIN_ZOOM}
            maxZoom={MAX_ZOOM}
            onZoomIn={handleZoomIn}
            onZoomOut={handleZoomOut}
            onReset={resetViewport}
            onCenterOnActor={handleCenterOnActor}
            hasActorPosition={!!currentActor?.position?.coord}
          />
        </div>

        <div className="flex flex-col h-full max-h-[calc(100vh-100px)]">
          {/* Sticky header area with current actor and controls */}
          <div className="sticky top-0 z-10 bg-background pb-2 space-y-2">
            {/* Current Actor Panel - most prominent (E9-US-006) */}
            <CurrentActorPanel
              actor={currentActor}
              isTurnActive={turnActive}
            />

            {/* Initiative Strip - compact horizontal (E9-US-006) */}
            <InitiativeStrip
              combatants={combatants}
              currentActorId={currentActor?.id ?? null}
              turnIndex={currentTurnIndex}
              roundNumber={currentRound}
            />

            {/* Turn Controls (Start/End Turn buttons) */}
            <TurnControls
              currentActorName={currentActor?.name ?? null}
              roundNumber={currentRound}
              turnIndex={currentTurnIndex}
              turnState={turnState}
              onStartTurn={handleStartTurn}
              onEndTurn={handleEndTurn}
              onAutoNpcTurn={handleAutoNpcTurn}
              isStarting={startTurn.isPending}
              isEnding={endTurn.isPending}
              isAutoNpc={autoNpcTurn.isPending}
              isCurrentActorAI={currentActor?.ai_controlled ?? false}
              economy={economy}
              canOvercharge={availableActions?.can_overcharge ?? false}
              overchargeLevel={availableActions?.overcharge_level ?? 0}
              error={actionError}
              confirmEndTurn={settings.confirmEndTurn}
            />
          </div>

          {/* Mission Controls - moved from header to side panel (E9-US-005) */}
          {data.status === "active" && (
            <div className="rounded-md border border-border bg-muted/30 p-2 space-y-2 mb-2">
              <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
                Mission Controls
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="flex-1 text-xs"
                  onClick={() => openModal({ type: "missionComplete" })}
                >
                  End Mission
                </Button>
                <Button
                  variant="destructive"
                  size="sm"
                  className="flex-1 text-xs"
                  onClick={() => openModal({ type: "forfeit" })}
                  disabled={forfeitCombat.isPending || !isPlayerTurn}
                  title={!isPlayerTurn ? "Can only forfeit during your turn" : ""}
                >
                  Forfeit
                </Button>
              </div>
            </div>
          )}

          {/* Scrollable content area */}
          <div className="flex-1 overflow-y-auto space-y-2 pr-1">
            {/* Action Log - collapsible (E9-US-006) */}
            <CollapsibleActionLog
              rounds={rounds}
              currentRound={currentRound}
              currentTurnIndex={currentTurnIndex}
              combatantNames={combatantNameById}
              selectedAction={selectedAction}
              onSelectAction={(roundIdx, turnIdx, actionIdx) =>
                setSelectedAction({ roundIdx, turnIdx, actionIdx })
              }
              defaultCollapsed={!hasRecentActions}
            />

            {/* Voice Transcription Display */}
            <VoiceTranscriptDisplay
              isListening={speechRecognition.isListening}
              transcript={speechRecognition.transcript}
              error={speechRecognition.error}
              recognitionSupported={speechRecognition.recognitionSupported}
              voiceEnabled={settings.enableVoiceInput}
              onRetry={() => {
                speechRecognition.stopListening();
                speechRecognition.resetTranscript();
                setTimeout(() => speechRecognition.startListening(), 100);
              }}
            />

            {/* AI Reasoning Panel */}
            {showReasoningPanel && aiReasoning && (
              <div className="rounded-md border border-border bg-muted/30 p-2 space-y-1">
                <div className="flex items-center justify-between">
                  <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
                    AI Reasoning
                  </div>
                  <button
                    type="button"
                    onClick={() => setShowReasoningPanel(false)}
                    className="text-xs text-muted-foreground hover:text-foreground"
                  >
                    ×
                  </button>
                </div>
                <div className="space-y-2 text-xs max-h-32 overflow-y-auto">
                  {aiReasoning.situation_assessment && (
                    <div>
                      <div className="font-medium">Situation Assessment</div>
                      <div className="text-muted-foreground whitespace-pre-wrap">
                        {aiReasoning.situation_assessment}
                      </div>
                    </div>
                  )}
                  {aiReasoning.considered_options && (
                    <div>
                      <div className="font-medium">Considered Options</div>
                      <div className="text-muted-foreground whitespace-pre-wrap">
                        {aiReasoning.considered_options}
                      </div>
                    </div>
                  )}
                  {aiReasoning.rationale && (
                    <div>
                      <div className="font-medium">Rationale</div>
                      <div className="text-muted-foreground whitespace-pre-wrap">
                        {aiReasoning.rationale}
                      </div>
                    </div>
                  )}
                  {aiReasoning.confidence !== undefined && (
                    <div>
                      <div className="font-medium">Confidence</div>
                      <div className="text-muted-foreground">
                        {aiReasoning.confidence.toFixed(2)}
                      </div>
                    </div>
                  )}
                  {aiReasoning.decision_reasoning && (
                    <div>
                      <div className="font-medium">Decision Reasoning</div>
                      <div className="text-muted-foreground whitespace-pre-wrap">
                        {aiReasoning.decision_reasoning}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Action Panel (only when turn is active) */}
            {turnActive && (
              <ActionPanel
                ref={actionPanelRef}
                availableActions={availableActions ?? null}
                economy={economy}
                onActionSelect={handleActionSelect}
                onExecuteAction={handleExecuteAction}
                onTargetModeChange={handleTargetModeChange}
                onPathModeChange={handlePathModeChange}
                onAreaPreviewChange={handleAreaPreviewChange}
                onMovementRangeChange={handleMovementRangeChange}
                isExecuting={executeAction.isPending}
                selectedTargetIds={selectedTargetIds}
                actorInventory={currentActor?.inventory}
                weaponDefinitions={weaponDefinitions}
                actorSpeed={currentActor?.stats?.speed ?? 4}
                actorPosition={currentActor?.position?.coord ?? null}
                hexClickCoord={pathHexClick}
                triggeredAction={triggeredAction}
                onTriggeredActionProcessed={() => setTriggeredAction(null)}
                onActionHover={setPreviewAction}
                onConfirmationReady={handleConfirmationReady}
                useConfirmationOverlay={true}
              />
            )}

            {/* Victory Conditions (if SITREP active) */}
            <VictoryConditionPanel
              sitrepResolution={scenario?.sitrep_resolution}
            />

            {/* Mission Objectives (if available) */}
            <ObjectiveTracker objectives={scenario?.objectives} />

            {/* Mission Reserves (if available) */}
            <ReservesPanel
              reserves={scenario?.mission_reserves}
              onSpendReserve={handleSpendReserve}
              isSpending={spendReserve.isPending}
            />

            {/* Combatants List with HP bars (E9-US-006) */}
            <CombatantList
              combatants={combatants}
              currentActorId={currentActor?.id ?? null}
              selectedTargetIds={selectedTargetIds}
              onCombatantClick={handleTokenClick}
            />
          </div>
        </div>

        {/* Modals - outside sidebar */}
        {/* Overcharge Confirmation Modal */}
        {isOpen("overcharge") && currentActor && (
          <OverchargeConfirm
            currentLevel={availableActions?.overcharge_level ?? 0}
            heatCurrent={currentActor.resources?.heat_current ?? 0}
            heatCap={currentActor.resources?.heat_cap ?? 6}
            onConfirm={handleOverchargeConfirm}
            onCancel={closeModal}
            isOpen={isOpen("overcharge")}
          />
        )}

        {/* AI Thinking Indicator */}
        <AIThinkingIndicator
          isThinking={autoNpcTurn.isPending}
          reducedMotion={settings.reducedMotion}
        />

        {/* Reaction Prompt (when not our turn and reaction opportunity exists) */}
        <Modal
          isOpen={
            !turnActive &&
            reactionOpportunity?.pending_triggers?.length !== undefined &&
            reactionOpportunity.pending_triggers.length > 0 &&
            !!firstPlayerCombatant
          }
          disableBackdropClose
          urgent
        >
          {reactionOpportunity?.pending_triggers?.[0] &&
            firstPlayerCombatant && (
              <ReactionPrompt
                triggerType={
                  reactionOpportunity.pending_triggers[0].trigger_type
                }
                reactorId={reactionOpportunity.combatant_id}
                reactorName={reactionOpportunity.combatant_name}
                triggeringActorName={
                  reactionOpportunity.pending_triggers[0].triggering_actor_name
                }
                availableReactions={
                  reactionOpportunity.pending_triggers[0].available_reactions
                }
                inventory={firstPlayerCombatant.inventory}
                validTargets={combatants
                  .filter((c) => c.side !== "players")
                  .map((c) => ({ id: c.id, name: c.name }))}
                onSubmit={handleReactionSubmit}
                onDecline={() => {
                  // User declined the reaction opportunity
                  // Could track this if needed
                }}
                isOpen={true}
                isSubmitting={submitReaction.isPending}
              />
            )}
        </Modal>

        {/* Pending Decision Prompts (save checks, system trauma) */}
        {pendingDecisions?.has_pending &&
          pendingDecisions.pending_decisions.map((decision) => {
            const isUrgent = decision.decision_type === "hull_save";
            // Render appropriate prompt based on decision type
            if (decision.decision_type === "system_trauma") {
              return (
                <Modal
                  key={decision.decision_id}
                  isOpen={true}
                  disableBackdropClose
                  urgent={isUrgent}
                >
                  <TraumaSelectionPrompt
                    decision={decision}
                    combatantId={pendingDecisions.combatant_id}
                    combatantName={pendingDecisions.combatant_name}
                    inventory={firstPlayerCombatant?.inventory}
                    onSubmit={handleDecisionSubmit}
                    onDecline={() => {
                      // User cancelled - no action taken
                    }}
                    isOpen={true}
                    isSubmitting={submitDecision.isPending}
                  />
                </Modal>
              );
            }
            // Save prompts (hull_save, engineering_save, engineering_check)
            return (
              <Modal
                key={decision.decision_id}
                isOpen={true}
                disableBackdropClose
                urgent={isUrgent}
              >
                <SaveCheckPrompt
                  decision={decision}
                  combatantId={pendingDecisions.combatant_id}
                  combatantName={pendingDecisions.combatant_name}
                  onSubmit={handleDecisionSubmit}
                  onDecline={() => {
                    // User cancelled - no action taken
                  }}
                  isOpen={true}
                  isSubmitting={submitDecision.isPending}
                />
              </Modal>
            );
          })}

        {/* Voice Confirmation Dialog */}
        <VoiceActionConfirmationDialog
          isOpen={showVoiceConfirmation}
          transcript={voiceTranscript}
          parsedAction={parsedAction}
          error={voiceError}
          isExecuting={executeAction.isPending}
          getCombatantName={getCombatantName}
          getWeaponName={getWeaponName}
          onClose={() => {
            setShowVoiceConfirmation(false);
            setVoiceError(null);
          }}
          onConfirm={(action) => {
            executeAction.mutate(action as ActionRequest, {
              onSuccess: () => {
                setShowVoiceConfirmation(false);
                setParsedAction(null);
                setVoiceTranscript("");
                speechRecognition.resetTranscript();
              },
              onError: (error) => {
                setVoiceError(error.message || "Action execution failed");
              },
            });
          }}
        />

        {/* Hover Tooltip */}
        <MapTooltip
          target={hoverTooltip?.target ?? null}
          position={hoverTooltip?.position ?? null}
          delay={300}
        />

        {/* Action Preview Panel */}
        <ActionPreviewPanel
          target={previewTargetCombatant}
          previewAction={previewAction}
          previewResponse={previewResponse}
          position={hoverTooltip?.position ?? null}
          isLoading={isPreviewLoading}
          error={previewError}
          delay={200}
        />

        {/* Context Menu (right-click on canvas) */}
        {contextMenu && (
          <ContextMenu
            position={contextMenu.position}
            target={contextMenu.target}
            availableActions={availableActions ?? null}
            actorPosition={currentActor?.position?.coord}
            onSelect={handleContextMenuSelect}
            onClose={() => setContextMenu(null)}
            isTurnActive={turnActive}
          />
        )}

        {/* Bottom Action Bar (WoW-style) */}
        <ActionBar
          sessionId={combatId}
          availableActions={availableActions ?? null}
          economy={economy}
          onActionSelect={handleActionSelect}
          onOvercharge={() => openModal({ type: "overcharge" })}
          canOvercharge={availableActions?.can_overcharge ?? false}
          overchargeLevel={availableActions?.overcharge_level ?? 0}
          isExecuting={executeAction.isPending}
          visible={turnActive}
          // Minimize
          minimized={actionBarMinimized}
          onMinimizeToggle={() => setActionBarMinimized((prev) => !prev)}
          // Preview targeting
          previewTargetId={previewTargetId}
          currentActor={currentActor}
          // Voice control
          onVoiceToggle={speechRecognition.toggleListening}
          isVoiceListening={speechRecognition.isListening}
          voiceEnabled={settings.enableVoiceInput}
          voiceSupported={speechRecognition.recognitionSupported}
          // Help
          onHelpClick={() => openModal({ type: "help" })}
        />

        {/* Contextual Help Overlay (E9-US-004) */}
        <ContextualHelpOverlay
          isOpen={isOpen("help")}
          onClose={closeModal}
          economy={economy}
          availableActions={availableActions ?? null}
          currentRound={currentRound}
        />

        {/* First Combat Tutorial (E9-US-004) */}
        <FirstCombatTutorial
          isOpen={isOpen("tutorial")}
          onClose={() => {
            closeModal();
            setTutorialDismissedThisSession(true);
          }}
          onDontShowAgain={handleTutorialDontShowAgain}
        />
      </div>
    </div>
  );
}

function clampIndex<T>(value: number, list: T[]): number {
  if (!list.length) {
    return 0;
  }
  if (value < 0) {
    return 0;
  }
  if (value >= list.length) {
    return list.length - 1;
  }
  return value;
}
