import { useCallback, useMemo, useRef, useState } from "react";
import { createFileRoute, Link, useNavigate, useSearch } from "@tanstack/react-router";
import { toast } from "sonner";
import { useCanvasViewport } from "../../lib/hooks/useCanvasViewport";

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
  useSpendReserve,
  useWeapons,
  useAutoNpcTurn,
  type ActionRequest,
  type ActionEconomyState,
  type AvailableActionItem,
  type ReactionRequest,
  type DecisionSubmitRequest,
  type CombatCompleteRequest,
} from "../../lib/api";
import { CombatCanvas, type TargetingMode, type ContextMenuInfo } from "../../components/combat/CombatCanvas";
import { ViewportControls } from "../../components/combat/ViewportControls";
import { ContextMenu, type ContextMenuTarget, type ContextMenuOption } from "../../components/combat/ContextMenu";
import { MapTooltip, type HoverTarget } from "../../components/combat/MapTooltip";
import {
  ActionLog,
  type SelectedAction,
} from "../../components/combat/ActionLog";
import { TurnControls, type TurnState } from "../../components/combat/TurnControls";
import { TurnIndicator } from "../../components/combat/TurnIndicator";
import { ActionPanel, type TargetMode } from "../../components/combat/ActionPanel";
import { ActionBar } from "../../components/combat/ActionBar";
import { OverchargeConfirm } from "../../components/combat/OverchargeConfirm";
import { ReactionPrompt } from "../../components/combat/ReactionPrompt";
import { SaveCheckPrompt } from "../../components/combat/SaveCheckPrompt";
import { TraumaSelectionPrompt } from "../../components/combat/TraumaSelectionPrompt";
import { MissionCompleteModal } from "../../components/combat/MissionCompleteModal";
import { VictoryConditionPanel } from "../../components/combat/VictoryConditionPanel";
import { ObjectiveTracker } from "../../components/combat/ObjectiveTracker";
import { ReservesPanel } from "../../components/combat/ReservesPanel";
import {
  adaptCombatScenario,
  buildMovementRangeOverlays,
  type CombatRenderAdapterOutput,
} from "../../lib/combat-render/adapter";
import { createHexLayout } from "../../lib/combat-render/hex";
import type { HexCoord, AttackPatternDefinition } from "../../lib/types/lancer";
import {
  Button,
  Card,
  CardContent,
  Modal,
} from "../../components/ui";
import { CombatSessionSkeleton } from "../../components/skeletons";

interface CombatSearch {
  missionId?: string;
}

export const Route = createFileRoute("/combat/$combatId")({
  component: CombatSessionPage,
  validateSearch: (search: Record<string, unknown>): CombatSearch => ({
    missionId: typeof search.missionId === "string" ? search.missionId : undefined,
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
  const spendReserve = useSpendReserve(combatId);
  const autoNpcTurn = useAutoNpcTurn(combatId);
  const navigate = useNavigate();

  // Mission completion state
  const [showMissionCompleteModal, setShowMissionCompleteModal] = useState(false);

  // Turn state tracking
  const [turnActive, setTurnActive] = useState(false);
  const [economy, setEconomy] = useState<ActionEconomyState | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  // Available actions query (only when turn is active)
  const { data: availableActions } = useAvailableActions(combatId, {
    enabled: turnActive,
  });
  const weaponsQuery = useWeapons();

  // Canvas interaction state
  const [hovered, setHovered] = useState<HexCoord | null>(null);
  const [selectedAction, setSelectedAction] = useState<SelectedAction | null>(null);

  // Targeting mode state
  const [targetMode, setTargetMode] = useState<TargetMode | null>(null);
  const [selectedTargetIds, setSelectedTargetIds] = useState<string[]>([]);
  const [maxTargets, setMaxTargets] = useState<number>(1);

  // Overcharge confirmation state
  const [showOverchargeConfirm, setShowOverchargeConfirm] = useState(false);

  // Area targeting state for line/cone attacks
  const [areaPattern, setAreaPattern] = useState<AttackPatternDefinition | null>(null);
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
  const canvasSizeRef = useRef<{ width: number; height: number }>({ width: 720, height: 520 });

  // Context menu state
  const [contextMenu, setContextMenu] = useState<{
    position: { x: number; y: number };
    target: ContextMenuTarget;
  } | null>(null);

  // Hover tooltip state
  const [hoverTooltip, setHoverTooltip] = useState<{
    target: HoverTarget;
    position: { x: number; y: number };
  } | null>(null);

  const scenario = data?.scenario;
  const rounds = scenario?.rounds ?? [];
  const currentRound = data?.current_round ?? 1;
  const currentTurnIndex = data?.current_turn_index ?? 0;
  const combatants = scenario?.combatants ?? [];

  // Determine current actor from turn order
  const currentActor = useMemo(() => {
    if (!scenario) return null;
    // Find combatant based on turn order in the current round
    const round = scenario.rounds?.[currentRound - 1];
    const turn = round?.turns?.[currentTurnIndex];
    if (turn?.actor_id) {
      return combatants.find((c) => c.id === turn.actor_id) ?? null;
    }
    // Fall back to first combatant if no turn data
    return combatants[0] ?? null;
  }, [scenario, currentRound, currentTurnIndex, combatants]);

  // Get player combatants for reaction polling (when not our turn)
  const playerCombatants = useMemo(
    () => combatants.filter((c) => c.side === "players"),
    [combatants]
  );

  // Poll for reaction opportunities when it's not our turn
  const firstPlayerCombatant = playerCombatants[0];
  const { data: reactionOpportunity } = useReactionOpportunity(
    combatId,
    firstPlayerCombatant?.id ?? null,
    {
      enabled: !turnActive && !!firstPlayerCombatant,
      pollingInterval: 3000, // Poll every 3 seconds
    }
  );

  // Poll for pending decisions (save prompts, system trauma)
  const { data: pendingDecisions } = usePendingDecisions(
    combatId,
    firstPlayerCombatant?.id ?? null,
    {
      enabled: !!firstPlayerCombatant,
      pollingInterval: 3000, // Poll every 3 seconds
    }
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
      },
      onError: (err) => {
        setActionError(err.message || "Failed to start turn");
        toast.error(err.message || "Failed to start turn");
      },
    });
  }, [startTurn]);

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
    autoNpcTurn.mutate(undefined, {
      onSuccess: () => {
        // Turn state remains inactive since the full turn cycle completed
        setTurnActive(false);
        setEconomy(null);
        setTargetMode(null);
        setSelectedTargetIds([]);
        setMaxTargets(1);
        toast.success("NPC turn completed");
      },
      onError: (err) => toast.error(err.message || "NPC turn failed"),
    });
  }, [autoNpcTurn]);

  // Action triggered from ActionBar (to pass to ActionPanel)
  const [triggeredAction, setTriggeredAction] = useState<AvailableActionItem | null>(null);

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
        setShowOverchargeConfirm(true);
        return;
      }

      // Check if any target is a deployable (Phase 60)
      const deployableIds = new Set(Object.keys(scenario?.deployables ?? {}));
      const deployableTargets = (request.target_ids ?? []).filter((id) => deployableIds.has(id));
      const combatantTargets = (request.target_ids ?? []).filter((id) => !deployableIds.has(id));

      // Build final request with deployable targeting if applicable
      const finalRequest: ActionRequest = deployableTargets.length > 0
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

            // Auto-end turn if all actions exhausted
            if (newEconomy) {
              const fullExhausted = newEconomy.full_actions_used >= 1;
              const quickTotal = 2 + (newEconomy.overcharge_used ? 1 : 0);
              const quickExhausted = newEconomy.quick_actions_used >= quickTotal;

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
    [executeAction, scenario?.deployables, handleEndTurn]
  );

  // Handle overcharge confirmation
  const handleOverchargeConfirm = useCallback(() => {
    setShowOverchargeConfirm(false);
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
      }
    );
  }, [executeAction]);

  // Handle reaction submission
  const handleReactionSubmit = useCallback(
    (reaction: ReactionRequest) => {
      submitReaction.mutate(reaction, {
        onSuccess: () => toast.success("Reaction executed"),
        onError: (err) => toast.error(err.message || "Reaction failed"),
      });
    },
    [submitReaction]
  );

  // Handle decision submission (save prompts, system trauma)
  const handleDecisionSubmit = useCallback(
    (request: DecisionSubmitRequest) => {
      submitDecision.mutate(request, {
        onSuccess: () => toast.success("Decision submitted"),
        onError: (err) => toast.error(err.message || "Decision failed"),
      });
    },
    [submitDecision]
  );

  // Handle mission completion
  const handleMissionComplete = useCallback(
    (request: CombatCompleteRequest) => {
      completeCombat.mutate(request, {
        onSuccess: (result) => {
          setShowMissionCompleteModal(false);
          toast.success("Mission completed");
          
          // Redirect to debrief if missionId is known, otherwise to campaign/dashboard
          if (search.missionId) {
            // Compute statistics
            const turnsTaken = scenario?.rounds?.reduce((acc, round) => acc + (round.turns?.length || 0), 0) || 0;
            const enemyCount = scenario?.combatants?.filter(c => c.side !== "players").length || 0;
            const damageDealt = enemyCount * 300; // placeholder
            const damageReceived = 1200; // placeholder
            
            navigate({
              to: "/missions/$missionId/debrief",
              params: { missionId: search.missionId },
              search: {
                outcome: request.outcome === "success" || request.outcome === "partial" ? "victory" : "defeat",
                turns: turnsTaken,
                damageDealt,
                damageReceived,
              },
            });
          } else if (result.campaign_id) {
            navigate({ to: "/campaigns/$campaignId", params: { campaignId: result.campaign_id } });
          } else {
            navigate({ to: "/" });
          }
        },
        onError: (err) => toast.error(err.message || "Failed to complete mission"),
      });
    },
    [completeCombat, navigate, search, scenario]
  );

  // Handle reserve spending
  const handleSpendReserve = useCallback(
    (reserveId: string) => {
      spendReserve.mutate(
        { reserve_id: reserveId },
        {
          onSuccess: () => toast.success("Reserve spent"),
          onError: (err) => toast.error(err.message || "Failed to spend reserve"),
        }
      );
    },
    [spendReserve]
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
  const handlePathModeChange = useCallback((isActive: boolean, path: HexCoord[]) => {
    setIsPathMode(isActive);
    setMovementPath(path);
    if (!isActive) {
      setPathHexClick(null);
    }
  }, []);

  // Handle AoE preview changes from ActionPanel (weapon selection)
  const handleAreaPreviewChange = useCallback(
    (pattern: AttackPatternDefinition | null, origin: HexCoord | null, direction: HexCoord | null) => {
      setAreaPattern(pattern);
      setPreviewOrigin(origin);
      setAreaDirection(direction);
    },
    []
  );

  // Handle hex click for path building (from CombatCanvas)
  const handlePathHexClick = useCallback((coord: HexCoord) => {
    if (isPathMode) {
      setPathHexClick(coord);
    }
  }, [isPathMode]);

  // Handle movement range preview changes from ActionPanel
  const handleMovementRangeChange = useCallback((show: boolean, speed: number) => {
    setShowMovementRange(show);
    setMovementRangeSpeed(speed);
  }, []);

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
    [setZoom, viewport.zoom]
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
    [targetMode, maxTargets]
  );

  // Handle right-click context menu on canvas
  const handleContextMenu = useCallback(
    (info: ContextMenuInfo) => {
      // Don't show context menu during path mode or certain states
      if (isPathMode) return;

      // Determine what was clicked
      let target: ContextMenuTarget;

      if (info.tokenId) {
        // Clicked on a combatant token
        const combatant = combatants.find((c) => c.id === info.tokenId);
        if (combatant) {
          const isEnemy = combatant.side !== "players";
          target = isEnemy
            ? { type: "enemy", combatantId: combatant.id, combatantName: combatant.name, coord: info.coord }
            : { type: "friendly", combatantId: combatant.id, combatantName: combatant.name, coord: info.coord };
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
    [combatants, scenario?.deployables, isPathMode]
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
        if (contextMenu?.target.type === "enemy" || contextMenu?.target.type === "friendly") {
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
    [handleActionSelect, contextMenu]
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
  }, [targetMode, combatants, currentActor, selectedTargetIds, maxTargets, scenario?.deployables]);

  // Derive active indices from selectedAction or fall back to current position
  const activeRoundIndex = selectedAction?.roundIdx ?? clampIndex(currentRound - 1, rounds);
  const round = rounds[activeRoundIndex] ?? null;
  const turns = round?.turns ?? [];
  const activeTurnIndex = selectedAction?.turnIdx ?? clampIndex(currentTurnIndex, turns);
  const turn = turns[activeTurnIndex] ?? null;
  const actions = turn?.actions ?? [];
  const activeActionIndex = selectedAction?.actionIdx ?? 0;
  const action = actions[activeActionIndex] ?? null;

  const combatantNameById = useMemo(
    () => new Map((scenario?.combatants ?? []).map((c) => [c.id, c.name])),
    [scenario?.combatants],
  );
  const weaponDefinitions = useMemo(
    () => new Map((weaponsQuery.data ?? []).map((weapon) => [weapon.id, weapon])),
    [weaponsQuery.data],
  );

  // Build movement range overlays when in path mode
  const movementRangeOverlays = useMemo(() => {
    if (!showMovementRange || !currentActor?.position?.coord || !scenario) {
      return [];
    }

    const origin = currentActor.position.coord;
    const speed = movementRangeSpeed;

    if (speed <= 0) return [];

    // Build valid hex set from grid (we need to compute the grid first)
    // Use a temporary grid calculation matching what adaptCombatScenario would use
    const combatants = scenario.combatants ?? [];
    let maxDistance = 0;
    for (const combatant of combatants) {
      if (combatant.position?.coord) {
        const dist = Math.abs(combatant.position.coord.q) + Math.abs(combatant.position.coord.r);
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
        blockedHexes.add(`${combatant.position.coord.q},${combatant.position.coord.r}`);
      }
    }

    // Build difficult terrain set from scenario
    const difficultHexes = new Set<string>();
    for (const tile of scenario.terrain?.tiles ?? []) {
      if (tile.difficult) {
        difficultHexes.add(`${tile.coord.q},${tile.coord.r}`);
      }
    }

    return buildMovementRangeOverlays(origin, speed, validHexes, blockedHexes, difficultHexes);
  }, [showMovementRange, currentActor, scenario, movementRangeSpeed]);

  const renderOutput: CombatRenderAdapterOutput | null = useMemo(() => {
    if (!scenario) {
      return null;
    }
    // For blast patterns, use preview origin (follows cursor) instead of actor position
    const effectivePatternOrigin = areaPattern?.pattern === "blast" && previewOrigin
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
    });

    // Add movement range overlays (before other overlays so they appear underneath)
    if (movementRangeOverlays.length > 0) {
      result.state.overlays = [
        ...movementRangeOverlays,
        ...(result.state.overlays ?? []),
      ];
    }

    return result;
  }, [action, hovered, round, scenario, turn, areaPattern, areaDirection, currentActor, previewOrigin, movementRangeOverlays]);

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
            <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Reconnecting...
          </div>
        </div>
      )}

      {/* Compact header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link to="/" className="text-primary hover:underline text-sm">←</Link>
          <div>
            <h1 className="text-lg font-heading font-semibold text-foreground">{data.name}</h1>
            <p className="text-xs text-muted-foreground">Round {data.current_round} · {data.status}</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <span className={`w-2 h-2 rounded-full ${wsConnected ? "bg-green-500" : "bg-amber-500"}`} />
            {wsConnected ? "Live" : "Polling"}
          </div>
          {data.status === "active" && (
            <Button variant="outline" size="sm" onClick={() => setShowMissionCompleteModal(true)}>
              End Mission
            </Button>
          )}
        </div>
      </div>

      {/* Mission Complete Modal */}
      <MissionCompleteModal
        isOpen={showMissionCompleteModal}
        onComplete={handleMissionComplete}
        onCancel={() => setShowMissionCompleteModal(false)}
        isSubmitting={completeCombat.isPending}
        campaignId={data.campaign_id}
        missionReserves={scenario?.mission_reserves}
      />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
        {/* Canvas area - reduced height when action bar visible */}
        <div className="relative rounded-md border border-border bg-muted/30 p-2">
          <div className={`min-h-[400px] w-full ${turnActive ? "h-[calc(100vh-280px)]" : "h-[calc(100vh-180px)]"}`}>
                {renderOutput ? (
                  <CombatCanvas
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
                      if (areaPattern?.pattern === "blast" && targetMode?.requiresTarget) {
                        setPreviewOrigin(coord);
                      }

                      // Update hover tooltip
                      if (!coord) {
                        setHoverTooltip(null);
                        return;
                      }

                      // Determine what's at this hex
                      const combatant = combatants.find(
                        (c) => c.position?.coord?.q === coord.q && c.position?.coord?.r === coord.r
                      );
                      const deployableEntry = Object.entries(scenario?.deployables ?? {}).find(
                        ([_, d]) => d.position?.coord?.q === coord.q && d.position?.coord?.r === coord.r
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
                          position: { x: rect.left + point.x, y: rect.top + point.y },
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
          {/* Sticky Turn Controls with integrated Economy */}
          <div className="sticky top-0 z-10 bg-background pb-2 space-y-2">
            {/* Turn Indicator with initiative order */}
            <TurnIndicator
              currentActor={currentActor}
              combatants={combatants}
              roundNumber={currentRound}
              turnIndex={currentTurnIndex}
              isTurnActive={turnActive}
              isPlayerTurn={currentActor?.side === "players"}
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
            />
          </div>

          {/* Scrollable content area */}
          <div className="flex-1 overflow-y-auto space-y-3 pr-1">
            {/* Action Log - at top for visibility */}
            <div className="rounded-md border border-border bg-muted/30 p-2 space-y-1">
              <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
                Action Log
              </div>
              <div className="max-h-24 overflow-y-auto">
                <ActionLog
                  rounds={rounds}
                  currentRound={currentRound}
                  currentTurnIndex={currentTurnIndex}
                  combatantNames={combatantNameById}
                  selectedAction={selectedAction}
                  onSelectAction={(roundIdx, turnIdx, actionIdx) =>
                    setSelectedAction({ roundIdx, turnIdx, actionIdx })
                  }
                />
              </div>
            </div>

            {/* Action Panel (only when turn is active) */}
            {turnActive && (
              <ActionPanel
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
              />
            )}

            {/* Victory Conditions (if SITREP active) */}
            <VictoryConditionPanel sitrepResolution={scenario?.sitrep_resolution} />

            {/* Mission Objectives (if available) */}
            <ObjectiveTracker objectives={scenario?.objectives} />

            {/* Mission Reserves (if available) */}
            <ReservesPanel
              reserves={scenario?.mission_reserves}
              onSpendReserve={handleSpendReserve}
              isSpending={spendReserve.isPending}
            />

            {/* Combatants List - compact */}
            <div className="rounded-md border border-border bg-muted/30 p-2 space-y-1">
              <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
                Combatants
              </div>
              <div className="space-y-0.5">
                {combatants.map((combatant) => (
                  <div
                    key={combatant.id}
                    className={`flex items-center justify-between rounded px-2 py-1 text-xs ${
                      combatant.id === currentActor?.id
                        ? "bg-primary/10 border-l-2 border-primary"
                        : "bg-muted/40"
                    } ${
                      selectedTargetIds.includes(combatant.id)
                        ? "ring-1 ring-green-500"
                        : ""
                    }`}
                  >
                    <div className="font-medium truncate">
                      {combatant.name}
                      {combatant.id === currentActor?.id && (
                        <span className="ml-1 text-[10px] text-primary font-normal">●</span>
                      )}
                    </div>
                    <div className="text-[10px] text-muted-foreground ml-2">
                      {combatant.position?.coord
                        ? `${combatant.position.coord.q},${combatant.position.coord.r}`
                        : "--"}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Modals - outside sidebar */}
        {/* Overcharge Confirmation Modal */}
        {showOverchargeConfirm && currentActor && (
          <OverchargeConfirm
            currentLevel={availableActions?.overcharge_level ?? 0}
            heatCurrent={currentActor.resources?.heat_current ?? 0}
            heatCap={currentActor.resources?.heat_cap ?? 6}
            onConfirm={handleOverchargeConfirm}
            onCancel={() => setShowOverchargeConfirm(false)}
            isOpen={showOverchargeConfirm}
          />
        )}

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
            {reactionOpportunity?.pending_triggers?.[0] && firstPlayerCombatant && (
              <ReactionPrompt
                triggerType={reactionOpportunity.pending_triggers[0].trigger_type}
                reactorId={reactionOpportunity.combatant_id}
                reactorName={reactionOpportunity.combatant_name}
                triggeringActorName={reactionOpportunity.pending_triggers[0].triggering_actor_name}
                availableReactions={reactionOpportunity.pending_triggers[0].available_reactions}
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

        {/* Hover Tooltip */}
        <MapTooltip
          target={hoverTooltip?.target ?? null}
          position={hoverTooltip?.position ?? null}
          delay={300}
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
          availableActions={availableActions ?? null}
          economy={economy}
          onActionSelect={handleActionSelect}
          onOvercharge={() => setShowOverchargeConfirm(true)}
          canOvercharge={availableActions?.can_overcharge ?? false}
          overchargeLevel={availableActions?.overcharge_level ?? 0}
          isExecuting={executeAction.isPending}
          visible={turnActive}
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
