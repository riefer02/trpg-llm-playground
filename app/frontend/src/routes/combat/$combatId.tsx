import { useCallback, useMemo, useState } from "react";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { toast } from "sonner";

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
import { CombatCanvas, type TargetingMode } from "../../components/combat/CombatCanvas";
import {
  ActionLog,
  type SelectedAction,
} from "../../components/combat/ActionLog";
import { TerrainLegend } from "../../components/combat/TerrainLegend";
import { TurnControls, type TurnState } from "../../components/combat/TurnControls";
import { ActionPanel, type TargetMode } from "../../components/combat/ActionPanel";
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
  type CombatRenderAdapterOutput,
} from "../../lib/combat-render/adapter";
import { createHexLayout } from "../../lib/combat-render/hex";
import type { HexCoord, AttackPatternDefinition } from "../../lib/types/lancer";
import {
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Modal,
} from "../../components/ui";
import { CombatSessionSkeleton } from "../../components/skeletons";

export const Route = createFileRoute("/combat/$combatId")({
  component: CombatSessionPage,
});

/** Polling interval when WebSocket is disconnected (5 seconds) */
const FALLBACK_POLLING_INTERVAL = 5000;

function CombatSessionPage() {
  const { combatId } = Route.useParams();

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
  const [selected, setSelected] = useState<HexCoord | null>(null);
  const [targeted, setTargeted] = useState<HexCoord | null>(null);
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

  // Handle action selection from ActionPanel
  const handleActionSelect = useCallback((action: AvailableActionItem) => {
    // Reset targets when selecting new action and set max targets from action
    setSelectedTargetIds([]);
    setMaxTargets(action.max_targets);
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
            setEconomy(result.economy);
            setTargetMode(null);
            setSelectedTargetIds([]);
            setMaxTargets(1);
            setAreaPattern(null);
            setAreaDirection(null);
            setPreviewOrigin(null);
            setActionError(null);
            toast.success("Action executed");
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
    [executeAction, scenario?.deployables]
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
          // Redirect to campaign if linked, otherwise to dashboard
          if (result.campaign_id) {
            navigate({ to: "/campaigns/$campaignId", params: { campaignId: result.campaign_id } });
          } else {
            navigate({ to: "/" });
          }
        },
        onError: (err) => toast.error(err.message || "Failed to complete mission"),
      });
    },
    [completeCombat, navigate]
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

  const renderOutput: CombatRenderAdapterOutput | null = useMemo(() => {
    if (!scenario) {
      return null;
    }
    // For blast patterns, use preview origin (follows cursor) instead of actor position
    const effectivePatternOrigin = areaPattern?.pattern === "blast" && previewOrigin
      ? { coord: previewOrigin }
      : currentActor?.position;

    return adaptCombatScenario({
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
  }, [action, hovered, round, scenario, turn, areaPattern, areaDirection, currentActor, previewOrigin]);

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
    <div className="px-6 py-8 max-w-7xl mx-auto space-y-6">
      {/* WebSocket disconnection banner */}
      {!wsConnected && (
        <div className="fixed top-0 left-0 right-0 z-40 bg-amber-500 text-amber-950 px-4 py-2 text-sm text-center font-medium shadow-lg animate-in slide-in-from-top duration-300">
          <div className="flex items-center justify-center gap-2">
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Live connection lost — Polling for updates every {FALLBACK_POLLING_INTERVAL / 1000}s
          </div>
        </div>
      )}

      <section className="dashboard-surface p-6 animate-rise">
        <Link to="/" className="text-primary hover:underline text-sm">
          ← Back to Dashboard
        </Link>
        <div className="mt-3 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h1 className="text-3xl font-heading font-semibold text-foreground">
              {data.name}
            </h1>
            <p className="text-muted-foreground">
              Status: {data.status} · Round {data.current_round}
            </p>
          </div>
          <div className="flex items-center gap-4">
            {data.status === "active" && (
              <Button
                variant="outline"
                onClick={() => setShowMissionCompleteModal(true)}
              >
                End Mission
              </Button>
            )}
            <div className="flex flex-col items-end gap-1">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span
                  className={`w-2 h-2 rounded-full ${wsConnected ? "bg-green-500" : "bg-amber-500"}`}
                />
                {wsConnected ? "Live" : "Polling"}
              </div>
              <div className="text-xs text-muted-foreground">
                Session ID: {data.id}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Mission Complete Modal */}
      <MissionCompleteModal
        isOpen={showMissionCompleteModal}
        onComplete={handleMissionComplete}
        onCancel={() => setShowMissionCompleteModal(false)}
        isSubmitting={completeCombat.isPending}
        campaignId={data.campaign_id}
        missionReserves={scenario?.mission_reserves}
      />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
        <Card className="h-full">
          <CardHeader>
            <CardTitle>Combat Canvas</CardTitle>
            <CardDescription>
              {isPathMode
                ? "Click adjacent hexes to build movement path. Click last hex to undo."
                : targetMode?.requiresTarget
                  ? "Click a combatant to select as target"
                  : "Hover for hex highlight, left click to select, right click to target."}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="rounded-md border border-border bg-muted/30 p-3">
              <div className="h-[520px] w-full">
                {renderOutput ? (
                  <CombatCanvas
                    width={720}
                    height={520}
                    resizeToParent
                    layout={(size) =>
                      createHexLayout(30, {
                        x: size.width / 2,
                        y: size.height / 2,
                      })
                    }
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
                    onHover={(coord) => {
                      setHovered(coord);
                      // Update blast preview origin when hovering (blast follows cursor)
                      if (areaPattern?.pattern === "blast" && targetMode?.requiresTarget) {
                        setPreviewOrigin(coord);
                      }
                    }}
                    onSelect={(coord) => setSelected(coord)}
                    onTarget={(coord) => setTargeted(coord)}
                    onTokenClick={handleTokenClick}
                    onHexClick={handlePathHexClick}
                    className="h-full w-full"
                  />
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                    No scenario data available yet.
                  </div>
                )}
              </div>
              <div className="mt-3 flex flex-wrap gap-4 text-sm text-muted-foreground">
                <div>Hover: {formatCoord(hovered)}</div>
                <div>Selected: {formatCoord(selected)}</div>
                <div>Targeted: {formatCoord(targeted)}</div>
              </div>
              <TerrainLegend className="mt-3" />
            </div>
          </CardContent>
        </Card>

        <div className="flex flex-col h-full max-h-[calc(100vh-220px)]">
          {/* Sticky Turn Controls with integrated Economy */}
          <div className="sticky top-0 z-10 bg-background pb-3">
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
          <div className="flex-1 overflow-y-auto space-y-4 pr-1">
            {/* Action Panel (only when turn is active) - prioritized at top */}
            {turnActive && (
              <ActionPanel
                availableActions={availableActions ?? null}
                economy={economy}
                onActionSelect={handleActionSelect}
                onExecuteAction={handleExecuteAction}
                onTargetModeChange={handleTargetModeChange}
                onPathModeChange={handlePathModeChange}
                onAreaPreviewChange={handleAreaPreviewChange}
                isExecuting={executeAction.isPending}
                selectedTargetIds={selectedTargetIds}
                actorInventory={currentActor?.inventory}
                weaponDefinitions={weaponDefinitions}
                actorSpeed={currentActor?.stats?.speed ?? 4}
                actorPosition={currentActor?.position?.coord ?? null}
                hexClickCoord={pathHexClick}
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
            <div className="rounded-md border border-border bg-muted/30 p-3 space-y-2">
              <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Combatants
              </div>
              <div className="space-y-1">
                {combatants.map((combatant) => (
                  <div
                    key={combatant.id}
                    className={`flex items-center justify-between rounded px-2 py-1.5 text-sm ${
                      combatant.id === currentActor?.id
                        ? "bg-primary/10 border-l-2 border-primary"
                        : "bg-muted/40"
                    } ${
                      selectedTargetIds.includes(combatant.id)
                        ? "ring-1 ring-green-500"
                        : ""
                    }`}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="font-medium truncate">
                        {combatant.name}
                        {combatant.id === currentActor?.id && (
                          <span className="ml-1 text-[10px] text-primary font-normal">(turn)</span>
                        )}
                      </div>
                    </div>
                    <div className="text-xs text-muted-foreground ml-2">
                      {combatant.position?.coord
                        ? `${combatant.position.coord.q},${combatant.position.coord.r}`
                        : "--"}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Action Log - compact */}
            <div className="rounded-md border border-border bg-muted/30 p-3 space-y-2">
              <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Action Log
              </div>
              <div className="max-h-32 overflow-y-auto">
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

function formatCoord(coord: HexCoord | null): string {
  if (!coord) {
    return "--";
  }
  return `${coord.q},${coord.r}`;
}
