import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "./client";
import { characterKeys, type CharacterResponse } from "./characters";
import type { FullTechOptionSelection, MechCombatScenario } from "../types/lancer";
import { getActiveCharacterId, autoSave } from '../save/saveSystem';

// =============================================================================
// Request/Response Types for Combat Execution
// =============================================================================

export interface TurnStartResponse {
  actor_id: string;
  actor_name: string;
  economy: ActionEconomyState;
  available_actions: string[];
  prepared_action_expired: boolean;
  cooldowns_decremented: string[];
  scenario: MechCombatScenario;
}

// =============================================================================
// Voice Intent Parsing Types and Hooks
// =============================================================================

export interface VoiceIntentRequest {
  transcript: string;
  actor_id?: string;
}

export interface VoiceIntentResponse {
  success: boolean;
  transcript: string;
  action?: Record<string, unknown>;
  confidence?: number;
  fallback_prompt?: string;
  error?: string;
  scenario: MechCombatScenario;
}

export function useParseVoiceIntent(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: VoiceIntentRequest) =>
      api.post<VoiceIntentResponse>(`/combat/${sessionId}/voice-intent`, request),
    onSuccess: (data) => {
      if (data.success) {
        // Update combat session with latest scenario (if changed)
        queryClient.setQueryData<CombatSessionResponse>(
          combatKeys.detail(sessionId),
          (prev) => {
            if (!prev) return prev;
            return {
              ...prev,
              scenario: data.scenario,
            };
          },
        );
      }
    },
  });
}

export function useActionPreview(sessionId: string) {
  return useMutation({
    mutationFn: (request: ActionPreviewRequest) =>
      api.post<ActionPreviewResponse>(`/combat/${sessionId}/action-preview`, request),
  });
}

export interface ActionResponse {
  success: boolean;
  error?: string;
  action_use?: Record<string, unknown>;
  effects_applied: Record<string, unknown>[];
  damage_dealt: number;
  heat_generated: number;
  economy: ActionEconomyState;
  scenario: MechCombatScenario;
}

export interface TurnEndResponse {
  actor_id: string;
  next_actor_id: string | null;
  next_actor_name: string | null;
  round_advanced: boolean;
  new_round_number: number | null;
  end_of_turn_effects: Record<string, unknown>[];
  scenario: MechCombatScenario;
}

export interface ReactionRequest {
  reactor_id: string;
  reaction_type: "brace" | "overwatch";
  trigger_action_id?: string;
  target_ids?: string[];
  weapon_id?: string;
}

export interface ReactionResponse {
  success: boolean;
  error?: string;
  reaction_used?: string;
  effects_applied: Record<string, unknown>[];
  damage_dealt: number;
  scenario: MechCombatScenario;
}

export interface AvailableActionItem {
  action_id: string;
  action_name: string;
  action_type: string;
  is_available: boolean;
  unavailable_reason?: string;
  requires_target: boolean;
  requires_weapon: boolean;
  requires_system: boolean;
  requires_path: boolean;
  max_targets: number;
}

export interface AvailableActionsResponse {
  actor_id: string;
  economy: ActionEconomyState;
  full_actions: AvailableActionItem[];
  quick_actions: AvailableActionItem[];
  free_actions: AvailableActionItem[];
  reactions: AvailableActionItem[];
  protocols: AvailableActionItem[];
  can_overcharge: boolean;
  overcharge_level: number;
}

export interface ReactionTrigger {
  trigger_type: "attack_incoming" | "enemy_movement";
  triggering_actor_id: string;
  triggering_actor_name: string;
  triggering_action_id?: string;
  available_reactions: ("brace" | "overwatch")[];
}

export interface ReactionOpportunityResponse {
  combatant_id: string;
  combatant_name: string;
  has_reaction_available: boolean;
  pending_triggers: ReactionTrigger[];
}

export interface ActionPreviewRequest {
  action_id: string;
  actor_id: string;
  target_id: string;
  weapon_id?: string;
}

export interface ActionPreviewResponse {
  action_id: string;
  actor_id: string;
  target_id: string;
  weapon_id?: string;
  damage_min: number;
  damage_max: number;
  damage_average: number;
  damage_types: string[];
  hit_probability: number;
  predicted_effects: Record<string, unknown>[];
  is_valid: boolean;
  validation_errors: string[];
}

export interface ActionEconomyState {
  /** Full actions used this turn (0 or 1) */
  full_actions_used: number;
  /** Quick actions used this turn (0-2+) */
  quick_actions_used: number;
  /** Whether overcharge was used this turn */
  overcharge_used: boolean;
  /** Reactions used this turn (0 or 1) */
  reactions_used_this_turn: number;
}

export interface CombatSessionResponse {
  id: string;
  gm_user_id: string;
  campaign_id: string | null;
  created_at: string;
  updated_at: string;
  name: string;
  status: string;
  current_round: number;
  current_turn_index: number;
  notes: string;
  xp_awarded?: number;
  salvage_awarded?: number;
  scenario: MechCombatScenario;
}

export const combatKeys = {
  all: ["combat"] as const,
  detail: (sessionId: string) => [...combatKeys.all, sessionId] as const,
};

export interface UseCombatSessionOptions {
  /** Polling interval in milliseconds. Undefined disables polling. */
  pollingInterval?: number;
}

export function useCombatSession(
  sessionId: string,
  options?: UseCombatSessionOptions,
) {
  return useQuery({
    queryKey: combatKeys.detail(sessionId),
    queryFn: () => api.get<CombatSessionResponse>(`/combat/${sessionId}`),
    enabled: Boolean(sessionId),
    refetchInterval: options?.pollingInterval,
  });
}

// =============================================================================
// Turn Management Mutations
// =============================================================================

export function useStartTurn(sessionId: string) {
  // Note: We don't manually update the cache here because the WebSocket
  // broadcast will handle it. This avoids race conditions where both
  // the mutation onSuccess and WebSocket update the cache simultaneously,
  // which can cause React DOM reconciliation errors.
  return useMutation({
    mutationFn: () =>
      api.post<TurnStartResponse>(`/combat/${sessionId}/turns/start`),
  });
}

export function useEndTurn(sessionId: string) {
  // Note: WebSocket broadcast handles cache updates to avoid race conditions
  return useMutation({
    mutationFn: () =>
      api.post<TurnEndResponse>(`/combat/${sessionId}/turns/end`),
  });
}

// =============================================================================
// Action Execution Mutations
// =============================================================================

export function useExecuteAction(sessionId: string) {
  // Note: WebSocket broadcast handles cache updates to avoid race conditions
  return useMutation({
    mutationFn: (action: ActionRequest) =>
      api.post<ActionResponse>(`/combat/${sessionId}/actions`, action),
  });
}

export function useSubmitReaction(sessionId: string) {
  // Note: WebSocket broadcast handles cache updates to avoid race conditions
  return useMutation({
    mutationFn: (reaction: ReactionRequest) =>
      api.post<ReactionResponse>(`/combat/${sessionId}/reactions`, reaction),
  });
}

// =============================================================================
// Available Actions Query
// =============================================================================

export interface UseAvailableActionsOptions {
  enabled?: boolean;
}

export function useAvailableActions(
  sessionId: string,
  options?: UseAvailableActionsOptions,
) {
  return useQuery({
    queryKey: [...combatKeys.detail(sessionId), "available-actions"] as const,
    queryFn: () =>
      api.get<AvailableActionsResponse>(`/combat/${sessionId}/available-actions`),
    enabled: Boolean(sessionId) && (options?.enabled ?? true),
  });
}

// =============================================================================
// Reaction Opportunity Query
// =============================================================================

export interface UseReactionOpportunityOptions {
  enabled?: boolean;
  pollingInterval?: number;
}

export function useReactionOpportunity(
  sessionId: string,
  combatantId: string | null,
  options?: UseReactionOpportunityOptions,
) {
  return useQuery({
    queryKey: [
      ...combatKeys.detail(sessionId),
      "reaction-opportunity",
      combatantId,
    ] as const,
    queryFn: () =>
      api.get<ReactionOpportunityResponse>(
        `/combat/${sessionId}/reaction-opportunities/${combatantId}`
      ),
    enabled: Boolean(sessionId) && Boolean(combatantId) && (options?.enabled ?? true),
    refetchInterval: options?.pollingInterval,
  });
}

// =============================================================================
// Pending Decisions Types and Hooks
// =============================================================================

export type DecisionType =
  | "hull_save"
  | "engineering_save"
  | "engineering_check"
  | "system_trauma";

export type DecisionChoice = "roll" | "voluntary_fail" | "use_reroll";

export type SaveType = "hull" | "agility" | "systems" | "engineering";

export interface PendingDecisionItem {
  decision_id: string;
  decision_type: DecisionType;
  trigger_source: string;
  trigger_round: number;

  // Save-specific
  save_type?: SaveType;
  save_target?: number;
  save_bonus: number;

  // Trauma-specific
  trauma_target?: "mount" | "system";
  eligible_mounts: number[];
  eligible_systems: string[];

  // Reroll availability
  reroll_available: boolean;
  reroll_source?: string;
}

export interface PendingDecisionsResponse {
  combatant_id: string;
  combatant_name: string;
  pending_decisions: PendingDecisionItem[];
  has_pending: boolean;
}

export interface DecisionSubmitRequest {
  decision_id: string;
  combatant_id: string;
  choice: DecisionChoice;
  selected_mount_index?: number;
  selected_system_id?: string;
}

export interface DecisionResultResponse {
  success: boolean;
  error?: string;
  roll_result?: number;
  save_succeeded?: boolean;
  effects_applied: Record<string, unknown>[];
  scenario: MechCombatScenario;
}

export interface UsePendingDecisionsOptions {
  enabled?: boolean;
  pollingInterval?: number;
}

export function usePendingDecisions(
  sessionId: string,
  combatantId: string | null,
  options?: UsePendingDecisionsOptions,
) {
  return useQuery({
    queryKey: [
      ...combatKeys.detail(sessionId),
      "pending-decisions",
      combatantId,
    ] as const,
    queryFn: () =>
      api.get<PendingDecisionsResponse>(
        `/combat/${sessionId}/pending-decisions/${combatantId}`
      ),
    enabled: Boolean(sessionId) && Boolean(combatantId) && (options?.enabled ?? true),
    refetchInterval: options?.pollingInterval,
  });
}

export function useSubmitDecision(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (decision: DecisionSubmitRequest) =>
      api.post<DecisionResultResponse>(`/combat/${sessionId}/decisions`, decision),
    onSuccess: (data) => {
      if (data.success) {
        queryClient.setQueryData<CombatSessionResponse>(
          combatKeys.detail(sessionId),
          (prev) => {
            if (!prev) return prev;
            return {
              ...prev,
              scenario: data.scenario,
            };
          },
        );
        // Invalidate pending decisions query to refresh
        queryClient.invalidateQueries({
          queryKey: [...combatKeys.detail(sessionId), "pending-decisions"],
        });
      }
    },
  });
}

// =============================================================================
// Mission Completion Types and Hooks
// =============================================================================

export type MissionOutcome = "success" | "partial" | "failure" | "catastrophic";

export interface CombatCompleteRequest {
  outcome: MissionOutcome;
  completion_score?: number;
  debrief_notes?: string;
  reserves_spent?: Record<string, unknown>[];
  reserves_earned?: Record<string, unknown>[];
  rewards?: string[];
  notes?: string;
}

export function useCompleteCombat(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: CombatCompleteRequest) =>
      api.post<CombatSessionResponse>(`/combat/${sessionId}/complete`, request),
    onSuccess: (data) => {
      queryClient.setQueryData<CombatSessionResponse>(
        combatKeys.detail(sessionId),
        data,
      );
      // Invalidate characters query to reflect XP/salvage changes
      queryClient.invalidateQueries({ queryKey: characterKeys.lists() });
      
      // Auto-save updated character
      const characterId = getActiveCharacterId();
      if (characterId) {
        queryClient.fetchQuery({
          queryKey: characterKeys.detail(characterId),
        }).then((character) => {
          autoSave(character as CharacterResponse);
        }).catch(() => {
          // Ignore errors (character may not exist)
        });
      }
    },
  });
}

// =============================================================================
// Mission Forfeit Types and Hooks
// =============================================================================

export function useForfeitCombat(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () =>
      api.post<CombatSessionResponse>(`/combat/${sessionId}/forfeit`),
    onSuccess: (data) => {
      queryClient.setQueryData<CombatSessionResponse>(
        combatKeys.detail(sessionId),
        data,
      );
      // Invalidate characters query to reflect XP/salvage changes
      queryClient.invalidateQueries({ queryKey: characterKeys.lists() });
      
      // Auto-save updated character
      const characterId = getActiveCharacterId();
      if (characterId) {
        queryClient.fetchQuery({
          queryKey: characterKeys.detail(characterId),
        }).then((character) => {
          autoSave(character as CharacterResponse);
        }).catch(() => {
          // Ignore errors (character may not exist)
        });
      }
    },
  });
}

// =============================================================================
// Reserve Spending Types and Hooks
// =============================================================================

export interface SpendReserveRequest {
  reserve_id: string;
}

export function useSpendReserve(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: SpendReserveRequest) =>
      api.post<CombatSessionResponse>(`/combat/${sessionId}/reserves/spend`, request),
    onSuccess: (data) => {
      queryClient.setQueryData<CombatSessionResponse>(
        combatKeys.detail(sessionId),
        data,
      );
    },
  });
}

// =============================================================================
// Demo Combat Types and Hooks
// =============================================================================

export type DemoScenarioType = "skirmish" | "control" | "boss";

export interface CreateDemoCombatOptions {
  scenarioType?: DemoScenarioType;
  missionId?: string;
  missionDifficulty?: number;
}

export function useCreateDemoCombat() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (options: CreateDemoCombatOptions | DemoScenarioType = "skirmish") => {
      // Backward compatibility: allow string scenarioType
      const params = typeof options === "string" ? { scenarioType: options } : options;
      const scenarioType = params.scenarioType || "skirmish";
      const queryParams = new URLSearchParams();
      queryParams.set("scenario_type", scenarioType);
      if (params.missionId) queryParams.set("mission_id", params.missionId);
      if (params.missionDifficulty) queryParams.set("mission_difficulty", params.missionDifficulty.toString());
      return api.post<CombatSessionResponse>(`/combat/demo?${queryParams.toString()}`);
    },
    onSuccess: () => {
      // Invalidate combat list to include the new demo session
      queryClient.invalidateQueries({ queryKey: combatKeys.all });
    },
  });
}

// =============================================================================
// Auto NPC Turn Types and Hooks
// =============================================================================

export interface AutoNPCTurnResponse {
  success: boolean;
  actor_id: string;
  actor_name: string;
  decision_action?: string;
  decision_target?: string;
  decision_reasoning?: string;
  // Detailed reasoning fields for AI reasoning display
  situation_assessment?: string;
  considered_options?: string;
  rationale?: string;
  confidence?: number;
  actions_taken: number;
  skipped: boolean;
  skip_reason?: string;
  scenario: MechCombatScenario;
}

export function useAutoNpcTurn(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () =>
      api.post<AutoNPCTurnResponse>(`/combat/${sessionId}/turns/auto-npc`),
    onSuccess: (data) => {
      queryClient.setQueryData<CombatSessionResponse>(
        combatKeys.detail(sessionId),
        (prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            scenario: data.scenario,
          };
        },
      );
    },
  });
}
