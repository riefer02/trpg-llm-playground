import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "./client";
import type { FullTechOptionSelection, MechCombatScenario } from "../types/lancer";

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

export interface ActionRequest {
  action_id: string;
  action_type: "full" | "quick" | "free" | "reaction" | "protocol" | "move";
  target_ids?: string[];
  target_position?: { coord: { q: number; r: number } };
  weapon_id?: string;
  weapon_profile_id?: string;
  system_id?: string;
  full_tech_first?: FullTechOptionSelection;
  full_tech_second?: FullTechOptionSelection;
  movement_path?: { coord: { q: number; r: number } }[];
  is_overcharge?: boolean;
  use_thrown?: boolean;
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

export interface ActionEconomyState {
  full_action_used: boolean;
  quick_actions_used: number;
  quick_actions_available: number;
  free_actions_used: string[];
  reaction_used: boolean;
  overcharge_used: boolean;
  protocol_used: boolean;
  movement_used: number;
  movement_available: number;
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
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () =>
      api.post<TurnStartResponse>(`/combat/${sessionId}/turns/start`),
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

export function useEndTurn(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () =>
      api.post<TurnEndResponse>(`/combat/${sessionId}/turns/end`),
    onSuccess: (data) => {
      queryClient.setQueryData<CombatSessionResponse>(
        combatKeys.detail(sessionId),
        (prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            scenario: data.scenario,
            current_round: data.new_round_number ?? prev.current_round,
          };
        },
      );
    },
  });
}

// =============================================================================
// Action Execution Mutations
// =============================================================================

export function useExecuteAction(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (action: ActionRequest) =>
      api.post<ActionResponse>(`/combat/${sessionId}/actions`, action),
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
      }
    },
  });
}

export function useSubmitReaction(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (reaction: ReactionRequest) =>
      api.post<ReactionResponse>(`/combat/${sessionId}/reactions`, reaction),
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
      }
    },
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
