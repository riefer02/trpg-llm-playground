/**
 * API module exports.
 *
 * Import from this file for cleaner imports:
 *   import { useHealth, useCharacters, api } from '@/lib/api'
 *
 * For core domain types, import from '@/lib/types/lancer'
 */

// Core client
export { apiClient, api, APIError } from './client'
export type { RequestOptions } from './client'

// Health hooks
export { useHealth, useDatabaseHealth, healthKeys } from './health'
export type { HealthResponse, DatabaseHealthResponse } from './health'

// Character hooks (primary user-facing API)
export {
  useCharacters,
  useCharacter,
  useCharacterValidation,
  useCreateCharacter,
  useUpdateCharacter,
  useUpdatePilotGear,
  useUpdateMechBuild,
  useDeleteCharacter,
  useAddMech,
  useRemoveMech,
  useSetActiveMech,
  characterKeys,
} from './characters'
export type {
  CharacterResponse,
  CharacterListResponse,
  CharacterCreateRequest,
  CharacterUpdateRequest,
  PilotGearUpdateRequest,
  MechBuildUpdateRequest,
  CharacterValidationResponse,
  MechAddRequest,
  MechStats,
  MechConfig,
  ValidationIssue,
} from './characters'

// Compendium hooks (reference data for character creation)
export {
  useBackgrounds,
  useTriggers,
  useTalents,
  useFrames,
  useWeapons,
  useSystems,
  usePilotGear,
  useLicenses,
  compendiumKeys,
} from './compendium'
// Note: Compendium types are simplified API responses, not full core types
export type {
  Background as CompendiumBackground,
  Trigger as CompendiumTrigger,
  Talent as CompendiumTalent,
  License as CompendiumLicense,
} from './compendium'

// Pilot hooks (internal/low-level primitive)
export {
  usePilots,
  usePilot,
  usePilotValidation,
  useCreatePilot,
  useUpdatePilot,
  useDeletePilot,
  pilotKeys,
} from './pilots'
export type {
  PilotResponse,
  PilotListResponse,
  PilotCreateRequest,
  PilotUpdateRequest,
  PilotValidationResponse,
} from './pilots'

// Campaign hooks
export {
  useCampaigns,
  useCampaign,
  useCreateCampaign,
  useCreateCampaignInvite,
  useAcceptCampaignInvite,
  useAttachCampaignCharacter,
  useUpdateCampaignMemberSettings,
  useUpdateCampaignIdentity,
  useUpdateCampaignLobby,
  useLaunchCampaignMission,
  useUpdateSessionLifecycle,
  usePreviewCampaignInvite,
  useRevokeCampaignInvite,
  useResendCampaignInvite,
  useRecordCampaignSessionOutcome,
  useReserveTemplates,
  useBeginDowntime,
  campaignKeys,
} from './campaigns'
export type {
  CampaignSummary,
  CampaignDetail,
  CampaignListResponse,
  CampaignCreateRequest,
  CampaignInviteCreateRequest,
  CampaignInvitePreviewResponse,
  CampaignInviteResendRequest,
  CampaignSessionOutcomeRequest,
  CampaignCharacterAttachRequest,
  CampaignMemberSettingsRequest,
  CampaignIdentityUpdateRequest,
  CampaignLobbyUpdateRequest,
  CampaignMissionLaunchRequest,
  SessionLifecycleUpdateRequest,
  CampaignReadinessSummary,
  CampaignMember,
  CampaignInvite,
  CampaignCharacter,
  ReserveTemplate,
} from './campaigns'

// Combat session hooks
export {
  useCombatSession,
  useStartTurn,
  useEndTurn,
  useExecuteAction,
  useSubmitReaction,
  useAvailableActions,
  useReactionOpportunity,
  usePendingDecisions,
  useSubmitDecision,
  useCompleteCombat,
  useSpendReserve,
  useCreateDemoCombat,
  useAutoNpcTurn,
  combatKeys,
} from './combat'
export type {
  CombatSessionResponse,
  UseCombatSessionOptions,
  TurnStartResponse,
  TurnEndResponse,
  ActionRequest,
  ActionResponse,
  ReactionRequest,
  ReactionResponse,
  AvailableActionsResponse,
  AvailableActionItem,
  ActionEconomyState,
  UseAvailableActionsOptions,
  UseReactionOpportunityOptions,
  ReactionOpportunityResponse,
  ReactionTrigger,
  // Decision types
  DecisionType,
  DecisionChoice,
  SaveType,
  PendingDecisionItem,
  PendingDecisionsResponse,
  DecisionSubmitRequest,
  DecisionResultResponse,
  UsePendingDecisionsOptions,
  // Mission completion types
  MissionOutcome,
  CombatCompleteRequest,
  // Demo combat types
  DemoScenarioType,
  // Auto NPC turn types
  AutoNPCTurnResponse,
} from './combat'

// Combat WebSocket hook for real-time updates
export { useCombatWebSocket } from './combat-ws'
export type {
  CombatWebSocketState,
  UseCombatWebSocketOptions,
} from './combat-ws'

// Quarters hooks (pilot quarters hub)
export { useActiveCharacter, useMissionCount } from './quarters'

// Missions hooks
export { useMissions, useMission } from './missions'
export type { Mission } from './missions'

// Re-export generated core types for convenience
// These are the source of truth - use these for domain data
export type {
  PilotTrigger,
  Talent,
  License,
  CoreBonus,
  Background,
  SkillSet,
  Character,
  MechConfiguration,
  Pilot,
} from '../types/lancer'
