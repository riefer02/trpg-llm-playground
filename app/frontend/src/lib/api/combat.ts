import { useQuery } from "@tanstack/react-query";

import { api } from "./client";
import type { MechCombatScenario } from "../types/lancer";

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

export function useCombatSession(sessionId: string) {
  return useQuery({
    queryKey: combatKeys.detail(sessionId),
    queryFn: () => api.get<CombatSessionResponse>(`/combat/${sessionId}`),
    enabled: Boolean(sessionId),
  });
}
