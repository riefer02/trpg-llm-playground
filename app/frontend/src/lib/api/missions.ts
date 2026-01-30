/**
 * Missions API hooks.
 * Provides hooks for accessing mission data and selection.
 */

import { useQuery } from "@tanstack/react-query";
import { api } from "./client";
import { useActiveCharacter } from "./quarters";

export interface Mission {
  id: string;
  name: string;
  difficulty: number; // 1-3 stars
  sitrep: string; // SITREP type: "control", "escort", "extract", "hold_out", "gauntlet", "recon"
  terrain: string; // "urban", "forest", "desert", "facility", "space station"
  enemyCount: number;
  description?: string;
  briefing: string; // 2-3 paragraph mission briefing
  objectives: string[]; // primary and secondary objectives
  enemyIntel: string; // enemy composition and threat assessment
  mapPreviewUrl?: string; // placeholder image URL
}

/**
 * Static mission data for MVP.
 * Returns 3 missions with varying difficulty, SITREP types, and terrain.
 */
const STATIC_MISSIONS: Mission[] = [
  {
    id: "mission-1",
    name: "Operation Glass Hammer",
    difficulty: 1,
    sitrep: "control",
    terrain: "urban",
    enemyCount: 4,
    description: "Secure the central plaza from hostile forces.",
    briefing: "Union intelligence reports hostile corporate forces have occupied the central plaza of Haven-7. These mercenaries are equipped with light mechs and are testing our response times. The plaza provides excellent defensive positions with high ground advantage.\n\nYour mission is to disrupt their occupation and secure the central control node. Expect resistance from Striker-class mechs supported by Artillery units. Civilian infrastructure is at risk—minimize collateral damage where possible.\n\nExtraction will be available at LZ Alpha once the control node is secured. Keep comms open for updates on enemy reinforcements.",
    objectives: [
      "Secure the central control node",
      "Neutralize all hostile mechs",
      "Minimize collateral damage (optional)"
    ],
    enemyIntel: "Enemy force consists of 4 mechs: 2 Strikers (Assault-class frames), 1 Defender (Bulwark frame), and 1 Artillery (Longshot frame). Threat assessment: Moderate. The Defender will attempt to hold the control node while Strikers flank. Artillery provides long-range support from elevated positions.",
    mapPreviewUrl: "https://placehold.co/600x400/1e293b/94a3b8?text=Urban+Plaza+Map",
  },
  {
    id: "mission-2",
    name: "Shadow Extraction",
    difficulty: 2,
    sitrep: "extract",
    terrain: "facility",
    enemyCount: 6,
    description: "Infiltrate the research facility and extract the VIP.",
    briefing: "A high-value Union scientist, Dr. Aris Thorne, has been captured by SSC security forces during a diplomatic incident. He is being held in the underground research facility on Vega Station. The facility is heavily guarded with automated turrets and patrol mechs.\n\nYour mission is to infiltrate the facility, locate Dr. Thorne, and escort him to the extraction point. The facility's layout is complex with multiple security layers. Expect resistance from fast-moving Skirmisher mechs and defensive turrets.\n\nDr. Thorne's research is critical to Union interests. His safe extraction takes priority over enemy elimination. Use stealth and speed to avoid prolonged engagements.",
    objectives: [
      "Locate and extract Dr. Aris Thorne",
      "Reach extraction point within time limit",
      "Disable security systems (optional)"
    ],
    enemyIntel: "6 enemy mechs: 3 Skirmishers (Swift-class frames), 2 Defenders (Guardian frames), and 1 Commander (Overseer frame). Threat assessment: High. Skirmishers are fast and will attempt to flank. The Commander provides tactical coordination and can call reinforcements. Automated turrets cover key corridors.",
    mapPreviewUrl: "https://placehold.co/600x400/1e293b/94a3b8?text=Research+Facility+Map",
  },
  {
    id: "mission-3",
    name: "Gauntlet Run",
    difficulty: 3,
    sitrep: "gauntlet",
    terrain: "desert",
    enemyCount: 8,
    description: "Survive waves of enemies while traversing the canyon.",
    briefing: "The Martian canyon network is a known smuggling route used by pirate factions. Intel suggests a large pirate convoy is moving through Canyon Sigma carrying stolen Union technology. Your mission is to intercept and destroy the convoy while surviving the gauntlet of enemy forces.\n\nYou will be deployed at the canyon entrance and must fight your way through multiple waves of enemies. The terrain provides limited cover but numerous choke points. Enemy reinforcements will arrive at timed intervals.\n\nThis is a high-intensity combat scenario. Ammunition and heat management will be critical. Extraction is only available at the far end of the canyon after all waves are cleared.",
    objectives: [
      "Survive all enemy waves",
      "Destroy the convoy command vehicle",
      "Reach extraction point at canyon exit"
    ],
    enemyIntel: "8+ enemy mechs in waves: 4 Strikers (varied frames), 2 Artillery (Siege-class), 1 Commander (Tactician frame), and 1 Boss (Heavy-class). Threat assessment: Severe. Waves are coordinated and will attempt to overwhelm with combined arms. The Boss mech is heavily armored and requires focused fire.",
    mapPreviewUrl: "https://placehold.co/600x400/1e293b/94a3b8?text=Canyon+Sigma+Map",
  },
];

/**
 * Hook to get available missions with lock status based on pilot license level.
 * Returns loading state, error, and mission list.
 */
export function useMissions() {
  const { character, isLoading: characterLoading } = useActiveCharacter();
  const pilotLevel = character?.level ?? 0;

  // Compute locked status: mission is locked if difficulty > pilotLevel + 1 (adjust as needed)
  // For MVP: mission difficulty > pilotLevel + 1 is locked (e.g., LL0 can access difficulty 1)
  const missions = STATIC_MISSIONS.map(mission => ({
    ...mission,
    locked: mission.difficulty > pilotLevel + 1,
  }));

  // Use query for consistency with other hooks (though data is static)
  const query = useQuery({
    queryKey: ["missions", pilotLevel],
    queryFn: () => Promise.resolve(missions),
    enabled: !characterLoading,
  });

  return {
    ...query,
    missions: query.data || [],
  };
}

/**
 * Hook to get a specific mission by ID.
 */
export function useMission(id: string) {
  const { missions, isLoading, error } = useMissions();
  const mission = missions.find(m => m.id === id);
  return {
    mission,
    isLoading,
    error,
  };
}