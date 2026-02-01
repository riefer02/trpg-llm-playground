"""Combat statistics tracking for mission debrief and player feedback.

This module provides tracking of combat statistics during a mission,
including damage dealt/received, enemies destroyed, actions taken,
and notable moments like closest calls and overkill damage.
"""

from pydantic import Field
from typing import Literal
from core.shared.models import FrozenModel


class ActionTypeCount(FrozenModel):
    """Count of actions taken by type."""

    attacks: int = Field(default=0, ge=0, description="Number of attack actions")
    moves: int = Field(default=0, ge=0, description="Number of move/boost actions")
    techs: int = Field(default=0, ge=0, description="Number of tech actions")
    full_actions: int = Field(default=0, ge=0, description="Number of full actions")
    quick_actions: int = Field(default=0, ge=0, description="Number of quick actions")
    reactions: int = Field(default=0, ge=0, description="Number of reactions used")
    overcharges: int = Field(
        default=0, ge=0, description="Number of overcharge actions"
    )


class CombatantStatistics(FrozenModel):
    """Statistics for an individual combatant."""

    combatant_id: str = Field(description="ID of the combatant")
    combatant_name: str = Field(description="Name of the combatant")
    side: Literal["players", "hostiles", "neutral"] = Field(
        description="Which side the combatant is on"
    )

    # Damage tracking
    damage_dealt: int = Field(
        default=0, ge=0, description="Total damage dealt to enemies"
    )
    damage_received: int = Field(default=0, ge=0, description="Total damage received")
    overkill_dealt: int = Field(
        default=0, ge=0, description="Excess damage beyond enemy HP (overkill)"
    )

    # HP tracking for "closest call"
    lowest_hp_reached: int = Field(
        default=0, description="Lowest HP reached during combat"
    )
    hp_at_start: int = Field(
        default=0, description="HP at combat start for calculating closest call"
    )

    # Action tracking
    actions_taken: ActionTypeCount = Field(
        default_factory=ActionTypeCount, description="Actions taken by type"
    )

    # Kills tracking
    enemies_destroyed: int = Field(
        default=0, ge=0, description="Number of enemies destroyed"
    )
    destroyed_enemy_ids: list[str] = Field(
        default_factory=list, description="IDs of destroyed enemies"
    )

    # Turns
    turns_taken: int = Field(default=0, ge=0, description="Number of turns taken")


class CombatStatistics(FrozenModel):
    """Complete combat statistics for a mission.

    Tracks aggregate and per-combatant statistics for post-mission debriefing.
    """

    # Mission metadata
    rounds_completed: int = Field(
        default=0, ge=0, description="Number of combat rounds completed"
    )
    total_turns: int = Field(
        default=0, ge=0, description="Total turns taken by all combatants"
    )

    # Aggregate statistics
    total_damage_dealt_by_players: int = Field(
        default=0, ge=0, description="Total damage dealt by player side"
    )
    total_damage_received_by_players: int = Field(
        default=0, ge=0, description="Total damage received by player side"
    )
    total_enemies_destroyed: int = Field(
        default=0, ge=0, description="Total enemies destroyed"
    )

    # Notable moments
    closest_call_hp: int = Field(
        default=0, description="Lowest HP reached by any player (closest call)"
    )
    closest_call_combatant: str = Field(
        default="", description="Name of combatant with closest call"
    )
    max_overkill: int = Field(
        default=0, ge=0, description="Maximum overkill damage dealt"
    )

    # Per-combatant breakdown
    combatant_stats: dict[str, CombatantStatistics] = Field(
        default_factory=dict, description="Statistics per combatant ID"
    )

    # Action type totals
    action_totals: ActionTypeCount = Field(
        default_factory=ActionTypeCount,
        description="Total actions taken by all combatants",
    )

    def get_player_stats(self) -> list[CombatantStatistics]:
        """Get statistics for all player-side combatants."""
        return [
            stats for stats in self.combatant_stats.values() if stats.side == "players"
        ]

    def get_hostile_stats(self) -> list[CombatantStatistics]:
        """Get statistics for all hostile combatants."""
        return [
            stats for stats in self.combatant_stats.values() if stats.side == "hostiles"
        ]

    def get_total_actions(self) -> int:
        """Get total number of actions taken."""
        return (
            self.action_totals.attacks
            + self.action_totals.moves
            + self.action_totals.techs
            + self.action_totals.full_actions
            + self.action_totals.quick_actions
        )


class CombatStatisticsTracker:
    """Mutable tracker for combat statistics during a mission.

    This class tracks statistics during combat execution and can produce
    an immutable CombatStatistics snapshot at any point.
    """

    def __init__(self) -> None:
        self.rounds_completed: int = 0
        self.total_turns: int = 0
        self.total_damage_dealt_by_players: int = 0
        self.total_damage_received_by_players: int = 0
        self.total_enemies_destroyed: int = 0
        self.closest_call_hp: int = 999999  # Start high, track minimum
        self.closest_call_combatant: str = ""
        self.max_overkill: int = 0
        self.combatant_stats: dict[str, CombatantStatistics] = {}
        self.action_totals: ActionTypeCount = ActionTypeCount()

    def initialize_combatant(
        self,
        combatant_id: str,
        combatant_name: str,
        side: Literal["players", "hostiles", "neutral"],
        starting_hp: int,
    ) -> None:
        """Initialize statistics tracking for a combatant."""
        self.combatant_stats[combatant_id] = CombatantStatistics(
            combatant_id=combatant_id,
            combatant_name=combatant_name,
            side=side,
            lowest_hp_reached=starting_hp,
            hp_at_start=starting_hp,
        )

    def record_damage_dealt(
        self,
        dealer_id: str,
        target_id: str,
        damage: int,
        target_hp_before: int,
        target_hp_after: int,
        target_destroyed: bool,
    ) -> None:
        """Record damage dealt from one combatant to another.

        Args:
            dealer_id: ID of combatant dealing damage
            target_id: ID of combatant receiving damage
            damage: Amount of damage dealt
            target_hp_before: Target's HP before damage
            target_hp_after: Target's HP after damage
            target_destroyed: Whether the target was destroyed
        """
        if dealer_id in self.combatant_stats:
            dealer = self.combatant_stats[dealer_id]
            new_damage_dealt = dealer.damage_dealt + damage

            # Calculate overkill if target was destroyed
            overkill = 0
            if target_destroyed:
                overkill = damage - target_hp_before
                new_overkill = dealer.overkill_dealt + overkill
                if overkill > self.max_overkill:
                    self.max_overkill = overkill
            else:
                new_overkill = dealer.overkill_dealt

            # Update dealer stats
            new_enemies_destroyed = dealer.enemies_destroyed
            new_destroyed_ids = list(dealer.destroyed_enemy_ids)
            if target_destroyed:
                new_enemies_destroyed += 1
                new_destroyed_ids.append(target_id)
                self.total_enemies_destroyed += 1

            self.combatant_stats[dealer_id] = CombatantStatistics(
                combatant_id=dealer.combatant_id,
                combatant_name=dealer.combatant_name,
                side=dealer.side,
                damage_dealt=new_damage_dealt,
                damage_received=dealer.damage_received,
                overkill_dealt=new_overkill,
                lowest_hp_reached=dealer.lowest_hp_reached,
                hp_at_start=dealer.hp_at_start,
                actions_taken=dealer.actions_taken,
                enemies_destroyed=new_enemies_destroyed,
                destroyed_enemy_ids=new_destroyed_ids,
                turns_taken=dealer.turns_taken,
            )

            # Update aggregate if dealer is player
            if dealer.side == "players":
                self.total_damage_dealt_by_players += damage

        # Update target's damage received
        if target_id in self.combatant_stats:
            target = self.combatant_stats[target_id]
            new_damage_received = target.damage_received + damage

            # Track closest call for players
            new_lowest_hp = min(target.lowest_hp_reached, target_hp_after)
            if target.side == "players":
                if new_lowest_hp < self.closest_call_hp:
                    self.closest_call_hp = new_lowest_hp
                    self.closest_call_combatant = target.combatant_name

            self.combatant_stats[target_id] = CombatantStatistics(
                combatant_id=target.combatant_id,
                combatant_name=target.combatant_name,
                side=target.side,
                damage_dealt=target.damage_dealt,
                damage_received=new_damage_received,
                overkill_dealt=target.overkill_dealt,
                lowest_hp_reached=new_lowest_hp,
                hp_at_start=target.hp_at_start,
                actions_taken=target.actions_taken,
                enemies_destroyed=target.enemies_destroyed,
                destroyed_enemy_ids=target.destroyed_enemy_ids,
                turns_taken=target.turns_taken,
            )

            # Update aggregate if target is player
            if target.side == "players":
                self.total_damage_received_by_players += damage

    def record_action(
        self,
        combatant_id: str,
        action_type: Literal[
            "attack", "move", "tech", "full", "quick", "reaction", "overcharge"
        ],
    ) -> None:
        """Record an action taken by a combatant."""
        # Update action totals
        if action_type == "attack":
            self.action_totals = self.action_totals.model_copy(
                update={"attacks": self.action_totals.attacks + 1}
            )
        elif action_type == "move":
            self.action_totals = self.action_totals.model_copy(
                update={"moves": self.action_totals.moves + 1}
            )
        elif action_type == "tech":
            self.action_totals = self.action_totals.model_copy(
                update={"techs": self.action_totals.techs + 1}
            )
        elif action_type == "full":
            self.action_totals = self.action_totals.model_copy(
                update={"full_actions": self.action_totals.full_actions + 1}
            )
        elif action_type == "quick":
            self.action_totals = self.action_totals.model_copy(
                update={"quick_actions": self.action_totals.quick_actions + 1}
            )
        elif action_type == "reaction":
            self.action_totals = self.action_totals.model_copy(
                update={"reactions": self.action_totals.reactions + 1}
            )
        elif action_type == "overcharge":
            self.action_totals = self.action_totals.model_copy(
                update={"overcharges": self.action_totals.overcharges + 1}
            )

        # Update combatant action counts
        if combatant_id in self.combatant_stats:
            combatant = self.combatant_stats[combatant_id]
            new_actions = combatant.actions_taken.model_copy()

            if action_type == "attack":
                new_actions = new_actions.model_copy(
                    update={"attacks": new_actions.attacks + 1}
                )
            elif action_type == "move":
                new_actions = new_actions.model_copy(
                    update={"moves": new_actions.moves + 1}
                )
            elif action_type == "tech":
                new_actions = new_actions.model_copy(
                    update={"techs": new_actions.techs + 1}
                )
            elif action_type == "full":
                new_actions = new_actions.model_copy(
                    update={"full_actions": new_actions.full_actions + 1}
                )
            elif action_type == "quick":
                new_actions = new_actions.model_copy(
                    update={"quick_actions": new_actions.quick_actions + 1}
                )
            elif action_type == "reaction":
                new_actions = new_actions.model_copy(
                    update={"reactions": new_actions.reactions + 1}
                )
            elif action_type == "overcharge":
                new_actions = new_actions.model_copy(
                    update={"overcharges": new_actions.overcharges + 1}
                )

            self.combatant_stats[combatant_id] = CombatantStatistics(
                combatant_id=combatant.combatant_id,
                combatant_name=combatant.combatant_name,
                side=combatant.side,
                damage_dealt=combatant.damage_dealt,
                damage_received=combatant.damage_received,
                overkill_dealt=combatant.overkill_dealt,
                lowest_hp_reached=combatant.lowest_hp_reached,
                hp_at_start=combatant.hp_at_start,
                actions_taken=new_actions,
                enemies_destroyed=combatant.enemies_destroyed,
                destroyed_enemy_ids=combatant.destroyed_enemy_ids,
                turns_taken=combatant.turns_taken,
            )

    def record_turn_taken(self, combatant_id: str) -> None:
        """Record that a combatant took their turn."""
        self.total_turns += 1

        if combatant_id in self.combatant_stats:
            combatant = self.combatant_stats[combatant_id]
            self.combatant_stats[combatant_id] = CombatantStatistics(
                combatant_id=combatant.combatant_id,
                combatant_name=combatant.combatant_name,
                side=combatant.side,
                damage_dealt=combatant.damage_dealt,
                damage_received=combatant.damage_received,
                overkill_dealt=combatant.overkill_dealt,
                lowest_hp_reached=combatant.lowest_hp_reached,
                hp_at_start=combatant.hp_at_start,
                actions_taken=combatant.actions_taken,
                enemies_destroyed=combatant.enemies_destroyed,
                destroyed_enemy_ids=combatant.destroyed_enemy_ids,
                turns_taken=combatant.turns_taken + 1,
            )

    def record_round_completed(self) -> None:
        """Record completion of a combat round."""
        self.rounds_completed += 1

    def to_combat_statistics(self) -> CombatStatistics:
        """Convert tracker to immutable CombatStatistics snapshot."""
        return CombatStatistics(
            rounds_completed=self.rounds_completed,
            total_turns=self.total_turns,
            total_damage_dealt_by_players=self.total_damage_dealt_by_players,
            total_damage_received_by_players=self.total_damage_received_by_players,
            total_enemies_destroyed=self.total_enemies_destroyed,
            closest_call_hp=self.closest_call_hp
            if self.closest_call_hp < 999999
            else 0,
            closest_call_combatant=self.closest_call_combatant,
            max_overkill=self.max_overkill,
            combatant_stats=self.combatant_stats,
            action_totals=self.action_totals,
        )
