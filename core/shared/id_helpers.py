"""ID coercion helpers for typed ID migration.

Provides IdField[T] pattern for automatic string → typed ID coercion,
enabling backward compatibility with existing code while providing
type safety for new code.

Usage:
    from core.shared.id_helpers import PilotIdField, WeaponIdField
    from core.shared.ids import PilotId, WeaponId

    class Pilot(FrozenModel):
        id: PilotIdField  # Coerces "p1" → PilotId("p1")

    class MountedWeapon(FrozenModel):
        weapon_id: WeaponIdField  # Coerces "w1" → WeaponId("w1")

Type checkers will recognize the typed ID return types, catching mismatches
like passing WeaponId where SystemId is expected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, TypeVar

from pydantic import BeforeValidator


if TYPE_CHECKING:
    from core.shared.ids import (
        PilotId,
        MechId,
        CharacterId,
        NpcId,
        CombatantId,
        ActorId,
        EntityId,
        FrameId,
        WeaponId,
        SystemId,
        LicenseId,
        TalentId,
        CoreBonusId,
        ActionId,
        EffectId,
        StatusId,
        ProtocolId,
        TriggerId,
        ReactionId,
        DroneId,
        DeployableId,
        ObjectId,
        ZoneId,
        MissionId,
        ObjectiveId,
        SessionId,
        SceneId,
        ConsequenceId,
        TemplateId,
        NominatorId,
        NomineeId,
    )

T = TypeVar("T")


def _coerce_id(raw_id: Any, id_type: type[T]) -> T:
    """Coerce value to typed ID.

    Args:
        raw_id: The raw string ID value
        id_type: The target NewType (e.g., PilotId, WeaponId)

    Returns:
        The typed ID instance

    Raises:
        ValueError: If the value cannot be coerced
    """
    if isinstance(raw_id, str):
        return id_type(raw_id)  # type: ignore[call-arg]
    if isinstance(raw_id, id_type):
        return raw_id
    raise ValueError(f"Cannot coerce {raw_id!r} to {id_type.__name__}")


def IdField(id_type: type[T]) -> Any:
    """Create a field type that coerces strings to typed IDs.

    Args:
        id_type: A NewType such as PilotId, WeaponId, etc.

    Returns:
        Annotated type with BeforeValidator for coercion

    Example:
        class Pilot(FrozenModel):
            id: IdField[PilotId]
    """
    validator = BeforeValidator(lambda x: _coerce_id(x, id_type))
    return Annotated[str, validator]


def _id_field_factory(id_type: type[T]) -> Any:
    """Factory for creating ID field types at runtime."""
    validator = BeforeValidator(lambda x: _coerce_id(x, id_type))
    return Annotated[str, validator]


if TYPE_CHECKING:
    PilotIdField = Annotated[str, BeforeValidator]
    MechIdField = Annotated[str, BeforeValidator]
    CharacterIdField = Annotated[str, BeforeValidator]
    NpcIdField = Annotated[str, BeforeValidator]
    CombatantIdField = Annotated[str, BeforeValidator]
    ActorIdField = Annotated[str, BeforeValidator]
    EntityIdField = Annotated[str, BeforeValidator]
    FrameIdField = Annotated[str, BeforeValidator]
    WeaponIdField = Annotated[str, BeforeValidator]
    SystemIdField = Annotated[str, BeforeValidator]
    LicenseIdField = Annotated[str, BeforeValidator]
    TalentIdField = Annotated[str, BeforeValidator]
    CoreBonusIdField = Annotated[str, BeforeValidator]
    ActionIdField = Annotated[str, BeforeValidator]
    EffectIdField = Annotated[str, BeforeValidator]
    StatusIdField = Annotated[str, BeforeValidator]
    ProtocolIdField = Annotated[str, BeforeValidator]
    TriggerIdField = Annotated[str, BeforeValidator]
    ReactionIdField = Annotated[str, BeforeValidator]
    DroneIdField = Annotated[str, BeforeValidator]
    DeployableIdField = Annotated[str, BeforeValidator]
    ObjectIdField = Annotated[str, BeforeValidator]
    ZoneIdField = Annotated[str, BeforeValidator]
    MissionIdField = Annotated[str, BeforeValidator]
    ObjectiveIdField = Annotated[str, BeforeValidator]
    SessionIdField = Annotated[str, BeforeValidator]
    SceneIdField = Annotated[str, BeforeValidator]
    ConsequenceIdField = Annotated[str, BeforeValidator]
    TemplateIdField = Annotated[str, BeforeValidator]
    NominatorIdField = Annotated[str, BeforeValidator]
    NomineeIdField = Annotated[str, BeforeValidator]
else:
    from core.shared.ids import (
        PilotId,
        MechId,
        CharacterId,
        NpcId,
        CombatantId,
        ActorId,
        EntityId,
        FrameId,
        WeaponId,
        SystemId,
        LicenseId,
        TalentId,
        CoreBonusId,
        ActionId,
        EffectId,
        StatusId,
        ProtocolId,
        TriggerId,
        ReactionId,
        DroneId,
        DeployableId,
        ObjectId,
        ZoneId,
        MissionId,
        ObjectiveId,
        SessionId,
        SceneId,
        ConsequenceId,
        TemplateId,
        NominatorId,
        NomineeId,
    )

    PilotIdField = _id_field_factory(PilotId)
    MechIdField = _id_field_factory(MechId)
    CharacterIdField = _id_field_factory(CharacterId)
    NpcIdField = _id_field_factory(NpcId)
    CombatantIdField = _id_field_factory(CombatantId)
    ActorIdField = _id_field_factory(ActorId)
    EntityIdField = _id_field_factory(EntityId)
    FrameIdField = _id_field_factory(FrameId)
    WeaponIdField = _id_field_factory(WeaponId)
    SystemIdField = _id_field_factory(SystemId)
    LicenseIdField = _id_field_factory(LicenseId)
    TalentIdField = _id_field_factory(TalentId)
    CoreBonusIdField = _id_field_factory(CoreBonusId)
    ActionIdField = _id_field_factory(ActionId)
    EffectIdField = _id_field_factory(EffectId)
    StatusIdField = _id_field_factory(StatusId)
    ProtocolIdField = _id_field_factory(ProtocolId)
    TriggerIdField = _id_field_factory(TriggerId)
    ReactionIdField = _id_field_factory(ReactionId)
    DroneIdField = _id_field_factory(DroneId)
    DeployableIdField = _id_field_factory(DeployableId)
    ObjectIdField = _id_field_factory(ObjectId)
    ZoneIdField = _id_field_factory(ZoneId)
    MissionIdField = _id_field_factory(MissionId)
    ObjectiveIdField = _id_field_factory(ObjectiveId)
    SessionIdField = _id_field_factory(SessionId)
    SceneIdField = _id_field_factory(SceneId)
    ConsequenceIdField = _id_field_factory(ConsequenceId)
    TemplateIdField = _id_field_factory(TemplateId)
    NominatorIdField = _id_field_factory(NominatorId)
    NomineeIdField = _id_field_factory(NomineeId)


__all__ = [
    "IdField",
    "PilotIdField",
    "MechIdField",
    "CharacterIdField",
    "NpcIdField",
    "CombatantIdField",
    "ActorIdField",
    "EntityIdField",
    "FrameIdField",
    "WeaponIdField",
    "SystemIdField",
    "LicenseIdField",
    "TalentIdField",
    "CoreBonusIdField",
    "ActionIdField",
    "EffectIdField",
    "StatusIdField",
    "ProtocolIdField",
    "TriggerIdField",
    "ReactionIdField",
    "DroneIdField",
    "DeployableIdField",
    "ObjectIdField",
    "ZoneIdField",
    "MissionIdField",
    "ObjectiveIdField",
    "SessionIdField",
    "SceneIdField",
    "ConsequenceIdField",
    "TemplateIdField",
    "NominatorIdField",
    "NomineeIdField",
]
