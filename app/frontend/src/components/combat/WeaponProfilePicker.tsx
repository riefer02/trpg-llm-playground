import type { MechWeaponDefinition, WeaponProfile } from "../../lib/types/lancer";
import { Button } from "../ui";

export interface WeaponProfilePickerProps {
  weapon: MechWeaponDefinition | null;
  onSelect: (profileId: string) => void;
  onCancel: () => void;
  isOpen: boolean;
}

export function WeaponProfilePicker({
  weapon,
  onSelect,
  onCancel,
  isOpen,
}: WeaponProfilePickerProps) {
  if (!isOpen) {
    return null;
  }

  const profiles = weapon?.dynamic?.profile_choice?.profiles ?? [];

  if (profiles.length === 0) {
    return (
      <div className="rounded-md border border-border bg-muted/30 p-3 space-y-2">
        <div className="text-sm font-medium text-foreground">Select Profile</div>
        <div className="text-xs text-muted-foreground">
          No profiles available
        </div>
        <Button variant="ghost" size="sm" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    );
  }

  return (
    <div className="rounded-md border border-border bg-muted/30 p-3 space-y-2">
      <div className="text-sm font-medium text-foreground">
        Select Profile for {weapon?.name ?? "Weapon"}
      </div>
      <div className="space-y-1">
        {profiles.map((profile) => (
          <ProfileItem
            key={profile.profile_id}
            profile={profile}
            onSelect={() => onSelect(profile.profile_id)}
          />
        ))}
      </div>
      <div className="pt-2">
        <Button variant="ghost" size="sm" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </div>
  );
}

interface ProfileItemProps {
  profile: WeaponProfile;
  onSelect: () => void;
}

function ProfileItem({ profile, onSelect }: ProfileItemProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className="w-full text-left px-2 py-1.5 rounded text-sm transition-colors hover:bg-primary/10 text-foreground cursor-pointer"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-primary" />
          <span>{profile.name}</span>
        </div>
        {profile.damage_type && (
          <span className="text-xs text-muted-foreground capitalize">
            {profile.damage_type}
          </span>
        )}
      </div>
    </button>
  );
}
