import { DEFAULT_TERRAIN_STYLE } from "../../lib/combat-render/canvas";

type TerrainLegendProps = {
  className?: string;
};

type LegendSwatchProps = {
  label: string;
  fill?: string;
  border?: string;
  borderWidth?: number;
  text?: string;
  rounded?: string;
};

function LegendSwatch({
  label,
  fill,
  border,
  borderWidth = 1,
  text,
  rounded = "rounded-sm",
}: LegendSwatchProps) {
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      <span
        className={`inline-flex h-4 w-4 items-center justify-center ${rounded}`}
        style={{
          backgroundColor: fill,
          borderColor: border ?? "transparent",
          borderWidth,
          borderStyle: "solid",
          color: DEFAULT_TERRAIN_STYLE.elevationTextColor,
          fontSize: "10px",
          fontWeight: 700,
        }}
      >
        {text}
      </span>
      <span>{label}</span>
    </div>
  );
}

export function TerrainLegend({ className }: TerrainLegendProps) {
  return (
    <div className={`flex flex-wrap gap-3 ${className ?? ""}`}>
      <LegendSwatch
        label="Difficult"
        fill={DEFAULT_TERRAIN_STYLE.difficultFill}
        border="rgba(15, 23, 42, 0.25)"
      />
      <LegendSwatch
        label="Dangerous"
        fill={DEFAULT_TERRAIN_STYLE.dangerousFill}
        border="rgba(15, 23, 42, 0.25)"
      />
      <LegendSwatch
        label="Soft cover"
        border={DEFAULT_TERRAIN_STYLE.softCoverStroke}
        borderWidth={2}
      />
      <LegendSwatch
        label="Hard cover"
        border={DEFAULT_TERRAIN_STYLE.hardCoverStroke}
        borderWidth={3}
      />
      <LegendSwatch
        label="Blocks LOS"
        fill={DEFAULT_TERRAIN_STYLE.blockingFill}
        border={DEFAULT_TERRAIN_STYLE.blockingStroke}
        borderWidth={2}
      />
      <LegendSwatch
        label="Elevation"
        fill={DEFAULT_TERRAIN_STYLE.elevationBadgeFill}
        border={DEFAULT_TERRAIN_STYLE.elevationBadgeStroke}
        borderWidth={1}
        text="2"
        rounded="rounded-full"
      />
    </div>
  );
}
