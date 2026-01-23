type ViewportControlsProps = {
  zoom: number;
  minZoom: number;
  maxZoom: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
  onCenterOnActor: () => void;
  hasActorPosition: boolean;
};

export function ViewportControls({
  zoom,
  minZoom,
  maxZoom,
  onZoomIn,
  onZoomOut,
  onReset,
  onCenterOnActor,
  hasActorPosition,
}: ViewportControlsProps) {
  const canZoomIn = zoom < maxZoom;
  const canZoomOut = zoom > minZoom;
  const zoomPercent = Math.round(zoom * 100);

  return (
    <div className="absolute bottom-3 right-3 flex flex-col gap-1 bg-background/80 backdrop-blur-sm rounded-md border border-border shadow-lg p-1">
      {/* Zoom In */}
      <button
        onClick={onZoomIn}
        disabled={!canZoomIn}
        className="w-8 h-8 flex items-center justify-center rounded hover:bg-muted disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        title="Zoom in (+)"
        aria-label="Zoom in"
      >
        <svg
          className="w-4 h-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 6v12M6 12h12"
          />
        </svg>
      </button>

      {/* Zoom Level Display */}
      <div className="w-8 h-6 flex items-center justify-center text-[10px] text-muted-foreground font-mono">
        {zoomPercent}%
      </div>

      {/* Zoom Out */}
      <button
        onClick={onZoomOut}
        disabled={!canZoomOut}
        className="w-8 h-8 flex items-center justify-center rounded hover:bg-muted disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        title="Zoom out (-)"
        aria-label="Zoom out"
      >
        <svg
          className="w-4 h-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M6 12h12"
          />
        </svg>
      </button>

      {/* Divider */}
      <div className="h-px bg-border mx-1" />

      {/* Reset View */}
      <button
        onClick={onReset}
        className="w-8 h-8 flex items-center justify-center rounded hover:bg-muted transition-colors"
        title="Reset view (Home)"
        aria-label="Reset view"
      >
        <svg
          className="w-4 h-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"
          />
        </svg>
      </button>

      {/* Center on Actor */}
      <button
        onClick={onCenterOnActor}
        disabled={!hasActorPosition}
        className="w-8 h-8 flex items-center justify-center rounded hover:bg-muted disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        title="Center on current actor (C)"
        aria-label="Center on current actor"
      >
        <svg
          className="w-4 h-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <circle cx="12" cy="12" r="3" strokeWidth={2} />
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 2v4M12 18v4M2 12h4M18 12h4"
          />
        </svg>
      </button>
    </div>
  );
}
