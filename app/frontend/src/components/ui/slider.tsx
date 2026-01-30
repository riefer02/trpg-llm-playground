/**
 * Slider component for volume and other numeric settings.
 * 
 * Usage:
 *   <Slider
 *     value={volume}
 *     onChange={(value) => setVolume(value)}
 *     min={0}
 *     max={100}
 *     step={1}
 *     label="Master Volume"
 *     unit="%"
 *   />
 */

import type { ChangeEvent } from 'react'

export interface SliderProps {
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  label: string
  unit?: string
  className?: string
  disabled?: boolean
}

export function Slider({
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  label,
  unit = '',
  className = '',
  disabled = false,
}: SliderProps) {
  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    onChange(Number(e.target.value))
  }

  const percentage = ((value - min) / (max - min)) * 100

  return (
    <div className={`space-y-3 ${className}`}>
      <div className="flex justify-between items-center">
        <label className="text-sm font-medium text-foreground">
          {label}
        </label>
        <span className="text-sm font-mono text-muted-foreground">
          {value}
          {unit}
        </span>
      </div>
      <div className="relative">
        {/* Track */}
        <div className="h-2 bg-secondary rounded-full focus-within:ring-2 focus-within:ring-primary focus-within:ring-offset-2">
          {/* Filled portion */}
          <div
            className="h-full bg-primary rounded-full"
            style={{ width: `${percentage}%` }}
          />
        </div>
        {/* Thumb */}
        <input
          type="range"
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={handleChange}
          disabled={disabled}
          className="absolute top-1/2 left-0 w-full h-0 appearance-none cursor-pointer transform -translate-y-1/2 focus:outline-none"
          aria-label={label}
        />
      </div>
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{min}{unit}</span>
        <span>{max}{unit}</span>
      </div>
    </div>
  )
}