/**
 * Toggle switch component for boolean settings.
 * 
 * Usage:
 *   <Toggle
 *     checked={enabled}
 *     onChange={(checked) => setEnabled(checked)}
 *     label="Enable voice input"
 *     description="Use speech recognition for commands"
 *   />
 */

import type { ChangeEvent } from 'react'

export interface ToggleProps {
  checked: boolean
  onChange: (checked: boolean) => void
  label: string
  description?: string
  className?: string
  disabled?: boolean
}

export function Toggle({
  checked,
  onChange,
  label,
  description,
  className = '',
  disabled = false,
}: ToggleProps) {
  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    onChange(e.target.checked)
  }

  const id = `toggle-${label.replace(/\s+/g, '-')}`

  return (
    <div className={`flex items-start space-x-4 ${className}`}>
      <div className="relative flex-shrink-0 focus-within:ring-2 focus-within:ring-primary focus-within:ring-offset-2 rounded">
        <input
          type="checkbox"
          checked={checked}
          onChange={handleChange}
          disabled={disabled}
          className="sr-only"
          aria-label={label}
          id={id}
        />
        <label
          htmlFor={id}
          className={`w-12 h-6 rounded-full transition-colors duration-200 ${
            checked ? 'bg-primary' : 'bg-secondary'
          } ${disabled ? 'opacity-50' : 'cursor-pointer'}`}
          onClick={() => !disabled && onChange(!checked)}
          aria-hidden="true"
          tabIndex={-1}
        >
          {/* Thumb */}
          <div
            className={`absolute top-1 left-1 w-4 h-4 rounded-full bg-white transition-transform duration-200 ${
              checked ? 'transform translate-x-6' : ''
            }`}
          />
        </label>
      </div>
      <div className="space-y-1">
        <label className="text-sm font-medium text-foreground" htmlFor={id}>
          {label}
        </label>
        {description && (
          <p className="text-sm text-muted-foreground">
            {description}
          </p>
        )}
      </div>
    </div>
  )
}