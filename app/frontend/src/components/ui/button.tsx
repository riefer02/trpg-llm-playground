/**
 * Button component following shadcn/ui patterns.
 * 
 * Usage:
 *   <Button variant="primary">Click me</Button>
 *   <Button variant="secondary" size="sm">Small</Button>
 */

import type { ButtonHTMLAttributes, ReactNode } from 'react'

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'destructive' | 'outline' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
  children: ReactNode
}

const variantStyles: Record<string, string> = {
  primary: 'bg-primary text-primary-foreground hover:opacity-90',
  secondary: 'bg-secondary text-secondary-foreground hover:opacity-80',
  destructive: 'bg-destructive text-destructive-foreground hover:opacity-90',
  outline: 'border border-border bg-transparent hover:bg-secondary',
  ghost: 'bg-transparent hover:bg-secondary',
}

const sizeStyles: Record<string, string> = {
  sm: 'h-8 px-3 text-sm',
  md: 'h-10 px-4 text-base',
  lg: 'h-12 px-6 text-lg',
}

export function Button({
  variant = 'primary',
  size = 'md',
  className = '',
  children,
  ...props
}: ButtonProps) {
  const baseStyles = 'inline-flex items-center justify-center rounded-md font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:ring-offset-background disabled:opacity-50 disabled:pointer-events-none'
  
  return (
    <button
      className={`${baseStyles} ${variantStyles[variant]} ${sizeStyles[size]} ${className}`}
      {...props}
    >
      {children}
    </button>
  )
}
