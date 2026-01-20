/**
 * LicenseBadge component for displaying equipment license requirements.
 *
 * Usage:
 *   <LicenseBadge licenseId={null} />                    // Shows "GMS"
 *   <LicenseBadge licenseId="raleigh" licenseRank={1} /> // Shows "RALEIGH R1"
 */

import { formatLicenseId } from '~/lib/utils/license'

export interface LicenseBadgeProps {
  licenseId?: string | null
  licenseRank?: number | null
  size?: 'sm' | 'md'
}

const sizeStyles: Record<string, string> = {
  sm: 'px-1.5 py-0.5 text-xs',
  md: 'px-2 py-1 text-sm',
}

export function LicenseBadge({ licenseId, licenseRank, size = 'sm' }: LicenseBadgeProps) {
  const isGms = !licenseId
  const sizeClasses = sizeStyles[size]

  if (isGms) {
    return (
      <span className={`inline-flex items-center rounded ${sizeClasses} font-medium bg-gms/15 text-gms border border-gms/30`}>
        GMS
      </span>
    )
  }

  const formatted = formatLicenseId(licenseId)
  return (
    <span className={`inline-flex items-center rounded ${sizeClasses} font-medium bg-muted text-muted-foreground border border-border`}>
      {formatted} {licenseRank ? `R${licenseRank}` : ''}
    </span>
  )
}
