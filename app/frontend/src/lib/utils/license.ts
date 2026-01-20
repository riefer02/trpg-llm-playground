/**
 * License formatting utilities for displaying equipment license requirements
 */

/**
 * Check if an item is a GMS (General Massive Systems) item
 * GMS items have no license requirement and are always available
 */
export function isGmsItem(licenseId?: string | null): boolean {
  return !licenseId;
}

/**
 * Format a license ID for display (e.g., "raleigh" -> "RALEIGH")
 */
export function formatLicenseId(licenseId: string): string {
  return licenseId.replace(/_/g, ' ').toUpperCase();
}

/**
 * Format the full license requirement string
 * Returns "GMS" for GMS items, or "LICENSENAME R#" for licensed items
 */
export function formatLicenseRequirement(
  licenseId?: string | null,
  licenseRank?: number | null
): string {
  if (!licenseId) return 'GMS';
  const formatted = formatLicenseId(licenseId);
  return licenseRank ? `${formatted} R${licenseRank}` : formatted;
}
