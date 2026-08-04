export function formatNumber(value?: number | null, digits = 2): string {
    if (value == null || !Number.isFinite(value)) return '--'
    return new Intl.NumberFormat('zh-CN', {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
    }).format(value)
}

/** Format with Chinese 万/亿 units; 万-branch drops decimals only when baseDigits is 0. */
export function formatWithChineseUnit(value: number, baseDigits = 2): string {
    const abs = Math.abs(value)
    if (abs >= 1e8) {
        return `${formatNumber(value / 1e8, 2)}亿`
    }
    if (abs >= 1e4) {
        return `${formatNumber(value / 1e4, baseDigits === 0 ? 0 : 2)}万`
    }
    return formatNumber(value, baseDigits)
}

/**
 * Volume-specific variant (KlinePanel): null-safe and always 0 decimals below
 * 1e4, always 2 decimals in the 万 branch — intentionally distinct from
 * formatWithChineseUnit's digit policy.
 */
export function formatVolume(value?: number | null): string {
    if (value == null || !Number.isFinite(value)) return '--'
    if (Math.abs(value) >= 1e8) return `${formatNumber(value / 1e8, 2)}亿`
    if (Math.abs(value) >= 1e4) return `${formatNumber(value / 1e4, 2)}万`
    return formatNumber(value, 0)
}
