import { useState } from 'react'
import type { AnalysisHorizon } from '@/types'

export type HorizonOptionKey = 'short' | 'medium' | 'dual'

export interface HorizonOption {
    key: HorizonOptionKey
    label: string
    sublabel: string
    title: string
    horizons: AnalysisHorizon[]
}

export const HORIZON_OPTIONS: HorizonOption[] = [
    {
        key: 'short',
        label: '短线',
        sublabel: '短期',
        title: '短线分析（短期，默认）',
        horizons: ['short'],
    },
    {
        key: 'medium',
        label: '中线',
        sublabel: '中期',
        title: '中线分析（中期）',
        horizons: ['medium'],
    },
    {
        key: 'dual',
        label: '双档',
        sublabel: '短+中',
        title: '双档分析（短线 + 中线并发分析）',
        horizons: ['short', 'medium'],
    },
]

export const DEFAULT_HORIZONS: AnalysisHorizon[] = ['short']

export function resolveHorizonOption(key: HorizonOptionKey): AnalysisHorizon[] {
    const opt = HORIZON_OPTIONS.find((o) => o.key === key)
    return opt ? [...opt.horizons] : ['short']
}

/**
 * Determine the active option key from a horizons array.
 * Preserves illegal/unknown states truthfully without silent correction.
 */
export function getActiveHorizonKey(
    horizons?: AnalysisHorizon[] | string[],
): HorizonOptionKey | 'unknown' | 'empty' {
    if (!horizons) return 'short'
    if (horizons.length === 0) return 'empty'
    if (horizons.length === 1) {
        if (horizons[0] === 'short') return 'short'
        if (horizons[0] === 'medium') return 'medium'
        return 'unknown'
    }
    if (horizons.length === 2) {
        const hasShort = horizons.includes('short')
        const hasMedium = horizons.includes('medium')
        if (hasShort && hasMedium) return 'dual'
        return 'unknown'
    }
    return 'unknown'
}

export interface AnalysisHorizonSelectorProps {
    value?: AnalysisHorizon[] | string[]
    onChange?: (horizons: AnalysisHorizon[]) => void
    disabled?: boolean
    className?: string
}

export default function AnalysisHorizonSelector({
    value,
    onChange,
    disabled = false,
    className = '',
}: AnalysisHorizonSelectorProps) {
    const [internalValue, setInternalValue] = useState<AnalysisHorizon[]>(DEFAULT_HORIZONS)
    const isControlled = value !== undefined
    const currentHorizons = isControlled ? value : internalValue
    const activeKey = getActiveHorizonKey(currentHorizons)

    const handleSelect = (optionHorizons: AnalysisHorizon[]) => {
        if (disabled) return
        if (!isControlled) {
            setInternalValue(optionHorizons)
        }
        onChange?.(optionHorizons)
    }

    return (
        <div
            className={`flex items-center gap-1.5 ${className}`}
            data-testid="analysis-horizon-selector"
            role="group"
            aria-label="分析档位选择"
        >
            <span className="text-[11px] font-medium text-slate-500 dark:text-slate-400 select-none">
                分析档位:
            </span>
            <div className="inline-flex rounded-lg bg-slate-100 p-0.5 dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
                {HORIZON_OPTIONS.map((opt) => {
                    const isActive = activeKey === opt.key
                    return (
                        <button
                            key={opt.key}
                            type="button"
                            disabled={disabled}
                            onClick={() => handleSelect(opt.horizons)}
                            data-testid={`horizon-${opt.key}`}
                            aria-pressed={isActive}
                            title={opt.title}
                            className={`px-2 py-0.5 text-xs font-medium rounded-md transition-all ${
                                isActive
                                    ? 'bg-white text-blue-600 shadow-sm dark:bg-slate-700 dark:text-blue-400 font-semibold'
                                    : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200'
                            } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                        >
                            <span>{opt.label}</span>
                            <span className="text-[10px] ml-0.5 opacity-75">({opt.sublabel})</span>
                        </button>
                    )
                })}
            </div>
            {activeKey === 'empty' && (
                <span className="text-[10px] text-rose-500 font-medium" data-testid="horizon-empty-notice">
                    未选择档位
                </span>
            )}
            {activeKey === 'unknown' && (
                <span className="text-[10px] text-amber-500 font-medium" data-testid="horizon-unknown-notice">
                    非标准档位
                </span>
            )}
        </div>
    )
}
