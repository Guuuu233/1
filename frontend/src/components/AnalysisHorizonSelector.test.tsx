import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import AnalysisHorizonSelector, {
    getActiveHorizonKey,
    HORIZON_OPTIONS,
    resolveHorizonOption,
} from '@/components/AnalysisHorizonSelector'
import type { AnalysisHorizon } from '@/types'

describe('getActiveHorizonKey utility', () => {
    it('returns short for undefined or empty input without silent corruption', () => {
        expect(getActiveHorizonKey(undefined)).toBe('short')
        expect(getActiveHorizonKey([])).toBe('empty')
    })

    it('identifies single short and single medium', () => {
        expect(getActiveHorizonKey(['short'])).toBe('short')
        expect(getActiveHorizonKey(['medium'])).toBe('medium')
    })

    it('identifies dual horizons regardless of input order', () => {
        expect(getActiveHorizonKey(['short', 'medium'])).toBe('dual')
        expect(getActiveHorizonKey(['medium', 'short'])).toBe('dual')
    })

    it('returns unknown for unrecognized horizon combinations', () => {
        expect(getActiveHorizonKey(['bogus'])).toBe('unknown')
        expect(getActiveHorizonKey(['short', 'bogus'])).toBe('unknown')
        expect(getActiveHorizonKey(['short', 'medium', 'long'])).toBe('unknown')
    })
})

describe('resolveHorizonOption utility (H-03b)', () => {
    it('resolves short to ["short"]', () => {
        expect(resolveHorizonOption('short')).toEqual(['short'])
    })

    it('resolves medium to ["medium"]', () => {
        expect(resolveHorizonOption('medium')).toEqual(['medium'])
    })

    it('resolves dual to canonical preserved order ["short", "medium"]', () => {
        // 契约 1: 双档为显式 ['short', 'medium']（保序）
        expect(resolveHorizonOption('dual')).toEqual(['short', 'medium'])
    })
})

describe('AnalysisHorizonSelector component rendering (H-03b)', () => {
    it('renders default selection as short (短期) visible before submit', () => {
        const html = renderToStaticMarkup(<AnalysisHorizonSelector />)

        // 契约 1: 默认 short，提交前可见
        expect(html).toContain('分析档位:')
        expect(html).toContain('短线')
        expect(html).toContain('短期')
        expect(html).toContain('中线')
        expect(html).toContain('双档')

        // 验证短线选项处于激活状态 (aria-pressed="true")
        expect(html).toMatch(/data-testid="horizon-short"[^>]*aria-pressed="true"/)
        expect(html).toMatch(/data-testid="horizon-medium"[^>]*aria-pressed="false"/)
        expect(html).toMatch(/data-testid="horizon-dual"[^>]*aria-pressed="false"/)
    })

    it('renders controlled medium selection correctly', () => {
        const html = renderToStaticMarkup(
            <AnalysisHorizonSelector value={['medium']} />,
        )

        expect(html).toMatch(/data-testid="horizon-short"[^>]*aria-pressed="false"/)
        expect(html).toMatch(/data-testid="horizon-medium"[^>]*aria-pressed="true"/)
        expect(html).toMatch(/data-testid="horizon-dual"[^>]*aria-pressed="false"/)
    })

    it('renders controlled dual selection correctly', () => {
        const html = renderToStaticMarkup(
            <AnalysisHorizonSelector value={['short', 'medium']} />,
        )

        expect(html).toMatch(/data-testid="horizon-short"[^>]*aria-pressed="false"/)
        expect(html).toMatch(/data-testid="horizon-medium"[^>]*aria-pressed="false"/)
        expect(html).toMatch(/data-testid="horizon-dual"[^>]*aria-pressed="true"/)
    })

    it('does NOT silently rewrite illegal empty array to short', () => {
        const html = renderToStaticMarkup(
            <AnalysisHorizonSelector value={[]} />,
        )

        // 契约 4: 非法组合不要静默改档
        expect(html).toMatch(/data-testid="horizon-short"[^>]*aria-pressed="false"/)
        expect(html).toMatch(/data-testid="horizon-medium"[^>]*aria-pressed="false"/)
        expect(html).toMatch(/data-testid="horizon-dual"[^>]*aria-pressed="false"/)
        expect(html).toContain('data-testid="horizon-empty-notice"')
    })

    it('does NOT silently rewrite illegal unknown values to short', () => {
        const html = renderToStaticMarkup(
            <AnalysisHorizonSelector value={['invalid_value'] as unknown as AnalysisHorizon[]} />,
        )

        expect(html).toMatch(/data-testid="horizon-short"[^>]*aria-pressed="false"/)
        expect(html).toMatch(/data-testid="horizon-medium"[^>]*aria-pressed="false"/)
        expect(html).toMatch(/data-testid="horizon-dual"[^>]*aria-pressed="false"/)
        expect(html).toContain('data-testid="horizon-unknown-notice"')
    })

    it('respects disabled prop and disables all buttons', () => {
        const html = renderToStaticMarkup(
            <AnalysisHorizonSelector disabled={true} />,
        )

        // 所有三个选项按钮均包含 disabled 属性
        expect(html).toContain('disabled=""')
        expect(html).toMatch(/<button[^>]*disabled=""[^>]*data-testid="horizon-short"/)
        expect(html).toMatch(/<button[^>]*disabled=""[^>]*data-testid="horizon-medium"/)
        expect(html).toMatch(/<button[^>]*disabled=""[^>]*data-testid="horizon-dual"/)
    })
})

describe('AnalysisHorizonSelector user interactions (H-03b)', () => {
    it('triggers onChange with ["medium"] when medium option is selected', () => {
        const onChange = vi.fn()
        let onSelectHandler: ((h: AnalysisHorizon[]) => void) | undefined

        function Harness() {
            onSelectHandler = (h) => {
                onChange(h)
            }
            return <AnalysisHorizonSelector value={['short']} onChange={onSelectHandler} />
        }

        renderToStaticMarkup(<Harness />)
        expect(onSelectHandler).toBeDefined()
        onSelectHandler!(resolveHorizonOption('medium'))

        expect(onChange).toHaveBeenCalledTimes(1)
        expect(onChange).toHaveBeenCalledWith(['medium'])
    })

    it('triggers onChange with explicit preserved-order ["short", "medium"] when dual option is selected', () => {
        const onChange = vi.fn()
        let onSelectHandler: ((h: AnalysisHorizon[]) => void) | undefined

        function Harness() {
            onSelectHandler = (h) => {
                onChange(h)
            }
            return <AnalysisHorizonSelector value={['short']} onChange={onSelectHandler} />
        }

        renderToStaticMarkup(<Harness />)
        expect(onSelectHandler).toBeDefined()
        onSelectHandler!(resolveHorizonOption('dual'))

        // 契约 1: 双档为显式 ['short', 'medium']（保序）
        expect(onChange).toHaveBeenCalledTimes(1)
        expect(onChange).toHaveBeenCalledWith(['short', 'medium'])
    })

    it('triggers onChange with ["short"] when short option is clicked', () => {
        const onChange = vi.fn()
        let onSelectHandler: ((h: AnalysisHorizon[]) => void) | undefined

        function Harness() {
            onSelectHandler = (h) => {
                onChange(h)
            }
            return <AnalysisHorizonSelector value={['medium']} onChange={onSelectHandler} />
        }

        renderToStaticMarkup(<Harness />)
        expect(onSelectHandler).toBeDefined()
        onSelectHandler!(resolveHorizonOption('short'))

        expect(onChange).toHaveBeenCalledTimes(1)
        expect(onChange).toHaveBeenCalledWith(['short'])
    })

    it('does NOT trigger onChange when disabled in component', () => {
        const onChange = vi.fn()

        function Harness() {
            return (
                <AnalysisHorizonSelector
                    value={['short']}
                    onChange={onChange}
                    disabled={true}
                />
            )
        }

        const html = renderToStaticMarkup(<Harness />)
        expect(html).toContain('disabled=""')
        expect(onChange).not.toHaveBeenCalled()
    })
})
