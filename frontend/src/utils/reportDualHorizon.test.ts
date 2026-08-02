import { describe, expect, it } from 'vitest'

import type { AnalysisReport } from '@/types'
import { getDualHorizonSummary } from '@/utils/reportDualHorizon'

function makeDualReport(overrides: Partial<AnalysisReport> = {}): AnalysisReport {
    return {
        symbol: 'AAPL',
        trade_date: '2026-08-02',
        mode: 'dual_horizon',
        status: 'completed',
        requested_horizons: ['short', 'medium'],
        horizon_status: { short: 'completed', medium: 'completed' },
        failed_horizons: [],
        short_term: {
            status: 'completed',
            not_applicable: false,
            data_gaps: ['盘前数据缺失'],
            falsification_conditions: ['跌破支撑离场'],
        },
        medium_term: {
            status: 'completed',
            not_applicable: false,
            data_gaps: [],
            falsification_conditions: [],
        },
        data_gaps: ['盘前数据缺失'],
        falsification_conditions: ['跌破支撑离场'],
        not_applicable: false,
        not_applicable_by_horizon: { short: false, medium: false },
        ...overrides,
    }
}

describe('getDualHorizonSummary', () => {
    it('returns aggregate and per-horizon state for a complete dual report', () => {
        const summary = getDualHorizonSummary(makeDualReport())

        expect(summary).not.toBeNull()
        expect(summary?.status).toBe('completed')
        expect(summary?.statusLabel).toBe('已完成')
        expect(summary?.requestedHorizons).toEqual(['short', 'medium'])
        expect(summary?.failedHorizons).toEqual([])
        expect(summary?.dataGaps).toEqual(['盘前数据缺失'])
        expect(summary?.falsificationConditions).toEqual(['跌破支撑离场'])
        expect(summary?.notApplicable).toBe(false)
        expect(summary?.horizons).toHaveLength(2)
        expect(summary?.horizons[0]).toMatchObject({
            horizon: 'short',
            status: 'completed',
            dataGaps: ['盘前数据缺失'],
            falsificationConditions: ['跌破支撑离场'],
            dataGapsProvided: true,
            falsificationConditionsProvided: true,
        })
    })

    it('derives aggregate lists from nested horizon fields when aggregate lists are absent', () => {
        const summary = getDualHorizonSummary(makeDualReport({
            data_gaps: undefined,
            falsification_conditions: undefined,
        }))

        expect(summary?.dataGapsProvided).toBe(true)
        expect(summary?.dataGaps).toEqual(['盘前数据缺失'])
        expect(summary?.falsificationConditionsProvided).toBe(true)
        expect(summary?.falsificationConditions).toEqual(['跌破支撑离场'])
    })

    it('handles a partially failed dual report without fabricating success', () => {
        const summary = getDualHorizonSummary(makeDualReport({
            status: 'partial',
            failed_horizons: ['medium'],
            horizon_status: { short: 'completed', medium: 'failed' },
            not_applicable: undefined,
            not_applicable_by_horizon: { short: false, medium: null },
            short_term: {
                status: 'completed',
                not_applicable: false,
                data_gaps: ['缺少盘后数据'],
                falsification_conditions: [],
            },
            medium_term: {
                status: 'failed',
                error: '外部数据源超时',
                not_applicable: null,
            },
        }))

        expect(summary?.status).toBe('partial')
        expect(summary?.statusLabel).toBe('部分完成')
        expect(summary?.failedHorizons).toEqual(['medium'])
        expect(summary?.horizons[1]).toMatchObject({
            status: 'failed',
            error: '外部数据源超时',
            notApplicable: null,
        })
        expect(summary?.notApplicable).toBeNull()
    })

    it('distinguishes absent fields from explicitly empty arrays', () => {
        const missing = getDualHorizonSummary(makeDualReport({
            data_gaps: undefined,
            falsification_conditions: undefined,
            not_applicable: undefined,
            not_applicable_by_horizon: { short: null, medium: null },
            short_term: { status: 'completed' },
            medium_term: { status: 'completed' },
        }))

        expect(missing?.dataGapsProvided).toBe(false)
        expect(missing?.falsificationConditionsProvided).toBe(false)
        expect(missing?.horizons.every(item => !item.dataGapsProvided)).toBe(true)
        expect(missing?.horizons.every(item => !item.falsificationConditionsProvided)).toBe(true)

        const empty = getDualHorizonSummary(makeDualReport({
            data_gaps: [],
            falsification_conditions: [],
            short_term: {
                status: 'completed',
                not_applicable: false,
                data_gaps: [],
                falsification_conditions: [],
            },
            medium_term: {
                status: 'completed',
                not_applicable: false,
                data_gaps: [],
                falsification_conditions: [],
            },
        }))

        expect(empty?.dataGapsProvided).toBe(true)
        expect(empty?.dataGaps).toEqual([])
        expect(empty?.falsificationConditionsProvided).toBe(true)
        expect(empty?.falsificationConditions).toEqual([])
    })

    it('prefers by-horizon maps when nested horizon fields are omitted', () => {
        const summary = getDualHorizonSummary(makeDualReport({
            data_gaps: undefined,
            falsification_conditions: undefined,
            not_applicable: undefined,
            falsification_conditions_by_horizon: {
                short: [],
                medium: ['跌破支撑离场'],
            },
            not_applicable_by_horizon: { short: true, medium: false },
            short_term: { status: 'completed' },
            medium_term: { status: 'completed' },
        }))

        expect(summary?.falsificationConditionsProvided).toBe(true)
        expect(summary?.falsificationConditions).toEqual(['跌破支撑离场'])
        expect(summary?.horizons[0]).toMatchObject({
            falsificationConditions: [],
            falsificationConditionsProvided: true,
            notApplicable: true,
        })
        expect(summary?.horizons[1]).toMatchObject({
            falsificationConditions: ['跌破支撑离场'],
            falsificationConditionsProvided: true,
            notApplicable: false,
        })
    })

    it('returns null for single-horizon reports even with horizon metadata', () => {
        const single = makeDualReport({
            mode: 'single_horizon',
            requested_horizons: ['short'],
            horizon_status: { short: 'completed' },
            failed_horizons: [],
            medium_term: undefined,
            falsification_conditions_by_horizon: undefined,
            not_applicable_by_horizon: undefined,
        })

        const legacySingle = makeDualReport({
            mode: undefined,
            requested_horizons: ['short'],
            horizon_status: { short: 'completed' },
            failed_horizons: [],
            medium_term: undefined,
            falsification_conditions_by_horizon: undefined,
            not_applicable_by_horizon: undefined,
        })

        expect(getDualHorizonSummary(single)).toBeNull()
        expect(getDualHorizonSummary(legacySingle)).toBeNull()
    })
})
