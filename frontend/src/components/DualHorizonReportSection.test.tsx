import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import DualHorizonReportSection from '@/components/DualHorizonReportSection'
import type { ReportDetail } from '@/types'

function makeDetail(overrides: Partial<ReportDetail> = {}): ReportDetail {
    return {
        id: 'report-1',
        symbol: 'AAPL',
        trade_date: '2026-08-02',
        status: 'completed',
        result_data: {
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
        },
        ...overrides,
    }
}

describe('DualHorizonReportSection', () => {
    it('renders aggregate status, horizon cards, gaps, and falsification conditions', () => {
        const html = renderToStaticMarkup(
            <DualHorizonReportSection reportData={makeDetail()} />,
        )

        expect(html).toContain('双周期分析')
        expect(html).toContain('短线')
        expect(html).toContain('中线')
        expect(html).toContain('已完成')
        expect(html).toContain('盘前数据缺失')
        expect(html).toContain('跌破支撑离场')
        expect(html).toContain('聚合数据缺口')
        expect(html).toContain('聚合证伪条件')
    })

    it('shows unprovided instead of fake empty values when fields are absent', () => {
        const detail = makeDetail({
            result_data: {
                symbol: 'AAPL',
                trade_date: '2026-08-02',
                mode: 'dual_horizon',
                status: 'completed',
                requested_horizons: ['short', 'medium'],
                horizon_status: { short: 'completed', medium: 'completed' },
                failed_horizons: [],
                short_term: { status: 'completed' },
                medium_term: { status: 'completed' },
            },
        })

        const html = renderToStaticMarkup(<DualHorizonReportSection reportData={detail} />)

        expect(html).toContain('未提供')
        expect(html).not.toContain('盘前数据缺失')
        expect(html).not.toContain('跌破支撑离场')
    })

    it('renders nothing for a single-horizon report', () => {
        const detail = makeDetail({
            result_data: {
                symbol: 'AAPL',
                trade_date: '2026-08-02',
                mode: 'single_horizon',
                requested_horizons: ['short'],
                horizon_status: { short: 'completed' },
                short_term: { status: 'completed' },
            },
        })

        expect(renderToStaticMarkup(<DualHorizonReportSection reportData={detail} />)).toBe('')
    })
})
