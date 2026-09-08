import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import CalibrationPanel from '@/components/CalibrationPanel'
import type { CalibrationResponse } from '@/types'

function makeMockResponse(overrides: Partial<CalibrationResponse> = {}): CalibrationResponse {
    return {
        brier_score: null,
        sample_size: 1,
        probability_sample_size: 1,
        sample_sufficient: false,
        min_sample_size: 30,
        insufficient_reason: '样本量不足以支撑校准结论 (当前有效样本 1 份，低于最小阈值 30 份)',
        skipped_no_outcome: 0,
        truncated_before_filter: false,
        buckets: [
            {
                bucket: '0-50%',
                probability_min: 0.0,
                probability_max: 0.5,
                count: 0,
                rise_count: 0,
                rise_rate: null,
                avg_probability: null,
            },
            {
                bucket: '50-60%',
                probability_min: 0.5,
                probability_max: 0.6,
                count: 0,
                rise_count: 0,
                rise_rate: null,
                avg_probability: null,
            },
            {
                bucket: '60-70%',
                probability_min: 0.6,
                probability_max: 0.7,
                count: 1,
                rise_count: 1,
                rise_rate: null,
                avg_probability: null,
            },
            {
                bucket: '70-80%',
                probability_min: 0.7,
                probability_max: 0.8,
                count: 0,
                rise_count: 0,
                rise_rate: null,
                avg_probability: null,
            },
            {
                bucket: '80+%',
                probability_min: 0.8,
                probability_max: 1.0,
                count: 0,
                rise_count: 0,
                rise_rate: null,
                avg_probability: null,
            },
        ],
        filters: {
            start_date: null,
            end_date: null,
            symbol: null,
            prompt_version: null,
            model: null,
            hold_days: 5,
            limit: 500,
        },
        excluded_counts: {
            legacy_null: 2,
            invalid: 1,
            abstain: 0,
            no_trade: 3,
            incomplete_outcome: 0,
            incomplete: 0,
            total: 6,
        },
        ...overrides,
    }
}

describe('CalibrationPanel small sample guard (DAV-758)', () => {
    it('withholds reliability curve and displays warning alert when sample size is insufficient (n=1 < 30)', () => {
        const mockData = makeMockResponse({
            sample_size: 1,
            probability_sample_size: 1,
            sample_sufficient: false,
            min_sample_size: 30,
            brier_score: null,
        })

        const html = renderToStaticMarkup(<CalibrationPanel initialData={mockData} />)

        // 1. Must display small sample warning alert
        expect(html).toContain('data-testid="small-sample-warning"')
        expect(html).toContain('样本量不足以支撑校准结论')
        expect(html).toContain('最小阈值 30')
        expect(html).toContain('L3 自欺')

        // 2. Must NOT render reliability curve chart
        expect(html).not.toContain('data-testid="reliability-chart"')
        expect(html).toContain('data-testid="curve-withheld-placeholder"')
        expect(html).toContain('可靠性曲线已熔断隐藏')

        // 3. Brier Score value must be withheld
        expect(html).toContain('Brier Score')

        // 4. Must display both sample_size and excluded_counts in top metric tiles with equal prominence
        expect(html).toContain('已评估样本 (n)')
        expect(html).toContain('>1<')
        expect(html).toContain('排除样本 (Excluded)')
        expect(html).toContain('>6<')
        expect(html).toContain('校准有效性')
        expect(html).toContain('样本不足 (熔断)')

        // 5. Table must not display exact rise rate percentages
        expect(html).not.toContain('100.0%')
    })

    it('renders reliability curve and suppresses warning alert when sample size meets threshold (n=30)', () => {
        const mockData = makeMockResponse({
            sample_size: 30,
            probability_sample_size: 30,
            sample_sufficient: true,
            min_sample_size: 30,
            insufficient_reason: null,
            brier_score: 0.1852,
            buckets: [
                {
                    bucket: '0-50%',
                    probability_min: 0.0,
                    probability_max: 0.5,
                    count: 5,
                    rise_count: 2,
                    rise_rate: 40.0,
                    avg_probability: 0.42,
                },
                {
                    bucket: '50-60%',
                    probability_min: 0.5,
                    probability_max: 0.6,
                    count: 5,
                    rise_count: 3,
                    rise_rate: 60.0,
                    avg_probability: 0.55,
                },
                {
                    bucket: '60-70%',
                    probability_min: 0.6,
                    probability_max: 0.7,
                    count: 8,
                    rise_count: 5,
                    rise_rate: 62.5,
                    avg_probability: 0.65,
                },
                {
                    bucket: '70-80%',
                    probability_min: 0.7,
                    probability_max: 0.8,
                    count: 6,
                    rise_count: 4,
                    rise_rate: 66.7,
                    avg_probability: 0.75,
                },
                {
                    bucket: '80+%',
                    probability_min: 0.8,
                    probability_max: 1.0,
                    count: 6,
                    rise_count: 5,
                    rise_rate: 83.3,
                    avg_probability: 0.88,
                },
            ],
        })

        const html = renderToStaticMarkup(<CalibrationPanel initialData={mockData} />)

        // 1. Must NOT display small sample warning alert
        expect(html).not.toContain('data-testid="small-sample-warning"')
        expect(html).not.toContain('data-testid="curve-withheld-placeholder"')

        // 2. Must render reliability curve chart
        expect(html).toContain('data-testid="reliability-chart"')

        // 3. Must display calculated Brier Score
        expect(html).toContain('0.1852')

        // 4. Must display active calibration status
        expect(html).toContain('有效校准')

        // 5. Table must display calculated rates
        expect(html).toContain('62.5%')
    })

    it('renders clean zero-sample state when sample_size is 0', () => {
        const mockData = makeMockResponse({
            sample_size: 0,
            probability_sample_size: 0,
            sample_sufficient: false,
            min_sample_size: 30,
            insufficient_reason: '当前筛选条件下暂无带概率的历史报告，调整日期范围或过滤条件后重试。',
            brier_score: null,
            skipped_no_outcome: 0,
            buckets: [
                { bucket: '0-50%', probability_min: 0.0, probability_max: 0.5, count: 0, rise_count: 0, rise_rate: null, avg_probability: null },
                { bucket: '50-60%', probability_min: 0.5, probability_max: 0.6, count: 0, rise_count: 0, rise_rate: null, avg_probability: null },
                { bucket: '60-70%', probability_min: 0.6, probability_max: 0.7, count: 0, rise_count: 0, rise_rate: null, avg_probability: null },
                { bucket: '70-80%', probability_min: 0.7, probability_max: 0.8, count: 0, rise_count: 0, rise_rate: null, avg_probability: null },
                { bucket: '80+%', probability_min: 0.8, probability_max: 1.0, count: 0, rise_count: 0, rise_rate: null, avg_probability: null },
            ],
        })

        const html = renderToStaticMarkup(<CalibrationPanel initialData={mockData} />)

        // Does not render curve
        expect(html).not.toContain('data-testid="reliability-chart"')
        // Shows zero sample text
        expect(html).toContain('当前筛选条件下暂无带概率的历史报告')
        expect(html).toContain('无样本')
        expect(html).not.toContain('data-testid="small-sample-warning"')
    })
})
