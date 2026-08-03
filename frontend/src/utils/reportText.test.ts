import { describe, expect, it } from 'vitest'

import {
    detectLegacyEnglishContent,
    isLegacyEnglishDirection,
    isLegacyEnglishReport,
    localizeDirection,
    sanitizeReportMarkdown,
} from '@/utils/reportText'

describe('isLegacyEnglishDirection', () => {
    it('recognizes English directions from the old en.py prompt set', () => {
        expect(isLegacyEnglishDirection('BULLISH')).toBe(true)
        expect(isLegacyEnglishDirection('lean_bearish')).toBe(true)
        expect(isLegacyEnglishDirection('NEUTRAL')).toBe(true)
        expect(isLegacyEnglishDirection('cautious')).toBe(true)
    })

    it('rejects Chinese directions and empty values', () => {
        expect(isLegacyEnglishDirection('看多')).toBe(false)
        expect(isLegacyEnglishDirection('偏空')).toBe(false)
        expect(isLegacyEnglishDirection(null)).toBe(false)
        expect(isLegacyEnglishDirection(undefined)).toBe(false)
        expect(isLegacyEnglishDirection('')).toBe(false)
    })
})

describe('localizeDirection', () => {
    it('maps legacy English directions to Chinese display labels', () => {
        expect(localizeDirection('BULLISH')).toBe('看多')
        expect(localizeDirection('lean_bearish')).toBe('偏空')
        expect(localizeDirection('NEUTRAL')).toBe('中性')
        expect(localizeDirection('CAUTIOUS')).toBe('谨慎')
    })

    it('passes through Chinese directions and empty values unchanged', () => {
        expect(localizeDirection('看多')).toBe('看多')
        expect(localizeDirection('')).toBe(null)
        expect(localizeDirection(null)).toBe(null)
        expect(localizeDirection(undefined)).toBe(null)
    })
})

describe('detectLegacyEnglishContent', () => {
    it('flags reports written in English prose', () => {
        const englishProse = [
            '## Market Overview',
            'The stock has been consolidating near its 20-day moving average.',
            'Volume expansion confirms the breakout attempt.',
            'Key support sits at 45.00 with resistance at 52.00.',
            'We recommend watching for a close above resistance before entering.',
        ].join('\n')
        expect(detectLegacyEnglishContent(englishProse)).toBe(true)
    })

    it('does not flag Chinese reports with scattered English jargon', () => {
        const chineseProse = [
            '股价在 20 日均线附近震荡整理。',
            '技术面上 MACD 金叉，KDJ 指标走强，短线看多。',
            '资金面主力净流入，建议 BUY，目标价 52 元，止损 45 元。',
            '基本面 PE 估值合理，ROE 稳健，HOLD 观望亦可。',
        ].join('\n')
        expect(detectLegacyEnglishContent(chineseProse)).toBe(false)
    })

    it('handles empty or non-English content', () => {
        expect(detectLegacyEnglishContent('')).toBe(false)
        expect(detectLegacyEnglishContent('纯中文内容，没有任何英文')).toBe(false)
        expect(detectLegacyEnglishContent('BUY SELL HOLD MACD ROE PE')).toBe(false)
    })
})

describe('isLegacyEnglishReport', () => {
    it('flags a list row whose direction is an English legacy value', () => {
        expect(isLegacyEnglishReport({ symbol: '600519', direction: 'BULLISH' })).toBe(true)
        expect(isLegacyEnglishReport({ symbol: '600519', direction: '看多' })).toBe(false)
    })

    it('flags a detail whose sections are English even without an English direction', () => {
        const detail = {
            symbol: '600519',
            direction: '看多',
            market_report: '## Market Analysis\nThe stock rallied on strong volume.',
            final_trade_decision: '## Final Trade Decision\nBuy at open with a stop below support.',
        }
        expect(isLegacyEnglishReport(detail)).toBe(true)
    })

    it('keeps current Chinese reports unflagged', () => {
        const detail = {
            symbol: '600519',
            direction: '偏多',
            market_report: '市场报告：股价沿 20 日均线上行，量价配合。',
            final_trade_decision: '最终交易建议：买入，目标价 52 元。',
        }
        expect(isLegacyEnglishReport(detail)).toBe(false)
    })

    it('treats reports with no content and no direction as non-legacy', () => {
        expect(isLegacyEnglishReport({ symbol: '600519' })).toBe(false)
        expect(isLegacyEnglishReport({})).toBe(false)
    })
})

describe('sanitizeReportMarkdown', () => {
    it('still strips verdict tags and translates legacy English decision phrases', () => {
        const text = '<!-- VERDICT: {"direction": "BULLISH"} -->\nFINAL TRANSACTION PROPOSAL: **BUY**'
        expect(sanitizeReportMarkdown(text)).toBe('\n最终交易建议：买入')
    })
})
