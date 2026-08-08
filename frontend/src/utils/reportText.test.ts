import { describe, expect, it } from 'vitest'

import {
    detectLegacyEnglishContent,
    isLegacyEnglishDirection,
    isLegacyEnglishReport,
    localizeDirection,
    parseDecisionAction,
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

describe('parseDecisionAction', () => {
    it('maps English action words case-insensitively', () => {
        expect(parseDecisionAction('BUY')).toBe('buy')
        expect(parseDecisionAction('buy')).toBe('buy')
        expect(parseDecisionAction('SELL')).toBe('sell')
        expect(parseDecisionAction('HOLD')).toBe('hold')
        expect(parseDecisionAction('ADD')).toBe('add')
        expect(parseDecisionAction('REDUCE')).toBe('reduce')
        expect(parseDecisionAction('WATCH')).toBe('watch')
    })

    it('maps Chinese action words', () => {
        expect(parseDecisionAction('买入')).toBe('buy')
        expect(parseDecisionAction('卖出')).toBe('sell')
        expect(parseDecisionAction('持有')).toBe('hold')
        expect(parseDecisionAction('增持')).toBe('add')
        expect(parseDecisionAction('减持')).toBe('reduce')
        expect(parseDecisionAction('观望')).toBe('watch')
    })

    it('returns undefined for empty, nullish, and unmatched inputs', () => {
        expect(parseDecisionAction(undefined)).toBeUndefined()
        expect(parseDecisionAction(null)).toBeUndefined()
        expect(parseDecisionAction('')).toBeUndefined()
        expect(parseDecisionAction('看多')).toBeUndefined()
        expect(parseDecisionAction('未知操作')).toBeUndefined()
    })

    it('matches actions inside longer decision phrases', () => {
        expect(parseDecisionAction('短线买入')).toBe('buy')
        expect(parseDecisionAction('部分减持')).toBe('reduce')
        expect(parseDecisionAction('继续持有')).toBe('hold')
    })

    it('applies precedence when a phrase contains multiple actions', () => {
        // HOLD is checked before BUY, so "buy and hold" resolves to hold.
        expect(parseDecisionAction('BUY AND HOLD')).toBe('hold')
        // WATCH is checked before HOLD, so 观望 wins over 持有.
        expect(parseDecisionAction('观望持有')).toBe('watch')
        // SELL is checked before BUY.
        expect(parseDecisionAction('BUY, then SELL')).toBe('sell')
    })
})

describe('sanitizeReportMarkdown', () => {
    it('still strips verdict tags and translates legacy English decision phrases', () => {
        const text = '<!-- VERDICT: {"direction": "BULLISH"} -->\nFINAL TRANSACTION PROPOSAL: **BUY**'
        expect(sanitizeReportMarkdown(text)).toBe('\n最终交易建议：买入')
    })
})
