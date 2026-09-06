import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { api, ApiError, isNotFoundError } from '@/services/api'
import type { AnalysisRequest } from '@/types'

describe('ApiError', () => {
    it('carries the HTTP status alongside the message', () => {
        const error = new ApiError('job not found', 404)
        expect(error).toBeInstanceOf(Error)
        expect(error.message).toBe('job not found')
        expect(error.status).toBe(404)
        expect(error.name).toBe('ApiError')
    })
})

describe('isNotFoundError', () => {
    it('matches ApiError with status 404', () => {
        expect(isNotFoundError(new ApiError('job not found', 404))).toBe(true)
        expect(isNotFoundError(new ApiError('报告不存在', 404))).toBe(true)
    })

    it('rejects other statuses and plain errors', () => {
        expect(isNotFoundError(new ApiError('Internal error', 500))).toBe(false)
        expect(isNotFoundError(new ApiError('Unauthorized', 401))).toBe(false)
        expect(isNotFoundError(new Error('job not found'))).toBe(false)
        expect(isNotFoundError(null)).toBe(false)
        expect(isNotFoundError('job not found')).toBe(false)
    })
})

describe('api.chatCompletion horizons contract (H-03b)', () => {
    let originalFetch: typeof globalThis.fetch

    beforeEach(() => {
        originalFetch = globalThis.fetch
    })

    afterEach(() => {
        globalThis.fetch = originalFetch
        vi.restoreAllMocks()
    })

    it('sends explicit default ["short"] in request body when horizons is omitted', async () => {
        let capturedUrl = ''
        let capturedBody: Record<string, unknown> = {}

        globalThis.fetch = vi.fn().mockImplementation(async (url: string, init?: RequestInit) => {
            capturedUrl = url
            capturedBody = JSON.parse((init?.body as string) || '{}')
            return new Response(JSON.stringify({ ok: true }), {
                status: 200,
                headers: { 'Content-Type': 'application/json' },
            })
        })

        const messages = [{ role: 'user', content: '分析 600519.SH' }]
        await api.chatCompletion(messages, true, ['market'])

        expect(capturedUrl).toContain('/v1/chat/completions')
        expect(capturedBody.messages).toEqual(messages)
        expect(capturedBody.stream).toBe(true)
        expect(capturedBody.selected_analysts).toEqual(['market'])
        // 契约 2: 未改选择器/未传时发送 ["short"]（显式默认）
        expect(capturedBody.horizons).toEqual(['short'])
        // 契约 3: investment_horizon 不得写入 horizons
        expect(capturedBody.horizons).not.toContain('短线')
        expect(capturedBody.investment_horizon).toBeUndefined()
    })

    it('sends explicit ["medium"] when medium horizon is selected', async () => {
        let capturedBody: Record<string, unknown> = {}

        globalThis.fetch = vi.fn().mockImplementation(async (_url: string, init?: RequestInit) => {
            capturedBody = JSON.parse((init?.body as string) || '{}')
            return new Response(JSON.stringify({ ok: true }), {
                status: 200,
                headers: { 'Content-Type': 'application/json' },
            })
        })

        await api.chatCompletion(
            [{ role: 'user', content: '分析 600519.SH' }],
            true,
            ['macro'],
            ['medium'],
        )

        expect(capturedBody.horizons).toEqual(['medium'])
    })

    it('sends explicit dual horizons ["short", "medium"] in preserved canonical order', async () => {
        let capturedBody: Record<string, unknown> = {}

        globalThis.fetch = vi.fn().mockImplementation(async (_url: string, init?: RequestInit) => {
            capturedBody = JSON.parse((init?.body as string) || '{}')
            return new Response(JSON.stringify({ ok: true }), {
                status: 200,
                headers: { 'Content-Type': 'application/json' },
            })
        })

        await api.chatCompletion(
            [{ role: 'user', content: '分析 600519.SH 短线和中线' }],
            true,
            ['market', 'macro'],
            ['short', 'medium'],
        )

        // 契约 1: 双档为显式 ['short', 'medium']（保序）
        expect(capturedBody.horizons).toEqual(['short', 'medium'])
    })

    it('does NOT silently rewrite illegal empty array to ["short"]', async () => {
        let capturedBody: Record<string, unknown> = {}

        globalThis.fetch = vi.fn().mockImplementation(async (_url: string, init?: RequestInit) => {
            capturedBody = JSON.parse((init?.body as string) || '{}')
            return new Response(JSON.stringify({ ok: true }), {
                status: 200,
                headers: { 'Content-Type': 'application/json' },
            })
        })

        await api.chatCompletion(
            [{ role: 'user', content: '分析 600519.SH' }],
            true,
            undefined,
            [],
        )

        // 契约 4: 非法组合不要静默改档；必须原样发送由后端校验
        expect(capturedBody.horizons).toEqual([])
    })

    it('does NOT silently rewrite illegal unknown values to ["short"]', async () => {
        let capturedBody: Record<string, unknown> = {}

        globalThis.fetch = vi.fn().mockImplementation(async (_url: string, init?: RequestInit) => {
            capturedBody = JSON.parse((init?.body as string) || '{}')
            return new Response(JSON.stringify({ ok: true }), {
                status: 200,
                headers: { 'Content-Type': 'application/json' },
            })
        })

        await api.chatCompletion(
            [{ role: 'user', content: '分析 600519.SH' }],
            true,
            undefined,
            ['bogus'],
        )

        expect(capturedBody.horizons).toEqual(['bogus'])
    })

    it('surfaces 422 validation errors with details without treating as success', async () => {
        globalThis.fetch = vi.fn().mockImplementation(async () => {
            return new Response(
                JSON.stringify({ detail: 'Invalid horizons: bogus' }),
                {
                    status: 422,
                    headers: { 'Content-Type': 'application/json' },
                },
            )
        })

        // 契约 4: 422/400 错误必须抛出并携带详情，不把失败当成 short 成功
        await expect(
            api.chatCompletion(
                [{ role: 'user', content: 'test' }],
                true,
                undefined,
                ['bogus'],
            ),
        ).rejects.toThrow('Invalid horizons: bogus')
    })
})

describe('api.analyze horizons contract (H-03b)', () => {
    let originalFetch: typeof globalThis.fetch

    beforeEach(() => {
        originalFetch = globalThis.fetch
    })

    afterEach(() => {
        globalThis.fetch = originalFetch
        vi.restoreAllMocks()
    })

    it('sends explicit default ["short"] when request.horizons is undefined', async () => {
        let capturedUrl = ''
        let capturedBody: Record<string, unknown> = {}

        globalThis.fetch = vi.fn().mockImplementation(async (url: string, init?: RequestInit) => {
            capturedUrl = url
            capturedBody = JSON.parse((init?.body as string) || '{}')
            return new Response(JSON.stringify({ job_id: 'job-1', status: 'pending' }), {
                status: 200,
                headers: { 'Content-Type': 'application/json' },
            })
        })

        const req: AnalysisRequest = {
            symbol: '600519.SH',
            trade_date: '2026-09-07',
            selected_analysts: ['market'],
            investment_horizon: '中线',
        }

        await api.analyze(req)

        expect(capturedUrl).toContain('/v1/analyze')
        expect(capturedBody.symbol).toBe('600519.SH')
        // 契约 2: 显式发送 ["short"]
        expect(capturedBody.horizons).toEqual(['short'])
        // 契约 3: investment_horizon 与 horizons 分开，不得混淆
        expect(capturedBody.investment_horizon).toBe('中线')
        expect(capturedBody.horizons).not.toContain('中线')
    })

    it('sends explicit horizons when provided on request', async () => {
        let capturedBody: Record<string, unknown> = {}

        globalThis.fetch = vi.fn().mockImplementation(async (_url: string, init?: RequestInit) => {
            capturedBody = JSON.parse((init?.body as string) || '{}')
            return new Response(JSON.stringify({ job_id: 'job-2', status: 'pending' }), {
                status: 200,
                headers: { 'Content-Type': 'application/json' },
            })
        })

        const req: AnalysisRequest = {
            symbol: '000001.SZ',
            trade_date: '2026-09-07',
            selected_analysts: ['market'],
            horizons: ['short', 'medium'],
        }

        await api.analyze(req)
        expect(capturedBody.horizons).toEqual(['short', 'medium'])
    })

    it('does NOT silently rewrite empty horizons array in analyze helper', async () => {
        let capturedBody: Record<string, unknown> = {}

        globalThis.fetch = vi.fn().mockImplementation(async (_url: string, init?: RequestInit) => {
            capturedBody = JSON.parse((init?.body as string) || '{}')
            return new Response(JSON.stringify({ job_id: 'job-3', status: 'pending' }), {
                status: 200,
                headers: { 'Content-Type': 'application/json' },
            })
        })

        const req: AnalysisRequest = {
            symbol: '000001.SZ',
            trade_date: '2026-09-07',
            selected_analysts: ['market'],
            horizons: [],
        }

        await api.analyze(req)
        expect(capturedBody.horizons).toEqual([])
    })
})

describe('api.createScheduled horizon contract (H-03c)', () => {
    let originalFetch: typeof globalThis.fetch

    beforeEach(() => {
        originalFetch = globalThis.fetch
    })

    afterEach(() => {
        globalThis.fetch = originalFetch
        vi.restoreAllMocks()
    })

    it('sends explicit single horizon "short" in request body (契约 1)', async () => {
        let capturedUrl = ''
        let capturedBody: Record<string, unknown> = {}

        globalThis.fetch = vi.fn().mockImplementation(async (url: string, init?: RequestInit) => {
            capturedUrl = url
            capturedBody = JSON.parse((init?.body as string) || '{}')
            return new Response(
                JSON.stringify({
                    id: 'sched-1',
                    symbol: '600519.SH',
                    horizon: 'short',
                    trigger_time: '20:00',
                    is_active: true,
                }),
                {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' },
                },
            )
        })

        const res = await api.createScheduled('600519.SH', 'short', '20:00')

        expect(capturedUrl).toContain('/v1/scheduled')
        expect(capturedBody.symbol).toBe('600519.SH')
        // 契约 1: 显式发送单档 short
        expect(capturedBody.horizon).toBe('short')
        expect(capturedBody.trigger_time).toBe('20:00')
        // 契约 4: investment_horizon 不得写入 horizon 或随 body 污染
        expect(capturedBody.investment_horizon).toBeUndefined()
        expect(res.id).toBe('sched-1')
        expect(res.horizon).toBe('short')
    })

    it('sends explicit single horizon "medium" in request body when medium is selected (契约 1)', async () => {
        let capturedUrl = ''
        let capturedBody: Record<string, unknown> = {}

        globalThis.fetch = vi.fn().mockImplementation(async (url: string, init?: RequestInit) => {
            capturedUrl = url
            capturedBody = JSON.parse((init?.body as string) || '{}')
            return new Response(
                JSON.stringify({
                    id: 'sched-2',
                    symbol: '000001.SZ',
                    horizon: 'medium',
                    trigger_time: '20:00',
                    is_active: true,
                }),
                {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' },
                },
            )
        })

        const res = await api.createScheduled('000001.SZ', 'medium', '20:00')

        expect(capturedUrl).toContain('/v1/scheduled')
        expect(capturedBody.symbol).toBe('000001.SZ')
        // 契约 1: 可选 medium
        expect(capturedBody.horizon).toBe('medium')
        expect(capturedBody.trigger_time).toBe('20:00')
        expect(capturedBody.investment_horizon).toBeUndefined()
        expect(res.horizon).toBe('medium')
    })

    it('does NOT silently rewrite illegal dual horizon list or unknown string to "short" (契约 3)', async () => {
        let capturedBody: Record<string, unknown> = {}

        globalThis.fetch = vi.fn().mockImplementation(async (_url: string, init?: RequestInit) => {
            capturedBody = JSON.parse((init?.body as string) || '{}')
            return new Response(
                JSON.stringify({ id: 'sched-illegal' }),
                {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' },
                },
            )
        })

        // 不得把非法档或双档静默改成 short，须原样发送触发后端校验
        await api.createScheduled('600519.SH', ['short', 'medium'] as unknown as string, '20:00')
        expect(capturedBody.horizon).toEqual(['short', 'medium'])
        expect(capturedBody.horizon).not.toBe('short')

        await api.createScheduled('600519.SH', 'dual', '20:00')
        expect(capturedBody.horizon).toBe('dual')
        expect(capturedBody.horizon).not.toBe('short')
    })

    it('surfaces 400 validation error with backend detail when dual horizon is rejected (契约 3)', async () => {
        globalThis.fetch = vi.fn().mockImplementation(async () => {
            return new Response(
                JSON.stringify({ detail: '定时分析暂不支持多周期/双档，仅支持单周期 (short 或 medium)' }),
                {
                    status: 400,
                    headers: { 'Content-Type': 'application/json' },
                },
            )
        })

        await expect(
            api.createScheduled('600519.SH', ['short', 'medium'] as unknown as string, '20:00'),
        ).rejects.toThrow('定时分析暂不支持多周期/双档，仅支持单周期 (short 或 medium)')
    })
})

describe('api.updateScheduled horizon contract (H-03c)', () => {
    let originalFetch: typeof globalThis.fetch

    beforeEach(() => {
        originalFetch = globalThis.fetch
    })

    afterEach(() => {
        globalThis.fetch = originalFetch
        vi.restoreAllMocks()
    })

    it('sends explicit single horizon "medium" in PATCH body without omitting (契约 2)', async () => {
        let capturedUrl = ''
        let capturedMethod = ''
        let capturedBody: Record<string, unknown> = {}

        globalThis.fetch = vi.fn().mockImplementation(async (url: string, init?: RequestInit) => {
            capturedUrl = url
            capturedMethod = init?.method || ''
            capturedBody = JSON.parse((init?.body as string) || '{}')
            return new Response(
                JSON.stringify({
                    id: 'sched-10',
                    symbol: '600519.SH',
                    horizon: 'medium',
                    trigger_time: '20:00',
                    is_active: true,
                }),
                {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' },
                },
            )
        })

        const res = await api.updateScheduled('sched-10', { horizon: 'medium' })

        expect(capturedUrl).toContain('/v1/scheduled/sched-10')
        expect(capturedMethod).toBe('PATCH')
        // 契约 2: 行内改档 body 为所选单档，不得省略成后端缺省
        expect(capturedBody.horizon).toBe('medium')
        expect(capturedBody.investment_horizon).toBeUndefined()
        expect(res.horizon).toBe('medium')
    })

    it('sends explicit single horizon "short" in PATCH body (契约 2)', async () => {
        let capturedBody: Record<string, unknown> = {}

        globalThis.fetch = vi.fn().mockImplementation(async (_url: string, init?: RequestInit) => {
            capturedBody = JSON.parse((init?.body as string) || '{}')
            return new Response(
                JSON.stringify({
                    id: 'sched-11',
                    symbol: '000001.SZ',
                    horizon: 'short',
                    trigger_time: '20:00',
                    is_active: true,
                }),
                {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' },
                },
            )
        })

        const res = await api.updateScheduled('sched-11', { horizon: 'short' })

        expect(capturedBody.horizon).toBe('short')
        expect(capturedBody.investment_horizon).toBeUndefined()
        expect(res.horizon).toBe('short')
    })

    it('surfaces 400 error on invalid horizon update without silent fallback (契约 3)', async () => {
        globalThis.fetch = vi.fn().mockImplementation(async () => {
            return new Response(
                JSON.stringify({ detail: 'horizon 必须为 short 或 medium' }),
                {
                    status: 400,
                    headers: { 'Content-Type': 'application/json' },
                },
            )
        })

        await expect(
            api.updateScheduled('sched-12', { horizon: 'bogus' }),
        ).rejects.toThrow('horizon 必须为 short 或 medium')
    })
})

describe('api.updateScheduledBatch horizon contract (H-03c)', () => {
    let originalFetch: typeof globalThis.fetch

    beforeEach(() => {
        originalFetch = globalThis.fetch
    })

    afterEach(() => {
        globalThis.fetch = originalFetch
        vi.restoreAllMocks()
    })

    it('sends explicit single horizon "medium" with item_ids in batch PATCH body (契约 2)', async () => {
        let capturedUrl = ''
        let capturedMethod = ''
        let capturedBody: Record<string, unknown> = {}

        globalThis.fetch = vi.fn().mockImplementation(async (url: string, init?: RequestInit) => {
            capturedUrl = url
            capturedMethod = init?.method || ''
            capturedBody = JSON.parse((init?.body as string) || '{}')
            return new Response(
                JSON.stringify({
                    items: [
                        { id: 't-1', symbol: '600519.SH', horizon: 'medium', trigger_time: '20:00', is_active: true },
                        { id: 't-2', symbol: '000001.SZ', horizon: 'medium', trigger_time: '20:00', is_active: true },
                    ],
                }),
                {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' },
                },
            )
        })

        const res = await api.updateScheduledBatch(['t-1', 't-2'], { horizon: 'medium' })

        expect(capturedUrl).toContain('/v1/scheduled/batch')
        expect(capturedMethod).toBe('PATCH')
        // 契约 2: 批量改档 body 为所选单档，不得省略成后端缺省
        expect(capturedBody.item_ids).toEqual(['t-1', 't-2'])
        expect(capturedBody.horizon).toBe('medium')
        expect(capturedBody.investment_horizon).toBeUndefined()
        expect(res.items).toHaveLength(2)
        expect(res.items[0].horizon).toBe('medium')
        expect(res.items[1].horizon).toBe('medium')
    })

    it('sends explicit single horizon "short" in batch PATCH body (契约 2)', async () => {
        let capturedBody: Record<string, unknown> = {}

        globalThis.fetch = vi.fn().mockImplementation(async (_url: string, init?: RequestInit) => {
            capturedBody = JSON.parse((init?.body as string) || '{}')
            return new Response(
                JSON.stringify({
                    items: [
                        { id: 't-1', symbol: '600519.SH', horizon: 'short', trigger_time: '20:00', is_active: true },
                    ],
                }),
                {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' },
                },
            )
        })

        await api.updateScheduledBatch(['t-1'], { horizon: 'short' })

        expect(capturedBody.item_ids).toEqual(['t-1'])
        expect(capturedBody.horizon).toBe('short')
        expect(capturedBody.investment_horizon).toBeUndefined()
    })

    it('surfaces 400 error on batch update when backend rejects dual horizon (契约 3)', async () => {
        globalThis.fetch = vi.fn().mockImplementation(async () => {
            return new Response(
                JSON.stringify({ detail: '定时分析暂不支持多周期/双档，仅支持单周期 (short 或 medium)' }),
                {
                    status: 400,
                    headers: { 'Content-Type': 'application/json' },
                },
            )
        })

        await expect(
            api.updateScheduledBatch(['t-1'], { horizon: ['short', 'medium'] as unknown as string }),
        ).rejects.toThrow('定时分析暂不支持多周期/双档，仅支持单周期 (short 或 medium)')
    })
})
