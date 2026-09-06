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
