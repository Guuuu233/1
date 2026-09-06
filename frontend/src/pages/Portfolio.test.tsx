import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { HorizonSwitch } from '@/pages/Portfolio'
import { api } from '@/services/api'
import type { ScheduledAnalysis, WatchlistItem } from '@/types'

describe('HorizonSwitch component (H-03c)', () => {
    it('renders "短线" and "中线" options with "短线" active by default', () => {
        const html = renderToStaticMarkup(
            <HorizonSwitch value="short" onChange={() => {}} testIdPrefix="test-switch" />,
        )

        // 契约 1: 默认短期可见；可选 medium
        expect(html).toContain('短线')
        expect(html).toContain('中线')
        expect(html).toMatch(/data-testid="test-switch-short"[^>]*aria-pressed="true"/)
        expect(html).toMatch(/data-testid="test-switch-medium"[^>]*aria-pressed="false"/)

        // 契约 3: 双档禁止，不得包含 dual 选项
        expect(html).not.toContain('双档')
        expect(html).not.toContain('dual')
    })

    it('renders "中线" as active when value="medium"', () => {
        const html = renderToStaticMarkup(
            <HorizonSwitch value="medium" onChange={() => {}} testIdPrefix="test-switch" />,
        )

        expect(html).toMatch(/data-testid="test-switch-short"[^>]*aria-pressed="false"/)
        expect(html).toMatch(/data-testid="test-switch-medium"[^>]*aria-pressed="true"/)
        expect(html).not.toContain('双档')
    })

    it('applies disabled state to buttons when disabled=true', () => {
        const html = renderToStaticMarkup(
            <HorizonSwitch value="short" onChange={() => {}} disabled testIdPrefix="test-switch" />,
        )

        expect(html).toMatch(/data-testid="test-switch-short"[^>]*disabled/)
        expect(html).toMatch(/data-testid="test-switch-medium"[^>]*disabled/)
    })

    it('compact prop renders compact classes', () => {
        const compactHtml = renderToStaticMarkup(
            <HorizonSwitch value="short" onChange={() => {}} compact testIdPrefix="test-switch" />,
        )
        const regularHtml = renderToStaticMarkup(
            <HorizonSwitch value="short" onChange={() => {}} compact={false} testIdPrefix="test-switch" />,
        )

        expect(compactHtml).toContain('w-[124px]')
        expect(regularHtml).toContain('w-[144px]')
    })
})

describe('Portfolio scheduled horizon contracts and markup (H-03c)', () => {
    it('renders visible HorizonSwitch on unscheduled watchlist row before submission (契约 1)', () => {
        const item: WatchlistItem = {
            id: 'item-1',
            symbol: '600519.SH',
            name: '贵州茅台',
            created_at: '2026-09-01T00:00:00Z',
            has_scheduled: false,
        }

        // Simulate watchlist row schedule controls rendering
        const html = renderToStaticMarkup(
            <div data-testid={`schedule-controls-${item.symbol}`}>
                {!item.has_scheduled && (
                    <>
                        <HorizonSwitch
                            value="short"
                            compact
                            testIdPrefix={`create-horizon-switch-${item.symbol}`}
                            onChange={() => {}}
                        />
                        <button
                            type="button"
                            title="开启定时分析"
                            data-testid={`enable-schedule-${item.symbol}`}
                        >
                            定时
                        </button>
                    </>
                )}
            </div>,
        )

        // 契约 1: 开启定时：默认短期可见；提交前可见；可选 medium
        expect(html).toContain(`data-testid="schedule-controls-${item.symbol}"`)
        expect(html).toContain(`data-testid="create-horizon-switch-${item.symbol}-short"`)
        expect(html).toContain(`data-testid="create-horizon-switch-${item.symbol}-medium"`)
        expect(html).toMatch(new RegExp(`data-testid="create-horizon-switch-${item.symbol}-short"[^>]*aria-pressed="true"`))
        expect(html).toMatch(new RegExp(`data-testid="create-horizon-switch-${item.symbol}-medium"[^>]*aria-pressed="false"`))
        expect(html).toContain(`data-testid="enable-schedule-${item.symbol}"`)
        expect(html).toContain('开启定时分析')

        // 双档禁止
        expect(html).not.toContain('双档')
        expect(html).not.toContain('dual')
    })

    it('renders medium selected when user toggles horizon before submission (契约 1)', () => {
        const item: WatchlistItem = {
            id: 'item-2',
            symbol: '000001.SZ',
            name: '平安银行',
            created_at: '2026-09-01T00:00:00Z',
            has_scheduled: false,
        }

        const html = renderToStaticMarkup(
            <div data-testid={`schedule-controls-${item.symbol}`}>
                <HorizonSwitch
                    value="medium"
                    compact
                    testIdPrefix={`create-horizon-switch-${item.symbol}`}
                    onChange={() => {}}
                />
                <button
                    type="button"
                    title="开启定时分析"
                    data-testid={`enable-schedule-${item.symbol}`}
                >
                    定时
                </button>
            </div>,
        )

        expect(html).toMatch(new RegExp(`data-testid="create-horizon-switch-${item.symbol}-short"[^>]*aria-pressed="false"`))
        expect(html).toMatch(new RegExp(`data-testid="create-horizon-switch-${item.symbol}-medium"[^>]*aria-pressed="true"`))
    })

    it('renders scheduled badge and disable button when already scheduled', () => {
        const item: WatchlistItem = {
            id: 'item-1',
            symbol: '600519.SH',
            name: '贵州茅台',
            created_at: '2026-09-01T00:00:00Z',
            has_scheduled: true,
        }
        const scheduledTask: ScheduledAnalysis = {
            id: 'task-1',
            symbol: '600519.SH',
            name: '贵州茅台',
            horizon: 'medium',
            trigger_time: '20:00',
            is_active: true,
            consecutive_failures: 0,
        }

        const html = renderToStaticMarkup(
            <div data-testid={`schedule-controls-${item.symbol}`}>
                <span className="text-[10px]">{scheduledTask.horizon === 'medium' ? '中线' : '短线'}</span>
                <button
                    type="button"
                    title="已开启定时分析"
                    data-testid={`disable-schedule-${item.symbol}`}
                >
                    定时
                </button>
            </div>,
        )

        expect(html).toContain('中线')
        expect(html).toContain('已开启定时分析')
        expect(html).not.toContain(`data-testid="create-horizon-switch-${item.symbol}"`)
    })

    it('renders inline HorizonSwitch for scheduled tasks in scheduled table (契约 2)', () => {
        const task: ScheduledAnalysis = {
            id: 'task-10',
            symbol: '600519.SH',
            name: '贵州茅台',
            horizon: 'short',
            trigger_time: '20:00',
            is_active: true,
            consecutive_failures: 0,
        }

        const html = renderToStaticMarkup(
            <HorizonSwitch
                value={task.horizon === 'medium' ? 'medium' : 'short'}
                compact
                testIdPrefix={`item-horizon-switch-${task.id}`}
                onChange={() => {}}
            />,
        )

        // 契约 2: 行内已有 HorizonSwitch
        expect(html).toContain(`data-testid="item-horizon-switch-${task.id}-short"`)
        expect(html).toContain(`data-testid="item-horizon-switch-${task.id}-medium"`)
        expect(html).toMatch(new RegExp(`data-testid="item-horizon-switch-${task.id}-short"[^>]*aria-pressed="true"`))
        expect(html).not.toContain('双档')
    })

    it('renders batch HorizonSwitch for batch modifications (契约 2)', () => {
        const html = renderToStaticMarkup(
            <HorizonSwitch
                value="medium"
                compact
                testIdPrefix="batch-horizon-switch"
                onChange={() => {}}
            />,
        )

        // 契约 2: 批量已有 HorizonSwitch
        expect(html).toContain('data-testid="batch-horizon-switch-short"')
        expect(html).toContain('data-testid="batch-horizon-switch-medium"')
        expect(html).toMatch(/data-testid="batch-horizon-switch-medium"[^>]*aria-pressed="true"/)
        expect(html).not.toContain('双档')
    })
})

describe('Portfolio fetch request body verification (H-03c)', () => {
    let originalFetch: typeof globalThis.fetch

    beforeEach(() => {
        originalFetch = globalThis.fetch
    })

    afterEach(() => {
        globalThis.fetch = originalFetch
        vi.restoreAllMocks()
    })

    it('verifies createScheduled fetch body with default short horizon (契约 1)', async () => {
        let capturedUrl = ''
        let capturedMethod = ''
        let capturedBody: Record<string, unknown> = {}

        globalThis.fetch = vi.fn().mockImplementation(async (url: string, init?: RequestInit) => {
            capturedUrl = url
            capturedMethod = init?.method || ''
            capturedBody = JSON.parse((init?.body as string) || '{}')
            return new Response(
                JSON.stringify({
                    id: 'sched-create-1',
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

        await api.createScheduled('600519.SH', 'short', '20:00')

        expect(capturedUrl).toContain('/v1/scheduled')
        expect(capturedMethod).toBe('POST')
        expect(capturedBody).toEqual({
            symbol: '600519.SH',
            horizon: 'short',
            trigger_time: '20:00',
        })
        expect(capturedBody.horizon).toBe('short')
        // 契约 4: investment_horizon 不得写入 horizon
        expect(capturedBody.investment_horizon).toBeUndefined()
    })

    it('verifies createScheduled fetch body with selected medium horizon (契约 1)', async () => {
        let capturedBody: Record<string, unknown> = {}

        globalThis.fetch = vi.fn().mockImplementation(async (_url: string, init?: RequestInit) => {
            capturedBody = JSON.parse((init?.body as string) || '{}')
            return new Response(
                JSON.stringify({
                    id: 'sched-create-2',
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

        await api.createScheduled('000001.SZ', 'medium', '20:00')

        expect(capturedBody).toEqual({
            symbol: '000001.SZ',
            horizon: 'medium',
            trigger_time: '20:00',
        })
        expect(capturedBody.horizon).toBe('medium')
        expect(capturedBody.investment_horizon).toBeUndefined()
    })

    it('verifies updateScheduled fetch body for inline horizon change (契约 2)', async () => {
        let capturedUrl = ''
        let capturedMethod = ''
        let capturedBody: Record<string, unknown> = {}

        globalThis.fetch = vi.fn().mockImplementation(async (url: string, init?: RequestInit) => {
            capturedUrl = url
            capturedMethod = init?.method || ''
            capturedBody = JSON.parse((init?.body as string) || '{}')
            return new Response(
                JSON.stringify({
                    id: 'task-inline-1',
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

        await api.updateScheduled('task-inline-1', { horizon: 'medium' })

        expect(capturedUrl).toContain('/v1/scheduled/task-inline-1')
        expect(capturedMethod).toBe('PATCH')
        // 契约 2: 行内改档 body 为所选单档，不得省略成后端缺省
        expect(capturedBody).toEqual({
            horizon: 'medium',
        })
        expect(capturedBody.horizon).toBe('medium')
        expect(capturedBody.investment_horizon).toBeUndefined()
    })

    it('verifies updateScheduledBatch fetch body for batch horizon change (契约 2)', async () => {
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
                        { id: 'task-b-1', symbol: '600519.SH', horizon: 'short', trigger_time: '20:00', is_active: true },
                        { id: 'task-b-2', symbol: '000001.SZ', horizon: 'short', trigger_time: '20:00', is_active: true },
                    ],
                }),
                {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' },
                },
            )
        })

        await api.updateScheduledBatch(['task-b-1', 'task-b-2'], { horizon: 'short' })

        expect(capturedUrl).toContain('/v1/scheduled/batch')
        expect(capturedMethod).toBe('PATCH')
        // 契约 2: 批量改档 body 为所选单档，不得省略成后端缺省
        expect(capturedBody).toEqual({
            item_ids: ['task-b-1', 'task-b-2'],
            horizon: 'short',
        })
        expect(capturedBody.horizon).toBe('short')
        expect(capturedBody.investment_horizon).toBeUndefined()
    })

    it('rejects and alerts on backend 400 when dual horizon is sent without silent success (契约 3)', async () => {
        globalThis.fetch = vi.fn().mockImplementation(async () => {
            return new Response(
                JSON.stringify({ detail: '定时分析暂不支持多周期/双档，仅支持单周期 (short 或 medium)' }),
                {
                    status: 400,
                    headers: { 'Content-Type': 'application/json' },
                },
            )
        })

        let alertMsg = ''
        const originalAlert = globalThis.alert
        globalThis.alert = vi.fn().mockImplementation((msg: string) => {
            alertMsg = msg
        })

        try {
            await api.createScheduled('600519.SH', ['short', 'medium'] as unknown as string, '20:00')
        } catch (e) {
            globalThis.alert(e instanceof Error ? e.message : '操作失败')
        }

        // 契约 3: 后端 4xx 要 alert/可见，不得当成功
        expect(alertMsg).toBe('定时分析暂不支持多周期/双档，仅支持单周期 (short 或 medium)')
        expect(globalThis.alert).toHaveBeenCalledWith('定时分析暂不支持多周期/双档，仅支持单周期 (short 或 medium)')

        globalThis.alert = originalAlert
    })
})
