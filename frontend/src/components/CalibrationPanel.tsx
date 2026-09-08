import { Activity, AlertTriangle, Ban, BarChart3, Loader2, RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import {
    Bar,
    CartesianGrid,
    ComposedChart,
    Legend,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts'

import { api } from '@/services/api'
import type { CalibrationBucket, CalibrationResponse } from '@/types'

interface CalibrationPanelProps {
    compact?: boolean
    initialData?: CalibrationResponse | null
}

type ChartDatum = {
    bucket: string
    predicted: number | null // 预测概率中值（%）
    actual: number | null // 实际上涨率（%）
    count: number
}

function toChartData(buckets: CalibrationBucket[]): ChartDatum[] {
    return buckets.map(bucket => {
        // Empty buckets have no observed prediction; leave predicted null so the
        // chart never draws a bar for a bucket with zero samples.
        const predicted = bucket.avg_probability != null
            ? Number((bucket.avg_probability * 100).toFixed(1))
            : null
        return {
            bucket: bucket.bucket,
            predicted,
            actual: bucket.rise_rate,
            count: bucket.count,
        }
    })
}

type TooltipValue = number | string | readonly (string | number)[]

const tooltipFormatter = (value: TooltipValue | undefined, name: string | number | undefined): [string, string] => {
    const label = name === undefined ? '' : String(name)
    const display = Array.isArray(value) ? value.join(', ') : value ?? ''
    if (label === 'count') return [String(display), '样本数']
    return [`${display}%`, label === 'predicted' ? '预测概率(中值)' : '实际上涨率']
}

export default function CalibrationPanel({ compact = false, initialData = null }: CalibrationPanelProps) {
    const [data, setData] = useState<CalibrationResponse | null>(initialData)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    // Filters
    const [startDate, setStartDate] = useState('')
    const [endDate, setEndDate] = useState('')
    const [holdDays, setHoldDays] = useState(5)
    const [symbol, setSymbol] = useState('')
    const [promptVersion, setPromptVersion] = useState('')
    const [model, setModel] = useState('')

    // Single fetch source shared by the mount effect and the refresh button; it
    // returns a promise and never calls setState itself, so the effect applies
    // results in async callbacks (matching the codebase fetch pattern) and there
    // is no competing double-fetch path.
    const fetchCalibration = useCallback(
        () => api.getCalibration({
            start_date: startDate || undefined,
            end_date: endDate || undefined,
            symbol: symbol.trim() || undefined,
            prompt_version: promptVersion.trim() || undefined,
            model: model.trim() || undefined,
            hold_days: holdDays,
        }),
        [endDate, holdDays, model, promptVersion, startDate, symbol],
    )

    useEffect(() => {
        if (initialData) return
        let cancelled = false
        fetchCalibration()
            .then(res => {
                if (cancelled) return
                setError(null)
                setData(res)
            })
            .catch(err => {
                if (cancelled) return
                setError(err instanceof Error ? err.message : '加载校准度数据失败')
            })
        return () => {
            cancelled = true
        }
    }, [fetchCalibration, initialData])

    const handleRefresh = useCallback(() => {
        setLoading(true)
        fetchCalibration()
            .then(res => {
                setError(null)
                setData(res)
            })
            .catch(err => {
                setError(err instanceof Error ? err.message : '加载校准度数据失败')
            })
            .finally(() => setLoading(false))
    }, [fetchCalibration])

    const isSufficient = data?.sample_sufficient ?? false
    const minSamples = data?.min_sample_size ?? 30
    const probSamples = data?.probability_sample_size ?? 0
    const totalSamples = data?.sample_size ?? 0
    const skipped = data?.skipped_no_outcome ?? 0
    const excludedCounts = data?.excluded_counts
    const excludedTotal = excludedCounts?.total ?? (data?.skipped_no_outcome ?? 0)
    const chartData = data && isSufficient ? toChartData(data.buckets) : []

    return (
        <div className="card space-y-4">
            {/* Header */}
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-2">
                    <div className="rounded-lg bg-indigo-100 p-2 text-indigo-600 dark:bg-indigo-500/10 dark:text-indigo-400">
                        <BarChart3 className="h-4 w-4" />
                    </div>
                    <div>
                        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">校准度面板</h2>
                        <p className="text-xs text-slate-400 dark:text-slate-500">
                            历史报告命中率 vs 预测概率（可靠性曲线）
                        </p>
                    </div>
                </div>
                <button
                    type="button"
                    onClick={handleRefresh}
                    disabled={loading}
                    className="btn-secondary inline-flex items-center gap-1.5 px-3 py-1.5 text-sm disabled:opacity-50"
                >
                    {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                    刷新
                </button>
            </div>

            {/* Filters */}
            <div className="grid grid-cols-2 gap-3 md:grid-cols-6">
                <label className="block">
                    <span className="mb-1 block text-xs text-slate-400">起始日期</span>
                    <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} className="input w-full" />
                </label>
                <label className="block">
                    <span className="mb-1 block text-xs text-slate-400">结束日期</span>
                    <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} className="input w-full" />
                </label>
                <label className="block">
                    <span className="mb-1 block text-xs text-slate-400">股票代码</span>
                    <input
                        type="text"
                        value={symbol}
                        onChange={e => setSymbol(e.target.value)}
                        placeholder="600519.SH"
                        className="input w-full"
                    />
                </label>
                <label className="block">
                    <span className="mb-1 block text-xs text-slate-400">提示词版本</span>
                    <input
                        type="text"
                        value={promptVersion}
                        onChange={e => setPromptVersion(e.target.value)}
                        placeholder="resolved_hash"
                        className="input w-full"
                    />
                </label>
                <label className="block">
                    <span className="mb-1 block text-xs text-slate-400">模型</span>
                    <input
                        type="text"
                        value={model}
                        onChange={e => setModel(e.target.value)}
                        placeholder="gpt-4o-mini"
                        className="input w-full"
                    />
                </label>
                <label className="block">
                    <span className="mb-1 block text-xs text-slate-400">持有天数</span>
                    <select
                        value={holdDays}
                        onChange={e => setHoldDays(Number(e.target.value))}
                        className="input w-full"
                    >
                        {[1, 3, 5, 10, 20].map(days => (
                            <option key={days} value={days}>{days} 天</option>
                        ))}
                    </select>
                </label>
            </div>

            {error && (
                <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-600 dark:border-rose-900/50 dark:bg-rose-950/30 dark:text-rose-300">
                    {error}
                </div>
            )}

            {data?.truncated_before_filter && (
                <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-700 dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-300">
                    提示：快照过滤的候选扫描达到上限，当前结果可能未覆盖所有历史报告（有偏采样）。
                </div>
            )}

            {!data && !error && (
                <div className="flex items-center justify-center py-10">
                    <Loader2 className="h-6 w-6 animate-spin text-indigo-500" />
                </div>
            )}

            {data && (
                <>
                    {/* Summary stats with equal prominence */}
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
                        <StatTile
                            icon={Activity}
                            label="Brier Score"
                            value={isSufficient && data.brier_score != null ? data.brier_score.toFixed(4) : '—'}
                            hint={isSufficient ? '越低越准（0 完美 / 1 最差）' : (data.insufficient_reason ?? '样本不足，指标熔断')}
                        />
                        <StatTile
                            icon={BarChart3}
                            label="概率校准样本 (n)"
                            value={String(probSamples)}
                            hint={isSufficient ? `已达统计门槛 (≥${minSamples})${totalSamples > probSamples ? ` · 全部可评估样本 ${totalSamples}` : ''}` : `门槛需 ≥${minSamples} 份（缺 ${Math.max(0, minSamples - probSamples)} 份）${totalSamples > probSamples ? ` · 全部可评估样本 ${totalSamples}` : ''}`}
                        />
                        <StatTile
                            icon={Ban}
                            label="排除样本 (Excluded)"
                            value={String(excludedTotal)}
                            hint={`无效 ${excludedCounts?.invalid ?? 0} · 弃权 ${excludedCounts?.abstain ?? 0} · 观望/未交易 ${excludedCounts?.no_trade ?? 0} · 未到期 ${excludedCounts?.incomplete ?? skipped}`}
                        />
                        <StatTile
                            icon={isSufficient ? Activity : AlertTriangle}
                            label="校准有效性"
                            value={isSufficient ? '有效校准' : (probSamples === 0 && totalSamples === 0 ? '无样本' : '样本不足 (熔断)')}
                            hint={isSufficient ? `含样本分桶 ${chartData.filter(d => d.count > 0).length}/${chartData.length}` : (probSamples === 0 && totalSamples === 0 ? '暂无有效评估报告' : (probSamples === 0 ? '无有效概率样本，不具统计学效力' : '样本极小，不具统计学效力'))}
                        />
                    </div>

                    {/* Small sample warning alert */}
                    {(probSamples > 0 || totalSamples > 0) && !isSufficient && (
                        <div
                            className="rounded-xl border border-amber-300 bg-amber-50 p-4 text-amber-800 dark:border-amber-700/50 dark:bg-amber-950/40 dark:text-amber-200"
                            data-testid="small-sample-warning"
                        >
                            <div className="flex items-start gap-3">
                                <AlertTriangle className="h-5 w-5 flex-shrink-0 text-amber-600 dark:text-amber-400 mt-0.5" />
                                <div className="space-y-1">
                                    <h4 className="font-semibold text-sm">
                                        样本量不足以支撑校准结论（当前概率样本 n={probSamples}，最小阈值 {minSamples}）
                                    </h4>
                                    <p className="text-xs leading-relaxed opacity-90">
                                        为防止小样本下的偶然涨跌制造虚假校准曲线（L3 自欺），在样本量达到统计显著性下限前，系统拒绝呈现 Brier Score 与可靠性曲线柱体。{totalSamples > probSamples ? `（当前全部可评估样本共 ${totalSamples} 份，其中方向命中等非概率样本不计入校准分母）` : ''}
                                    </p>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Reliability curve */}
                    {isSufficient ? (
                        <div className="h-72 w-full" data-testid="reliability-chart">
                            <ResponsiveContainer width="100%" height="100%">
                                <ComposedChart data={chartData} margin={{ top: 8, right: 16, bottom: 8, left: -12 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.12} />
                                    <XAxis
                                        dataKey="bucket"
                                        tick={{ fill: 'currentColor', opacity: 0.6, fontSize: 12 }}
                                        axisLine={false}
                                        tickLine={false}
                                    />
                                    <YAxis
                                        domain={[0, 100]}
                                        tickFormatter={value => `${value}%`}
                                        tick={{ fill: 'currentColor', opacity: 0.6, fontSize: 12 }}
                                        axisLine={false}
                                        tickLine={false}
                                    />
                                    <Tooltip formatter={tooltipFormatter} />
                                    <Legend wrapperStyle={{ fontSize: 12 }} />
                                    <Bar dataKey="predicted" name="预测概率(中值)" fill="#6366f1" radius={[4, 4, 0, 0]} />
                                    <Bar dataKey="actual" name="实际上涨率" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                                </ComposedChart>
                            </ResponsiveContainer>
                        </div>
                    ) : (
                        <div
                            className="flex h-48 w-full flex-col items-center justify-center rounded-xl border border-dashed border-amber-200 bg-amber-50/40 p-6 text-center dark:border-amber-900/40 dark:bg-amber-950/20"
                            data-testid="curve-withheld-placeholder"
                        >
                            <AlertTriangle className="mb-2 h-7 w-7 text-amber-500 opacity-80" />
                            <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
                                {probSamples === 0 && totalSamples === 0 ? '暂无可评估样本，未绘制可靠性曲线' : '样本量不足，可靠性曲线已熔断隐藏'}
                            </p>
                            <p className="mt-1 max-w-md text-xs text-slate-400 dark:text-slate-500">
                                {probSamples === 0 && totalSamples === 0
                                    ? (skipped > 0
                                        ? '最近报告持有期尚未结束，或缺少价格数据，暂无可评估样本。'
                                        : '当前筛选条件下暂无带概率的历史报告，调整日期范围或过滤条件后重试。')
                                    : `概率样本量低于统计显著性阈值（当前概率 n=${probSamples}，阈值 ${minSamples}，缺 ${Math.max(0, minSamples - probSamples)} 份），拒绝呈现可靠性曲线。${totalSamples > probSamples ? ` 全部可评估样本 ${totalSamples} 份。` : ''}`}
                            </p>
                        </div>
                    )}

                    {!compact && (
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="border-b border-slate-200 text-left text-slate-400 dark:border-slate-700">
                                        <th className="py-2 pr-4 font-medium">分桶</th>
                                        <th className="py-2 pr-4 font-medium">样本数</th>
                                        <th className="py-2 pr-4 font-medium">实际上涨</th>
                                        <th className="py-2 pr-4 font-medium">实际上涨率</th>
                                        <th className="py-2 pr-4 font-medium">平均预测概率</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                                    {data.buckets.map(bucket => (
                                        <tr key={bucket.bucket} className="text-slate-600 dark:text-slate-300">
                                            <td className="py-2 pr-4 font-medium text-slate-900 dark:text-slate-100">{bucket.bucket}</td>
                                            <td className="py-2 pr-4 tabular-nums">{bucket.count}</td>
                                            <td className="py-2 pr-4 tabular-nums">{isSufficient ? bucket.rise_count : '—'}</td>
                                            <td className="py-2 pr-4 tabular-nums">
                                                {isSufficient && bucket.rise_rate != null ? `${bucket.rise_rate}%` : '—'}
                                            </td>
                                            <td className="py-2 pr-4 tabular-nums">
                                                {isSufficient && bucket.avg_probability != null ? `${(bucket.avg_probability * 100).toFixed(1)}%` : '—'}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </>
            )}

            {data && data.sample_size === 0 && !error && (
                <p className="py-4 text-center text-sm text-slate-400">
                    {data.skipped_no_outcome > 0
                        ? '最近报告持有期尚未结束，或缺少价格数据，暂无可评估样本。'
                        : '当前筛选条件下暂无带概率的历史报告，调整日期范围或过滤条件后重试。'}
                </p>
            )}
        </div>
    )
}

function StatTile({
    icon: Icon,
    label,
    value,
    hint,
}: {
    icon: React.ComponentType<{ className?: string }>
    label: string
    value: string
    hint: string
}) {
    return (
        <div className="rounded-2xl border border-slate-200 bg-slate-50/80 px-4 py-3 dark:border-slate-700 dark:bg-slate-800/40">
            <div className="flex items-center gap-2">
                <Icon className="h-4 w-4 text-indigo-500 dark:text-indigo-400" />
                <p className="text-xs uppercase tracking-[0.14em] text-slate-400">{label}</p>
            </div>
            <p className="mt-2 text-xl font-semibold text-slate-900 dark:text-slate-100">{value}</p>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{hint}</p>
        </div>
    )
}
