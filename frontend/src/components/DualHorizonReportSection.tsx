import { AlertCircle, AlertTriangle, BarChart3, CheckCircle2, Clock3, Info, ShieldCheck } from 'lucide-react'
import type { ReportDetail } from '@/types'
import { getDualHorizonSummary, HORIZON_LABELS } from '@/utils/reportDualHorizon'
import type { HorizonSummaryItem } from '@/utils/reportDualHorizon'

const STATUS_BADGE: Record<string, string> = {
    completed: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-300',
    failed: 'bg-rose-100 text-rose-700 dark:bg-rose-500/20 dark:text-rose-300',
    partial: 'bg-amber-100 text-amber-700 dark:bg-amber-500/20 dark:text-amber-300',
    pending: 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300',
    running: 'bg-blue-100 text-blue-700 dark:bg-blue-500/20 dark:text-blue-300',
    not_requested: 'bg-slate-100 text-slate-500 dark:bg-slate-700 dark:text-slate-400',
}

function getStatusBadge(status: string): string {
    return STATUS_BADGE[status] || 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300'
}

function StatusIcon({ status }: { status: string }) {
    const className = 'h-3.5 w-3.5'
    switch (status) {
        case 'completed':
            return <CheckCircle2 className={`${className} text-emerald-500`} />
        case 'failed':
            return <AlertCircle className={`${className} text-rose-500`} />
        case 'partial':
            return <AlertTriangle className={`${className} text-amber-500`} />
        case 'running':
            return <Clock3 className={`${className} text-blue-500`} />
        case 'not_requested':
        case 'pending':
        default:
            return <Info className={`${className} text-slate-400`} />
    }
}

function StringList({
    title,
    items,
    provided,
}: {
    title: string
    items: string[]
    provided: boolean
}) {
    return (
        <div className="rounded-xl border border-slate-200 bg-slate-50/80 p-3.5 dark:border-slate-700/60 dark:bg-slate-800/40">
            <p className="text-xs font-semibold text-slate-700 dark:text-slate-300">{title}</p>
            {!provided ? (
                <p className="mt-1.5 text-xs text-slate-400">未提供</p>
            ) : items.length === 0 ? (
                <p className="mt-1.5 text-xs text-emerald-600 dark:text-emerald-400">无</p>
            ) : (
                <ul className="mt-1.5 space-y-1">
                    {items.map(item => (
                        <li key={item} className="flex items-start gap-1.5 text-xs leading-5 text-slate-600 dark:text-slate-400">
                            <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-slate-400" />
                            <span className="min-w-0">{item}</span>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    )
}

function HorizonCard({ item }: { item: HorizonSummaryItem }) {
    const applicability = item.notApplicable === true
        ? '不适用'
        : item.notApplicable === false
            ? '适用'
            : '未提供'

    return (
        <div className="rounded-xl border border-slate-200 bg-white p-3.5 shadow-sm dark:border-slate-700/60 dark:bg-slate-800/50">
            <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                    <StatusIcon status={item.status} />
                    <span className="text-sm font-semibold text-slate-900 dark:text-slate-100">{item.label}</span>
                </div>
                <span className={`badge shrink-0 ${getStatusBadge(item.status)}`}>{item.statusLabel}</span>
            </div>

            {item.error ? (
                <p className="mt-2.5 rounded-lg bg-rose-50 px-2.5 py-2 text-xs leading-5 text-rose-600 dark:bg-rose-500/10 dark:text-rose-300">
                    {item.error}
                </p>
            ) : null}

            <div className="mt-3 flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
                <ShieldCheck className="h-3.5 w-3.5 text-slate-400" />
                <span>适用性：{applicability}</span>
            </div>

            <div className="mt-2 space-y-2">
                <StringList
                    title="数据缺口"
                    items={item.dataGaps}
                    provided={item.dataGapsProvided}
                />
                <StringList
                    title="证伪条件"
                    items={item.falsificationConditions}
                    provided={item.falsificationConditionsProvided}
                />
            </div>
        </div>
    )
}

interface DualHorizonReportSectionProps {
    reportData?: ReportDetail | null
}

export default function DualHorizonReportSection({ reportData }: DualHorizonReportSectionProps) {
    const summary = getDualHorizonSummary(reportData?.result_data)
    if (!summary) return null

    const requestedLabels = summary.requestedHorizons.map(horizon => HORIZON_LABELS[horizon])

    return (
        <div className="card space-y-4 p-4 sm:p-5">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-2">
                    <div className="p-1.5 rounded-lg bg-indigo-500/20">
                        <BarChart3 className="h-4 w-4 text-indigo-500" />
                    </div>
                    <div>
                        <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">双周期分析</h2>
                        <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                            请求周期：{requestedLabels.join('、') || '未提供'}
                        </p>
                    </div>
                </div>
                <span className={`badge shrink-0 self-start sm:self-auto ${getStatusBadge(summary.status)}`}>
                    {summary.statusLabel}
                </span>
            </div>

            {summary.failedHorizons.length > 0 ? (
                <div className="flex items-start gap-2 rounded-xl border border-amber-200/80 bg-amber-50/80 px-3 py-2.5 text-xs leading-5 text-amber-800 dark:border-amber-500/20 dark:bg-amber-500/10 dark:text-amber-200">
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                    <span>
                        失败周期：{summary.failedHorizons.map(horizon => HORIZON_LABELS[horizon]).join('、')}
                        ，结果不完整，请参考已完成周期。
                    </span>
                </div>
            ) : null}

            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                {summary.horizons.map(item => (
                    <HorizonCard key={item.horizon} item={item} />
                ))}
            </div>

            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                <StringList
                    title="聚合数据缺口"
                    items={summary.dataGaps}
                    provided={summary.dataGapsProvided}
                />
                <StringList
                    title="聚合证伪条件"
                    items={summary.falsificationConditions}
                    provided={summary.falsificationConditionsProvided}
                />
            </div>
        </div>
    )
}
