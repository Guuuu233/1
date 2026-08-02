import type { AnalysisHorizon, AnalysisHorizonResult, AnalysisReport } from '@/types'

export const HORIZON_LABELS: Record<AnalysisHorizon, string> = {
    short: '短线',
    medium: '中线',
}

const STATUS_LABELS: Record<string, string> = {
    completed: '已完成',
    failed: '失败',
    not_requested: '未请求',
    partial: '部分完成',
    pending: '排队中',
    running: '分析中',
    error: '异常',
    unknown: '未知',
}

const REQUESTED_ORDER: AnalysisHorizon[] = ['short', 'medium']

export interface HorizonSummaryItem {
    horizon: AnalysisHorizon
    label: string
    status: string
    statusLabel: string
    notApplicable: boolean | null
    dataGaps: string[]
    dataGapsProvided: boolean
    falsificationConditions: string[]
    falsificationConditionsProvided: boolean
    error?: string
    impact?: string
}

export interface DualHorizonSummaryData {
    status: string
    statusLabel: string
    requestedHorizons: AnalysisHorizon[]
    failedHorizons: AnalysisHorizon[]
    dataGaps: string[]
    dataGapsProvided: boolean
    falsificationConditions: string[]
    falsificationConditionsProvided: boolean
    notApplicable: boolean | null
    notApplicableByHorizon: Partial<Record<AnalysisHorizon, boolean | null>>
    horizons: HorizonSummaryItem[]
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function getByHorizonRecord(value: unknown): Record<string, unknown> | null {
    return isRecord(value) ? value : null
}

function asStringArray(value: unknown): string[] {
    return Array.isArray(value)
        ? value.filter((item): item is string => typeof item === 'string')
        : []
}

function asBoolean(value: unknown): boolean | null {
    return typeof value === 'boolean' ? value : null
}

function uniqueHorizons(horizons: AnalysisHorizon[]): AnalysisHorizon[] {
    return Array.from(new Set(horizons))
}

export function getHorizonStatusLabel(status?: string): string {
    if (!status) return '未知'
    return STATUS_LABELS[status] || status
}

function getHorizonData(report: AnalysisReport, horizon: AnalysisHorizon): AnalysisHorizonResult | undefined {
    const byKey = report.horizons?.[horizon]
    if (byKey) return byKey
    return horizon === 'short' ? report.short_term : report.medium_term
}

function getHorizonStatus(
    report: AnalysisReport,
    horizon: AnalysisHorizon,
    result: AnalysisHorizonResult | undefined,
): string {
    const statusMap = getByHorizonRecord(report.horizon_status)
    const explicit = statusMap?.[horizon]
    return typeof explicit === 'string' ? explicit : (result?.status || '')
}

function getRequestedHorizons(report: AnalysisReport): AnalysisHorizon[] {
    if (Array.isArray(report.requested_horizons)) {
        const list = report.requested_horizons.filter((horizon): horizon is AnalysisHorizon => {
            return horizon === 'short' || horizon === 'medium'
        })
        if (list.length > 0) return uniqueHorizons(list)
    }

    const statusMap = getByHorizonRecord(report.horizon_status)
    if (statusMap) {
        const fromStatus = REQUESTED_ORDER.filter(horizon => statusMap[horizon] !== undefined)
        if (fromStatus.length > 0) return fromStatus
    }

    const falsificationMap = getByHorizonRecord(report.falsification_conditions_by_horizon)
    const notApplicableMap = getByHorizonRecord(report.not_applicable_by_horizon)
    return REQUESTED_ORDER.filter(horizon => {
        const result = getHorizonData(report, horizon)
        const status = getHorizonStatus(report, horizon, result)
        const hasMeaningfulStatus = status !== '' && status !== 'not_requested'
        const hasMetadata = falsificationMap?.[horizon] !== undefined || notApplicableMap?.[horizon] !== undefined
        return hasMeaningfulStatus || hasMetadata
    })
}

function getNotApplicableByHorizon(
    report: AnalysisReport,
    requested: AnalysisHorizon[],
): Partial<Record<AnalysisHorizon, boolean | null>> {
    const explicit = getByHorizonRecord(report.not_applicable_by_horizon)
    const result: Partial<Record<AnalysisHorizon, boolean | null>> = {}
    for (const horizon of requested) {
        const horizonResult = getHorizonData(report, horizon)
        const explicitValue = explicit?.[horizon]
        result[horizon] = explicitValue === undefined
            ? (getHorizonStatus(report, horizon, horizonResult) === 'completed'
                ? asBoolean(horizonResult?.not_applicable)
                : null)
            : asBoolean(explicitValue)
    }
    return result
}

function getDataGapsByHorizon(
    report: AnalysisReport,
    requested: AnalysisHorizon[],
): Partial<Record<AnalysisHorizon, string[]>> {
    const result: Partial<Record<AnalysisHorizon, string[]>> = {}
    for (const horizon of requested) {
        result[horizon] = asStringArray(getHorizonData(report, horizon)?.data_gaps)
    }
    return result
}

function getFalsificationByHorizon(
    report: AnalysisReport,
    requested: AnalysisHorizon[],
): Partial<Record<AnalysisHorizon, string[]>> {
    const explicit = getByHorizonRecord(report.falsification_conditions_by_horizon)
    const result: Partial<Record<AnalysisHorizon, string[]>> = {}
    for (const horizon of requested) {
        const explicitValue = explicit?.[horizon]
        result[horizon] = Array.isArray(explicitValue)
            ? asStringArray(explicitValue)
            : asStringArray(getHorizonData(report, horizon)?.falsification_conditions)
    }
    return result
}

function flattenByHorizon(
    byHorizon: Partial<Record<AnalysisHorizon, string[]>>,
    requested: AnalysisHorizon[],
): string[] {
    const flattened: string[] = []
    for (const horizon of requested) {
        for (const item of byHorizon[horizon] ?? []) {
            if (!flattened.includes(item)) flattened.push(item)
        }
    }
    return flattened
}

function getFailedHorizons(report: AnalysisReport, requested: AnalysisHorizon[]): AnalysisHorizon[] {
    if (Array.isArray(report.failed_horizons)) {
        const list = report.failed_horizons.filter((horizon): horizon is AnalysisHorizon => {
            return horizon === 'short' || horizon === 'medium'
        })
        if (list.length > 0) return uniqueHorizons(list)
    }
    return requested.filter(horizon => {
        return getHorizonStatus(report, horizon, getHorizonData(report, horizon)) === 'failed'
    })
}

function getAggregateStatus(
    report: AnalysisReport,
    requested: AnalysisHorizon[],
    failed: AnalysisHorizon[],
): string {
    if (typeof report.status === 'string' && report.status) return report.status
    if (requested.length === 0) return 'unknown'
    if (failed.length === requested.length) return 'failed'
    if (failed.length > 0) return 'partial'
    return 'completed'
}

function getAggregateNotApplicable(
    report: AnalysisReport,
    requested: AnalysisHorizon[],
    byHorizon: Partial<Record<AnalysisHorizon, boolean | null>>,
): boolean | null {
    if (typeof report.not_applicable === 'boolean') return report.not_applicable
    if (requested.length === 0) return null
    const values = requested.map(horizon => byHorizon[horizon] ?? null)
    if (values.some(value => value === null)) return null
    return values.every(value => value === true)
}

function isReportDualHorizon(report: AnalysisReport): boolean {
    if (report.mode === 'single_horizon') return false
    if (Array.isArray(report.requested_horizons) && report.requested_horizons.length > 1) return true

    const statusMap = getByHorizonRecord(report.horizon_status)
    const falsificationMap = getByHorizonRecord(report.falsification_conditions_by_horizon)
    const notApplicableMap = getByHorizonRecord(report.not_applicable_by_horizon)

    const statuses = REQUESTED_ORDER
        .map(horizon => getHorizonStatus(report, horizon, getHorizonData(report, horizon)))
        .filter(status => status && status !== 'not_requested')

    if (report.mode === 'dual_horizon') {
        if (
            (statusMap && Object.keys(statusMap).length > 0)
            || (falsificationMap && Object.keys(falsificationMap).length > 0)
            || (notApplicableMap && Object.keys(notApplicableMap).length > 0)
        ) {
            return true
        }
        return statuses.length > 0
    }

    return statuses.length > 1
        || (statusMap ? Object.keys(statusMap).length > 1 : false)
        || (falsificationMap ? Object.keys(falsificationMap).length > 0 : false)
        || (notApplicableMap ? Object.keys(notApplicableMap).length > 0 : false)
}

export function getDualHorizonSummary(report?: AnalysisReport): DualHorizonSummaryData | null {
    if (!report || !isReportDualHorizon(report)) return null

    const requested = getRequestedHorizons(report)
    const failed = getFailedHorizons(report, requested)
    const dataGapsByHorizon = getDataGapsByHorizon(report, requested)
    const falsificationByHorizon = getFalsificationByHorizon(report, requested)
    const notApplicableByHorizon = getNotApplicableByHorizon(report, requested)
    const status = getAggregateStatus(report, requested, failed)

    const horizons: HorizonSummaryItem[] = requested.map(horizon => {
        const result = getHorizonData(report, horizon)
        const horizonStatus = getHorizonStatus(report, horizon, result)
        return {
            horizon,
            label: HORIZON_LABELS[horizon],
            status: horizonStatus,
            statusLabel: getHorizonStatusLabel(horizonStatus),
            notApplicable: notApplicableByHorizon[horizon] ?? null,
            dataGaps: dataGapsByHorizon[horizon] ?? [],
            dataGapsProvided: Array.isArray(result?.data_gaps),
            falsificationConditions: falsificationByHorizon[horizon] ?? [],
            falsificationConditionsProvided:
                Array.isArray(result?.falsification_conditions)
                || Array.isArray(getByHorizonRecord(report.falsification_conditions_by_horizon)?.[horizon]),
            error: result?.error,
            impact: result?.impact,
        }
    })

    const dataGapsProvided = Array.isArray(report.data_gaps)
        || horizons.some(item => item.dataGapsProvided)
    const falsificationConditionsProvided = Array.isArray(report.falsification_conditions)
        || getByHorizonRecord(report.falsification_conditions_by_horizon) !== null
        || horizons.some(item => item.falsificationConditionsProvided)

    return {
        status,
        statusLabel: getHorizonStatusLabel(status),
        requestedHorizons: requested,
        failedHorizons: failed,
        dataGaps: Array.isArray(report.data_gaps)
            ? asStringArray(report.data_gaps)
            : flattenByHorizon(dataGapsByHorizon, requested),
        dataGapsProvided,
        falsificationConditions: Array.isArray(report.falsification_conditions)
            ? asStringArray(report.falsification_conditions)
            : flattenByHorizon(falsificationByHorizon, requested),
        falsificationConditionsProvided,
        notApplicable: getAggregateNotApplicable(report, requested, notApplicableByHorizon),
        notApplicableByHorizon,
        horizons,
    }
}
