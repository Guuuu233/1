export type AnalysisRunState = 'idle' | 'running' | 'completed' | 'failed'

export interface JobLifecycleUpdate {
    isAnalyzing: boolean
    runState: AnalysisRunState
    overtimeNotice: string | null
}

export type RecoveredJobDisposition = 'running' | 'completed' | 'failed'
export type DbReportDisposition = 'completed' | 'failed' | 'active' | 'not_found'

export const DEFAULT_OVERTIME_NOTICE = '分析耗时较长，后台仍在继续，正在等待最终结果，请勿重复提交。'
// Recovery polls once every 3s. 5 minutes (100 attempts) bounds how long a
// user can stay locked in "isAnalyzing" after an interrupted stream — the old
// 2-hour cap meant a stuck in-memory job froze the input for hours.
export const RECOVERY_POLL_MAX_ATTEMPTS = 5 * 60 / 3
export const RECOVERY_POLL_TIMEOUT_MESSAGE = '任务状态异常，已停止自动等待。请刷新页面到「历史报告」查看最终结果，或重新发起分析。'
export const JOB_NOT_FOUND_MESSAGE = '任务已不存在（服务可能已重启）。已停止轮询，请重新发起分析，或到历史报告中查看已生成的结果。'

export function hasRecoveryPollingReachedLimit(
    attempts: number,
    maxAttempts: number = RECOVERY_POLL_MAX_ATTEMPTS,
): boolean {
    return attempts >= maxAttempts
}

export function getJobLifecycleUpdate(
    eventName: string,
    data: Record<string, unknown> = {},
): JobLifecycleUpdate | null {
    switch (eventName) {
        case 'job.running':
            return { isAnalyzing: true, runState: 'running', overtimeNotice: null }
        case 'job.overtime': {
            const suppliedMessage = data.message ?? data.msg
            return {
                isAnalyzing: true,
                runState: 'running',
                overtimeNotice: typeof suppliedMessage === 'string' && suppliedMessage.trim()
                    ? suppliedMessage
                    : DEFAULT_OVERTIME_NOTICE,
            }
        }
        case 'job.completed':
            return { isAnalyzing: false, runState: 'completed', overtimeNotice: null }
        case 'job.failed':
            if (classifyRecoveredJobStatus('failed', typeof data.error === 'string' ? data.error : null) === 'running') {
                return {
                    isAnalyzing: true,
                    runState: 'running',
                    overtimeNotice: DEFAULT_OVERTIME_NOTICE,
                }
            }
            return { isAnalyzing: false, runState: 'failed', overtimeNotice: null }
        default:
            return null
    }
}

export function classifyRecoveredJobStatus(status: string, error?: string | null): RecoveredJobDisposition {
    if (status === 'completed') return 'completed'
    if (status !== 'failed') return 'running'

    // Older servers marked the outer 1800-second watchdog as failed even though
    // the inner analysis continued and could still persist a completed report.
    const legacySoftTimeout = /(?:任务超时[^\n]*(?:1800|秒)|(?:timeout|timed out)[^\n]*1800)/i.test(error || '')
    return legacySoftTimeout ? 'running' : 'failed'
}

/**
 * Classify the *persisted* reports-table row for a job. The backend keys the
 * reports table by job id (report.id == job_id), so recovery polling can fall
 * back to it when the in-memory job store is gone (restart) or the job entry
 * is stuck in "running" while the report already reached a terminal state.
 */
export function classifyDbReportStatus(
    report: { status?: string; error?: string | null } | null | undefined,
): DbReportDisposition {
    if (!report) return 'not_found'
    if (report.status === 'completed') return 'completed'
    if (report.status === 'failed') return 'failed'
    // pending / running / unknown → 任务仍在进行或状态未知，继续轮询
    return 'active'
}
