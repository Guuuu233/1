export interface MarkdownSection {
    key: string
    title: string
}

/**
 * Build a markdown document from the string sections of a report. Callers pass
 * their own section list so historical vs live report shapes can differ.
 */
export function buildReportMarkdown(
    report: unknown,
    sections: MarkdownSection[],
    footer?: string,
): string {
    if (!report || typeof report !== 'object') return ''
    const record = report as Record<string, unknown>
    const parts: string[] = []
    for (const section of sections) {
        const content = record[section.key]
        if (typeof content === 'string' && content.length > 0) {
            parts.push(`## ${section.title}\n\n${content}`)
        }
    }
    if (footer) parts.push(footer)
    return parts.join('\n\n---\n\n')
}

/** Trigger a browser download of a markdown string. */
export function downloadMarkdown(filename: string, markdown: string): void {
    const blob = new Blob([markdown], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
}
