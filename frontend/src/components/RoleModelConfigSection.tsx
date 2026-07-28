import { useState, useEffect, useMemo } from 'react'
import {
    Users,
    Cpu,
    Plus,
        AlertTriangle,
    Check,
    Loader2,
    RefreshCw,
            Sliders,
        Info,
    ChevronDown,
    ChevronRight,
} from 'lucide-react'
import { api } from '@/services/api'
import type {
    Provider,
    ModelProfile,
    RoleBinding,
    RoleBindingItem,
    ResolvedRole,
} from '@/types'

const ROLE_LABELS: Record<string, { name: string; desc: string }> = {
    bull_researcher: { name: '多头研究员', desc: '看多立场，挖掘看涨逻辑与利好因素' },
    bear_researcher: { name: '空头研究员', desc: '看空立场，挖掘风险点与下行压力' },
    research_manager: { name: '研究总监', desc: '主持多空辩论并做出最终综合裁决' },
    market: { name: '市场趋势分析师', desc: '分析大盘与板块趋势' },
    social: { name: '情绪与舆情分析师', desc: '监测散户与社交媒体情绪' },
    news: { name: '新闻事件分析师', desc: '解读政策、公告与突发新闻' },
    fundamentals: { name: '基本面分析师', desc: '评估财务指标与估值水平' },
    macro: { name: '宏观经济分析师', desc: '分析宏观经济环境与货币政策' },
    smart_money: { name: '主力资金分析师', desc: '追踪北向资金、机构持仓与主力流向' },
    volume_price: { name: '量价形态分析师', desc: '分析量价结构与支撑压力位' },
    trader: { name: '交易员', desc: '制定具体买卖仓位与操作策略' },
    aggressive_analyst: { name: '激进风控员', desc: '偏向高风险高收益策略' },
    neutral_analyst: { name: '中性风控员', desc: '平衡风险与收益' },
    conservative_analyst: { name: '稳健风控员', desc: '严格控制回撤与风险暴露' },
    risk_manager: { name: '风控总监', desc: '汇总风控意见，设定止损止盈' },
}

const ROLE_GROUPS = [
    {
        key: 'researchers',
        title: '核心红蓝对抗组 (Researchers)',
        desc: '多空辩论的主角，建议绑定不同厂商模型实现独立思考',
        roles: ['bull_researcher', 'bear_researcher'],
    },
    {
        key: 'arbiter',
        title: '辩论裁决组 (Arbiter)',
        desc: '主持多空对抗并产出终审报告的高阶角色',
        roles: ['research_manager'],
    },
    {
        key: 'analysts',
        title: '基础数据分析师组 (Analysts)',
        desc: '负责并行提取各维度数据，通常使用性价比高的常规模型',
        roles: [
            'market',
            'social',
            'news',
            'fundamentals',
            'macro',
            'smart_money',
            'volume_price',
        ],
    },
    {
        key: 'trader',
        title: '交易执行组 (Trader)',
        desc: '生成交易指令与仓位建议',
        roles: ['trader'],
    },
    {
        key: 'risk',
        title: '风险控制团队 (Risk Management)',
        desc: '多角度风控质询与风控决策',
        roles: ['aggressive_analyst', 'neutral_analyst', 'conservative_analyst', 'risk_manager'],
    },
]

interface RoleModelConfigSectionProps {
    fetchedModels?: string[]
    onRefreshRequired?: () => void
}

export default function RoleModelConfigSection({ fetchedModels = [] }: RoleModelConfigSectionProps) {
    const [loading, setLoading] = useState(false)
    const [saving, setSaving] = useState(false)
    const [presetLoading, setPresetLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [successMsg, setSuccessMsg] = useState<string | null>(null)

    const [providers, setProviders] = useState<Provider[]>([])
    const [profiles, setProfiles] = useState<ModelProfile[]>([])
    const [bindings, setBindings] = useState<RoleBinding[]>([])
    const [resolvedRoles, setResolvedRoles] = useState<Record<string, ResolvedRole>>({})

    // Draft role bindings state: role_key -> profile_id (empty string means inherit)
    const [draftBindings, setDraftBindings] = useState<Record<string, string>>({})
    const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({})
    const [newCustomModel, setNewCustomModel] = useState('')
    const [addingCustomModel, setAddingCustomModel] = useState(false)

    // Heterogeneous selection modal/inline states for Presets
    const [selectedPreset, setSelectedPreset] = useState<'single' | 'bull_bear_hetero' | 'three_way_hetero'>('single')
    const [presetBullProfile, setPresetBullProfile] = useState<string>('')
    const [presetBearProfile, setPresetBearProfile] = useState<string>('')
    const [presetManagerProfile, setPresetManagerProfile] = useState<string>('')

    const fetchData = async () => {
        setLoading(true)
        setError(null)
        try {
            const [provRes, profRes, bindRes, resRes] = await Promise.all([
                api.getProviders(),
                api.getModelProfiles(),
                api.getRoleBindings(),
                api.getResolvedRoles(),
            ])
            setProviders(provRes)
            setProfiles(profRes)
            setBindings(bindRes)
            setResolvedRoles(resRes)

            // Populate draft bindings
            const initialDraft: Record<string, string> = {}
            bindRes.forEach((b) => {
                if (b.target_type === 'role') {
                    initialDraft[b.target_key] = b.model_profile_id
                }
            })
            setDraftBindings(initialDraft)
        } catch (err: any) {
            console.error('Failed to load role routing data', err)
            setError(err.message || '加载分角色模型配置失败')
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        fetchData()
    }, [])

    useEffect(() => {
        if (fetchedModels && fetchedModels.length > 0) {
            api.syncModelProfiles(fetchedModels)
                .then((updatedProfiles) => {
                    setProfiles(updatedProfiles)
                })
                .catch((err) => console.warn('Failed auto sync fetched models', err))
        }
    }, [fetchedModels])

    const handleAddCustomModel = async () => {
        const trimmed = newCustomModel.trim()
        if (!trimmed) return
        setAddingCustomModel(true)
        setError(null)
        try {
            await api.createModelProfile({
                model_name: trimmed,
                display_name: trimmed,
            })
            setNewCustomModel('')
            setSuccessMsg("模型 Profile 已成功添加")
            await fetchData()
        } catch (err: any) {
            setError(err.message || '创建模型 Profile 失败')
        } finally {
            setAddingCustomModel(false)
        }
    }


    const handleSaveBindings = async () => {
        setSaving(true)
        setError(null)
        setSuccessMsg(null)
        try {
            const items: RoleBindingItem[] = Object.entries(draftBindings)
                .filter(([_, profileId]) => Boolean(profileId))
                .map(([roleKey, profileId]) => ({
                    target_type: 'role',
                    target_key: roleKey,
                    model_profile_id: profileId,
                }))

            await api.updateRoleBindings(items)
            setSuccessMsg('分角色模型配置已保存')
            await fetchData()
        } catch (err: any) {
            setError(err.message || '保存配置失败')
        } finally {
            setSaving(false)
        }
    }

    const handleApplyPreset = async (presetMode: 'single' | 'bull_bear_hetero' | 'three_way_hetero') => {
        setPresetLoading(true)
        setError(null)
        setSuccessMsg(null)
        try {
            let payload: Record<string, any> = {}
            if (presetMode === 'bull_bear_hetero' || presetMode === 'three_way_hetero') {
                if (presetBullProfile) payload.bull_profile_id = presetBullProfile
                if (presetBearProfile) payload.bear_profile_id = presetBearProfile
            }
            if (presetMode === 'three_way_hetero') {
                if (presetManagerProfile) payload.manager_profile_id = presetManagerProfile
            }

            await api.applyPreset(presetMode, payload)
            setSelectedPreset(presetMode)
            setSuccessMsg(`预设「${presetMode === 'single' ? '单模型' : presetMode === 'bull_bear_hetero' ? '多空异构' : '三方异构'}」应用成功`)
            await fetchData()
        } catch (err: any) {
            setError(err.message || '应用预设失败')
        } finally {
            setPresetLoading(false)
        }
    }

    const toggleGroup = (groupKey: string) => {
        setCollapsedGroups((prev) => ({ ...prev, [groupKey]: !prev[groupKey] }))
    }

    // Estimate Cost & Tokens per analysis run
    const estimatedTokens = useMemo(() => {
        // ~7 analysts quick runs (~3k tokens each) = 21k
        // ~2 debate rounds (bull + bear x 2) = 15k
        // ~1 risk assessment = 5k
        // ~1 research manager synthesis = 8k
        return {
            quickCalls: 12,
            deepCalls: 3,
            minTokens: 35000,
            maxTokens: 75000,
        }
    }, [])

    return (
        <div className="card space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-2">
                    <Users className="w-5 h-5 text-indigo-500" />
                    <div>
                        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                            分角色模型配置 (Role-Based Model Routing)
                        </h2>
                        <p className="text-xs text-slate-500 dark:text-slate-400">
                            支持为多头、空头、研究总监等独立指定不同厂商的模型，规避同厂商模型的认知盲点共性。
                        </p>
                    </div>
                </div>
                <button
                    onClick={fetchData}
                    disabled={loading}
                    className="p-2 rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
                    title="刷新配置"
                >
                    <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                </button>
            </div>

            {error && (
                <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-600 dark:bg-rose-950/30 dark:border-rose-900/50 dark:text-rose-400 text-sm">
                    ⚠️ {error}
                </div>
            )}

            {successMsg && (
                <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-700 dark:bg-emerald-950/30 dark:border-emerald-900/50 dark:text-emerald-300 text-sm flex items-center gap-2">
                    <Check className="w-4 h-4 text-emerald-500" />
                    <span>{successMsg}</span>
                </div>
            )}

            {/* Info Banner explaining global defaults vs per-role overrides */}
            <div className="flex items-start gap-3 rounded-xl border border-blue-200/80 bg-blue-50/80 p-4 text-xs text-blue-900 dark:border-blue-900/50 dark:bg-blue-950/30 dark:text-blue-200">
                <Info className="mt-0.5 w-4 h-4 shrink-0 text-blue-500" />
                <div className="space-y-1.5 leading-relaxed">
                    <div className="font-semibold text-sm text-blue-950 dark:text-blue-100 flex items-center gap-1.5">
                        💡 常规/推理模型 与 分角色模型 协同工作说明（无冲突）
                    </div>
                    <p>
                        • <strong>常规模型 / 推理模型</strong>（设置页面上方）：作为<span className="font-semibold underline">全局默认兜底模型</span>。未单独指定角色的分析师将自动使用这里的模型。
                    </p>
                    <p>
                        • <strong>分角色模型</strong>（本区域下方）：用于为特定角色（如多头研究员、空头研究员）<span className="font-semibold underline">单独指定模型</span>（例如 Gemini、DeepSeek、Qwen）。选“继承 Tier 默认”则自动沿用常规/推理模型。
                    </p>
                    <p className="text-blue-700 dark:text-blue-300 font-medium">
                        两者层级分明、互为补充：全局默认兜底，角色精准覆盖，没有任何冲突！
                    </p>
                </div>
            </div>

            {/* Quick Custom Model Add Form */}
            <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-xl border border-slate-200/80 bg-slate-50/80 dark:border-slate-800 dark:bg-slate-900/50">
                <div className="flex items-center gap-2">
                    <Plus className="w-4 h-4 text-indigo-500" />
                    <span className="text-xs text-slate-700 dark:text-slate-300 font-medium">
                        手动添加/扩充模型 Profile：
                    </span>
                </div>
                <div className="flex items-center gap-2 flex-1 max-w-md">
                    <input
                        type="text"
                        value={newCustomModel}
                        onChange={(e) => setNewCustomModel(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleAddCustomModel()}
                        placeholder="输入模型 ID (如 gemini-2.5-flash 或 claude-3-5-sonnet)"
                        className="text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-3 py-1.5 w-full text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                    />
                    <button
                        type="button"
                        onClick={handleAddCustomModel}
                        disabled={addingCustomModel || !newCustomModel.trim()}
                        className="btn-secondary text-xs px-3 py-1.5 inline-flex items-center gap-1 shrink-0 disabled:opacity-50"
                    >
                        {addingCustomModel ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Plus className="w-3.5 h-3.5" />}
                        添加模型 Profile
                    </button>
                </div>
            </div>

            {/* Section 1: Quick Presets */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/50 p-4 space-y-3">
                <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                        <Sliders className="w-4 h-4 text-indigo-500" /> 一键预设模式
                    </span>
                    <span className="text-xs text-slate-400">点击按钮快速套用推荐角色组合</span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <button
                        type="button"
                        onClick={() => handleApplyPreset('single')}
                        disabled={presetLoading}
                        className={`p-3 rounded-xl border text-left transition-all ${
                            selectedPreset === 'single'
                                ? 'border-indigo-500 bg-indigo-50/50 dark:bg-indigo-950/30 dark:border-indigo-600'
                                : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-950 hover:border-slate-300 dark:hover:border-slate-600'
                        }`}
                    >
                        <div className="font-semibold text-sm text-slate-800 dark:text-slate-200">
                            🎯 单模型 (Single)
                        </div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                            所有角色遵循全局默认，兼容传统单厂商模式
                        </div>
                    </button>

                    <button
                        type="button"
                        onClick={() => handleApplyPreset('bull_bear_hetero')}
                        disabled={presetLoading}
                        className={`p-3 rounded-xl border text-left transition-all ${
                            selectedPreset === 'bull_bear_hetero'
                                ? 'border-indigo-500 bg-indigo-50/50 dark:bg-indigo-950/30 dark:border-indigo-600'
                                : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-950 hover:border-slate-300 dark:hover:border-slate-600'
                        }`}
                    >
                        <div className="font-semibold text-sm text-slate-800 dark:text-slate-200 flex items-center gap-1">
                            ⚖️ 多空异构 (Bull/Bear)
                        </div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                            多头与空头使用不同模型，提振辩论对抗真实度
                        </div>
                    </button>

                    <button
                        type="button"
                        onClick={() => handleApplyPreset('three_way_hetero')}
                        disabled={presetLoading}
                        className={`p-3 rounded-xl border text-left transition-all ${
                            selectedPreset === 'three_way_hetero'
                                ? 'border-indigo-500 bg-indigo-50/50 dark:bg-indigo-950/30 dark:border-indigo-600'
                                : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-950 hover:border-slate-300 dark:hover:border-slate-600'
                        }`}
                    >
                        <div className="font-semibold text-sm text-slate-800 dark:text-slate-200">
                            🔺 三方异构 (Three-Way)
                        </div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                            多头、空头与裁决总监三者完全独立模型
                        </div>
                    </button>
                </div>

                {/* Preset model profile selectors if heterogenous selected */}
                {profiles.length > 0 && (
                    <div className="pt-2 grid grid-cols-1 sm:grid-cols-3 gap-3 border-t border-slate-200/60 dark:border-slate-800">
                        <div>
                            <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                                🟢 多头研究员模型
                            </label>
                            <select
                                value={presetBullProfile}
                                onChange={(e) => setPresetBullProfile(e.target.value)}
                                className="w-full text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 p-2"
                            >
                                <option value="">（选择多头模型 Profile）</option>
                                {profiles.map((p) => (
                                    <option key={p.id} value={p.id}>
                                        [{p.provider_display_name || p.provider_type}] {p.display_name} ({p.model_name})
                                    </option>
                                ))}
                            </select>
                        </div>
                        <div>
                            <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                                🔴 空头研究员模型
                            </label>
                            <select
                                value={presetBearProfile}
                                onChange={(e) => setPresetBearProfile(e.target.value)}
                                className="w-full text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 p-2"
                            >
                                <option value="">（选择空头模型 Profile）</option>
                                {profiles.map((p) => (
                                    <option key={p.id} value={p.id}>
                                        [{p.provider_display_name || p.provider_type}] {p.display_name} ({p.model_name})
                                    </option>
                                ))}
                            </select>
                        </div>
                        <div>
                            <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                                ⚖️ 研究总监模型 (裁决)
                            </label>
                            <select
                                value={presetManagerProfile}
                                onChange={(e) => setPresetManagerProfile(e.target.value)}
                                className="w-full text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 p-2"
                            >
                                <option value="">（选择研究总监 Profile）</option>
                                {profiles.map((p) => (
                                    <option key={p.id} value={p.id}>
                                        [{p.provider_display_name || p.provider_type}] {p.display_name} ({p.model_name})
                                    </option>
                                ))}
                            </select>
                        </div>
                    </div>
                )}
            </div>

            {/* Heterogeneity Alert Banner */}
            <div className="flex items-start gap-2.5 rounded-xl border border-amber-200 bg-amber-50/80 px-4 py-3 text-xs text-amber-800 dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-300">
                <AlertTriangle className="mt-0.5 w-4 h-4 shrink-0 text-amber-500" />
                <div className="leading-relaxed">
                    <span className="font-semibold">异构模型提示：</span>
                    多空使用不同模型时，辩论结果可能受两个模型的表达能力差异影响，而不只是证据强弱。建议配合镜像运行对照。
                </div>
            </div>

            {/* Role Table */}
            <div className="space-y-4">
                <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
                        <Cpu className="w-4 h-4 text-purple-500" /> 角色模型绑定明细与 Fallback 状态
                    </h3>
                    <span className="text-xs text-slate-400">
                        厂商 {providers.length} | Profile {profiles.length} | 显式绑定 {bindings.length} | 角色 {Object.keys(ROLE_LABELS).length}
                    </span>
                </div>

                <div className="border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden divide-y divide-slate-200 dark:divide-slate-800">
                    {ROLE_GROUPS.map((group) => {
                        const isCollapsed = collapsedGroups[group.key]
                        return (
                            <div key={group.key} className="bg-white dark:bg-slate-950">
                                {/* Group Header */}
                                <button
                                    type="button"
                                    onClick={() => toggleGroup(group.key)}
                                    className="w-full flex items-center justify-between px-4 py-3 bg-slate-50 dark:bg-slate-900/80 hover:bg-slate-100 dark:hover:bg-slate-900 transition-colors text-left"
                                >
                                    <div className="flex items-center gap-2">
                                        {isCollapsed ? (
                                            <ChevronRight className="w-4 h-4 text-slate-400" />
                                        ) : (
                                            <ChevronDown className="w-4 h-4 text-slate-400" />
                                        )}
                                        <span className="font-semibold text-xs text-slate-700 dark:text-slate-300">
                                            {group.title}
                                        </span>
                                        <span className="text-[11px] text-slate-400 font-normal">
                                            ({group.roles.length} 个角色)
                                        </span>
                                    </div>
                                    <span className="text-[11px] text-slate-400 hidden sm:inline">
                                        {group.desc}
                                    </span>
                                </button>

                                {/* Group Roles */}
                                {!isCollapsed && (
                                    <div className="divide-y divide-slate-100 dark:divide-slate-800/60">
                                        {group.roles.map((roleKey) => {
                                            const roleMeta = ROLE_LABELS[roleKey] || {
                                                name: roleKey,
                                                desc: '',
                                            }
                                            const currentBound = draftBindings[roleKey] || ''
                                            const resolved = resolvedRoles[roleKey]

                                            return (
                                                <div
                                                    key={roleKey}
                                                    className="px-4 py-3 grid grid-cols-1 md:grid-cols-12 gap-3 items-center hover:bg-slate-50/50 dark:hover:bg-slate-900/30 transition-colors"
                                                >
                                                    {/* Role Info */}
                                                    <div className="md:col-span-4">
                                                        <div className="font-medium text-xs text-slate-800 dark:text-slate-200">
                                                            {roleMeta.name}
                                                            <span className="ml-1 text-[10px] text-slate-400 font-mono">
                                                                ({roleKey})
                                                            </span>
                                                        </div>
                                                        <div className="text-[11px] text-slate-500 dark:text-slate-400 truncate">
                                                            {roleMeta.desc}
                                                        </div>
                                                    </div>

                                                    {/* Model Profile Selector */}
                                                    <div className="md:col-span-4">
                                                        <select
                                                            value={currentBound}
                                                            onChange={(e) =>
                                                                setDraftBindings((prev) => ({
                                                                    ...prev,
                                                                    [roleKey]: e.target.value,
                                                                }))
                                                            }
                                                            className="w-full text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 p-2 text-slate-800 dark:text-slate-200"
                                                        >
                                                            <option value="">
                                                                (继承 Tier 默认 / 全局降级)
                                                            </option>
                                                            {profiles.map((p) => (
                                                                <option key={p.id} value={p.id}>
                                                                    [{p.provider_display_name || p.provider_type}] {p.display_name} ({p.model_name})
                                                                </option>
                                                            ))}
                                                        </select>
                                                    </div>

                                                    {/* Effective Resolved Model Badge */}
                                                    <div className="md:col-span-4 flex items-center justify-between md:justify-end gap-2 text-xs">
                                                        {resolved ? (
                                                            <div className="text-right">
                                                                <div className="font-mono text-[11px] text-indigo-600 dark:text-indigo-400 font-medium">
                                                                    {resolved.provider_type}: {resolved.model_name}
                                                                </div>
                                                                <div className="text-[10px] text-slate-400 italic">
                                                                    解析路径: {resolved.resolved_via}{' '}
                                                                    {resolved.fallback_used ? ' (降级回退)' : ''}
                                                                </div>
                                                            </div>
                                                        ) : (
                                                            <span className="text-[11px] text-slate-400 italic">
                                                                尚未解析
                                                            </span>
                                                        )}
                                                    </div>
                                                </div>
                                            )
                                        })}
                                    </div>
                                )}
                            </div>
                        )
                    })}
                </div>
            </div>

            {/* Cost & Token Estimation Card */}
            <div className="rounded-xl border border-slate-200/80 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-900/60 p-4 space-y-2">
                <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-1">
                        <Info className="w-3.5 h-3.5 text-blue-500" /> 单次分析开销预估
                    </span>
                    <span className="text-[11px] font-mono text-slate-500">
                        估算消耗: {estimatedTokens.minTokens.toLocaleString()} ~ {estimatedTokens.maxTokens.toLocaleString()} Tokens / 次
                    </span>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs text-slate-600 dark:text-slate-400">
                    <div className="p-2 rounded-lg bg-white dark:bg-slate-950 border border-slate-200/60 dark:border-slate-800">
                        <div className="text-[10px] text-slate-400">Quick Tier 调用</div>
                        <div className="font-semibold text-slate-800 dark:text-slate-200">约 {estimatedTokens.quickCalls} 次</div>
                    </div>
                    <div className="p-2 rounded-lg bg-white dark:bg-slate-950 border border-slate-200/60 dark:border-slate-800">
                        <div className="text-[10px] text-slate-400">Deep Tier 调用</div>
                        <div className="font-semibold text-slate-800 dark:text-slate-200">约 {estimatedTokens.deepCalls} 次</div>
                    </div>
                    <div className="p-2 rounded-lg bg-white dark:bg-slate-950 border border-slate-200/60 dark:border-slate-800">
                        <div className="text-[10px] text-slate-400">多空辩论轮次</div>
                        <div className="font-semibold text-slate-800 dark:text-slate-200">1 ~ 3 轮可调</div>
                    </div>
                    <div className="p-2 rounded-lg bg-white dark:bg-slate-950 border border-slate-200/60 dark:border-slate-800">
                        <div className="text-[10px] text-slate-400">预估单次成本</div>
                        <div className="font-semibold text-emerald-600 dark:text-emerald-400">根据绑定的模型价格判定</div>
                    </div>
                </div>
            </div>

            {/* Action Buttons */}
            <div className="flex justify-end gap-3 pt-2">
                <button
                    type="button"
                    onClick={handleSaveBindings}
                    disabled={saving || loading}
                    className="btn-primary inline-flex items-center gap-2 text-xs px-4 py-2"
                >
                    {saving ? (
                        <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            正在保存分角色配置...
                        </>
                    ) : (
                        <>
                            <Check className="w-4 h-4" />
                            保存角色模型绑定
                        </>
                    )}
                </button>
            </div>
        </div>
    )
}
