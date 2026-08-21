"""动态知识库 RAG 检索与索引模块 (Knowledge Base Dynamic RAG Retrieval & Indexing).

本模块实现基于 27 行业全维图谱与 19 宏观情景三级传导图谱的高性能本地 RAG 检索：
1. 本地结构化倒排索引 (Field-Weighted Inverted Index + BM25/TF-IDF + Domain Vocabulary)；
2. 标的/股票代码/行业名称/宏观政策/产业链传导多通道检索；
3. 严格遵循产品契约：
   - 检索源权威对齐 `tradingagents.knowledge` 现有 27 行业 + 19 宏观情景，零另起文本；
   - 零新增 pip 依赖，零付费/云端 API，纯本地高效向量/关键词倒排索引；
   - 缺命中统一格式化输出 `【知识库未命中】`，严禁产生幻觉或编造行业事实；
   - 提供丰富完备的结构化检索与 Prompt 注入辅助函数。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from tradingagents.knowledge.industry_linkage import (
    INDUSTRY_PROFILES,
    IndustryProfile,
    format_industry_deep_context,
    get_industry_profile,
    search_industries,
)
from tradingagents.knowledge.macro_events import (
    MACRO_EVENT_SCENARIOS,
    MacroEventScenario,
    format_macro_event_context,
    get_macro_event_scenario,
    match_events_from_text,
    search_macro_events,
)

# 统一未命中提示常量
KNOWLEDGE_MISSING_FALLBACK: str = "【知识库未命中】"
INDUSTRY_KNOWLEDGE_MISSING_BLOCK: str = "【行业常识知识库】\n【知识库未命中】"
MACRO_EVENT_MISSING_BLOCK: str = "【宏观事件传导图谱】\n【知识库未命中】"

# 常见权威金融与宏观英文代码词表
_KNOWN_FINANCE_EN_CODES: Set[str] = {
    "sox", "cxo", "lpr", "mlf", "wti", "brent", "oled", "asp", "tips", "dxy",
    "fomc", "pmi", "cpi", "ppi", "nim", "roe", "capex", "aisc", "ivd", "eda",
    "ic", "shibor", "dr007", "cbam", "psy", "msy", "tc", "rc", "ev", "ebitda",
    "pe", "pb", "ps", "gpr", "wgc", "eia", "shfe", "lme", "comex",
}

# 停用单字集合（避免生成无意义跨词组合）
_STOP_CHARS: Set[str] = {
    "的", "了", "和", "是", "在", "有", "对", "与", "及", "等", "其", "此", "为", "于",
    "这", "那", "就", "中", "上", "下", "个", "也", "都", "要", "将", "从", "到", "以",
    "按", "把", "被", "让", "给", "但", "而", "或", "之", "后", "前", "已", "并", "向",
    "由", "更", "很", "最", "非常", "一", "二", "三", "四", "五", "年", "月", "日",
}

# 中文及金融通用停用词集合（过滤无判别力的泛化虚词与通用行政词汇）
_STOP_WORDS: Set[str] = {
    "", " ", "\t", "\n", "\r", "公司", "企业", "股份", "集团", "有限", "发布", "公告",
    "日常", "经营", "员工", "培训", "例行", "关于", "通知", "报告", "会议", "召开",
    "完成", "情况", "工作", "推进", "组织", "活动", "人员", "管理", "披露", "提示",
    "说明", "要求", "显示", "整体", "表示", "预计", "实现", "主要", "部分", "涉及",
    "影响", "未知", "xyz", "系统", "维护", "保洁", "记录", "服务", "支持", "发展",
    "推动", "规定", "建设", "项目", "落实", "指导", "加强", "采矿", "行业", "领域",
    "市场", "国家", "国内", "海外", "全球", "重点", "指标", "水平", "走势", "趋势",
    "阶段", "行政", "周度", "内部", "后勤", "总结", "上周", "今日", "巡查", "标的",
    "外星", "比赛", "摄影", "工会", "庆祝", "生日", "结束", "圆满", "开启", "开展",
    "举行", "深入", "持续", "进一步", "全面", "一般", "通常", "相关", "方面", "来看",
    "显著", "大幅", "稳步", "加快", "探测", "种植", "采选", "制造", "使用", "利用",
    "处理", "加工",
}


def is_chinese_token(s: str) -> bool:
    """判断字符串是否由纯中文字符构成。"""
    return bool(s) and all("一" <= ch <= "鿿" for ch in s)


def _extract_domain_vocabulary(
    industry_profiles: Dict[str, IndustryProfile],
    macro_scenarios: Dict[str, MacroEventScenario],
) -> List[str]:
    """从知识库图谱中自动抽取全部专业术语并按长度降序排序。"""
    vocab: Set[str] = set()

    def _add_terms(t: Any) -> None:
        if not t:
            return
        clean = re.sub(r'[/,，、；;（）()| \-\n\t:：\"\'“”]+', ' ', str(t)).strip().lower()
        for word in clean.split():
            if is_chinese_token(word):
                if len(word) >= 2 and word not in _STOP_WORDS and not any(ch in _STOP_CHARS for ch in word):
                    vocab.add(word)
                for n in (2, 3, 4):
                    if len(word) >= n:
                        for i in range(len(word) - n + 1):
                            sub = word[i : i + n]
                            if sub not in _STOP_WORDS and not any(ch in _STOP_CHARS for ch in sub):
                                vocab.add(sub)
            else:
                if re.match(r"^[a-z0-9_]{2,}$", word) and word not in _STOP_WORDS:
                    vocab.add(word)

    for p in industry_profiles.values():
        _add_terms(p.industry_id)
        _add_terms(p.industry_name)
        _add_terms(p.category)
        for a in p.aliases:
            _add_terms(a)
        for s in p.representative_segments:
            _add_terms(s)
        for u in p.upstream:
            _add_terms(u)
        for d in p.downstream:
            _add_terms(d)
        for c in p.core_inputs:
            _add_terms(c)
        for pol in p.macro_sensitivity.policy_drivers:
            _add_terms(pol)
        for k in p.cycle_profile.key_cycle_indicators:
            _add_terms(k)
        for m in p.key_metrics:
            _add_terms(m)
        for r in (
            p.risks.geopolitical
            + p.risks.supply_chain_bottlenecks
            + p.risks.technology_substitution
            + p.risks.policy_regulatory
            + p.risks.demand_cliff
        ):
            _add_terms(r)

    for s in macro_scenarios.values():
        _add_terms(s.event_id)
        _add_terms(s.event_name)
        _add_terms(s.category)
        for a in s.aliases:
            _add_terms(a)
        for imp in s.direct_impact:
            _add_terms(imp)
        for b in s.beneficiary_sectors:
            _add_terms(b.sector)
            for d in b.key_drivers:
                _add_terms(d)
        for a in s.adversely_affected_sectors:
            _add_terms(a.sector)
            for d in a.key_drivers:
                _add_terms(d)
        for k in s.key_monitoring_indicators:
            _add_terms(k)
        for h in s.historical_reference_cases:
            _add_terms(h)

    # 过滤停用词并按长度降序排列
    filtered = [v for v in vocab if v not in _STOP_WORDS and len(v) >= 2]
    filtered.sort(key=lambda x: len(x), reverse=True)
    return filtered


def tokenize_cn_en(
    text: str,
    domain_vocab: Optional[Sequence[str]] = None,
) -> List[str]:
    """对输入文本基于金融领域词汇库进行高精准度分词。

    提取内容：
    1. 6 位证券代码与已知英文金融代码（如 '600519', 'SOX', 'CXO', 'LPR', 'MLF', 'WTI', 'OLED'）；
    2. 知识库领域专业词汇匹配（如 '半导体', '光刻机', '碳酸锂', '降准', '降息', '红海', '特别国债' 等）。
    """
    if not text or not isinstance(text, str):
        return []

    clean_text = text.strip()
    if not clean_text:
        return []

    text_lower = clean_text.lower()
    matched_tokens: List[str] = []
    seen: Set[str] = set()

    # 1. 提取 6 位证券代码与英文金融代码
    for m in re.findall(r"[A-Za-z0-9_]+", text_lower):
        if m not in _STOP_WORDS and m not in seen:
            if re.match(r"^\d{6}$", m) or m in _KNOWN_FINANCE_EN_CODES:
                seen.add(m)
                matched_tokens.append(m)

    # 2. 匹配领域词表
    vocab = domain_vocab or get_global_rag_index().domain_vocabulary
    for term in vocab:
        if is_chinese_token(term) and term in text_lower:
            if term not in seen:
                seen.add(term)
                matched_tokens.append(term)
        elif not is_chinese_token(term) and term in text_lower:
            if term not in seen and (term in _KNOWN_FINANCE_EN_CODES or re.match(r"^\d{6}$", term)):
                seen.add(term)
                matched_tokens.append(term)

    return matched_tokens


@dataclass
class _DocumentIndex:
    doc_id: str
    doc_type: str  # "industry" or "macro_event"
    obj: Any
    name: str
    aliases: List[str]
    term_frequencies: Dict[str, float] = field(default_factory=dict)
    doc_length: float = 0.0


class KnowledgeRAGIndex:
    """本地多维知识库倒排索引与 BM25-TFIDF 复合检索引擎。"""

    def __init__(
        self,
        industry_profiles: Optional[Dict[str, IndustryProfile]] = None,
        macro_scenarios: Optional[Dict[str, MacroEventScenario]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.industry_profiles = industry_profiles or INDUSTRY_PROFILES
        self.macro_scenarios = macro_scenarios or MACRO_EVENT_SCENARIOS

        self.domain_vocabulary: List[str] = _extract_domain_vocabulary(
            self.industry_profiles, self.macro_scenarios
        )

        self.industry_docs: Dict[str, _DocumentIndex] = {}
        self.macro_docs: Dict[str, _DocumentIndex] = {}

        self.industry_df: Dict[str, int] = {}
        self.macro_df: Dict[str, int] = {}

        self.industry_avgdl: float = 0.0
        self.macro_avgdl: float = 0.0

        self._build_index()

    def _build_index(self) -> None:
        """构建行业与宏观事件的多字段加权倒排索引。"""
        # 1. 构建行业图谱索引
        total_ind_len = 0.0
        for ind_id, p in self.industry_profiles.items():
            tf_map: Dict[str, float] = {}

            def _add_field_tokens(text: str, weight: float) -> None:
                for token in tokenize_cn_en(text, self.domain_vocabulary):
                    tf_map[token] = tf_map.get(token, 0.0) + weight

            _add_field_tokens(p.industry_id, 15.0)
            _add_field_tokens(p.industry_name, 20.0)
            _add_field_tokens(p.category, 8.0)
            for alias in p.aliases:
                _add_field_tokens(alias, 15.0)
            for seg in p.representative_segments:
                _add_field_tokens(seg, 10.0)
            for pol in p.macro_sensitivity.policy_drivers:
                _add_field_tokens(pol, 8.0)
            for up in p.upstream:
                _add_field_tokens(up, 5.0)
            for down in p.downstream:
                _add_field_tokens(down, 5.0)
            for inp in p.core_inputs:
                _add_field_tokens(inp, 4.0)
            _add_field_tokens(p.pricing_power, 3.0)
            _add_field_tokens(p.macro_sensitivity.global_macro_linkage, 3.0)
            for r in (
                p.risks.geopolitical
                + p.risks.supply_chain_bottlenecks
                + p.risks.technology_substitution
                + p.risks.policy_regulatory
                + p.risks.demand_cliff
            ):
                _add_field_tokens(r, 4.0)
            for ind_metric in p.cycle_profile.key_cycle_indicators:
                _add_field_tokens(ind_metric, 4.0)
            for metric in p.key_metrics:
                _add_field_tokens(metric, 3.0)

            doc_len = sum(tf_map.values())
            doc_idx = _DocumentIndex(
                doc_id=ind_id,
                doc_type="industry",
                obj=p,
                name=p.industry_name,
                aliases=list(p.aliases),
                term_frequencies=tf_map,
                doc_length=doc_len,
            )
            self.industry_docs[ind_id] = doc_idx
            total_ind_len += doc_len

            for term in tf_map.keys():
                self.industry_df[term] = self.industry_df.get(term, 0) + 1

        if self.industry_docs:
            self.industry_avgdl = total_ind_len / len(self.industry_docs)

        # 2. 构建宏观事件索引
        total_macro_len = 0.0
        for ev_id, s in self.macro_scenarios.items():
            tf_map_macro: Dict[str, float] = {}

            def _add_macro_tokens(text: str, weight: float) -> None:
                for token in tokenize_cn_en(text, self.domain_vocabulary):
                    tf_map_macro[token] = tf_map_macro.get(token, 0.0) + weight

            _add_macro_tokens(s.event_id, 15.0)
            _add_macro_tokens(s.event_name, 20.0)
            _add_macro_tokens(s.category, 8.0)
            for alias in s.aliases:
                _add_macro_tokens(alias, 15.0)
            _add_macro_tokens(s.description, 4.0)
            for step in s.transmission_mechanism:
                _add_macro_tokens(step, 6.0)
            for imp in s.direct_impact:
                _add_macro_tokens(imp, 8.0)
            for b in s.beneficiary_sectors:
                _add_macro_tokens(b.sector, 6.0)
                _add_macro_tokens(b.transmission_logic, 4.0)
                for d in b.key_drivers:
                    _add_macro_tokens(d, 5.0)
            for a in s.adversely_affected_sectors:
                _add_macro_tokens(a.sector, 6.0)
                _add_macro_tokens(a.transmission_logic, 4.0)
                for d in a.key_drivers:
                    _add_macro_tokens(d, 5.0)
            for mkt, desc in s.cross_market_spillovers.items():
                _add_macro_tokens(mkt, 5.0)
                _add_macro_tokens(desc, 4.0)
            for kmi in s.key_monitoring_indicators:
                _add_macro_tokens(kmi, 4.0)
            for hrc in s.historical_reference_cases:
                _add_macro_tokens(hrc, 3.0)

            doc_len_macro = sum(tf_map_macro.values())
            doc_idx_macro = _DocumentIndex(
                doc_id=ev_id,
                doc_type="macro_event",
                obj=s,
                name=s.event_name,
                aliases=list(s.aliases),
                term_frequencies=tf_map_macro,
                doc_length=doc_len_macro,
            )
            self.macro_docs[ev_id] = doc_idx_macro
            total_macro_len += doc_len_macro

            for term in tf_map_macro.keys():
                self.macro_df[term] = self.macro_df.get(term, 0) + 1

        if self.macro_docs:
            self.macro_avgdl = total_macro_len / len(self.macro_docs)

    def _score_bm25(
        self,
        query_tokens: List[str],
        doc_idx: _DocumentIndex,
        df_map: Dict[str, int],
        total_docs: int,
        avgdl: float,
        raw_query: str = "",
    ) -> float:
        """计算查询与文档的加权 BM25 得分及精确字串匹配加权。"""
        if not query_tokens:
            return 0.0

        score = 0.0
        doc_len = doc_idx.doc_length
        tf_map = doc_idx.term_frequencies

        # 1. 词项 BM25 打分
        for token in query_tokens:
            if token not in tf_map:
                continue
            tf = tf_map[token]
            df = df_map.get(token, 0)
            idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            if idf < 0.1:
                idf = 0.1

            token_len_mult = 1.0 if len(token) <= 2 else (1.5 if len(token) <= 4 else 2.0)
            denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (avgdl or 1.0)))
            if denom > 0:
                score += (idf * (tf * (self.k1 + 1.0)) / denom) * token_len_mult

        # 2. 精确全名/别名/ID 加权（当 raw_query 明确命中时）
        if raw_query and score > 0:
            q_lower = raw_query.lower()
            if doc_idx.name.lower() in q_lower:
                score += 30.0
            for alias in doc_idx.aliases:
                a_lower = alias.lower()
                if a_lower and len(a_lower) >= 2 and a_lower not in _STOP_WORDS:
                    if a_lower in q_lower:
                        score += 20.0
                        break
            if doc_idx.doc_id.lower() in q_lower:
                score += 30.0

        return score

    def retrieve_industries(
        self,
        query: str,
        top_k: int = 1,
        min_score: float = 3.0,
    ) -> List[Tuple[IndustryProfile, float]]:
        """根据查询文本检索最相关的行业图谱。"""
        if not query or not isinstance(query, str):
            return []

        q_clean = query.strip()
        if not q_clean:
            return []

        # 短词精确匹配 ID 或 名称/别名
        if len(q_clean) <= 12:
            exact_prof = get_industry_profile(q_clean)
            if exact_prof:
                return [(exact_prof, 100.0)]

        q_tokens = tokenize_cn_en(q_clean, self.domain_vocabulary)
        if not q_tokens:
            return []

        total_docs = len(self.industry_docs)
        scores: List[Tuple[IndustryProfile, float]] = []

        for doc_id, doc_idx in self.industry_docs.items():
            sc = self._score_bm25(
                query_tokens=q_tokens,
                doc_idx=doc_idx,
                df_map=self.industry_df,
                total_docs=total_docs,
                avgdl=self.industry_avgdl,
                raw_query=q_clean,
            )
            if sc >= min_score:
                scores.append((doc_idx.obj, sc))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def retrieve_industry(
        self,
        query: str,
        min_score: float = 3.0,
    ) -> Optional[IndustryProfile]:
        """检索单个最相关的行业图谱（若未命中返回 None）。"""
        results = self.retrieve_industries(query=query, top_k=1, min_score=min_score)
        if results:
            return results[0][0]
        return None

    def retrieve_macro_events(
        self,
        query: str,
        top_k: int = 2,
        min_score: float = 3.0,
    ) -> List[Tuple[MacroEventScenario, float]]:
        """根据查询文本检索最相关的宏观事件情景。"""
        if not query or not isinstance(query, str):
            return []

        q_clean = query.strip()
        if not q_clean:
            return []

        # 短词精确匹配
        if len(q_clean) <= 16:
            exact_sc = get_macro_event_scenario(q_clean)
            if exact_sc:
                return [(exact_sc, 100.0)]

        # 先收集文本中直接命中的情景
        direct_matched = match_events_from_text(q_clean)
        direct_ids = {s.event_id for s in direct_matched}

        q_tokens = tokenize_cn_en(q_clean, self.domain_vocabulary)
        if not q_tokens and not direct_matched:
            return []

        total_docs = len(self.macro_docs)
        scores: List[Tuple[MacroEventScenario, float]] = []

        for doc_id, doc_idx in self.macro_docs.items():
            sc = self._score_bm25(
                query_tokens=q_tokens,
                doc_idx=doc_idx,
                df_map=self.macro_df,
                total_docs=total_docs,
                avgdl=self.macro_avgdl,
                raw_query=q_clean,
            )
            if doc_id in direct_ids:
                sc += 35.0

            if sc >= min_score:
                scores.append((doc_idx.obj, sc))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def retrieve_macro_event(
        self,
        query: str,
        min_score: float = 3.0,
    ) -> Optional[MacroEventScenario]:
        """检索单个最相关的宏观事件情景（若未命中返回 None）。"""
        results = self.retrieve_macro_events(query=query, top_k=1, min_score=min_score)
        if results:
            return results[0][0]
        return None


# 全局默认索引单例
_GLOBAL_RAG_INDEX: Optional[KnowledgeRAGIndex] = None


def get_global_rag_index() -> KnowledgeRAGIndex:
    """获取全局知识库 RAG 索引单例。"""
    global _GLOBAL_RAG_INDEX
    if _GLOBAL_RAG_INDEX is None:
        _GLOBAL_RAG_INDEX = KnowledgeRAGIndex()
    return _GLOBAL_RAG_INDEX


# ─────────────────────────────────────────────────────────────────────────────
# 便捷检索与上下文格式化函数
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_industry_knowledge(
    query: str,
    top_k: int = 1,
    min_score: float = 3.0,
) -> List[Tuple[IndustryProfile, float]]:
    """检索行业图谱知识。"""
    index = get_global_rag_index()
    return index.retrieve_industries(query=query, top_k=top_k, min_score=min_score)


def retrieve_macro_event_knowledge(
    query: str,
    top_k: int = 2,
    min_score: float = 3.0,
) -> List[Tuple[MacroEventScenario, float]]:
    """检索宏观事件图谱知识。"""
    index = get_global_rag_index()
    return index.retrieve_macro_events(query=query, top_k=top_k, min_score=min_score)


def format_rag_industry_context(
    industry_or_query: Union[str, IndustryProfile, Sequence[IndustryProfile], None],
    fallback_on_miss: bool = True,
    min_score: float = 3.0,
) -> str:
    """将检索到的行业知识格式化为 Prompt 注入文本。

    若未命中且 fallback_on_miss=True，返回 '【行业常识知识库】\\n【知识库未命中】'；
    若 fallback_on_miss=False，返回空字符串。
    """
    if industry_or_query is None:
        return INDUSTRY_KNOWLEDGE_MISSING_BLOCK if fallback_on_miss else ""

    if isinstance(industry_or_query, IndustryProfile):
        return format_industry_deep_context(industry_or_query.industry_name)

    if isinstance(industry_or_query, (list, tuple)):
        blocks = []
        for item in industry_or_query:
            if isinstance(item, IndustryProfile):
                b = format_industry_deep_context(item.industry_name)
                if b:
                    blocks.append(b)
        if blocks:
            return "\n\n".join(blocks)
        return INDUSTRY_KNOWLEDGE_MISSING_BLOCK if fallback_on_miss else ""

    if isinstance(industry_or_query, str):
        q = industry_or_query.strip()
        if not q:
            return INDUSTRY_KNOWLEDGE_MISSING_BLOCK if fallback_on_miss else ""

        results = retrieve_industry_knowledge(q, top_k=1, min_score=min_score)
        if results:
            prof = results[0][0]
            return format_industry_deep_context(prof.industry_name)
        return INDUSTRY_KNOWLEDGE_MISSING_BLOCK if fallback_on_miss else ""

    return INDUSTRY_KNOWLEDGE_MISSING_BLOCK if fallback_on_miss else ""


def format_rag_macro_context(
    events_or_query: Union[str, MacroEventScenario, Sequence[MacroEventScenario], None],
    max_events: int = 2,
    fallback_on_miss: bool = True,
    min_score: float = 3.0,
) -> str:
    """将检索到的宏观事件情景格式化为 Prompt 注入文本。

    若未命中且 fallback_on_miss=True，返回 '【宏观事件传导图谱】\\n【知识库未命中】'；
    若 fallback_on_miss=False，返回空字符串。
    """
    if events_or_query is None:
        return MACRO_EVENT_MISSING_BLOCK if fallback_on_miss else ""

    if isinstance(events_or_query, MacroEventScenario):
        return format_macro_event_context(events_or_query.event_name)

    if isinstance(events_or_query, (list, tuple)):
        blocks = []
        for item in events_or_query[:max_events]:
            if isinstance(item, MacroEventScenario):
                b = format_macro_event_context(item.event_name)
                if b:
                    blocks.append(b)
        if blocks:
            return "\n\n".join(blocks)
        return MACRO_EVENT_MISSING_BLOCK if fallback_on_miss else ""

    if isinstance(events_or_query, str):
        q = events_or_query.strip()
        if not q:
            return MACRO_EVENT_MISSING_BLOCK if fallback_on_miss else ""

        results = retrieve_macro_event_knowledge(q, top_k=max_events, min_score=min_score)
        if results:
            blocks = [format_macro_event_context(sc.event_name) for sc, _ in results]
            valid_blocks = [b for b in blocks if b]
            if valid_blocks:
                return "\n\n".join(valid_blocks)
        return MACRO_EVENT_MISSING_BLOCK if fallback_on_miss else ""

    return MACRO_EVENT_MISSING_BLOCK if fallback_on_miss else ""
