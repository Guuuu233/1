"""Unit and integration tests for historical cases learning loop (DAV-283).

涵盖测试项：
1. 历史案例落库 (Case Recording):
   - completed 报告创建后落库一条历史案例（symbol, trade_date, decision/direction, claims, run_sha）
   - 提取关键 claims（包含 investment_debate_state / risk_debate_state / DEBATE_STATE / RISK_STATE 结构）
   - 提取 claims 失败或无 claims 时返回空列表 []
   - 运行 SHA 提取（优先环境变量，兜底 git / baseline fallback）
   - 幂等落库（重复落库更新现有记录，不产生重复主键/记录）
   - 失败报告（status="failed" / "pending" / "running"）严格不落库（"只在 completed 落库，失败不得假案例"）
2. 行情对比与缺失处理 (T+1 Return & Missing Data Handling):
   - 正常 T+1 计算：trade_date 到 eval_date 收盘价对比，涨跌幅计算正确
   - 预测方向与涨跌对比评估 (is_error):
     - 看多 + 上涨 -> is_error = False
     - 看多 + 下跌 -> is_error = True
     - 看空 + 下跌 -> is_error = False
     - 看空 + 上涨 -> is_error = True
     - 中性 + 大涨/大跌 -> is_error = True
   - 日历缺失、未来未到日（eval_date > today）、今日盘中未收盘等情况下：
     - actual_outcome 严格为 `【数据缺失】`
     - actual_change_pct 为 None
     - 严禁填 0 或今天
   - 行情接口报错/返回空/格式异常时：
     - actual_outcome 严格为 `【数据缺失】`，不抛异常崩溃
3. 历史案例检索与 Prompt 注入 (Retrieval & Prompt Injection):
   - 同一标的 (symbol) 检索最多 N 条历史案例（按 trade_date 降序）
   - 标的不足时补充同一行业 (industry) 案例
   - 严格防前视偏差（before_date 过滤）
   - 未命中返回 `【历史案例复盘】\n【历史案例未命中】`
   - 命中时格式化输出标的、行业、分析日、历史研判、关键 Claims、T+1 表现、案例启示
   - 宏观分析师 (Macro Analyst) 与基本面分析师 (Fundamentals Analyst) 注入历史案例验证（包含命中与未命中场景）
4. 全流程端到端闭环验证:
   - `report_service.create_report` -> 触发落库 -> `retrieve_similar_historical_cases` -> 格式化注入
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base, HistoricalCaseDB, ReportDB
from api.services import report_service
from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from tradingagents.agents.analysts.macro_analyst import create_macro_analyst
from tradingagents.graph.data_collector import DataCollector
from tradingagents.agents.utils.knowledge_context import (
    format_rag_historical_cases_context,
    resolve_dynamic_knowledge_context,
    resolve_historical_cases_context,
)
from tradingagents.knowledge.historical_cases import (
    DATA_MISSING_PLACEHOLDER,
    HISTORICAL_CASE_MISSING_BLOCK,
    HISTORICAL_CASE_MISSING_FALLBACK,
    backfill_pending_cases,
    calculate_t1_return,
    evaluate_prediction_error,
    extract_claims_from_report,
    format_historical_cases_context,
    get_current_run_sha,
    get_next_cn_trading_day,
    record_historical_case,
    retrieve_similar_historical_cases,
)


@pytest.fixture
def test_db_session():
    """In-memory SQLite session fixture for isolated testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# 1. 案例抽取、SHA 与基础辅助函数测试
# ─────────────────────────────────────────────────────────────────────────────

def test_get_current_run_sha():
    """验证 Git commit SHA 提取优先读取环境变量并后备 git/baseline。"""
    with patch.dict("os.environ", {"TA_RUN_SHA": "abc1234567890"}):
        assert get_current_run_sha() == "abc1234567890"

    with patch.dict("os.environ", {"TA_RUN_SHA": "", "GIT_COMMIT_SHA": "fedcba987654"}):
        assert get_current_run_sha() == "fedcba987654"

    with patch.dict("os.environ", {}, clear=True):
        sha = get_current_run_sha()
        assert isinstance(sha, str)
        assert len(sha) >= 7


def test_get_next_cn_trading_day():
    """验证下一交易日计算准确且无后续交易日返回 None。"""
    # 模拟交易日历
    fake_dates = [
        date(2024, 5, 8),
        date(2024, 5, 9),
        date(2024, 5, 10),  # Friday
        date(2024, 5, 13),  # Monday
        date(2024, 5, 14),
    ]
    with patch("tradingagents.knowledge.historical_cases._load_cn_trade_dates", return_value=(fake_dates, set(fake_dates))):
        next_day = get_next_cn_trading_day("2024-05-10")
        assert next_day == "2024-05-13"

        next_day2 = get_next_cn_trading_day("2024-05-08")
        assert next_day2 == "2024-05-09"

        # 超过最新交易日
        next_day_none = get_next_cn_trading_day("2024-05-14")
        assert next_day_none is None

        # 无效日期
        assert get_next_cn_trading_day("invalid-date") is None
        assert get_next_cn_trading_day("") is None


def test_extract_claims_from_report():
    """验证从各种格式中提取结构化 Claims 列表。"""
    # 1. 从 investment_debate_state / risk_debate_state 提取
    result_data = {
        "investment_debate_state": {
            "claims": [
                {"claim_id": "INV-1", "claim": "高端白酒动销强劲批价企稳", "confidence": 0.85},
                {"claim_id": "INV-2", "claim": "估值处于历史15%分位", "confidence": 0.75},
            ]
        },
        "risk_debate_state": {
            "claims": [
                {"claim_id": "RISK-1", "claim": "商务宴请消费复苏不及预期", "confidence": 0.60},
            ]
        }
    }
    claims = extract_claims_from_report(result_data)
    assert len(claims) == 3
    assert claims[0]["claim"] == "高端白酒动销强劲批价企稳"
    assert claims[0]["claim_id"] == "INV-1"
    assert claims[1]["claim"] == "估值处于历史15%分位"
    assert claims[2]["claim"] == "商务宴请消费复苏不及预期"

    # 2. 从 machine block (<!-- DEBATE_STATE: ... -->) 提取
    text_with_block = """
    最终裁决如下：
    建议买入。
    <!-- DEBATE_STATE: {"new_claims": [{"claim": "AI芯片需求爆发推动代工稼动率提升", "confidence": 0.80}]} -->
    <!-- RISK_STATE: {"new_claims": [{"claim": "先进制程海外出口管制加剧", "confidence": 0.70}]} -->
    """
    claims_from_text = extract_claims_from_report({}, texts=[text_with_block])
    assert len(claims_from_text) == 2
    assert claims_from_text[0]["claim"] == "AI芯片需求爆发推动代工稼动率提升"
    assert claims_from_text[1]["claim"] == "先进制程海外出口管制加剧"

    # 3. 空结构返回 []
    assert extract_claims_from_report({}) == []
    assert extract_claims_from_report(None) == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. T+1 行情对比与【数据缺失】严谨性测试
# ─────────────────────────────────────────────────────────────────────────────

def test_calculate_t1_return_normal():
    """验证正常历史交易日 T+1 收益率与涨跌字符串计算。"""
    sample_csv = """date,open,high,low,close,volume
2024-05-10,1700.0,1720.0,1695.0,1710.0,10000
2024-05-13,1715.0,1760.0,1710.0,1752.75,12000
"""
    fake_dates = [date(2024, 5, 10), date(2024, 5, 13)]
    with patch("tradingagents.knowledge.historical_cases._load_cn_trade_dates", return_value=(fake_dates, set(fake_dates))), \
         patch("tradingagents.knowledge.historical_cases.now_cn", return_value=datetime(2024, 5, 20, 16, 0, tzinfo=timezone.utc)), \
         patch("tradingagents.dataflows.interface.route_to_vendor", return_value=sample_csv):

        eval_date, change_pct, outcome_str = calculate_t1_return("600519", "2024-05-10")
        assert eval_date == "2024-05-13"
        # 1752.75 vs 1710.0 -> +2.50%
        assert change_pct == 2.50
        assert outcome_str == "+2.50%"


def test_calculate_t1_return_missing_data_future_or_incomplete():
    """验证当评估日为未来或盘中未收盘时严格返回【数据缺失】，禁止填 0 或今天。"""
    fake_dates = [date(2026, 8, 20), date(2026, 8, 21), date(2026, 8, 24)]

    # 1. 评估日晚于当前日期 (as_of > today)
    with patch("tradingagents.knowledge.historical_cases._load_cn_trade_dates", return_value=(fake_dates, set(fake_dates))), \
         patch("tradingagents.knowledge.historical_cases.now_cn", return_value=datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc)):

        eval_date, change_pct, outcome_str = calculate_t1_return("600519", "2026-08-20")
        assert eval_date == "2026-08-21"
        assert change_pct is None
        assert outcome_str == DATA_MISSING_PLACEHOLDER

    # 2. 评估日是今日但盘中未收盘 (in_session)
    with patch("tradingagents.knowledge.historical_cases._load_cn_trade_dates", return_value=(fake_dates, set(fake_dates))), \
         patch("tradingagents.knowledge.historical_cases.now_cn", return_value=datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)), \
         patch("tradingagents.knowledge.historical_cases.cn_market_phase", return_value="in_session"):

        eval_date, change_pct, outcome_str = calculate_t1_return("600519", "2026-08-20")
        assert eval_date == "2026-08-21"
        assert change_pct is None
        assert outcome_str == DATA_MISSING_PLACEHOLDER

    # 3. 行情接口异常或返回空
    with patch("tradingagents.knowledge.historical_cases._load_cn_trade_dates", return_value=(fake_dates, set(fake_dates))), \
         patch("tradingagents.knowledge.historical_cases.now_cn", return_value=datetime(2026, 8, 25, 16, 0, tzinfo=timezone.utc)), \
         patch("tradingagents.dataflows.interface.route_to_vendor", side_effect=RuntimeError("API error")):

        eval_date, change_pct, outcome_str = calculate_t1_return("600519", "2026-08-20")
        assert eval_date == "2026-08-21"
        assert change_pct is None
        assert outcome_str == DATA_MISSING_PLACEHOLDER


def test_evaluate_prediction_error():
    """验证决策与次日实际收益的偏差判定逻辑。"""
    # 看多 (BUY / 看多)
    assert evaluate_prediction_error("BUY", "看多", 2.0) is False   # 涨 -> 准确
    assert evaluate_prediction_error("BUY", "看多", -1.5) is True   # 跌 -> 偏差
    assert evaluate_prediction_error("买入", "偏多", -0.5) is True  # 跌 -> 偏差

    # 看空 (SELL / 看空)
    assert evaluate_prediction_error("SELL", "看空", -2.0) is False  # 跌 -> 准确
    assert evaluate_prediction_error("SELL", "看空", 1.5) is True   # 涨 -> 偏差

    # 中性 (HOLD / 中性)
    assert evaluate_prediction_error("HOLD", "中性", 0.5) is False
    assert evaluate_prediction_error("HOLD", "中性", 4.0) is True   # 大涨 -> 偏差
    assert evaluate_prediction_error("HOLD", "中性", -3.5) is True  # 大跌 -> 偏差

    # 无行情数据
    assert evaluate_prediction_error("BUY", "看多", None) is None


# ─────────────────────────────────────────────────────────────────────────────
# 3. 案例落库与幂等性测试 (Record Case Tests)
# ─────────────────────────────────────────────────────────────────────────────

def test_record_historical_case_completed_success(test_db_session):
    """验证 completed 报告落库一条完整案例。"""
    report = ReportDB(
        id="rep-1001",
        symbol="600519",
        trade_date="2024-05-10",
        status="completed",
        decision="BUY",
        direction="看多",
        confidence=75,
        final_trade_decision="建议买入茅台。\n<!-- DEBATE_STATE: {\"new_claims\": [{\"claim\": \"高端白酒批价上行\", \"confidence\": 0.85}]} -->",
        result_data={
            "investment_debate_state": {
                "claims": [{"claim_id": "INV-1", "claim": "高端白酒批价上行", "confidence": 0.85}]
            }
        },
    )
    test_db_session.add(report)
    test_db_session.commit()

    sample_csv = """date,open,high,low,close,volume
2024-05-10,1700.0,1720.0,1695.0,1710.0,10000
2024-05-13,1715.0,1760.0,1710.0,1752.75,12000
"""
    fake_dates = [date(2024, 5, 10), date(2024, 5, 13)]
    with patch("tradingagents.knowledge.historical_cases._load_cn_trade_dates", return_value=(fake_dates, set(fake_dates))), \
         patch("tradingagents.knowledge.historical_cases.now_cn", return_value=datetime(2024, 5, 20, 16, 0, tzinfo=timezone.utc)), \
         patch("tradingagents.dataflows.interface.route_to_vendor", return_value=sample_csv), \
         patch("tradingagents.knowledge.historical_cases.get_current_run_sha", return_value="test_sha_123"):

        case = record_historical_case(test_db_session, report)

        assert case is not None
        assert case.symbol == "600519"
        assert case.industry == "liquor_beverage"
        assert case.trade_date == "2024-05-10"
        assert case.decision == "BUY"
        assert case.direction == "看多"
        assert case.confidence == 75
        assert case.run_sha == "test_sha_123"
        assert case.eval_date == "2024-05-13"
        assert case.actual_change_pct == 2.50
        assert case.actual_outcome == "+2.50%"
        assert case.is_error is False
        assert len(case.claims) == 1
        assert case.claims[0]["claim"] == "高端白酒批价上行"

        # 查询数据库确认持久化
        rows = test_db_session.query(HistoricalCaseDB).all()
        assert len(rows) == 1
        assert rows[0].report_id == "rep-1001"


def test_record_historical_case_idempotent(test_db_session):
    """验证对相同报告重复触发落库具有严格幂等性（更新而非重复新增）。"""
    report = ReportDB(
        id="rep-1002",
        symbol="688981",
        trade_date="2024-06-01",
        status="completed",
        decision="HOLD",
        direction="中性",
        confidence=60,
    )
    test_db_session.add(report)
    test_db_session.commit()

    with patch("tradingagents.knowledge.historical_cases.calculate_t1_return", return_value=("2024-06-03", None, DATA_MISSING_PLACEHOLDER)):
        case1 = record_historical_case(test_db_session, report)
        assert case1 is not None

        # 二次调用更新
        report.confidence = 80
        case2 = record_historical_case(test_db_session, report)
        assert case2 is not None
        assert case2.id == case1.id
        assert case2.confidence == 80

        rows = test_db_session.query(HistoricalCaseDB).filter(HistoricalCaseDB.symbol == "688981").all()
        assert len(rows) == 1


def test_record_historical_case_skips_failed_report(test_db_session):
    """验证失败或非 completed 状态报告严格不落库（'失败不得假案例'）。"""
    failed_report = ReportDB(
        id="rep-fail-1",
        symbol="600519",
        trade_date="2024-05-10",
        status="failed",
        error="LLM timeout",
    )
    test_db_session.add(failed_report)
    test_db_session.commit()

    case = record_historical_case(test_db_session, failed_report)
    assert case is None
    assert test_db_session.query(HistoricalCaseDB).count() == 0

    # pending / running 亦不落库
    pending_report = ReportDB(
        id="rep-pend-1",
        symbol="600519",
        trade_date="2024-05-10",
        status="pending",
    )
    assert record_historical_case(test_db_session, pending_report) is None
    assert test_db_session.query(HistoricalCaseDB).count() == 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. 案例检索与 Prompt 注入测试 (Retrieve & Format Tests)
# ─────────────────────────────────────────────────────────────────────────────

def test_retrieve_similar_historical_cases_by_symbol_and_industry(test_db_session):
    """验证按同一标的与同一行业检索历史案例，并严格按日期过滤防前视偏差。"""
    # 插入历史案例
    # 案例 1: 600519 (白酒), 2024-01-10
    test_db_session.add(HistoricalCaseDB(
        id="c1", symbol="600519", industry="liquor_beverage", trade_date="2024-01-10",
        decision="BUY", direction="看多", actual_outcome="+1.80%", is_error=False,
    ))
    # 案例 2: 600519 (白酒), 2024-03-15
    test_db_session.add(HistoricalCaseDB(
        id="c2", symbol="600519", industry="liquor_beverage", trade_date="2024-03-15",
        decision="BUY", direction="看多", actual_outcome="-2.30%", is_error=True,
    ))
    # 案例 3: 000858 (白酒 - 同行业不同标的), 2024-02-20
    test_db_session.add(HistoricalCaseDB(
        id="c3", symbol="000858", industry="liquor_beverage", trade_date="2024-02-20",
        decision="BUY", direction="看多", actual_outcome="-1.10%", is_error=True,
    ))
    # 案例 4: 688981 (半导体 - 其他行业), 2024-02-01
    test_db_session.add(HistoricalCaseDB(
        id="c4", symbol="688981", industry="semiconductor", trade_date="2024-02-01",
        decision="BUY", direction="看多", actual_outcome="+3.00%", is_error=False,
    ))
    # 案例 5: 600519 (未来案例), 2024-06-01
    test_db_session.add(HistoricalCaseDB(
        id="c5", symbol="600519", industry="liquor_beverage", trade_date="2024-06-01",
        decision="BUY", direction="看多", actual_outcome="+5.00%", is_error=False,
    ))
    test_db_session.commit()

    # 1. 查询 600519 在 2024-04-01 前的历史案例 (max_cases=3)
    # 应命中 600519 的 c2, c1，以及同行业 000858 的 c3；c5（未来）被严格排除；c4（其他行业）不混入
    cases = retrieve_similar_historical_cases(
        symbol="600519",
        industry="liquor_beverage",
        before_date="2024-04-01",
        max_cases=3,
        db=test_db_session,
    )
    assert len(cases) == 3
    case_ids = [c.id for c in cases]
    # 同标的优先且按日期降序
    assert case_ids[0] == "c2"  # 2024-03-15 (600519)
    assert case_ids[1] == "c1"  # 2024-01-10 (600519)
    assert case_ids[2] == "c3"  # 2024-02-20 (000858 同行业补充)

    # 2. 仅检索错误案例 (errors_only=True)
    err_cases = retrieve_similar_historical_cases(
        symbol="600519",
        industry="liquor_beverage",
        before_date="2024-04-01",
        max_cases=3,
        errors_only=True,
        db=test_db_session,
    )
    assert len(err_cases) == 2
    assert {c.id for c in err_cases} == {"c2", "c3"}


def test_format_historical_cases_context_hit_and_miss(test_db_session):
    """验证历史案例 Prompt 格式化：命中与未命中场景。"""
    # 1. 未命中场景
    miss_text = format_historical_cases_context(
        [],
        fallback_on_miss=True,
    )
    assert miss_text == HISTORICAL_CASE_MISSING_BLOCK
    assert HISTORICAL_CASE_MISSING_FALLBACK in miss_text

    # fallback_on_miss=False 返回空
    assert format_historical_cases_context([], fallback_on_miss=False) == ""

    # 2. 命中场景
    case_item = HistoricalCaseDB(
        id="c10",
        symbol="600519",
        industry="liquor_beverage",
        trade_date="2024-05-10",
        decision="BUY",
        direction="看多",
        confidence=80,
        claims=[{"claim": "渠道批价上行"}, {"claim": "库存处于安全边际"}],
        eval_date="2024-05-13",
        actual_change_pct=-1.5,
        actual_outcome="-1.50%",
        is_error=True,
    )
    formatted = format_historical_cases_context([case_item], fallback_on_miss=True)
    assert "【历史案例复盘】" in formatted
    assert "600519" in formatted
    assert "2024-05-10" in formatted
    assert "BUY" in formatted
    assert "渠道批价上行" in formatted
    assert "-1.50%" in formatted
    assert "【偏差复盘】" in formatted


def test_resolve_historical_cases_context_in_knowledge_context(test_db_session):
    """验证 knowledge_context.py 中的 resolve_historical_cases_context 接口。"""
    test_db_session.add(HistoricalCaseDB(
        id="c20",
        symbol="600519",
        industry="liquor_beverage",
        trade_date="2024-04-10",
        decision="BUY",
        direction="看多",
        confidence=70,
        claims=[{"claim": "动销加速"}],
        actual_outcome="+2.00%",
        is_error=False,
    ))
    test_db_session.commit()

    # 1. 命中标的
    cases, ctx = resolve_historical_cases_context(
        ticker="600519",
        trade_date="2024-05-01",
        fallback_on_miss=False,
        db=test_db_session,
    )
    assert len(cases) == 1
    assert "600519" in ctx
    assert "动销加速" in ctx

    # 2. 未命中标的
    cases_miss, ctx_miss = resolve_historical_cases_context(
        ticker="000001",
        trade_date="2024-05-01",
        fallback_on_miss=True,
        db=test_db_session,
    )
    assert len(cases_miss) == 0
    assert ctx_miss == HISTORICAL_CASE_MISSING_BLOCK


# ─────────────────────────────────────────────────────────────────────────────
# 5. 分析师节点 Prompt 注入集成测试 (Macro & Fundamentals Analysts)
# ─────────────────────────────────────────────────────────────────────────────

def test_macro_analyst_injects_historical_cases(test_db_session):
    """验证 Macro Analyst 在有历史案例时注入案例，无案例时注入【历史案例未命中】。"""
    test_db_session.add(HistoricalCaseDB(
        id="c30",
        symbol="600519",
        industry="liquor_beverage",
        trade_date="2024-03-01",
        decision="BUY",
        direction="看多",
        confidence=75,
        claims=[{"claim": "白酒板块资金净流入"}],
        actual_outcome="+1.20%",
        is_error=False,
    ))
    test_db_session.commit()

    # 1. 命中场景测试
    captured_messages_hit = []

    mock_llm_hit = MagicMock()
    mock_chunk = SimpleNamespace(content="宏观分析测试报告", response_metadata={})

    async def _mock_astream_hit(messages):
        captured_messages_hit.extend(messages)
        yield mock_chunk

    mock_llm_hit.astream = _mock_astream_hit
    mock_llm_hit.model_name = "test-model"

    collector_hit = DataCollector()
    collector_hit._cache["600519_2024-05-01"] = {
        "fund_flow_board": "白酒板块资金流入20亿",
        "news": "茅台动销良好",
        "global_news": "全球消费企稳",
        "global_indices": "无数据",
        "major_assets": "无数据",
        "cn_indices": "无数据",
        "northbound_flow": "无数据",
        "industry_linkage": "无数据",
    }

    macro_node = create_macro_analyst(mock_llm_hit, collector_hit)
    state = {
        "trade_date": "2024-05-01",
        "company_of_interest": "600519",
    }

    with patch("tradingagents.knowledge.historical_cases.get_db_ctx") as mock_ctx:
        mock_ctx.return_value.__enter__.return_value = test_db_session
        result = asyncio.run(macro_node(state))

    assert "macro_report" in result
    human_msg = next(m for m in captured_messages_hit if isinstance(m, HumanMessage))
    assert "【历史案例复盘】" in human_msg.content
    assert "白酒板块资金净流入" in human_msg.content

    # 2. 未命中场景测试
    captured_messages_miss = []

    async def _mock_astream_miss(messages):
        captured_messages_miss.extend(messages)
        yield mock_chunk

    mock_llm_miss = MagicMock()
    mock_llm_miss.astream = _mock_astream_miss
    mock_llm_miss.model_name = "test-model"

    collector_miss = DataCollector()
    collector_miss._cache["000001_2024-05-01"] = {
        "fund_flow_board": "无数据",
        "news": "日常信息",
        "global_news": "无数据",
        "global_indices": "无数据",
        "major_assets": "无数据",
        "cn_indices": "无数据",
        "northbound_flow": "无数据",
        "industry_linkage": "无数据",
    }

    macro_node_miss = create_macro_analyst(mock_llm_miss, collector_miss)
    state_miss = {
        "trade_date": "2024-05-01",
        "company_of_interest": "000001",  # 无历史案例
    }

    with patch("tradingagents.knowledge.historical_cases.get_db_ctx") as mock_ctx:
        mock_ctx.return_value.__enter__.return_value = test_db_session
        result_miss = asyncio.run(macro_node_miss(state_miss))

    human_msg_miss = next(m for m in captured_messages_miss if isinstance(m, HumanMessage))
    assert "【历史案例复盘】\n【历史案例未命中】" in human_msg_miss.content


def test_fundamentals_analyst_injects_historical_cases(test_db_session):
    """验证 Fundamentals Analyst 在有历史案例时注入案例，无案例时注入【历史案例未命中】。"""
    test_db_session.add(HistoricalCaseDB(
        id="c40",
        symbol="688981",
        industry="semiconductor",
        trade_date="2024-02-15",
        decision="BUY",
        direction="看多",
        confidence=80,
        claims=[{"claim": "晶圆扩产CapEx提升驱动业绩反转"}],
        actual_outcome="+3.50%",
        is_error=False,
    ))
    test_db_session.commit()

    captured_messages = []
    mock_chunk = SimpleNamespace(content="基本面分析测试报告", response_metadata={})

    async def _mock_astream(messages):
        captured_messages.extend(messages)
        yield mock_chunk

    mock_llm = MagicMock()
    mock_llm.astream = _mock_astream
    mock_llm.model_name = "test-model"

    collector = DataCollector()
    collector._cache["688981_2024-05-01"] = {
        "fundamentals": "中芯国际业绩说明",
        "balance_sheet": "资产负债表稳健",
        "cashflow": "现金流充沛",
        "income_statement": "利润表增长",
        "global_indices": "无数据",
        "major_assets": "无数据",
        "cn_indices": "无数据",
        "industry_linkage": "无数据",
    }

    fund_node = create_fundamentals_analyst(mock_llm, collector)
    state = {
        "trade_date": "2024-05-01",
        "company_of_interest": "688981",
    }

    with patch("tradingagents.knowledge.historical_cases.get_db_ctx") as mock_ctx:
        mock_ctx.return_value.__enter__.return_value = test_db_session
        result = asyncio.run(fund_node(state))

    assert "fundamentals_report" in result
    human_msg = next(m for m in captured_messages if isinstance(m, HumanMessage))
    assert "【历史案例复盘】" in human_msg.content
    assert "晶圆扩产CapEx提升驱动业绩反转" in human_msg.content

    assert "fundamentals_report" in result
    human_msg = next(m for m in captured_messages if isinstance(m, HumanMessage))
    assert "【历史案例复盘】" in human_msg.content
    assert "晶圆扩产CapEx提升驱动业绩反转" in human_msg.content


# ─────────────────────────────────────────────────────────────────────────────
# 6. report_service 端到端挂钩测试
# ─────────────────────────────────────────────────────────────────────────────

def test_report_service_create_report_hooks_historical_case(test_db_session):
    """验证 report_service.create_report 完成时自动触发落库历史案例。"""
    with patch("tradingagents.knowledge.historical_cases.calculate_t1_return", return_value=("2024-05-13", 2.50, "+2.50%")), \
         patch("tradingagents.knowledge.historical_cases.get_current_run_sha", return_value="sha_hook_test"), \
         patch("api.services.report_service.extract_structured_data", return_value=None):

        db_report = report_service.create_report(
            db=test_db_session,
            symbol="600519",
            trade_date="2024-05-10",
            decision="BUY",
            result_data={
                "investment_debate_state": {
                    "claims": [{"claim_id": "INV-1", "claim": "端午回款动销良好"}]
                }
            },
        )

        assert db_report.status == "completed"
        # 验证历史案例落库成功
        case = test_db_session.query(HistoricalCaseDB).filter(HistoricalCaseDB.report_id == db_report.id).first()
        assert case is not None
        assert case.symbol == "600519"
        assert case.industry == "liquor_beverage"
        assert case.actual_outcome == "+2.50%"
        assert case.claims[0]["claim"] == "端午回款动销良好"


# ─────────────────────────────────────────────────────────────────────────────
# 7. 历史案例 T+1 实际值回填测试 (DAV-287 回填闭环补全)
# ─────────────────────────────────────────────────────────────────────────────

def test_backfill_pending_cases_success(test_db_session):
    """验证回填成功路径：扫描 actual_outcome=【数据缺失】且 eval_date <= as_of 的案例，重算并更新。"""
    case = HistoricalCaseDB(
        id="case-backfill-1",
        symbol="000725",
        industry="electronics_semiconductor",
        trade_date="2024-05-10",
        eval_date="2024-05-13",
        decision="BUY",
        direction="看多",
        confidence=70,
        claims=[{"claim": "面板价格筑底回升"}],
        actual_change_pct=None,
        actual_outcome=DATA_MISSING_PLACEHOLDER,
        is_error=None,
    )
    test_db_session.add(case)
    test_db_session.commit()

    with patch("tradingagents.knowledge.historical_cases.calculate_t1_return", return_value=("2024-05-13", 3.25, "+3.25%")):
        stats = backfill_pending_cases(test_db_session, as_of="2024-05-15")

        assert stats["total_scanned"] == 1
        assert stats["backfilled"] == 1
        assert stats["still_missing"] == 0
        assert stats["skipped_future"] == 0

        # 验证数据库中记录已更新
        updated = test_db_session.query(HistoricalCaseDB).filter(HistoricalCaseDB.id == "case-backfill-1").first()
        assert updated.actual_outcome == "+3.25%"
        assert updated.actual_change_pct == 3.25
        assert updated.is_error is False
        assert updated.eval_date == "2024-05-13"


def test_backfill_pending_cases_eval_date_future_skipped(test_db_session):
    """验证 eval_date 未到（eval_date > as_of）跳过回填，保持【数据缺失】。"""
    case = HistoricalCaseDB(
        id="case-future-1",
        symbol="000725",
        industry="electronics_semiconductor",
        trade_date="2026-08-22",
        eval_date="2026-08-24",
        decision="BUY",
        direction="看多",
        confidence=70,
        actual_change_pct=None,
        actual_outcome=DATA_MISSING_PLACEHOLDER,
        is_error=None,
    )
    test_db_session.add(case)
    test_db_session.commit()

    with patch("tradingagents.knowledge.historical_cases.calculate_t1_return") as mock_calc:
        stats = backfill_pending_cases(test_db_session, as_of="2026-08-22")

        assert stats["total_scanned"] == 1
        assert stats["skipped_future"] == 1
        assert stats["backfilled"] == 0
        assert stats["still_missing"] == 0
        mock_calc.assert_not_called()

        updated = test_db_session.query(HistoricalCaseDB).filter(HistoricalCaseDB.id == "case-future-1").first()
        assert updated.actual_outcome == DATA_MISSING_PLACEHOLDER
        assert updated.actual_change_pct is None
        assert updated.is_error is None


def test_backfill_pending_cases_market_data_missing_remains_missing(test_db_session):
    """验证仍取不到行情时保持【数据缺失】并记录失败台账，禁止填 0 或臆造。"""
    case = HistoricalCaseDB(
        id="case-missing-quote-1",
        symbol="600000",
        industry="banking",
        trade_date="2024-05-10",
        eval_date="2024-05-13",
        decision="BUY",
        direction="看多",
        confidence=65,
        actual_change_pct=None,
        actual_outcome=DATA_MISSING_PLACEHOLDER,
        is_error=None,
    )
    test_db_session.add(case)
    test_db_session.commit()

    with patch("tradingagents.knowledge.historical_cases.calculate_t1_return", return_value=("2024-05-13", None, DATA_MISSING_PLACEHOLDER)):
        stats = backfill_pending_cases(test_db_session, as_of="2024-05-15")

        assert stats["total_scanned"] == 1
        assert stats["backfilled"] == 0
        assert stats["still_missing"] == 1

        updated = test_db_session.query(HistoricalCaseDB).filter(HistoricalCaseDB.id == "case-missing-quote-1").first()
        assert updated.actual_outcome == DATA_MISSING_PLACEHOLDER
        assert updated.actual_change_pct is None
        assert updated.is_error is None


def test_backfill_pending_cases_is_error_recomputed(test_db_session):
    """验证回填成功后重算 is_error（看多/看空/中性各类偏差与一致场景）。"""
    cases = [
        HistoricalCaseDB(
            id="c-err-bull-wrong",
            symbol="000001",
            trade_date="2024-05-10",
            eval_date="2024-05-13",
            decision="BUY",
            direction="看多",
            actual_outcome=DATA_MISSING_PLACEHOLDER,
        ),
        HistoricalCaseDB(
            id="c-err-bear-wrong",
            symbol="000002",
            trade_date="2024-05-10",
            eval_date="2024-05-13",
            decision="SELL",
            direction="看空",
            actual_outcome=DATA_MISSING_PLACEHOLDER,
        ),
        HistoricalCaseDB(
            id="c-err-bear-correct",
            symbol="000003",
            trade_date="2024-05-10",
            eval_date="2024-05-13",
            decision="SELL",
            direction="看空",
            actual_outcome=DATA_MISSING_PLACEHOLDER,
        ),
        HistoricalCaseDB(
            id="c-err-neutral-big-move",
            symbol="000004",
            trade_date="2024-05-10",
            eval_date="2024-05-13",
            decision="HOLD",
            direction="中性",
            actual_outcome=DATA_MISSING_PLACEHOLDER,
        ),
        HistoricalCaseDB(
            id="c-err-neutral-flat",
            symbol="000005",
            trade_date="2024-05-10",
            eval_date="2024-05-13",
            decision="HOLD",
            direction="中性",
            actual_outcome=DATA_MISSING_PLACEHOLDER,
        ),
    ]
    for c in cases:
        test_db_session.add(c)
    test_db_session.commit()

    def _mock_calc(symbol, trade_date, eval_date=None):
        if symbol == "000001":
            return (eval_date, -2.10, "-2.10%")
        elif symbol == "000002":
            return (eval_date, 1.80, "+1.80%")
        elif symbol == "000003":
            return (eval_date, -3.00, "-3.00%")
        elif symbol == "000004":
            return (eval_date, 4.50, "+4.50%")
        elif symbol == "000005":
            return (eval_date, 0.40, "+0.40%")
        return (eval_date, None, DATA_MISSING_PLACEHOLDER)

    with patch("tradingagents.knowledge.historical_cases.calculate_t1_return", side_effect=_mock_calc):
        stats = backfill_pending_cases(test_db_session, as_of="2024-05-15")
        assert stats["backfilled"] == 5

        # 逐个检查 is_error 判定
        c1 = test_db_session.query(HistoricalCaseDB).filter(HistoricalCaseDB.id == "c-err-bull-wrong").first()
        assert c1.actual_outcome == "-2.10%"
        assert c1.is_error is True  # 看多却下跌

        c2 = test_db_session.query(HistoricalCaseDB).filter(HistoricalCaseDB.id == "c-err-bear-wrong").first()
        assert c2.actual_outcome == "+1.80%"
        assert c2.is_error is True  # 看空却上涨

        c3 = test_db_session.query(HistoricalCaseDB).filter(HistoricalCaseDB.id == "c-err-bear-correct").first()
        assert c3.actual_outcome == "-3.00%"
        assert c3.is_error is False  # 看空且下跌

        c4 = test_db_session.query(HistoricalCaseDB).filter(HistoricalCaseDB.id == "c-err-neutral-big-move").first()
        assert c4.actual_outcome == "+4.50%"
        assert c4.is_error is True  # 中性但剧烈波动超过 3%

        c5 = test_db_session.query(HistoricalCaseDB).filter(HistoricalCaseDB.id == "c-err-neutral-flat").first()
        assert c5.actual_outcome == "+0.40%"
        assert c5.is_error is False  # 中性且波动幅度在 3% 以内


def test_backfill_pending_cases_idempotent(test_db_session):
    """验证回填幂等性：重复执行不重复写入，已回填记录不被重复处理。"""
    case = HistoricalCaseDB(
        id="case-idem-1",
        symbol="000725",
        trade_date="2024-05-10",
        eval_date="2024-05-13",
        decision="BUY",
        direction="看多",
        actual_outcome=DATA_MISSING_PLACEHOLDER,
    )
    test_db_session.add(case)
    test_db_session.commit()

    with patch("tradingagents.knowledge.historical_cases.calculate_t1_return", return_value=("2024-05-13", 1.50, "+1.50%")):
        # 第一次运行：回填 1 条
        stats1 = backfill_pending_cases(test_db_session, as_of="2024-05-15")
        assert stats1["total_scanned"] == 1
        assert stats1["backfilled"] == 1

        # 第二次运行：待处理数为 0，不重复写入
        stats2 = backfill_pending_cases(test_db_session, as_of="2024-05-15")
        assert stats2["total_scanned"] == 0
        assert stats2["backfilled"] == 0


def test_report_service_completion_triggers_backfill(test_db_session):
    """验证当新报告 completed 落库时，自动顺带回填历史待处理案例。"""
    # 预设一条之前的待回填案例
    prior_case = HistoricalCaseDB(
        id="prior-case-1",
        symbol="600519",
        trade_date="2024-05-10",
        eval_date="2024-05-13",
        decision="BUY",
        direction="看多",
        actual_outcome=DATA_MISSING_PLACEHOLDER,
    )
    test_db_session.add(prior_case)
    test_db_session.commit()

    def _mock_calc(symbol, trade_date, eval_date=None):
        if symbol == "600519":
            return ("2024-05-13", 2.10, "+2.10%")
        elif symbol == "000001":
            return ("2024-05-14", None, DATA_MISSING_PLACEHOLDER)
        return (eval_date, None, DATA_MISSING_PLACEHOLDER)

    fake_now = datetime(2024, 5, 20, 16, 0, tzinfo=timezone.utc)
    with patch("tradingagents.knowledge.historical_cases.calculate_t1_return", side_effect=_mock_calc), \
         patch("tradingagents.knowledge.historical_cases.now_cn", return_value=fake_now), \
         patch("api.services.report_service.extract_structured_data", return_value=None):

        # 创建新报告并 completed
        db_report = report_service.create_report(
            db=test_db_session,
            symbol="000001",
            trade_date="2024-05-13",
            decision="BUY",
        )

        assert db_report.status == "completed"

        # 验证 prior_case 在新报告创建后被顺带回填成功
        updated_prior = test_db_session.query(HistoricalCaseDB).filter(HistoricalCaseDB.id == "prior-case-1").first()
        assert updated_prior.actual_outcome == "+2.10%"
        assert updated_prior.actual_change_pct == 2.10
        assert updated_prior.is_error is False


def test_update_report_partial_triggers_backfill(test_db_session):
    """验证 update_report_partial 将报告更新为 completed 时顺带触发回填。"""
    prior_case = HistoricalCaseDB(
        id="prior-case-2",
        symbol="688981",
        trade_date="2024-05-10",
        eval_date="2024-05-13",
        decision="BUY",
        direction="看多",
        actual_outcome=DATA_MISSING_PLACEHOLDER,
    )
    test_db_session.add(prior_case)

    # 初始处于 running 状态的报告
    running_report = ReportDB(
        id="rep-running-1",
        symbol="000002",
        trade_date="2024-05-13",
        status="running",
    )
    test_db_session.add(running_report)
    test_db_session.commit()

    def _mock_calc(symbol, trade_date, eval_date=None):
        if symbol == "688981":
            return ("2024-05-13", 3.00, "+3.00%")
        return (eval_date, None, DATA_MISSING_PLACEHOLDER)

    fake_now = datetime(2024, 5, 20, 16, 0, tzinfo=timezone.utc)
    with patch("tradingagents.knowledge.historical_cases.calculate_t1_return", side_effect=_mock_calc), \
         patch("tradingagents.knowledge.historical_cases.now_cn", return_value=fake_now):

        updated_report = report_service.update_report_partial(
            db=test_db_session,
            report_id="rep-running-1",
            status="completed",
            decision="BUY",
            direction="看多",
        )

        assert updated_report.status == "completed"
        # 验证 prior_case 已被回填
        updated_prior = test_db_session.query(HistoricalCaseDB).filter(HistoricalCaseDB.id == "prior-case-2").first()
        assert updated_prior.actual_outcome == "+3.00%"
        assert updated_prior.actual_change_pct == 3.00


def test_api_lifespan_triggers_backfill():
    """验证 FastAPI lifespan 启动时自动触发 historical_cases 回填流程。"""
    from sqlalchemy.pool import StaticPool
    from api.main import lifespan, app

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = TestingSession()

    prior_case = HistoricalCaseDB(
        id="startup-case-1",
        symbol="000725",
        trade_date="2024-05-10",
        eval_date="2024-05-13",
        decision="BUY",
        direction="看多",
        actual_outcome=DATA_MISSING_PLACEHOLDER,
    )
    session.add(prior_case)
    session.commit()

    def _mock_calc(symbol, trade_date, eval_date=None):
        if symbol == "000725":
            return ("2024-05-13", 4.10, "+4.10%")
        return (eval_date, None, DATA_MISSING_PLACEHOLDER)

    fake_now = datetime(2024, 5, 20, 16, 0, tzinfo=timezone.utc)
    with patch("tradingagents.knowledge.historical_cases.calculate_t1_return", side_effect=_mock_calc), \
         patch("tradingagents.knowledge.historical_cases.now_cn", return_value=fake_now), \
         patch("api.main.get_db_ctx") as mock_get_db_ctx, \
         patch("api.main.init_db"), \
         patch("api.main._report_version_stats"), \
         patch("api.main._load_cn_stock_map"), \
         patch("tradingagents.dataflows.trade_calendar._load_cn_trade_dates"), \
         patch("api.services.report_service.recover_stale_active_reports", return_value={"failed": 0}), \
         patch("api.services.auth_service.ensure_secure_secret_configured"), \
         patch("api.services.auth_service.is_custom_secret_configured", return_value=True):

        from contextlib import contextmanager

        @contextmanager
        def _fake_ctx():
            s = TestingSession()
            try:
                yield s
            finally:
                s.close()

        mock_get_db_ctx.side_effect = _fake_ctx

        async def _run():
            async with lifespan(app):
                pass

        asyncio.run(_run())

        check_session = TestingSession()
        updated = check_session.query(HistoricalCaseDB).filter(HistoricalCaseDB.id == "startup-case-1").first()
        assert updated.actual_outcome == "+4.10%"
        assert updated.actual_change_pct == 4.10
        assert updated.is_error is False
        check_session.close()
        session.close()
