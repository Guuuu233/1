"""Tests for tradingagents.llm_clients.thinking_cleaner (Bug D)."""

from tradingagents.llm_clients.thinking_cleaner import (
    clean_report_result_data,
    clean_thinking_traces,
)


def test_removes_pure_thinking_lines():
    text = (
        "Let me think about the key levels here.\n"
        "## 交易计划\n"
        "**方向：** 看多\n"
        "wait, I need to reconsider the entry.\n"
        "- 支撑位：11.40\n"
        "OK.\n"
        "- 目标价：12.50\n"
    )
    cleaned = clean_thinking_traces(text)
    assert "Let me think" not in cleaned
    assert "wait, I need to reconsider" not in cleaned
    assert "OK." not in cleaned
    assert "## 交易计划" in cleaned
    assert "**方向：** 看多" in cleaned
    assert "- 支撑位：11.40" in cleaned
    assert "- 目标价：12.50" in cleaned


def test_strips_thinking_prefix_but_keeps_substantive_content():
    text = (
        "wait, the high-volume node at 11.80 is the key accumulation level that institutions defend\n"
        "I think the support is at 11.40\n"
        "Hmm, the volume profile confirms accumulation\n"
    )
    cleaned = clean_thinking_traces(text)
    assert "wait," not in cleaned
    assert "I think" not in cleaned
    assert "Hmm," not in cleaned
    assert "the high-volume node at 11.80 is the key accumulation level that institutions defend" in cleaned
    assert "the support is at 11.40" in cleaned
    assert "the volume profile confirms accumulation" in cleaned


def test_normal_report_text_is_unchanged():
    text = (
        "## 量价分析\n\n"
        "**方向：** 偏多\n\n"
        "- 高量节点：11.80，机构在此防守\n"
        "- 上方套牢区：12.00-12.30\n\n"
        "> 成交量显示吸筹迹象。\n\n"
        "基本面显示 I think 这只股票被低估。"
    )
    assert clean_thinking_traces(text) == text


def test_removes_chinese_thinking_fillers():
    text = (
        "基本面分析显示营收增长 15%。\n"
        "让我想想，先看看数据再说。\n"
        "估值处于合理区间。\n"
    )
    cleaned = clean_thinking_traces(text)
    assert "让我想想" not in cleaned
    assert "基本面分析显示营收增长 15%。" in cleaned
    assert "估值处于合理区间。" in cleaned


def test_clean_report_result_data_cleans_sections_and_preserves_fields():
    result = {
        "symbol": "600206.SH",
        "trade_date": "2026-08-05",
        "trader_investment_plan": "Let me think about the entry.\nPlan: buy at 11.40.\n",
        "fundamentals_report": "Hmm, wait.\nRevenue growing 15%.\n",
        "volume_price_report": "I think volume confirms accumulation.\n",
        "short_term": {
            "final_trade_decision": "OK.\nDecision: BUY\n",
            "volume_price_report": "wait, high node at 11.80\n",
        },
    }
    cleaned = clean_report_result_data(result)
    assert cleaned["symbol"] == "600206.SH"
    assert cleaned["trade_date"] == "2026-08-05"
    assert "Let me think" not in cleaned["trader_investment_plan"]
    assert "Plan: buy at 11.40." in cleaned["trader_investment_plan"]
    assert "Hmm" not in cleaned["fundamentals_report"]
    assert "Revenue growing 15%." in cleaned["fundamentals_report"]
    assert "I think" not in cleaned["volume_price_report"]
    assert cleaned["short_term"]["final_trade_decision"] == "Decision: BUY"
    assert "high node at 11.80" in cleaned["short_term"]["volume_price_report"]
