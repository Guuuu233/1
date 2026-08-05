"""Regression coverage for CN stock name search (Bug A, DAV-88).

Root cause (Hermes):
1. The stock map built by ``_load_cn_stock_map`` keeps names verbatim, so a
   share-class letter can be full-width (e.g. 京东方Ａ). ``_search_cn_stock_by_name``
   matched without normalization, so a half-width "京东方A" query never matched
   the full-width map key.
2. The LLM-failure fallback passes the whole original sentence ("分析京东方") to
   ``_search_cn_stock_by_name``; a sentence is not a substring of any name, so it
   failed too.

Fix: NFKC-normalize both sides before exact/substring matching, and as a final
fallback strip common intent words from the sentence and retry.
"""

from unittest.mock import MagicMock, patch

import pytest

from api import main

_EMPTY_EXTRACTION = (
    '{"stock_name": null, "date": null, "horizons": ["short"], '
    '"focus_areas": [], "specific_questions": [], "user_context": {}}'
)


def _patch_stock_map(names_to_codes):
    """Serve a synthetic name→code map to _search_cn_stock_by_name."""
    return patch.object(main, "_load_cn_stock_map", return_value=dict(names_to_codes))


# ── _search_cn_stock_by_name unit tests ──────────────────────────────────────


def test_exact_name_still_hits():
    with _patch_stock_map({"贵州茅台": "600519.SH"}):
        assert main._search_cn_stock_by_name("贵州茅台") == "600519.SH"


@pytest.mark.parametrize("query", ["京东方A", "京东方Ａ"])
def test_half_and_full_width_a_suffix_hits(query):
    # Bug A #1: full-width Ａ in the map must match a half-width A query.
    with _patch_stock_map({"京东方Ａ": "000725.SZ"}):
        assert main._search_cn_stock_by_name(query) == "000725.SZ"


def test_whole_sentence_intent_prefix_fallback_hits():
    # Bug A #2: LLM-failure fallback passes the raw sentence; intent words
    # ("分析") must be stripped before matching.
    with _patch_stock_map({"京东方Ａ": "000725.SZ"}):
        assert main._search_cn_stock_by_name("分析京东方") == "000725.SZ"


def test_whole_sentence_full_message_fallback_hits():
    with _patch_stock_map({"京东方Ａ": "000725.SZ"}):
        assert main._search_cn_stock_by_name("帮我分析一下京东方怎么样") == "000725.SZ"


def test_extended_nonstandard_name_is_known_limitation():
    # "京东方科技" is neither a map name nor a substring of one — registered as a
    # known limitation (requires fuzzy/prefix matching beyond this bug's scope).
    with _patch_stock_map({"京东方Ａ": "000725.SZ"}):
        assert main._search_cn_stock_by_name("京东方科技") is None


def test_unrelated_sentence_returns_none():
    with _patch_stock_map({"京东方Ａ": "000725.SZ"}):
        assert main._search_cn_stock_by_name("今天天气不错") is None


# ── Integration: streaming extraction LLM-failure original-text fallback ──────


class _FakeLLM:
    async def astream(self, prompt):
        yield MagicMock(content=_EMPTY_EXTRACTION)


class _FakeClient:
    def __init__(self, llm):
        self._llm = llm

    def get_llm(self):
        return self._llm


def test_streaming_extraction_original_text_fallback_hits():
    """LLM returns null stock_name → whole sentence "分析京东方" must resolve."""
    import asyncio

    llm = _FakeLLM()
    with (
        patch(
            "tradingagents.llm_clients.factory.create_llm_client",
            return_value=_FakeClient(llm),
        ),
        patch.object(main, "_emit_job_event"),
        _patch_stock_map({"京东方Ａ": "000725.SZ"}),
    ):
        result = asyncio.run(
            main._ai_extract_symbol_and_date_streaming("分析京东方", {}, "job-1")
        )
    assert result[0] == "000725.SZ"
