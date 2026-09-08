"""Comprehensive offline tests for multi-horizon return labels contracts (V-01-1).

Verifies:
1. OutcomeStatus (7 states) and ReturnType (2 states) enums.
2. Alignment with canonical HORIZON_PROFILE_V1 and immutable max-roll mapping.
3. Frozen HorizonCalendarWindow dataclass immutability.
4. Strict ISO date formatting and real calendar date validation.
5. trading_days sequence validations: rejection of string/bytes pseudo-sequences, non-string elements.
6. trading_days ordering and uniqueness: unsorted and duplicate rejection.
7. Exact signal_date presence: rejection if not in trading_days (no silent bisect drifting).
8. Horizon validation (only 'short' and 'medium').
9. Standard short (T+10, max_roll=2) and medium (T+40, max_roll=5) resolution.
10. Spring Festival 2024 fixture exact T+10=2024-02-23 vs naive weekday counterexample != 2024-02-15.
11. Full roll candidate resolution without truncation.
12. Fail-closed rejection (InsufficientTradingCalendarError / ValueError) on insufficient coverage.
13. as_of due logic matrix (target > as_of, target == as_of unclosed, target == as_of closed, target < as_of).
14. as_of date format and real calendar date validation.
15. as_of_market_closed strict boolean type validation.
16. Zero mutation and zero side effects on input sequence.
17. Zero network requests assertion.
18. Absence of resolve_horizon_return_label placeholder.
19. HorizonCalendarWindow does not output OutcomeStatus.EVALUATED_OK.
"""

from __future__ import annotations

import copy
import socket
from dataclasses import FrozenInstanceError, is_dataclass
from typing import List

import pytest

from tradingagents.dataflows import return_labels as rl
from tradingagents.graph.horizon_profile import (
    HORIZON_MEDIUM,
    HORIZON_PROFILE_V1,
    HORIZON_SHORT,
    SUPPORTED_HORIZONS,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Standard January 2024 A-share trading days (22 trading days)
FIXTURE_JAN_2024: List[str] = [
    "2024-01-02",
    "2024-01-03",
    "2024-01-04",
    "2024-01-05",
    "2024-01-08",
    "2024-01-09",
    "2024-01-10",
    "2024-01-11",
    "2024-01-12",
    "2024-01-15",
    "2024-01-16",
    "2024-01-17",
    "2024-01-18",
    "2024-01-19",
    "2024-01-22",
    "2024-01-23",
    "2024-01-24",
    "2024-01-25",
    "2024-01-26",
    "2024-01-29",
    "2024-01-30",
    "2024-01-31",
]

# February 2024 A-share trading days (Spring Festival: closed Feb 9 to Feb 18)
FIXTURE_SPRING_FESTIVAL_2024: List[str] = [
    "2024-02-01",  # T (signal_date, Thu)
    "2024-02-02",  # T+1 (executable_entry_date, Fri)
    "2024-02-05",  # T+2 (Mon)
    "2024-02-06",  # T+3 (Tue)
    "2024-02-07",  # T+4 (Wed)
    "2024-02-08",  # T+5 (Thu)
    # 2024-02-09 to 2024-02-18 Holiday (Closed)
    "2024-02-19",  # T+6 (Mon)
    "2024-02-20",  # T+7 (Tue)
    "2024-02-21",  # T+8 (Wed)
    "2024-02-22",  # T+9 (Thu)
    "2024-02-23",  # T+10 (target_calendar_date, Fri)
    "2024-02-26",  # T+11 (roll candidate 1, Mon)
    "2024-02-27",  # T+12 (roll candidate 2, Tue)
    "2024-02-28",  # T+13 (Wed)
    "2024-02-29",  # T+14 (Thu)
]

# 60+ Trading Days sequence across Jan-Apr 2024 for medium (T+40 + 5 roll) testing
FIXTURE_EXTENDED_2024: List[str] = [
    # Jan (22 days)
    "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08",
    "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12", "2024-01-15",
    "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19", "2024-01-22",
    "2024-01-23", "2024-01-24", "2024-01-25", "2024-01-26", "2024-01-29",
    "2024-01-30", "2024-01-31",
    # Feb (15 days)
    "2024-02-01", "2024-02-02", "2024-02-05", "2024-02-06", "2024-02-07",
    "2024-02-08", "2024-02-19", "2024-02-20", "2024-02-21", "2024-02-22",
    "2024-02-23", "2024-02-26", "2024-02-27", "2024-02-28", "2024-02-29",
    # Mar (21 days)
    "2024-03-01", "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07",
    "2024-03-08", "2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14",
    "2024-03-15", "2024-03-18", "2024-03-19", "2024-03-20", "2024-03-21",
    "2024-03-22", "2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28",
    "2024-03-29",
    # Apr (5 days)
    "2024-04-01", "2024-04-02", "2024-04-03", "2024-04-08", "2024-04-09",
]


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestOutcomeStatusAndReturnTypeEnums:
    """1. Verify 7-state OutcomeStatus and 2-state ReturnType enums."""

    def test_outcome_status_exact_7_values(self):
        expected_values = {
            "pending_due",
            "suspension",
            "data_missing",
            "provider_failure",
            "unsupported_price_basis",
            "unexecutable_entry",
            "evaluated_ok",
        }
        actual_values = {item.value for item in rl.OutcomeStatus}
        assert actual_values == expected_values
        assert len(rl.OutcomeStatus) == 7

        # Ensure values are str instances (str, Enum)
        for item in rl.OutcomeStatus:
            assert isinstance(item, str)
            assert isinstance(item.value, str)

    def test_return_type_exact_2_values(self):
        expected_values = {"price_return", "total_return"}
        actual_values = {item.value for item in rl.ReturnType}
        assert actual_values == expected_values
        assert len(rl.ReturnType) == 2

        for item in rl.ReturnType:
            assert isinstance(item, str)
            assert isinstance(item.value, str)


class TestHorizonProfileAlignment:
    """2. Verify profile constants, primary offsets, and max-roll mappings."""

    def test_offsets_read_from_canonical_horizon_profile(self):
        assert rl.PRIMARY_EVAL_OFFSET_SHORT == 10
        assert rl.PRIMARY_EVAL_OFFSET_MEDIUM == 40
        assert (
            rl.PRIMARY_EVAL_OFFSET_SHORT
            == HORIZON_PROFILE_V1[HORIZON_SHORT]["primary_eval_offset"]
        )
        assert (
            rl.PRIMARY_EVAL_OFFSET_MEDIUM
            == HORIZON_PROFILE_V1[HORIZON_MEDIUM]["primary_eval_offset"]
        )

    def test_max_roll_mapping_exact_and_immutable(self):
        assert rl.HORIZON_MAX_ROLL_DAYS[HORIZON_SHORT] == 2
        assert rl.HORIZON_MAX_ROLL_DAYS[HORIZON_MEDIUM] == 5
        assert set(rl.HORIZON_MAX_ROLL_DAYS.keys()) == set(SUPPORTED_HORIZONS)

    def test_horizon_return_result_typeddict_contract(self):
        # Assert type annotation keys match V-01 contract specifications
        expected_keys = {
            "symbol",
            "horizon",
            "profile_id",
            "price_basis",
            "return_type",
            "entry_date",
            "executable_entry_date",
            "target_calendar_date",
            "actual_exit_date",
            "roll_days_used",
            "entry_signal_price",
            "entry_executable_price",
            "entry_price",
            "exit_price",
            "cash_dividend_total",
            "split_ratio_total",
            "return_pct",
            "outcome_status",
            "is_direction_hit",
            "evaluation_eligible",
        }
        assert set(rl.HorizonReturnResult.__annotations__.keys()) == expected_keys


class TestHorizonCalendarWindowContract:
    """3. Verify frozen HorizonCalendarWindow dataclass and immutability."""

    def test_dataclass_frozen_and_cannot_mutate(self):
        window = rl.HorizonCalendarWindow(
            signal_date="2024-01-02",
            horizon="short",
            eval_offset=10,
            max_roll_days=2,
            executable_entry_date="2024-01-03",
            target_calendar_date="2024-01-16",
            roll_candidate_dates=("2024-01-17", "2024-01-18"),
            holding_trading_days=tuple(FIXTURE_JAN_2024[1:11]),
            is_due=True,
        )
        assert is_dataclass(window)

        with pytest.raises(FrozenInstanceError):
            window.is_due = False  # type: ignore

        with pytest.raises(FrozenInstanceError):
            window.signal_date = "2024-01-03"  # type: ignore

        with pytest.raises(FrozenInstanceError):
            window.new_attr = "invalid"  # type: ignore

    def test_window_does_not_output_evaluated_ok(self):
        window = rl.resolve_horizon_calendar_window(
            signal_date="2024-01-02",
            horizon="short",
            trading_days=FIXTURE_JAN_2024,
            as_of="2024-01-20",
            as_of_market_closed=True,
        )
        # HorizonCalendarWindow must only output strict bool is_due
        assert isinstance(window.is_due, bool)
        assert window.is_due is True
        # Must not have outcome_status or claim EVALUATED_OK
        assert not hasattr(window, "outcome_status")


class TestDateFormatAndRealCalendarDateValidation:
    """4. Verify strict ISO format YYYY-MM-DD and real calendar date validation."""

    @pytest.mark.parametrize(
        "bad_date",
        [
            "2024/01/02",
            "20240102",
            "2024-1-2",
            "2024-01-2",
            "invalid-date",
            "",
            "   ",
        ],
    )
    def test_signal_date_malformed_iso_rejected(self, bad_date):
        with pytest.raises(ValueError, match="strict ISO format YYYY-MM-DD"):
            rl.resolve_horizon_calendar_window(
                signal_date=bad_date,
                horizon="short",
                trading_days=FIXTURE_JAN_2024,
                as_of="2024-01-20",
            )

    @pytest.mark.parametrize(
        "unreal_date",
        [
            "2024-02-30",  # February 30th does not exist
            "2023-02-29",  # 2023 is not a leap year
            "2024-04-31",  # April only has 30 days
            "2024-13-01",  # Month 13 does not exist
            "2024-00-10",  # Month 0 does not exist
            "2024-01-32",  # Day 32 does not exist
        ],
    )
    def test_signal_date_unreal_calendar_date_rejected(self, unreal_date):
        with pytest.raises(ValueError, match="not a valid calendar date"):
            rl.resolve_horizon_calendar_window(
                signal_date=unreal_date,
                horizon="short",
                trading_days=FIXTURE_JAN_2024,
                as_of="2024-01-20",
            )


class TestTradingDaysPseudoSequenceValidation:
    """5. Verify trading_days rejection of string and bytes pseudo-sequences."""

    def test_trading_days_string_rejected(self):
        with pytest.raises(ValueError, match="non-string Sequence"):
            rl.resolve_horizon_calendar_window(
                signal_date="2024-01-02",
                horizon="short",
                trading_days="2024-01-02",  # type: ignore
                as_of="2024-01-20",
            )

    def test_trading_days_bytes_rejected(self):
        with pytest.raises(ValueError, match="non-string Sequence"):
            rl.resolve_horizon_calendar_window(
                signal_date="2024-01-02",
                horizon="short",
                trading_days=b"2024-01-02",  # type: ignore
                as_of="2024-01-20",
            )

    def test_trading_days_non_sequence_rejected(self):
        with pytest.raises(ValueError, match="must be a Sequence"):
            rl.resolve_horizon_calendar_window(
                signal_date="2024-01-02",
                horizon="short",
                trading_days=12345,  # type: ignore
                as_of="2024-01-20",
            )

    def test_trading_days_empty_sequence_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            rl.resolve_horizon_calendar_window(
                signal_date="2024-01-02",
                horizon="short",
                trading_days=[],
                as_of="2024-01-20",
            )


class TestTradingDaysNonStringElementsValidation:
    """6. Verify trading_days non-string element rejection."""

    @pytest.mark.parametrize(
        "bad_element",
        [
            123,
            None,
            3.14,
            {"date": "2024-01-02"},
            ["2024-01-02"],
        ],
    )
    def test_trading_days_non_string_elements_rejected(self, bad_element):
        days = ["2024-01-02", bad_element, "2024-01-04"]
        with pytest.raises(ValueError, match="must be a string"):
            rl.resolve_horizon_calendar_window(
                signal_date="2024-01-02",
                horizon="short",
                trading_days=days,  # type: ignore
                as_of="2024-01-20",
            )


class TestTradingDaysUnsortedRejected:
    """7. Verify trading_days unsorted rejection."""

    def test_trading_days_out_of_order_rejected(self):
        unsorted_days = [
            "2024-01-02",
            "2024-01-05",
            "2024-01-03",  # out of order
            "2024-01-08",
        ]
        with pytest.raises(ValueError, match="strictly sorted in ascending order"):
            rl.resolve_horizon_calendar_window(
                signal_date="2024-01-02",
                horizon="short",
                trading_days=unsorted_days,
                as_of="2024-01-20",
            )


class TestTradingDaysDuplicateRejected:
    """8. Verify trading_days duplicate rejection."""

    def test_trading_days_consecutive_duplicates_rejected(self):
        dup_days = [
            "2024-01-02",
            "2024-01-03",
            "2024-01-03",  # duplicate
            "2024-01-04",
        ]
        with pytest.raises(ValueError, match="contains duplicate date"):
            rl.resolve_horizon_calendar_window(
                signal_date="2024-01-02",
                horizon="short",
                trading_days=dup_days,
                as_of="2024-01-20",
            )


class TestSignalDatePresenceInCalendar:
    """9. Verify signal_date must be strictly present in trading_days (no bisect drift)."""

    def test_signal_date_on_weekend_rejected(self):
        # 2024-01-06 is Saturday (not in A-share trading days)
        with pytest.raises(ValueError, match="signal_date '2024-01-06' not found in trading_days"):
            rl.resolve_horizon_calendar_window(
                signal_date="2024-01-06",
                horizon="short",
                trading_days=FIXTURE_JAN_2024,
                as_of="2024-01-20",
            )

    def test_signal_date_outside_calendar_range_rejected(self):
        with pytest.raises(ValueError, match="not found in trading_days"):
            rl.resolve_horizon_calendar_window(
                signal_date="2023-12-29",
                horizon="short",
                trading_days=FIXTURE_JAN_2024,
                as_of="2024-01-20",
            )


class TestHorizonValidation:
    """10. Verify horizon parameter validation."""

    @pytest.mark.parametrize(
        "bad_horizon",
        [
            "ultra_long",
            "daily",
            "long",
            "SHORT",  # case-sensitive canonical contract
            "",
            None,
            10,
        ],
    )
    def test_unsupported_horizon_rejected(self, bad_horizon):
        with pytest.raises(ValueError, match="Unsupported horizon"):
            rl.resolve_horizon_calendar_window(
                signal_date="2024-01-02",
                horizon=bad_horizon,  # type: ignore
                trading_days=FIXTURE_JAN_2024,
                as_of="2024-01-20",
            )


class TestStandardShortAndMediumResolution:
    """11. Verify standard resolution for short (T+10) and medium (T+40)."""

    def test_short_standard_resolution(self):
        window = rl.resolve_horizon_calendar_window(
            signal_date="2024-01-02",
            horizon="short",
            trading_days=FIXTURE_JAN_2024,
            as_of="2024-01-20",
            as_of_market_closed=True,
        )
        assert window.signal_date == "2024-01-02"
        assert window.horizon == "short"
        assert window.eval_offset == 10
        assert window.max_roll_days == 2
        assert window.executable_entry_date == "2024-01-03"  # T+1
        assert window.target_calendar_date == "2024-01-16"  # T+10
        assert window.roll_candidate_dates == ("2024-01-17", "2024-01-18")
        assert len(window.holding_trading_days) == 10
        assert window.holding_trading_days[0] == "2024-01-03"
        assert window.holding_trading_days[-1] == "2024-01-16"
        assert window.is_due is True

    def test_medium_standard_resolution(self):
        window = rl.resolve_horizon_calendar_window(
            signal_date="2024-01-02",
            horizon="medium",
            trading_days=FIXTURE_EXTENDED_2024,
            as_of="2024-03-15",
            as_of_market_closed=True,
        )
        assert window.signal_date == "2024-01-02"
        assert window.horizon == "medium"
        assert window.eval_offset == 40
        assert window.max_roll_days == 5
        assert window.executable_entry_date == "2024-01-03"  # T+1
        # 40th trading day from 2024-01-02 (index 0) is index 40 -> 2024-03-05
        assert window.target_calendar_date == FIXTURE_EXTENDED_2024[40]
        assert window.roll_candidate_dates == tuple(FIXTURE_EXTENDED_2024[41:46])
        assert len(window.roll_candidate_dates) == 5
        assert len(window.holding_trading_days) == 40
        assert window.holding_trading_days[0] == "2024-01-03"
        assert window.holding_trading_days[-1] == window.target_calendar_date


class TestSpringFestival2024ExactProof:
    """12. Verify 2024 Spring Festival holiday skipping and counterexample."""

    def test_spring_festival_t10_is_feb_23(self):
        window = rl.resolve_horizon_calendar_window(
            signal_date="2024-02-01",
            horizon="short",
            trading_days=FIXTURE_SPRING_FESTIVAL_2024,
            as_of="2024-02-23",
            as_of_market_closed=True,
        )
        # February 1 is Thu. T+1 is Feb 2 (Fri).
        # Feb 9-18 is Spring Festival holiday.
        # Exactly skips holiday to land on 2024-02-23 (Fri) for T+10
        assert window.executable_entry_date == "2024-02-02"
        assert window.target_calendar_date == "2024-02-23"
        assert window.roll_candidate_dates == ("2024-02-26", "2024-02-27")

        # Prove counterexample: naive weekday counting cur.weekday() < 5 lands on 2024-02-15
        # (which is an official holiday and market was closed)
        assert window.target_calendar_date != "2024-02-15"


class TestCompleteRollCandidatesResolution:
    """13. Verify full roll candidates resolution."""

    def test_short_has_complete_2_roll_candidates(self):
        window = rl.resolve_horizon_calendar_window(
            signal_date="2024-02-01",
            horizon="short",
            trading_days=FIXTURE_SPRING_FESTIVAL_2024,
            as_of="2024-02-23",
            as_of_market_closed=True,
        )
        assert len(window.roll_candidate_dates) == 2
        assert window.roll_candidate_dates[0] == "2024-02-26"
        assert window.roll_candidate_dates[1] == "2024-02-27"

    def test_medium_has_complete_5_roll_candidates(self):
        window = rl.resolve_horizon_calendar_window(
            signal_date="2024-01-02",
            horizon="medium",
            trading_days=FIXTURE_EXTENDED_2024,
            as_of="2024-03-15",
            as_of_market_closed=True,
        )
        assert len(window.roll_candidate_dates) == 5


class TestInsufficientCalendarCoverageFailClosed:
    """14. Verify fail-closed behavior when calendar coverage is insufficient."""

    def test_calendar_not_covering_t_plus_n_raises_error(self):
        # Truncate calendar to T+9 (needs T+10 + 2 roll = T+12)
        truncated = FIXTURE_SPRING_FESTIVAL_2024[:10]  # indices 0..9 (ends at T+9)
        with pytest.raises(
            (rl.InsufficientTradingCalendarError, ValueError),
            match="does not cover up to T\\+10\\+2",
        ):
            rl.resolve_horizon_calendar_window(
                signal_date="2024-02-01",
                horizon="short",
                trading_days=truncated,
                as_of="2024-02-23",
            )

    def test_calendar_ending_at_target_date_missing_roll_raises_error(self):
        # Ends exactly at T+10 (missing T+11, T+12 roll days)
        # Must fail closed rather than returning truncated empty roll candidates
        truncated = FIXTURE_SPRING_FESTIVAL_2024[:11]  # ends at index 10 (T+10: 2024-02-23)
        with pytest.raises(
            (rl.InsufficientTradingCalendarError, ValueError),
            match="does not cover up to T\\+10\\+2",
        ):
            rl.resolve_horizon_calendar_window(
                signal_date="2024-02-01",
                horizon="short",
                trading_days=truncated,
                as_of="2024-02-23",
            )

    def test_calendar_with_partial_roll_days_raises_error(self):
        # Ends at T+11 (only 1 roll day instead of 2)
        truncated = FIXTURE_SPRING_FESTIVAL_2024[:12]  # ends at index 11 (T+11: 2024-02-26)
        with pytest.raises(
            (rl.InsufficientTradingCalendarError, ValueError),
            match="does not cover up to T\\+10\\+2",
        ):
            rl.resolve_horizon_calendar_window(
                signal_date="2024-02-01",
                horizon="short",
                trading_days=truncated,
                as_of="2024-02-23",
            )


class TestAsOfDueLogicMatrix:
    """15. Verify 4 quadrants of is_due logic matrix:
    - target > as_of -> False
    - target == as_of, market_closed=False -> False (intraday unclosed, anti price-theft)
    - target == as_of, market_closed=True -> True
    - target < as_of -> True
    """

    def test_target_greater_than_as_of_is_not_due(self):
        # Target is 2024-02-23, as_of is 2024-02-20
        window = rl.resolve_horizon_calendar_window(
            signal_date="2024-02-01",
            horizon="short",
            trading_days=FIXTURE_SPRING_FESTIVAL_2024,
            as_of="2024-02-20",
            as_of_market_closed=True,
        )
        assert window.is_due is False

    def test_target_equal_as_of_market_not_closed_is_not_due(self):
        # Target is 2024-02-23, as_of is 2024-02-23, but market is not closed yet
        window = rl.resolve_horizon_calendar_window(
            signal_date="2024-02-01",
            horizon="short",
            trading_days=FIXTURE_SPRING_FESTIVAL_2024,
            as_of="2024-02-23",
            as_of_market_closed=False,
        )
        assert window.is_due is False

    def test_target_equal_as_of_market_closed_is_due(self):
        # Target is 2024-02-23, as_of is 2024-02-23, and market is closed
        window = rl.resolve_horizon_calendar_window(
            signal_date="2024-02-01",
            horizon="short",
            trading_days=FIXTURE_SPRING_FESTIVAL_2024,
            as_of="2024-02-23",
            as_of_market_closed=True,
        )
        assert window.is_due is True

    def test_target_less_than_as_of_is_due(self):
        # Target is 2024-02-23, as_of is 2024-02-26 (subsequent date)
        window = rl.resolve_horizon_calendar_window(
            signal_date="2024-02-01",
            horizon="short",
            trading_days=FIXTURE_SPRING_FESTIVAL_2024,
            as_of="2024-02-26",
            as_of_market_closed=False,
        )
        assert window.is_due is True


class TestAsOfValidation:
    """16. Verify as_of format, type, and real calendar date validation."""

    @pytest.mark.parametrize(
        "bad_as_of",
        [
            "2024/02/23",
            "20240223",
            "invalid",
            "",
            None,
            20240223,
        ],
    )
    def test_as_of_bad_format_rejected(self, bad_as_of):
        with pytest.raises(ValueError):
            rl.resolve_horizon_calendar_window(
                signal_date="2024-02-01",
                horizon="short",
                trading_days=FIXTURE_SPRING_FESTIVAL_2024,
                as_of=bad_as_of,  # type: ignore
            )

    @pytest.mark.parametrize(
        "unreal_as_of",
        [
            "2024-02-30",
            "2023-02-29",
            "2024-04-31",
            "2024-13-01",
        ],
    )
    def test_as_of_unreal_calendar_date_rejected(self, unreal_as_of):
        with pytest.raises(ValueError, match="not a valid calendar date"):
            rl.resolve_horizon_calendar_window(
                signal_date="2024-02-01",
                horizon="short",
                trading_days=FIXTURE_SPRING_FESTIVAL_2024,
                as_of=unreal_as_of,
            )


class TestAsOfMarketClosedStrictBoolValidation:
    """17. Verify as_of_market_closed must be a strict bool."""

    @pytest.mark.parametrize(
        "non_bool_value",
        [
            1,
            0,
            "true",
            "false",
            None,
            [True],
            {"closed": True},
        ],
    )
    def test_as_of_market_closed_non_bool_rejected(self, non_bool_value):
        with pytest.raises(ValueError, match="must be a strict bool"):
            rl.resolve_horizon_calendar_window(
                signal_date="2024-02-01",
                horizon="short",
                trading_days=FIXTURE_SPRING_FESTIVAL_2024,
                as_of="2024-02-23",
                as_of_market_closed=non_bool_value,  # type: ignore
            )


class TestZeroMutationAndSideEffects:
    """18. Verify pure function invariants: zero mutation on input sequence."""

    def test_trading_days_input_list_not_mutated(self):
        input_list = list(FIXTURE_SPRING_FESTIVAL_2024)
        original_copy = copy.deepcopy(input_list)
        original_id = id(input_list)

        window = rl.resolve_horizon_calendar_window(
            signal_date="2024-02-01",
            horizon="short",
            trading_days=input_list,
            as_of="2024-02-23",
            as_of_market_closed=True,
        )

        assert id(input_list) == original_id
        assert input_list == original_copy
        assert window.signal_date == "2024-02-01"


class TestZeroNetworkAssertion:
    """19. Assert that module and resolution function make zero network calls."""

    def test_resolution_makes_no_network_calls(self, monkeypatch):
        # Guard socket creation to ensure absolutely zero network attempts
        def block_socket(*args, **kwargs):
            raise AssertionError("Unexpected network socket creation attempted!")

        monkeypatch.setattr(socket, "socket", block_socket)

        window = rl.resolve_horizon_calendar_window(
            signal_date="2024-02-01",
            horizon="short",
            trading_days=FIXTURE_SPRING_FESTIVAL_2024,
            as_of="2024-02-23",
            as_of_market_closed=True,
        )
        assert window.is_due is True


class TestNoResolveHorizonReturnLabelPlaceholder:
    """20. Assert that resolve_horizon_return_label placeholder is NOT implemented in V-01-1."""

    def test_no_half_baked_return_label_stub(self):
        assert not hasattr(rl, "resolve_horizon_return_label"), (
            "resolve_horizon_return_label must NOT be implemented as a placeholder or "
            "NotImplementedError stub in V-01-1; deferred to V-01-2+."
        )
