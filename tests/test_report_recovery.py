from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base, ReportDB
from api.services import report_service


def _make_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return TestingSessionLocal()


def _add_report(
    db,
    *,
    status: str = "running",
    decision=None,
    final_trade_decision=None,
    result_data=None,
    error=None,
):
    now = datetime.now(timezone.utc)
    report = ReportDB(
        id=uuid4().hex,
        user_id=uuid4().hex,
        symbol="600519.SH",
        trade_date="2026-04-01",
        status=status,
        decision=decision,
        final_trade_decision=final_trade_decision,
        result_data=result_data,
        error=error,
        created_at=now,
        updated_at=now,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def test_recover_stale_active_reports_marks_empty_running_report_failed():
    db = _make_session()
    try:
        report = _add_report(db, status="running")

        result = report_service.recover_stale_active_reports(db)

        refreshed = db.query(ReportDB).filter(ReportDB.id == report.id).first()
        assert result == {"total": 1, "failed": 1}
        assert refreshed is not None
        assert refreshed.status == "failed"
        assert refreshed.error == report_service.STALE_REPORT_ERROR_MESSAGE
    finally:
        db.close()


def test_recover_stale_active_reports_marks_partial_running_report_failed():
    db = _make_session()
    try:
        report = _add_report(
            db,
            status="running",
            final_trade_decision="结论：持有\n目标价：1750\n止损价：1650",
            result_data={"final_trade_decision": "结论：持有"},
        )

        result = report_service.recover_stale_active_reports(db)

        refreshed = db.query(ReportDB).filter(ReportDB.id == report.id).first()
        assert result == {"total": 1, "failed": 1}
        assert refreshed is not None
        assert refreshed.status == "failed"
        assert refreshed.error == report_service.STALE_REPORT_ERROR_MESSAGE
    finally:
        db.close()


def test_finalize_orphan_report_marks_pending_report_failed():
    db = _make_session()
    try:
        report = _add_report(db, status="pending")

        refreshed = report_service.finalize_orphan_report(db, report)

        assert refreshed.status == "failed"
        assert refreshed.error == report_service.STALE_REPORT_ERROR_MESSAGE
    finally:
        db.close()


def test_create_report_clears_previous_failure_error_on_success():
    db = _make_session()
    try:
        report = _add_report(
            db,
            status="failed",
            error="任务超时（旧策略）",
        )

        finalized = report_service.create_report(
            db=db,
            symbol=report.symbol,
            trade_date=report.trade_date,
            decision="BUY",
            result_data={"final_trade_decision": "结论：买入"},
            report_id=report.id,
        )

        assert finalized.status == "completed"
        assert finalized.error is None
        assert finalized.decision == "BUY"
    finally:
        db.close()


def test_structured_report_new_fields_default_when_missing_or_null():
    missing = report_service.StructuredReport()
    explicit_nulls = report_service.StructuredReport(
        data_gaps=None,
        falsification_conditions=None,
        not_applicable=None,
    )

    for structured in (missing, explicit_nulls):
        assert structured.probability is None
        assert structured.data_gaps == []
        assert structured.falsification_conditions == []
        assert structured.not_applicable is False


def test_create_report_persists_new_structured_fields():
    db = _make_session()
    try:
        report = report_service.create_report(
            db=db,
            symbol="600519.SH",
            trade_date="2026-07-29",
            decision="HOLD",
            result_data={"final_trade_decision": "结论：持有"},
            probability=0.42,
            data_gaps=["缺少盘中资金流"],
            falsification_conditions=["收入增速连续两个季度回升"],
            not_applicable=True,
        )

        persisted = db.query(ReportDB).filter(ReportDB.id == report.id).one()
        assert persisted.probability == 0.42
        assert persisted.data_gaps == ["缺少盘中资金流"]
        assert persisted.falsification_conditions == ["收入增速连续两个季度回升"]
        assert persisted.not_applicable is True
        assert persisted.to_dict()["not_applicable"] is True
    finally:
        db.close()
