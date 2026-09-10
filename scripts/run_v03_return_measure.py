#!/usr/bin/env python3
"""Runner script for V-03a Return Measurement Engine (DAV-802).

Strict Hard Constraints:
1. READ-ONLY connection to production database (mode=ro, physical write protection).
2. Measurement runs strictly on an atomic sqlite3 .backup() copy in work/.
3. Zero mutations to production database.
4. Generates comprehensive measurement report with baseline stamps and disclaimer:
   "半成品基线,非定性判断" in work/.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.eval.v03_return_measure import (
    DEFAULT_BENCHMARK_SYMBOL,
    DEFAULT_HOLD_DAYS,
    DEFAULT_STATUS_FILTER,
    DEFAULT_TARGET_USER_ID,
    CostModel,
    VendorPriceDataProvider,
    V03ReturnMeasureEngine,
)

DEFAULT_PROD_DB = "/Users/davidliu/Documents/TradingAgents-AShare/data/tradingagents.db"
DEFAULT_REPLICA_DB = str(PROJECT_ROOT / "work" / "tradingagents_v03_replica.db")
DEFAULT_REPORT_MD = str(PROJECT_ROOT / "work" / "v03_return_measurement_report.md")
DEFAULT_REPORT_JSON = str(PROJECT_ROOT / "work" / "v03_return_measurement_report.json")


def create_sqlite_backup(prod_db_path: str, backup_db_path: str) -> None:
    """Create an exact SQLite backup from production DB using sqlite3.backup().

    Production DB is opened strictly in read-only mode (mode=ro).
    """
    prod_path = Path(prod_db_path).resolve()
    if not prod_path.exists():
        raise FileNotFoundError(f"Production database not found: {prod_path}")

    backup_path = Path(backup_db_path).resolve()
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if backup_path.exists():
        backup_path.unlink()

    print(f"[1/4] Connecting to production DB (READ-ONLY mode=ro): {prod_path}")
    prod_uri = f"file:{prod_path}?mode=ro"
    prod_conn = sqlite3.connect(prod_uri, uri=True)

    print(f"[2/4] Performing sqlite3 .backup() to replica: {backup_path}")
    backup_conn = sqlite3.connect(str(backup_path))
    try:
        prod_conn.backup(backup_conn)
    finally:
        prod_conn.close()
        backup_conn.close()
    print("      Atomic backup completed. Production DB connection closed safely.")


def run_measurement(
    replica_db_path: str,
    output_md: str,
    output_json: str,
    hold_days: int = DEFAULT_HOLD_DAYS,
    limit: int | None = None,
    target_user_id: str = DEFAULT_TARGET_USER_ID,
    status_filter: str = DEFAULT_STATUS_FILTER,
) -> None:
    """Execute measurement engine on replica database with scope filtering."""
    user_stats = V03ReturnMeasureEngine.get_user_report_counts(
        replica_db_path, target_user_id=target_user_id
    )
    print(f"[3/4] Initializing V-03a measurement engine on replica: {replica_db_path}")
    print(f"      Target User: {target_user_id}")
    print(f"      Status Scope: {status_filter} (仅 completed)")
    print(f"      Account Stats: Total={user_stats['total']}, Completed={user_stats['completed']}, Failed={user_stats['failed']}")

    engine = V03ReturnMeasureEngine(
        cost_model=CostModel(),
        hold_days=hold_days,
        benchmark_symbol=DEFAULT_BENCHMARK_SYMBOL,
        price_provider=VendorPriceDataProvider(),
        target_user_id=target_user_id,
        status_filter=status_filter,
        target_user_stats=user_stats,
    )

    reports = engine.load_reports_from_db(
        replica_db_path,
        limit=limit,
        target_user_id=target_user_id,
        status_filter=status_filter,
    )
    print(f"      Loaded {len(reports)} completed reports from replica database for target account.")

    print("[4/4] Executing measurement across OOS segments (DEV / HISTORICAL / FORWARD)...")
    result = engine.measure_dataset(reports)

    # Print summary to console
    m_all = result.all_metrics
    print("\n" + "=" * 60)
    print("V-03a 测量结果摘要 (V-03a Progress Baseline Summary)")
    print(f"核心定位: {result.stamp.disclaimer}")
    print(f"评测账号: {result.stamp.target_user_id} | 状态限定: {result.stamp.status_filter} ({result.stamp.scope_filter_description})")
    print(f"账号分布: 总计={result.stamp.account_stats.get('total')} | completed={result.stamp.account_stats.get('completed')} | failed={result.stamp.account_stats.get('failed')}")
    print("=" * 60)
    print(f"总报告数: {m_all.total_reports}")
    print(f"评估候选数: {m_all.directional_candidate_count}")
    print(f"规范化合格: {m_all.mappable_count} | 隔离未规范: {m_all.unmappable_count}")
    print(f"入池数 (In-Pool): {m_all.in_pool_count} | 池排除数: {m_all.excluded_pool_count}")
    print(f"可交易数: {m_all.tradable_count} | 停牌/封死不可交易: {m_all.untradable_count}")
    print(f"有效评测数: {m_all.evaluated_count} | 数据缺口 (Typed-Missing): {m_all.typed_missing_count}")
    print(f"覆盖率 (Coverage Rate): {m_all.coverage_rate * 100:.2f}%")
    print(f"可评估率 (Evaluability Rate): {m_all.evaluability_rate * 100:.2f}%")
    if m_all.mean_net_return is not None:
        print(f"平均净收益率: {m_all.mean_net_return * 100:.2f}%")
    if m_all.mean_excess_return is not None:
        print(f"平均超额收益 (Alpha vs 沪深300): {m_all.mean_excess_return * 100:.2f}%")
    if m_all.win_rate is not None:
        print(f"胜率 (Win Rate): {m_all.win_rate * 100:.2f}%")
    print("=" * 60 + "\n")

    # Generate output files
    md_content = engine.generate_report_markdown(result)
    Path(output_md).write_text(md_content, encoding="utf-8")
    print(f"Wrote Markdown report to: {output_md}")

    json_dict = result.to_dict()
    Path(output_json).write_text(
        json.dumps(json_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote JSON report to: {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V-03a Return Measurement on Replica DB")
    parser.add_argument("--prod-db", default=DEFAULT_PROD_DB, help="Production DB path")
    parser.add_argument("--replica-db", default=DEFAULT_REPLICA_DB, help="Replica DB path")
    parser.add_argument("--output-md", default=DEFAULT_REPORT_MD, help="Output markdown path")
    parser.add_argument("--output-json", default=DEFAULT_REPORT_JSON, help="Output JSON path")
    parser.add_argument("--hold-days", type=int, default=DEFAULT_HOLD_DAYS, help="Holding days")
    parser.add_argument("--limit", type=int, default=None, help="Limit records for quick audit")
    parser.add_argument(
        "--target-user-id",
        type=str,
        default=DEFAULT_TARGET_USER_ID,
        help="Target user ID to evaluate (default: David)",
    )
    parser.add_argument(
        "--status-filter",
        type=str,
        default=DEFAULT_STATUS_FILTER,
        help="Status filter (default: completed)",
    )
    parser.add_argument(
        "--skip-backup", action="store_true", help="Skip backup if replica already exists"
    )
    args = parser.parse_args()

    if not args.skip_backup or not Path(args.replica_db).exists():
        create_sqlite_backup(args.prod_db, args.replica_db)

    run_measurement(
        replica_db_path=args.replica_db,
        output_md=args.output_md,
        output_json=args.output_json,
        hold_days=args.hold_days,
        limit=args.limit,
        target_user_id=args.target_user_id,
        status_filter=args.status_filter,
    )


if __name__ == "__main__":
    main()
