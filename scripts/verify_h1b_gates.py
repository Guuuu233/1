#!/usr/bin/env python3
"""P3-H1b: 信用加权门槛校验与分层隔离只读检查脚本 (verify_h1b_gates.py).

功能：
1. 扫描当前数据库 (ReportDB / reports) 及历史 v2 结构化辩论报告；
2. 提取并核验 7 维门槛指标矩阵 (N, 分侧, 时间, T+5 完整率, 平衡, 偏置冻结, 幅度)；
3. 输出 7 维 PASS/FAIL 结构化矩阵与汇总 JSON；
4. 依据已批准门槛给出系统级决策建议（样本未达标时明确建议保持关 flag）。
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

# Ensure project root in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tradingagents.agents.utils.shadow_credit import (
    H1B_THRESHOLDS,
    calculate_shadow_credit_metrics,
    evaluate_h1b_system_gates,
    evaluate_model_bias_and_weights,
    extract_report_industry,
    extract_sample_cohort,
    filter_reports_by_cohort,
    filter_v2_completed_reports,
    is_cohort_homogeneous,
    is_qualifying_v2_report,
    normalize_report_for_evaluation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_h1b_gates")


def load_reports_from_db(
    db_path: Optional[str] = None,
    input_file: Optional[str] = None,
    input_dir: Optional[str] = None,
    return_excluded_counts: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, int]]]:
    """Load reports from SQLite database, input file/dir, or fallback paths, filtering strictly for completed v2 samples."""
    raw_reports: List[Dict[str, Any]] = []

    # 1. Explicit SQLite DB path (prioritized and strict: fails explicitly if invalid/inaccessible)
    if db_path and str(db_path).strip():
        p = Path(db_path)
        if not p.is_absolute():
            if not p.exists():
                p_root = Path(project_root) / p
                if p_root.exists():
                    p = p_root
        if not p.exists():
            logger.error("指定的 SQLite 数据库路径不存在: %s", db_path)
            raise FileNotFoundError(f"指定的 SQLite 数据库路径不存在: {db_path}")

        abs_path = str(p.resolve())
        db_url = f"sqlite:///{abs_path}"
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from api.database import ReportDB, _ensure_report_schema

            engine = create_engine(db_url, connect_args={"check_same_thread": False})
            try:
                _ensure_report_schema(target_engine=engine)
                SessionCls = sessionmaker(autocommit=False, autoflush=False, bind=engine)
                session = SessionCls()
                try:
                    db_reports = session.query(ReportDB).filter(ReportDB.status == "completed").all()
                    for r in db_reports:
                        data = r.to_dict()
                        raw_reports.append(data)
                    logger.info("从 SQLite 数据库 %s 中加载了 %d 份 completed 报告记录", abs_path, len(raw_reports))
                finally:
                    session.close()
            finally:
                engine.dispose()
        except Exception as exc:
            logger.error("读取指定 SQLite 数据库 %s 失败: %s", db_path, exc)
            raise RuntimeError(f"读取指定 SQLite 数据库 {db_path} 失败: {exc}") from exc

        # When explicit db_path is provided, strictly return filtered results from this db without fallback
        v2_reports, excluded_counts = filter_v2_completed_reports(raw_reports, return_excluded_counts=True)
        logger.info(
            "从指定数据库共检索到 %d 份原始样本，筛选出 %d 份合格 v2 结构化辩论样本 (排除 %d 份: legacy_null=%d, abstain=%d, invalid_run=%d, data_error=%d, no_trade=%d, wait=%d)",
            len(raw_reports),
            len(v2_reports),
            sum(excluded_counts.values()),
            excluded_counts.get("legacy_null", 0),
            excluded_counts.get("abstain", 0),
            excluded_counts.get("invalid_run", 0),
            excluded_counts.get("data_error", 0),
            excluded_counts.get("no_trade", 0),
            excluded_counts.get("wait", 0),
        )
        if return_excluded_counts:
            return v2_reports, excluded_counts
        return v2_reports

    # 2. Explicit input file
    if input_file:
        p = Path(input_file)
        if not p.is_absolute():
            p = Path(project_root) / p
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if "samples" in data and isinstance(data["samples"], list):
                        raw_reports.extend([dict(s) for s in data["samples"] if isinstance(s, dict)])
                    else:
                        raw_reports.append(data)
                elif isinstance(data, list):
                    raw_reports.extend([dict(s) for s in data if isinstance(s, dict)])
                logger.info("从 JSON 文件 %s 中加载了 %d 份原始样本", p.name, len(raw_reports))
            except Exception as e:
                logger.warning("读取 input-file %s 失败: %s", p, e)

    # 3. Explicit input directory
    if input_dir and not raw_reports:
        p_dir = Path(input_dir)
        if not p_dir.is_absolute():
            p_dir = Path(project_root) / p_dir
        if p_dir.exists() and p_dir.is_dir():
            for fpath in sorted(p_dir.glob("*.json")):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        if "samples" in data and isinstance(data["samples"], list):
                            raw_reports.extend([dict(s) for s in data["samples"] if isinstance(s, dict)])
                        else:
                            raw_reports.append(data)
                    elif isinstance(data, list):
                        raw_reports.extend([dict(s) for s in data if isinstance(s, dict)])
                except Exception as e:
                    logger.debug("读取 input-dir 文件 %s 失败: %s", fpath.name, e)
            if raw_reports:
                logger.info("从目录 %s 中加载了 %d 份原始样本", p_dir.name, len(raw_reports))

    # 4. Try loading via sqlalchemy ReportDB if db exists
    if not raw_reports:
        try:
            from sqlalchemy import inspect as sa_inspect
            from api.database import get_db_ctx, ReportDB, _ensure_report_schema, engine
            insp = sa_inspect(engine)
            if insp.has_table("reports"):
                _ensure_report_schema(target_engine=engine)
                with get_db_ctx() as db:
                    db_reports = db.query(ReportDB).filter(ReportDB.status == "completed").all()
                    for r in db_reports:
                        data = r.to_dict()
                        raw_reports.append(data)
                if raw_reports:
                    logger.info("从数据库中加载了 %d 份 completed 报告记录", len(raw_reports))
        except Exception as exc:
            logger.error("从数据库加载报告或 schema 迁移失败: %s", exc)
            raise RuntimeError(f"从数据库加载报告或 schema 迁移失败: {exc}") from exc

    # 5. Check golden audit reports as fallback/supplement
    if not raw_reports:
        golden_dir = os.path.join(project_root, "tests", "golden", "audit_20260823")
        if os.path.exists(golden_dir):
            for fname in sorted(os.listdir(golden_dir)):
                if fname.endswith("_result_data.json"):
                    fpath = os.path.join(golden_dir, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                raw_reports.append(data)
                    except Exception as e:
                        logger.debug("读取 golden 报告 %s 失败: %s", fname, e)

    # 6. Filter strictly for completed v2 reports with winner (and extract industry without fabrication)
    v2_reports, excluded_counts = filter_v2_completed_reports(raw_reports, return_excluded_counts=True)
    logger.info(
        "共检索到 %d 份原始样本，筛选出 %d 份合格 v2 结构化辩论样本 (排除 %d 份: legacy_null=%d, abstain=%d, invalid_run=%d, data_error=%d, no_trade=%d, wait=%d)",
        len(raw_reports),
        len(v2_reports),
        sum(excluded_counts.values()),
        excluded_counts.get("legacy_null", 0),
        excluded_counts.get("abstain", 0),
        excluded_counts.get("invalid_run", 0),
        excluded_counts.get("data_error", 0),
        excluded_counts.get("no_trade", 0),
        excluded_counts.get("wait", 0),
    )
    if return_excluded_counts:
        return v2_reports, excluded_counts
    return v2_reports


def format_gates_matrix_text(evaluation: Dict[str, Any], cohort_info: Optional[Dict[str, Any]] = None) -> str:
    """Format evaluation matrix to clean terminal table."""
    matrix = evaluation.get("matrix", {})
    summary = evaluation.get("summary", {})
    passed = evaluation.get("passed", False)
    rec = evaluation.get("recommendation", "KEEP_FALSE")
    cohort_key = evaluation.get("cohort") or (cohort_info.get("canonical_key") if cohort_info else "unspecified")
    shas = (cohort_info.get("commit_shas") if cohort_info else None) or evaluation.get("commit_shas") or summary.get("commit_shas") or []
    sha_str = f"{len(shas)} SHAs ({', '.join(shas[:2])}...)" if len(shas) > 2 else (', '.join(shas) if shas else "None")

    lines = [
        "=" * 80,
        "P3-H1b 信用加权门槛校验报告 (7维门槛矩阵 - Cohort 隔离版)",
        f"【评测 Cohort】: {cohort_key}",
        f"【Commit SHAs】: {sha_str}",
        "=" * 80,
    ]

    d_n = matrix.get("dimension_n", {})
    d_side = matrix.get("dimension_side", {})
    d_time = matrix.get("dimension_time", {})
    d_t5 = matrix.get("dimension_t5", {})
    d_balance = matrix.get("dimension_balance", {})
    d_bias = matrix.get("dimension_bias", {})
    d_mag = matrix.get("dimension_magnitude", {})

    def _st(p: bool) -> str:
        return "[ PASS ]" if p else "[ FAIL ]"

    # 1. N
    det_n = d_n.get("details", {})
    lines.append(
        f"1. 样本与覆盖 (N)      : {_st(d_n.get('passed', False))} "
        f"样本量={det_n.get('sample_count', 0)}/{det_n.get('min_required', 60)}, "
        f"标的数={det_n.get('unique_symbols', 0)}/{det_n.get('min_unique_symbols', 20)}, "
        f"行业数={det_n.get('unique_industries', 0)}/{det_n.get('min_industries', 5)}, "
        f"单标的占比={det_n.get('max_symbol_share', 0.0):.1%}<={det_n.get('max_allowed_share', 0.15):.1%}"
    )

    # 2. Side
    det_side = d_side.get("details", {})
    lines.append(
        f"2. 分侧样本与Claims    : {_st(d_side.get('passed', False))} "
        f"多空样本={det_side.get('bull_samples', 0)}/{det_side.get('bear_samples', 0)} (各>={det_side.get('min_side_samples', 25)}), "
        f"Verified Claims={det_side.get('bull_verified_claims', 0)}/{det_side.get('bear_verified_claims', 0)} (各>={det_side.get('min_verified_claims', 100)})"
    )

    # 3. Time
    det_time = d_time.get("details", {})
    lines.append(
        f"3. 时间跨度与市场状态  : {_st(d_time.get('passed', False))} "
        f"自然日={det_time.get('calendar_days', 0)}/{det_time.get('min_calendar_days', 45)}, "
        f"交易日={det_time.get('trading_days', 0)}/{det_time.get('min_trading_days', 30)}, "
        f"覆盖状态={det_time.get('market_regimes_covered', [])}"
    )

    # 4. T+5
    det_t5 = d_t5.get("details", {})
    lines.append(
        f"4. T+5 完整率          : {_st(d_t5.get('passed', False))} "
        f"完整率={det_t5.get('completeness_rate', 0.0):.1%}/{det_t5.get('min_required_rate', 0.95):.1%} "
        f"(已评估={det_t5.get('completed_count', 0)}/到期={det_t5.get('due_count', 0)})"
    )

    # 5. Balance
    det_bal = d_balance.get("details", {})
    lines.append(
        f"5. 多空平衡性          : {_st(d_balance.get('passed', False))} "
        f"多头占比={det_bal.get('bull_ratio', 0.0):.1%} (区间 [40.0%, 60.0%]), "
        f"多空差值={det_bal.get('side_diff', 0)}<={det_bal.get('max_allowed_diff', 10)}"
    )

    # 6. Bias Freeze
    det_bias = d_bias.get("details", {})
    lines.append(
        f"6. 偏置冻结指标        : {_st(d_bias.get('passed', False))} "
        f"Δverified={det_bias.get('delta_verified_rate', 0.0):.1%}<={det_bias.get('max_allowed_delta_v', 0.18):.1%}, "
        f"Δchallenge={det_bias.get('delta_challenge_adoption_rate', 0.0):.1%}<={det_bias.get('max_allowed_delta_ch', 0.25):.1%}, "
        f"克隆率={det_bias.get('clone_rate', 0.0):.1%}<={det_bias.get('max_allowed_clone_rate', 0.05):.1%}, "
        f"自洽硬闸触发率={det_bias.get('consistency_trigger_rate', 0.0):.1%}<={det_bias.get('max_allowed_consistency_rate', 0.05):.1%}"
    )

    # 7. Magnitude
    det_mag = d_mag.get("details", {})
    lines.append(
        f"7. 加权幅度范围        : {_st(d_mag.get('passed', True))} "
        f"系数范围={det_mag.get('range', [0.85, 1.15])}"
    )

    ex_counts = evaluation.get("excluded_counts") or summary.get("excluded_counts") or {}
    total_excluded = sum(ex_counts.values()) if ex_counts else 0
    lines.append("-" * 80)
    lines.append(f"【排除样本分类计数 (D-009 §5)】: 总排除={total_excluded}")
    lines.append(
        f"  legacy_null={ex_counts.get('legacy_null', 0)}, "
        f"abstain={ex_counts.get('abstain', 0)}, "
        f"invalid_run={ex_counts.get('invalid_run', 0)}, "
        f"data_error={ex_counts.get('data_error', 0)}, "
        f"no_trade={ex_counts.get('no_trade', 0)}, "
        f"wait={ex_counts.get('wait', 0)}"
    )

    lines.append("-" * 80)
    lines.append(f"【系统级门槛总状态】: {'PASS (全部通过)' if passed else 'FAIL (未达标)'}")
    lines.append(f"【Feature Flag 建议】: {'可开启 (ELIGIBLE_FOR_ACTIVATION)' if passed else '严禁开启，保持默认关闭 (KEEP_FALSE)'}")
    lines.append("=" * 80)

    return "\n".join(lines)


def run_verify(
    db_path: Optional[str] = None,
    output_json: Optional[str] = None,
    input_file: Optional[str] = None,
    input_dir: Optional[str] = None,
    as_of: Optional[str] = None,
    cohort: Optional[Union[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Execute gate verification and generate structured report under cohort isolation."""
    loaded_res = load_reports_from_db(
        db_path=db_path,
        input_file=input_file,
        input_dir=input_dir,
        return_excluded_counts=True,
    )
    if isinstance(loaded_res, tuple):
        reports, initial_excluded = loaded_res
    else:
        reports, initial_excluded = loaded_res, {}

    cohort_meta: Dict[str, Any] = {}
    if cohort is not None and str(cohort).strip():
        filtered_reports, cohort_meta = filter_reports_by_cohort(reports, cohort=cohort)
        reports = filtered_reports
    else:
        # Check homogeneity if cohort not specified
        is_homo, c_key = is_cohort_homogeneous(reports)
        cohort_meta = {
            "cohort_type": c_key or "unspecified",
            "canonical_key": c_key or "unspecified",
            "commit_shas": sorted({
                extract_sample_cohort(r)["generated_by_commit_sha"]
                for r in reports
                if extract_sample_cohort(r)["generated_by_commit_sha"]
            }),
        }

    # 1. 7-dimension gate evaluation
    gate_eval = evaluate_h1b_system_gates(reports, as_of=as_of, cohort=cohort, excluded_counts=initial_excluded)

    # 2. Model isolation evaluation
    isolation_eval = evaluate_model_bias_and_weights(
        reports,
        system_gate_passed=gate_eval["passed"],
    )

    cohort_label = cohort_meta.get("canonical_key") or (str(cohort) if cohort else "unspecified")
    report_result: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "P3-H1b",
        "cohort": cohort_label,
        "cohort_info": cohort_meta,
        "sample_count": len(reports),
        "excluded_counts": gate_eval.get("excluded_counts", initial_excluded),
        "gate_evaluation": gate_eval,
        "model_isolation": isolation_eval,
        "recommendation": gate_eval["recommendation"],
    }

    # Print terminal output
    print(format_gates_matrix_text(gate_eval, cohort_info=cohort_meta))
    print(f"分层隔离状态: credit_weighting_active={isolation_eval.get('credit_weighting_active')}, global_fallback_shadow={isolation_eval.get('global_fallback_shadow')}")
    print(f"模型权重分配: {isolation_eval.get('model_weights')}")
    if isolation_eval.get("bias_freeze_reasons"):
        print(f"偏置冻结原因: {isolation_eval.get('bias_freeze_reasons')}")

    if output_json:
        out_dir = os.path.dirname(output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report_result, f, ensure_ascii=False, indent=2)
        print(f"\n[产物生成] 结构化门槛校验 JSON 已写入: {output_json}")

    return report_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P3-H1b 信用加权门槛校验脚本")
    parser.add_argument(
        "--cohort",
        type=str,
        default=None,
        help="Cohort 标识 (必填, 如 legacy_unversioned 或三元版本 decision_model.v1:evidence_contract.v0:price_basis.unspecified)",
    )
    parser.add_argument("--db-path", type=str, default=None, help="SQLite 数据库路径")
    parser.add_argument("--input-file", type=str, default=None, help="指定评测 JSON 文件路径")
    parser.add_argument("--input-dir", type=str, default=None, help="指定评测 JSON 目录路径")
    parser.add_argument("--output-json", type=str, default="work/h1b_gates_report.json", help="输出汇总 JSON 路径")
    parser.add_argument("--as-of", type=str, default=None, help="评估基准日期 (YYYY-MM-DD)")
    args = parser.parse_args()

    # Fail-closed: 未传 --cohort 必须非零退出，严禁输出 PASS 报告
    if not args.cohort or not args.cohort.strip():
        logger.error("未传 --cohort 参数：必须显式指定 cohort（如 --cohort=legacy_unversioned 或三元版本）（fail-closed）")
        sys.exit(2)

    try:
        res = run_verify(
            db_path=args.db_path,
            output_json=args.output_json,
            input_file=args.input_file,
            input_dir=args.input_dir,
            as_of=args.as_of,
            cohort=args.cohort,
        )
    except Exception as exc:
        logger.error("门槛校验执行失败: %s", exc)
        sys.exit(1)
    # Exit code: 0 if valid execution
    sys.exit(0)
