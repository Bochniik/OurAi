"""CPU-only postmortem for the failed v5 direction development run.

Reads only the v4 TRAIN split, its manifest, and the existing v5 train-only
development report. This diagnostic does not train a model, change the frozen
experiment, evaluate holdouts, or choose hyperparameters.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from statistics import fmean


EXPECTED_DATASET_ID = "7d5f3b066a30ba878bba"
EXPECTED_REPORT = "v5_temporal_market_context_gru_direction_train_only_v1"
HORIZON_MINUTES = 15
SEQUENCE_LENGTH = 60


@dataclass(slots=True)
class AuditRow:
    symbol: str
    market_date: str
    session: str
    minute_utc: int
    direction: str | None
    magnitude_bps: float | None
    sequence_eligible: bool = False


def parse_training_row(payload: dict) -> AuditRow:
    stamp = datetime.fromisoformat(str(payload["timestamp"]).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("training timestamp must be timezone-aware")
    minute = int(stamp.astimezone(timezone.utc).timestamp() // 60)
    target = payload.get("targets", {}).get("15m") or {}
    direction = str(target.get("direction") or "").upper()
    if direction not in {"UP", "DOWN"}:
        direction = None
    future_return = target.get("future_return")
    magnitude = None
    if future_return is not None:
        candidate = float(future_return)
        if isfinite(candidate):
            magnitude = abs(candidate) * 10000.0
    return AuditRow(
        symbol=str(payload["symbol"]),
        market_date=str(payload["session_date"]),
        session=str(payload["session"]),
        minute_utc=minute,
        direction=direction,
        magnitude_bps=magnitude,
    )


def mark_sequence_eligible(rows: list[AuditRow], *, length: int = SEQUENCE_LENGTH) -> None:
    """Require the same exact-minute 60-step target run as the frozen v5 corpus."""
    previous_group = None
    previous_minute = None
    streak = 0
    for row in rows:
        group = (row.symbol, row.market_date, row.session)
        if group == previous_group and previous_minute is not None and row.minute_utc - previous_minute == 1:
            streak += 1
        else:
            streak = 1
        row.sequence_eligible = streak >= length
        previous_group = group
        previous_minute = row.minute_utc


def nonoverlapping_label_count(rows: list[AuditRow], *, horizon_minutes: int = HORIZON_MINUTES) -> int:
    """Greedy bound on distinct, nonoverlapping forecast intervals per symbol/day."""
    last_start: dict[tuple[str, str, str], int] = {}
    count = 0
    for row in rows:
        key = (row.symbol, row.market_date, row.session)
        last = last_start.get(key)
        if last is None or row.minute_utc - last >= horizon_minutes:
            count += 1
            last_start[key] = row.minute_utc
    return count


def summarize_fold(rows: list[AuditRow], fold_report: dict) -> dict:
    validation_dates = fold_report["validation_market_dates"]
    if not validation_dates:
        raise ValueError("missing development-fold dates")
    first_date = min(validation_dates)
    threshold = float(fold_report["actual_tail_threshold_bps"])
    train_rows = [row for row in rows if row.market_date < first_date]
    tail_rows = [
        row for row in train_rows
        if row.sequence_eligible and row.direction in {"UP", "DOWN"}
        and row.magnitude_bps is not None and row.magnitude_bps >= threshold
    ]
    if not tail_rows:
        raise ValueError("no eligible tail training rows")
    independent_count = nonoverlapping_label_count(tail_rows)
    observed = len(tail_rows)
    reported = int(fold_report["gru_training_sequences"])
    losses = [float(item) for item in fold_report["gru_epoch_losses"]]
    if not losses:
        raise ValueError("missing GRU epoch losses")
    valid_directional = sum(row.direction is not None for row in train_rows)
    return {
        "fold": int(fold_report["fold"]),
        "training_dates": len({row.market_date for row in train_rows}),
        "train_rows": len(train_rows),
        "train_directional_fraction": valid_directional / len(train_rows),
        "tail_sequences_reconstructed": observed,
        "tail_sequences_reported": reported,
        "sequence_count_matches_report": observed == reported,
        "tail_up_fraction": sum(row.direction == "UP" for row in tail_rows) / observed,
        "approx_nonoverlapping_tail_intervals": independent_count,
        "nonoverlap_fraction_of_tail": independent_count / observed,
        "train_loss_epoch_1": losses[0],
        "train_loss_epoch_last": losses[-1],
        "train_loss_drop": losses[0] - losses[-1],
        "development_brier": float(fold_report["gru_brier"]),
        "development_null_brier": float(fold_report["tail_null_brier"]),
        "selected_coverage": float(fold_report["selected_coverage"]),
        "magnitude_enrichment_ratio": float(fold_report["magnitude_enrichment_ratio"]),
    }


def run_audit(dataset_dir: Path, report_path: Path, output_path: Path) -> dict:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != EXPECTED_DATASET_ID or report.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("expected canonical v4 dataset ID")
    if report.get("experiment") != EXPECTED_REPORT or report.get("source") != "train_splits_only":
        raise ValueError("wrong development report; audit cannot use other data")
    for key in ("validation_used", "test_lockbox_accessed", "context_validation_used", "context_test_used", "context_targets_used", "pnl_used"):
        if report.get(key) is not False:
            raise ValueError(f"development report must explicitly declare {key}=false")
    folds = report.get("fold_results") or []
    if len(folds) != 7 or report.get("summary", {}).get("supported") is not False:
        raise ValueError("audit expects the failed seven-fold v5 experiment")

    rows: list[AuditRow] = []
    with (dataset_dir / "train.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(parse_training_row(json.loads(line)))
    if not rows:
        raise ValueError("empty train split")
    rows.sort(key=lambda row: (row.symbol, row.market_date, row.session, row.minute_utc))
    mark_sequence_eligible(rows)
    fold_rows = [summarize_fold(rows, item) for item in folds]
    payload = {
        "schema_version": 1,
        "audit": "v5_gru_failed_development_overlap_diagnostic_v1",
        "dataset_id": EXPECTED_DATASET_ID,
        "source": "v4_train_only_and_existing_v5_development_report",
        "validation_used": False,
        "test_lockbox_accessed": False,
        "context_validation_used": False,
        "context_test_used": False,
        "model_retrained": False,
        "model_or_threshold_selected": False,
        "fold_results": fold_rows,
        "summary": {
            "mean_tail_up_fraction": fmean(row["tail_up_fraction"] for row in fold_rows),
            "mean_nonoverlap_fraction": fmean(row["nonoverlap_fraction_of_tail"] for row in fold_rows),
            "all_sequence_counts_match": all(row["sequence_count_matches_report"] for row in fold_rows),
            "mean_loss_drop": fmean(row["train_loss_drop"] for row in fold_rows),
            "mean_development_brier": fmean(row["development_brier"] for row in fold_rows),
            "mean_development_null_brier": fmean(row["development_null_brier"] for row in fold_rows),
        },
        "limitations": [
            "Nonoverlap count is a greedy independent-interval proxy, not a statistically estimated effective sample size.",
            "The v5 report does not retain raw model probabilities: confidence distribution and calibration cannot be reconstructed.",
            "Tail-conditioned training versus prediction-gated evaluation mismatch is a hypothesis, not proven by this audit.",
            "Do not tune or promote v5 based on these post-hoc diagnostics.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit frozen failed v5 GRU using v4 train only.")
    parser.add_argument("--dataset-dir", default="data/processed/market/first-real-direction-v4")
    parser.add_argument("--report", default="runs/first-real-direction-v5/temporal_gru_development_v1.json")
    parser.add_argument("--output", default="runs/first-real-direction-v5/gru_overlap_audit_v1.json")
    args = parser.parse_args()
    result = run_audit(Path(args.dataset_dir), Path(args.report), Path(args.output))
    for row in result["fold_results"]:
        print(
            f"FOLD {row['fold']} train_tail={row['tail_sequences_reconstructed']} "
            f"matches={row['sequence_count_matches_report']} "
            f"independent_proxy={row['approx_nonoverlapping_tail_intervals']} "
            f"nonoverlap_fraction={row['nonoverlap_fraction_of_tail']:.3f} "
            f"UP_share={row['tail_up_fraction']:.3f} "
            f"loss_drop={row['train_loss_drop']:.3f}"
        )
    s = result["summary"]
    print("\n=== V5 GRU TRAIN-ONLY POSTMORTEM ===")
    print("Sequence counts match recorded report:", s["all_sequence_counts_match"])
    print("Mean independent-interval proxy fraction:", s["mean_nonoverlap_fraction"])
    print("Mean tail UP fraction:", s["mean_tail_up_fraction"])
    print("Mean train loss drop:", s["mean_loss_drop"])
    print("GRU Brier / null:", s["mean_development_brier"], "/", s["mean_development_null_brier"])
    print("V4 validation/test: NOT ACCESSED")
    print("GRU retraining: NO")
    print("Report:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
