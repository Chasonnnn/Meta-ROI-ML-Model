from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import RunConfig
from .connectors import meta_csv
from .modeling.train import save_model_bundle, train_and_evaluate
from .profile import build_data_profile, write_data_profile
from .reporting import write_report
from .run_artifacts import RunContext, write_metadata
from .scoring import compute_leaderboards, load_model_bundle, score_table
from .table_builder import build_table
from .utils import write_json


def write_table(run_dir: Path, table: pd.DataFrame) -> Path:
    out = run_dir / "table.csv.gz"
    table.to_csv(out, index=False, compression="gzip")
    return out


def run_build_table(cfg: RunConfig, ctx: RunContext) -> dict[str, Any]:
    res = build_table(cfg)
    write_table(ctx.run_dir, res.table)

    profile = build_data_profile(res)
    write_data_profile(ctx.run_dir, profile)

    metadata = {
        "schema_version": cfg.schema_version,
        "join": {"strategy": res.join_strategy, "match_rate": res.join_match_rate},
        "labeling": {"as_of_date": res.as_of_date.isoformat(), "counts": res.label_summary},
        "leaderboards": {"min_segment_leads": int(cfg.reporting.min_segment_leads)},
    }
    write_metadata(ctx.run_dir, metadata)
    return {"result": res, "profile": profile, "metadata": metadata}


def run_train(cfg: RunConfig, ctx: RunContext, *, max_labeled_rows: int | None = None) -> dict[str, Any]:
    if cfg.paths.outcomes_path is None:
        raise ValueError(
            "outcomes_path is required for training. Provide paths.outcomes_path in config, "
            "or use `elv score` for inference-only scoring."
        )
    if not Path(cfg.paths.outcomes_path).exists():
        raise ValueError(f"outcomes_path does not exist: {cfg.paths.outcomes_path}")

    built = run_build_table(cfg, ctx)
    table = built["result"].table

    artifacts = train_and_evaluate(cfg, table, max_labeled_rows=max_labeled_rows)
    model_path = ctx.run_dir / "model.joblib"
    save_model_bundle(model_path, artifacts.model_bundle)

    metrics_path = ctx.run_dir / "metrics.json"
    write_json(metrics_path, artifacts.metrics)

    # Score full table (including unknowns)
    bundle = load_model_bundle(model_path)
    scored = score_table(cfg, table, bundle)

    # Persist predictions with a conservative column set.
    keep_cols = [
        c
        for c in [
            "lead_id",
            "created_time",
            "campaign_id",
            "campaign_name",
            "adset_id",
            "adset_name",
            "ad_id",
            "ad_name",
            "label_status",
            "label",
            "p_qualified_14d",
            "value_per_qualified",
            "elv",
            "score_rank",
        ]
        if c in scored.columns
    ]
    scored_out = scored[keep_cols].copy()
    scored_out.to_csv(ctx.run_dir / "predictions.csv", index=False)

    # Leaderboards (spend from ads.csv over the lead date range)
    ads = meta_csv.load_ads(cfg.paths.ads_path).df
    lbs = compute_leaderboards(scored, ads, min_segment_leads=cfg.reporting.min_segment_leads)
    lbs["campaign"].to_csv(ctx.run_dir / "leaderboard_campaign.csv", index=False)
    # Only write adset leaderboard when we have adset keys.
    if "adset_id" in scored.columns and scored["adset_id"].notna().any():
        lbs["adset"].to_csv(ctx.run_dir / "leaderboard_adset.csv", index=False)

    # Update metadata with split boundaries and a metric summary
    meta = built["metadata"]
    meta["split"] = {"train_end_time": artifacts.split.train_end_time, "calib_end_time": artifacts.split.calib_end_time}
    meta["metrics_summary"] = {
        "pr_auc": artifacts.metrics.get("model", {}).get("pr_auc"),
        "brier": artifacts.metrics.get("model", {}).get("brier"),
        "lift": (artifacts.metrics.get("model", {}).get("lift_at_k") or {}).get("lift"),
        "capture": (artifacts.metrics.get("model", {}).get("lift_at_k") or {}).get("capture"),
    }
    write_metadata(ctx.run_dir, meta)

    report_path = write_report(ctx.run_dir)
    return {
        "run_dir": ctx.run_dir,
        "model_path": model_path,
        "metrics_path": metrics_path,
        "report_path": report_path,
    }


def run_score(cfg: RunConfig, ctx: RunContext, model_path: Path) -> dict[str, Any]:
    built = run_build_table(cfg, ctx)
    table = built["result"].table

    bundle = load_model_bundle(model_path)
    scored = score_table(cfg, table, bundle)

    keep_cols = [
        c
        for c in [
            "lead_id",
            "created_time",
            "campaign_id",
            "campaign_name",
            "adset_id",
            "adset_name",
            "ad_id",
            "ad_name",
            "label_status",
            "label",
            "p_qualified_14d",
            "value_per_qualified",
            "elv",
            "score_rank",
        ]
        if c in scored.columns
    ]
    scored[keep_cols].to_csv(ctx.run_dir / "predictions.csv", index=False)

    ads = meta_csv.load_ads(cfg.paths.ads_path).df
    lbs = compute_leaderboards(scored, ads, min_segment_leads=cfg.reporting.min_segment_leads)
    lbs["campaign"].to_csv(ctx.run_dir / "leaderboard_campaign.csv", index=False)
    if "adset_id" in scored.columns and scored["adset_id"].notna().any():
        lbs["adset"].to_csv(ctx.run_dir / "leaderboard_adset.csv", index=False)

    report_path = write_report(ctx.run_dir)
    return {"run_dir": ctx.run_dir, "report_path": report_path}
