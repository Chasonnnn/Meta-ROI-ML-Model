from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel

from .config import RunConfig, load_config
from .demo import DemoDataSpec, generate_demo_data
from .modeling.metrics import evaluate_binary
from .modeling.train import train_model
from .profile import build_data_profile, write_data_profile
from .reporting import write_report_html
from .run_artifacts import create_run_context, snapshot_config, write_metadata
from .scoring import compute_leaderboards, load_model_bundle, score_table
from .table_builder import build_table
from .utils import ensure_dir, write_json
from .validate import render_validation_summary, validate_from_config

app = typer.Typer(add_completion=False, help="Meta Lead Ads Expected Lead Value (ELV) kit")
console = Console()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_cfg(config_path: Path) -> RunConfig:
    return load_config(config_path)


@app.command()
def validate(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False, help="Path to run config YAML"),
) -> None:
    cfg = _load_cfg(config)
    result = validate_from_config(cfg)
    summary = render_validation_summary(result)
    if result.ok:
        console.print(Panel.fit(summary, title="validate", border_style="green"))
        raise typer.Exit(0)
    console.print(Panel.fit(summary, title="validate", border_style="red"))
    raise typer.Exit(1)


@app.command("build-table")
def build_table_cmd(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    cfg = _load_cfg(config)
    v = validate_from_config(cfg)
    if not v.ok:
        console.print(render_validation_summary(v))
        raise typer.Exit(1)

    ctx = create_run_context(_repo_root())
    snapshot_config(ctx.run_dir, config)

    res = build_table(cfg)
    profile = build_data_profile(res)
    write_data_profile(ctx.run_dir, profile)

    # Save table
    table_path = ctx.run_dir / "table.csv"
    res.table.to_csv(table_path, index=False)

    metadata = {
        "run_id": ctx.run_id,
        "schema_version": cfg.schema_version,
        "join_strategy": res.join_strategy,
        "join_match_rate": res.join_match_rate,
        "labeling": {
            "as_of_date": res.as_of_date.isoformat(),
            "label_window_days": cfg.label.label_window_days,
            "require_label_maturity": cfg.label.require_label_maturity,
            "counts": res.label_summary,
        },
    }
    write_metadata(ctx.run_dir, metadata)
    console.print(f"[green]Wrote:[/green] {ctx.run_dir}")


@app.command()
def train(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    cfg = _load_cfg(config)
    v = validate_from_config(cfg)
    if not v.ok:
        console.print(render_validation_summary(v))
        raise typer.Exit(1)
    if cfg.paths.outcomes_path is None:
        console.print("[red]Training requires outcomes_path in config.[/red]")
        raise typer.Exit(1)

    ctx = create_run_context(_repo_root())
    snapshot_config(ctx.run_dir, config)

    res = build_table(cfg)
    profile = build_data_profile(res)
    write_data_profile(ctx.run_dir, profile)

    # Save table
    res.table.to_csv(ctx.run_dir / "table.csv", index=False)

    tr = train_model(cfg, res.table, ctx.run_dir)

    # Score all leads with the trained model
    bundle = load_model_bundle(tr.model_path)
    scored = score_table(cfg, res.table, bundle)
    scored_path = ctx.run_dir / "predictions.csv"
    scored.to_csv(scored_path, index=False)

    # Leaderboards (campaign/adset)
    from .connectors import meta_csv

    ads_df = meta_csv.load_ads(cfg.paths.ads_path).df
    leaderboards = compute_leaderboards(scored, ads_df)
    leaderboards["campaign"].to_csv(ctx.run_dir / "leaderboard_campaign.csv", index=False)
    leaderboards["adset"].to_csv(ctx.run_dir / "leaderboard_adset.csv", index=False)

    metadata = {
        "run_id": ctx.run_id,
        "schema_version": cfg.schema_version,
        "join_strategy": res.join_strategy,
        "join_match_rate": res.join_match_rate,
        "as_of_date": res.as_of_date.isoformat(),
        "label_counts": res.label_summary,
        "model_path": str(tr.model_path),
    }
    write_metadata(ctx.run_dir, metadata)

    metrics = tr.metrics
    write_json(ctx.run_dir / "metrics.json", metrics)
    if metrics.get("drift_psi") is not None:
        write_json(ctx.run_dir / "drift.json", metrics["drift_psi"])

    write_report_html(
        ctx.run_dir,
        metadata=metadata,
        data_profile=profile,
        metrics=metrics,
        scored=scored,
        leaderboards=leaderboards,
        drift=metrics.get("drift_psi") or {},
    )
    console.print(f"[green]Wrote:[/green] {ctx.run_dir}")


@app.command()
def score(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    model: Path = typer.Option(..., "--model", exists=True, dir_okay=False),
) -> None:
    cfg = _load_cfg(config)
    v = validate_from_config(cfg)
    if not v.ok:
        console.print(render_validation_summary(v))
        raise typer.Exit(1)

    ctx = create_run_context(_repo_root())
    snapshot_config(ctx.run_dir, config)

    res = build_table(cfg)
    profile = build_data_profile(res)
    write_data_profile(ctx.run_dir, profile)

    bundle = load_model_bundle(model)
    scored = score_table(cfg, res.table, bundle)
    scored.to_csv(ctx.run_dir / "predictions.csv", index=False)

    from .connectors import meta_csv

    ads_df = meta_csv.load_ads(cfg.paths.ads_path).df
    leaderboards = compute_leaderboards(scored, ads_df)
    leaderboards["campaign"].to_csv(ctx.run_dir / "leaderboard_campaign.csv", index=False)
    leaderboards["adset"].to_csv(ctx.run_dir / "leaderboard_adset.csv", index=False)

    # Optional eval if labels exist
    metrics = None
    labeled = scored[scored["label"].notna()] if "label" in scored.columns else scored.iloc[0:0]
    if len(labeled) >= 50:
        em = evaluate_binary(
            y_true=labeled["label"].astype(int).to_numpy(),
            y_prob=labeled["p_qualified_14d"].to_numpy(),
            topk_frac=cfg.reporting.topk_frac,
            ece_bins=cfg.reporting.ece_bins,
        )
        metrics = {
            "eval_labeled": {
                "n": int(len(labeled)),
                "pr_auc": em.pr_auc,
                "brier": em.brier,
                "roc_auc": em.roc_auc,
                "lift": em.lift,
                "ece": em.ece,
                "calibration": em.calibration,
            }
        }
        write_json(ctx.run_dir / "metrics.json", metrics)

    metadata = {
        "run_id": ctx.run_id,
        "schema_version": cfg.schema_version,
        "model_path": str(model),
        "join_strategy": res.join_strategy,
        "join_match_rate": res.join_match_rate,
        "as_of_date": res.as_of_date.isoformat(),
        "label_counts": res.label_summary,
    }
    write_metadata(ctx.run_dir, metadata)

    write_report_html(
        ctx.run_dir,
        metadata=metadata,
        data_profile=profile,
        metrics=metrics or {},
        scored=scored,
        leaderboards=leaderboards,
        drift={},
    )
    console.print(f"[green]Wrote:[/green] {ctx.run_dir}")


@app.command()
def report(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False),
) -> None:
    run_dir = run_dir.resolve()
    metadata = json.loads((run_dir / "metadata.json").read_text()) if (run_dir / "metadata.json").exists() else {}
    profile = json.loads((run_dir / "data_profile.json").read_text()) if (run_dir / "data_profile.json").exists() else {}
    metrics = json.loads((run_dir / "metrics.json").read_text()) if (run_dir / "metrics.json").exists() else {}
    drift = json.loads((run_dir / "drift.json").read_text()) if (run_dir / "drift.json").exists() else {}

    scored = pd.read_csv(run_dir / "predictions.csv") if (run_dir / "predictions.csv").exists() else None
    lbs = {}
    if (run_dir / "leaderboard_campaign.csv").exists():
        lbs["campaign"] = pd.read_csv(run_dir / "leaderboard_campaign.csv")
    if (run_dir / "leaderboard_adset.csv").exists():
        lbs["adset"] = pd.read_csv(run_dir / "leaderboard_adset.csv")

    write_report_html(
        run_dir,
        metadata=metadata,
        data_profile=profile,
        metrics=metrics,
        scored=scored,
        leaderboards=lbs,
        drift=drift,
    )
    console.print(f"[green]Wrote:[/green] {run_dir / 'report.html'}")


@app.command()
def demo(
    value_per_qualified: float = typer.Option(250.0, "--value-per-qualified"),
    seed: int = typer.Option(7, "--seed"),
) -> None:
    """
    Generate synthetic demo data and run a score-first flow.
    Uses a bundled demo model if present; otherwise trains a quick logistic model locally.
    """
    ctx = create_run_context(_repo_root())

    demo_dir = ensure_dir(ctx.run_dir / "demo_data")
    paths = generate_demo_data(demo_dir, spec=DemoDataSpec(seed=seed))

    # Write a minimal resolved config into the run dir
    config_yaml = ctx.run_dir / "config.yaml"
    config_yaml.write_text(
        f"""schema_version: 1
paths:
  ads_path: {paths['ads']}
  leads_path: {paths['leads']}
  outcomes_path: {paths['outcomes']}
  lead_to_ad_map_path: null
label:
  label_window_days: 14
  as_of_date: null
  require_label_maturity: true
features:
  ads_granularity: daily
  feature_window_days: 7
  feature_lag_days: 1
business:
  value_per_qualified: {value_per_qualified}
splits:
  train_frac: 0.6
  calib_frac: 0.2
  test_frac: 0.2
model:
  model_type: logreg
  calibration_method: sigmoid
  random_seed: {seed}
  lgbm_params: {{}}
  logreg_params:
    C: 1.0
    max_iter: 2000
reporting:
  topk_frac: 0.10
  ece_bins: 10
"""
    )
    cfg = load_config(config_yaml)

    v = validate_from_config(cfg)
    if not v.ok:
        console.print(render_validation_summary(v))
        raise typer.Exit(1)

    res = build_table(cfg)
    profile = build_data_profile(res)
    write_data_profile(ctx.run_dir, profile)
    res.table.to_csv(ctx.run_dir / "table.csv", index=False)

    # Use bundled demo model if available
    asset_model = Path(__file__).resolve().parent / "assets" / "demo_model.joblib"
    if asset_model.exists():
        bundle = load_model_bundle(asset_model)
    else:
        console.print(
            "[yellow]Bundled demo model not found; training a quick logistic model for this demo run.[/yellow]"
        )
        tr = train_model(cfg, res.table, ctx.run_dir)
        bundle = load_model_bundle(tr.model_path)

    scored = score_table(cfg, res.table, bundle)
    scored.to_csv(ctx.run_dir / "predictions.csv", index=False)

    from .connectors import meta_csv

    ads_df = meta_csv.load_ads(cfg.paths.ads_path).df
    leaderboards = compute_leaderboards(scored, ads_df)
    leaderboards["campaign"].to_csv(ctx.run_dir / "leaderboard_campaign.csv", index=False)
    leaderboards["adset"].to_csv(ctx.run_dir / "leaderboard_adset.csv", index=False)

    metadata = {
        "run_id": ctx.run_id,
        "schema_version": cfg.schema_version,
        "mode": "demo",
        "join_strategy": res.join_strategy,
        "join_match_rate": res.join_match_rate,
        "as_of_date": res.as_of_date.isoformat(),
        "label_counts": res.label_summary,
    }
    write_metadata(ctx.run_dir, metadata)

    write_report_html(
        ctx.run_dir,
        metadata=metadata,
        data_profile=profile,
        metrics={},
        scored=scored,
        leaderboards=leaderboards,
        drift={},
    )
    console.print(f"[green]Wrote:[/green] {ctx.run_dir}")
