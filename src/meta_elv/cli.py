from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .config import (
    BusinessConfig,
    FeaturesConfig,
    LabelConfig,
    ModelConfig,
    PathsConfig,
    ReportingConfig,
    RunConfig,
    SplitsConfig,
    load_config,
    save_config,
)
from .demo.generate import DemoDataSpec, generate_demo_data
from .pipeline import run_build_table, run_score, run_train
from .run_artifacts import create_run_context, snapshot_config
from .utils import ensure_dir
from .validate import render_validation_summary, validate_from_config


app = typer.Typer(add_completion=False, help="Meta Lead Ads ELV Kit (validate -> build-table -> train -> score -> report)")
console = Console()


def _repo_root() -> Path:
    # Prefer current working directory to keep runs/ local to where the user executes the CLI.
    return Path.cwd()


def _die(msg: str, code: int = 1) -> None:
    console.print(f"[red]Error:[/red] {msg}")
    raise typer.Exit(code=code)


@app.command()
def validate(config: Path = typer.Option(..., "--config", exists=True, dir_okay=False)):
    """Validate input CSVs against the v1 schema and print a friendly summary."""
    cfg = load_config(config)
    result = validate_from_config(cfg)
    console.print(render_validation_summary(result))
    if not result.ok:
        raise typer.Exit(code=1)


@app.command("build-table")
def build_table_cmd(config: Path = typer.Option(..., "--config", exists=True, dir_okay=False)):
    """Build the joined feature/label table and write artifacts under runs/<run_id>/."""
    cfg = load_config(config)
    ctx = create_run_context(_repo_root())
    snapshot_config(ctx.run_dir, config)
    out = run_build_table(cfg, ctx)
    console.print(f"[green]Wrote run:[/green] {ctx.run_dir}")
    console.print(f"Table: {ctx.run_dir / 'table.csv.gz'}")
    console.print(f"Profile: {ctx.run_dir / 'data_profile.md'}")
    if out.get("metadata"):
        console.print(f"Metadata: {ctx.run_dir / 'metadata.json'}")


@app.command()
def train(config: Path = typer.Option(..., "--config", exists=True, dir_okay=False)):
    """Train a calibrated model, score leads, and write a full report under runs/<run_id>/."""
    cfg = load_config(config)
    ctx = create_run_context(_repo_root())
    snapshot_config(ctx.run_dir, config)
    out = run_train(cfg, ctx)
    console.print(f"[green]Wrote run:[/green] {ctx.run_dir}")
    console.print(f"Model: {out['model_path']}")
    console.print(f"Metrics: {out['metrics_path']}")
    console.print(f"Predictions: {ctx.run_dir / 'predictions.csv'}")
    console.print(f"Report: {out['report_path']}")


@app.command()
def score(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    model: Path = typer.Option(..., "--model", exists=True, dir_okay=False),
):
    """Score leads using an existing model bundle and write predictions/leaderboards/report."""
    cfg = load_config(config)
    ctx = create_run_context(_repo_root())
    snapshot_config(ctx.run_dir, config)
    out = run_score(cfg, ctx, model_path=model)
    console.print(f"[green]Wrote run:[/green] {ctx.run_dir}")
    console.print(f"Predictions: {ctx.run_dir / 'predictions.csv'}")
    console.print(f"Report: {out['report_path']}")


@app.command()
def report(run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False)):
    """Regenerate report.html for an existing run directory."""
    from .reporting import write_report

    out = write_report(run_dir)
    console.print(f"[green]Wrote:[/green] {out}")


@app.command()
def demo(
    value_per_qualified: float = typer.Option(1.0, "--value-per-qualified"),
    model_path: Optional[Path] = typer.Option(None, "--model", help="Optional bundled model to score with (skip training)."),
    train: bool = typer.Option(
        False,
        "--train",
        help="Force local training (ignore bundled demo model, if present).",
    ),
):
    """
    Generate synthetic demo data and produce a full run directory.

    By default, demo is score-first if a bundled demo model exists (or if --model is provided).
    Use --train to force local training.
    """
    ctx = create_run_context(_repo_root())

    # Generate data inside the run dir to keep demo runs self-contained.
    demo_dir = ensure_dir(ctx.run_dir / "demo_data")
    paths = generate_demo_data(demo_dir, DemoDataSpec())

    cfg = RunConfig(
        schema_version=1,
        paths=PathsConfig(
            ads_path=paths["ads"],
            leads_path=paths["leads"],
            outcomes_path=paths["outcomes"],
            lead_to_ad_map_path=None,
        ),
        label=LabelConfig(label_window_days=14, as_of_date=None, require_label_maturity=True),
        features=FeaturesConfig(ads_granularity="daily", feature_window_days=7, feature_lag_days=1),
        business=BusinessConfig(value_per_qualified=float(value_per_qualified)),
        splits=SplitsConfig(train_frac=0.6, calib_frac=0.2, test_frac=0.2),
        model=ModelConfig(
            # Default demo uses logreg to avoid platform-specific LightGBM runtime deps (e.g., libomp on macOS).
            model_type="logreg",
            calibration_method="sigmoid",
            random_seed=7,
            lgbm_params={},
            logreg_params={"C": 1.0, "max_iter": 2000},
        ),
        reporting=ReportingConfig(topk_frac=0.10, ece_bins=10),
    )

    # Snapshot effective config to the run dir for reproducibility.
    save_config(cfg, ctx.run_dir / "config.yaml")

    # Validate (best-effort; demo should always pass).
    vr = validate_from_config(cfg)
    if not vr.ok:
        console.print(render_validation_summary(vr))
        _die("Demo data failed validation (unexpected).")

    if not train:
        if model_path is not None:
            out = run_score(cfg, ctx, model_path=model_path)
            console.print(f"[green]Demo (score-first) wrote run:[/green] {ctx.run_dir}")
            console.print(f"Report: {out['report_path']}")
            return

        # If available, use the bundled demo model asset.
        try:
            import importlib.resources as ir

            demo_model = ir.files("meta_elv.assets").joinpath("demo_model.joblib")
            if demo_model.is_file():
                with ir.as_file(demo_model) as mp:
                    out = run_score(cfg, ctx, model_path=mp)
                console.print(f"[green]Demo (score-first) wrote run:[/green] {ctx.run_dir}")
                console.print(f"Report: {out['report_path']}")
                return
        except Exception:
            # Fall back to local training.
            pass

    out = run_train(cfg, ctx)
    console.print(f"[green]Demo (trained locally) wrote run:[/green] {ctx.run_dir}")
    console.print(f"Report: {out['report_path']}")


def main() -> None:
    app()
