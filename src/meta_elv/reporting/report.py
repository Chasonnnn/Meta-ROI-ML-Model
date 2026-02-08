from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _plot_lift(df: pd.DataFrame) -> str | None:
    if "label" not in df.columns or df["label"].notna().sum() < 10:
        return None
    d = df[df["label"].notna()].copy()
    d = d.sort_values("p_qualified_14d", ascending=False).reset_index(drop=True)
    y = d["label"].astype(int).to_numpy()
    total_pos = y.sum()
    if total_pos <= 0:
        return None
    cum_pos = np.cumsum(y)
    frac_leads = (np.arange(len(d)) + 1) / len(d)
    frac_pos = cum_pos / total_pos
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(frac_leads, frac_pos, label="Model")
    ax.plot([0, 1], [0, 1], "--", label="Random")
    ax.set_title("Lift / Capture Curve")
    ax.set_xlabel("Fraction of leads contacted")
    ax.set_ylabel("Fraction of qualified leads captured")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _plot_calibration(cal: dict[str, Any] | None) -> str | None:
    if not cal:
        return None
    mean_pred = cal.get("mean_pred") or []
    frac_pos = cal.get("frac_pos") or []
    if len(mean_pred) < 2 or len(frac_pos) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mean_pred, frac_pos, marker="o", label="Calibration")
    ax.plot([0, 1], [0, 1], "--", label="Perfect")
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction positive")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None:
        return "<p>(missing)</p>"
    d = df.head(max_rows)
    return d.to_html(index=False, escape=True, float_format=lambda x: f"{x:.4g}")


def write_report_html(
    run_dir: Path,
    *,
    metadata: dict[str, Any] | None = None,
    data_profile: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    scored: pd.DataFrame | None = None,
    leaderboards: dict[str, pd.DataFrame] | None = None,
    drift: dict[str, Any] | None = None,
) -> Path:
    run_dir = Path(run_dir)

    lift_png = _plot_lift(scored) if scored is not None else None
    cal_png = None
    if metrics and isinstance(metrics.get("eval_test"), dict):
        cal_png = _plot_calibration(metrics["eval_test"].get("calibration"))

    lb_campaign = leaderboards.get("campaign") if leaderboards else None
    lb_adset = leaderboards.get("adset") if leaderboards else None

    meta_json = json.dumps(metadata or {}, indent=2, sort_keys=True)
    profile_json = json.dumps(data_profile or {}, indent=2, sort_keys=True)
    metrics_json = json.dumps(metrics or {}, indent=2, sort_keys=True)
    drift_json = json.dumps(drift or {}, indent=2, sort_keys=True)

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ELV Run Report</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }}
    h1, h2 {{ margin: 0.2em 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 14px; }}
    pre {{ background: #f7f7f7; padding: 10px; overflow-x: auto; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }}
    th {{ background: #fafafa; text-align: left; }}
    .muted {{ color: #666; }}
    img {{ max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>ELV Run Report</h1>
  <p class="muted">Run directory: <code>{run_dir}</code></p>

  <div class="grid">
    <div class="card">
      <h2>Leaderboards (Top)</h2>
      <h3>Campaign</h3>
      {_df_to_html_table(lb_campaign, max_rows=15)}
      <h3>Adset</h3>
      {_df_to_html_table(lb_adset, max_rows=15)}
    </div>
    <div class="card">
      <h2>Diagnostics</h2>
      <h3>Lift</h3>
      {f'<img src="data:image/png;base64,{lift_png}"/>' if lift_png else '<p>(no labels available)</p>'}
      <h3>Calibration</h3>
      {f'<img src="data:image/png;base64,{cal_png}"/>' if cal_png else '<p>(no calibration data)</p>'}
    </div>
  </div>

  <div class="card" style="margin-top:18px;">
    <h2>Scored Leads (Sample)</h2>
    {_df_to_html_table(scored, max_rows=20) if scored is not None else "<p>(missing)</p>"}
  </div>

  <div class="grid" style="margin-top:18px;">
    <div class="card">
      <h2>Metadata</h2>
      <pre>{meta_json}</pre>
    </div>
    <div class="card">
      <h2>Data Profile</h2>
      <pre>{profile_json}</pre>
    </div>
  </div>

  <div class="grid" style="margin-top:18px;">
    <div class="card">
      <h2>Metrics</h2>
      <pre>{metrics_json}</pre>
    </div>
    <div class="card">
      <h2>Drift (PSI)</h2>
      <pre>{drift_json}</pre>
    </div>
  </div>

</body>
</html>
"""
    out_path = run_dir / "report.html"
    out_path.write_text(html)
    return out_path

