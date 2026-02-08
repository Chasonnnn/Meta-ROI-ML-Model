from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from ..utils import read_json


def _fig_to_base64_png(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64


def _df_to_html(df: pd.DataFrame, max_rows: int = 20) -> str:
    d = df.head(max_rows).copy()
    return d.to_html(index=False, escape=True)


def _plot_lift(lift_data: dict[str, Any]) -> str | None:
    xs = lift_data.get("population_frac")
    ys = lift_data.get("positive_capture_frac")
    if not xs or not ys:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(xs, ys, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title("Lift Curve (Cumulative Capture)")
    ax.set_xlabel("Fraction of leads contacted")
    ax.set_ylabel("Fraction of qualified captured")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _fig_to_base64_png(fig)


def _plot_calibration(cal: dict[str, Any]) -> str | None:
    bins = cal.get("bins") or []
    xs = []
    ys = []
    for b in bins:
        if b.get("count", 0) and b.get("p_mean") is not None and b.get("y_rate") is not None:
            xs.append(b["p_mean"])
            ys.append(b["y_rate"])
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(xs, ys, marker="o", label="Observed")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ece = cal.get("ece")
    title = "Calibration"
    if ece is not None:
        title += f" (ECE={ece:.3f})"
    ax.set_title(title)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _fig_to_base64_png(fig)


def write_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)

    data_profile = None
    metrics = None
    try:
        data_profile = read_json(run_dir / "data_profile.json")
    except Exception:
        data_profile = None
    try:
        metrics = read_json(run_dir / "metrics.json")
    except Exception:
        metrics = None

    lb_campaign = None
    lb_adset = None
    if (run_dir / "leaderboard_campaign.csv").exists():
        lb_campaign = pd.read_csv(run_dir / "leaderboard_campaign.csv")
    if (run_dir / "leaderboard_adset.csv").exists():
        lb_adset = pd.read_csv(run_dir / "leaderboard_adset.csv")

    # Plots
    lift_b64 = None
    cal_b64 = None
    model_metrics = (metrics or {}).get("model", {})
    if model_metrics:
        lift_b64 = _plot_lift(model_metrics.get("lift_curve", {}) or {})
        cal_b64 = _plot_calibration(model_metrics.get("calibration", {}) or {})

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; color: #111; }
    h1 { margin: 0 0 8px; }
    .subtle { color: #444; margin: 0 0 18px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; }
    .card { border: 1px solid #e6e6e6; border-radius: 10px; padding: 14px; background: #fff; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #e6e6e6; padding: 6px 8px; font-size: 12px; }
    th { background: #fafafa; text-align: left; }
    .warn { background: #fff6e5; border: 1px solid #ffd18a; padding: 10px 12px; border-radius: 10px; }
    img { max-width: 100%; }
    """

    parts: list[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'/>")
    parts.append(f"<style>{css}</style></head><body>")
    parts.append("<h1>Meta Lead Ads ELV Report</h1>")

    if data_profile:
        join = (data_profile.get("join") or {})
        labeling = (data_profile.get("labeling") or {})
        counts = (labeling.get("counts") or {})
        parts.append(
            f"<p class='subtle'>Rows={data_profile.get('rows')} | Join={join.get('strategy')} (match {join.get('match_rate', 0):.1%}) | "
            f"Labels pos/neg/unk={counts.get('positive', 0)}/{counts.get('negative', 0)}/{counts.get('unknown', 0)} | "
            f"as_of_date={labeling.get('as_of_date')}</p>"
        )
    else:
        parts.append("<div class='warn'>Missing data_profile.json. Run `elv build-table` or `elv train` first.</div>")

    if metrics:
        m = metrics.get("model", {})
        base = metrics.get("baseline_campaign_rate", {})
        parts.append("<div class='grid'>")
        parts.append("<div class='card'>")
        parts.append("<h2>Metrics (Test)</h2>")
        parts.append("<table>")
        parts.append("<tr><th>Model</th><th>PR-AUC</th><th>Brier</th><th>Lift@k</th><th>Capture@k</th></tr>")

        def fmt(x: Any) -> str:
            if x is None:
                return ""
            if isinstance(x, float):
                return f"{x:.4f}"
            return str(x)

        lift = (m.get("lift_at_k") or {})
        parts.append(
            "<tr>"
            f"<td>calibrated {metrics.get('model_type', 'model')}</td>"
            f"<td>{fmt(m.get('pr_auc'))}</td>"
            f"<td>{fmt(m.get('brier'))}</td>"
            f"<td>{fmt(lift.get('lift'))}</td>"
            f"<td>{fmt(lift.get('capture'))}</td>"
            "</tr>"
        )
        liftb = (base.get("lift_at_k") or {})
        parts.append(
            "<tr>"
            "<td>baseline: campaign rate</td>"
            f"<td>{fmt(base.get('pr_auc'))}</td>"
            f"<td>{fmt(base.get('brier'))}</td>"
            f"<td>{fmt(liftb.get('lift'))}</td>"
            f"<td>{fmt(liftb.get('capture'))}</td>"
            "</tr>"
        )
        parts.append("</table>")
        parts.append("</div>")

        parts.append("<div class='card'>")
        parts.append("<h2>Calibrated ELV</h2>")
        parts.append("<p class='subtle'>ELV = P(qualified_within_14d) × value_per_qualified</p>")
        parts.append("<ul>")
        data = metrics.get("data", {})
        if data:
            parts.append(f"<li>n_labeled={data.get('n_labeled')} | positive_rate_labeled={data.get('positive_rate_labeled'):.2%}</li>")
        split = metrics.get("split", {})
        if split:
            parts.append(f"<li>train_end_time={split.get('train_end_time')} | calib_end_time={split.get('calib_end_time')}</li>")
        parts.append("</ul>")
        parts.append("</div>")
        parts.append("</div>")
    else:
        parts.append("<div class='warn'>Missing metrics.json. Run `elv train` first.</div>")

    parts.append("<div class='grid'>")
    parts.append("<div class='card'><h2>Lift</h2>")
    if lift_b64:
        parts.append(f"<img alt='lift' src='data:image/png;base64,{lift_b64}'/>")
    else:
        parts.append("<p class='subtle'>No lift curve available.</p>")
    parts.append("</div>")
    parts.append("<div class='card'><h2>Calibration</h2>")
    if cal_b64:
        parts.append(f"<img alt='calibration' src='data:image/png;base64,{cal_b64}'/>")
    else:
        parts.append("<p class='subtle'>No calibration data available.</p>")
    parts.append("</div>")
    parts.append("</div>")

    if lb_campaign is not None:
        parts.append("<div class='card'><h2>Campaign Leaderboard</h2>")
        parts.append(_df_to_html(lb_campaign, max_rows=20))
        parts.append("</div>")
    if lb_adset is not None:
        parts.append("<div class='card'><h2>Adset Leaderboard</h2>")
        parts.append(_df_to_html(lb_adset, max_rows=20))
        parts.append("</div>")

    parts.append("</body></html>")
    out = run_dir / "report.html"
    out.write_text("\n".join(parts))
    return out

