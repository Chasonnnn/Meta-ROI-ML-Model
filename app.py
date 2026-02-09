from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from meta_elv.config import (
    BusinessConfig,
    FeaturesConfig,
    LabelConfig,
    ModelConfig,
    PathsConfig,
    ReportingConfig,
    RunConfig,
    SplitsConfig,
    save_config,
)
from meta_elv.demo.generate import DemoDataSpec, generate_demo_data
from meta_elv.pipeline import run_score, run_train
from meta_elv.run_artifacts import create_run_context
from meta_elv.validate import render_validation_summary, validate_from_config


def _altair() -> Any | None:
    try:
        import altair as alt  # type: ignore

        return alt
    except Exception:
        return None


def _write_upload(upload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(upload.getvalue())


def _get_secret(name: str) -> str | None:
    # Works for local `.streamlit/secrets.toml` and Hugging Face Spaces secrets.
    try:
        val = st.secrets.get(name)  # type: ignore[attr-defined]
        if isinstance(val, str) and val.strip():
            return val.strip()
    except Exception:
        pass
    val = os.environ.get(name)
    return val.strip() if isinstance(val, str) and val.strip() else None


def _render_gemini_advisor(run_dir: Path) -> None:
    with st.expander("GenAI Advisor (Gemini, optional)", expanded=False):
        st.info(
            "Optional qualitative suggestions. This sends aggregate run summaries (leaderboards/metrics) and any "
            "media you upload to Gemini. Do not use this with sensitive client data on public deployments. "
            "If this is a public Space, do not embed a shared paid API key in Secrets."
        )

        api_key = _get_secret("GEMINI_API_KEY") or _get_secret("GOOGLE_API_KEY")
        if not api_key:
            api_key = st.text_input(
                "GEMINI_API_KEY",
                type="password",
                help="Provide your key for this session (not written to disk). "
                "On Hugging Face Spaces, prefer adding it as a Secret.",
            ).strip() or None

        model_default = _get_secret("GEMINI_MODEL") or "gemini-2.0-flash"
        model = st.text_input("Gemini model", value=model_default).strip() or model_default

        offer = st.text_area(
            "Offer context (required)",
            placeholder="What are you selling? Price point? Who is the ideal customer? What qualifies a lead?",
        ).strip()
        keywords = st.text_area(
            "Audience keywords / targeting notes (optional)",
            placeholder="Copy/paste interests, lookalike notes, exclusions, geos, etc.",
        ).strip()

        media = st.file_uploader(
            "Creative media (optional: image or short video clip)",
            type=["png", "jpg", "jpeg", "mp4", "mov"],
            accept_multiple_files=False,
        )

        save_to_run = st.checkbox("Save suggestions to run_dir/suggestions.md", value=False)

        if st.button("Generate suggestions", type="secondary"):
            if not api_key:
                st.error("Missing GEMINI_API_KEY. Add it as a Secret (recommended) or paste it above.")
                st.stop()
            if not offer:
                st.error("Offer context is required.")
                st.stop()

            media_path = None
            try:
                if media is not None:
                    # Write to a temp file because Gemini's SDK file upload expects a filesystem path.
                    suffix = Path(media.name).suffix or ".bin"
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmp.write(media.getvalue())
                    tmp.flush()
                    tmp.close()
                    media_path = Path(tmp.name)

                from meta_elv.genai.gemini_advisor import generate_gemini_suggestions

                with st.spinner("Calling Gemini..."):
                    out_md = generate_gemini_suggestions(
                        run_dir=Path(run_dir),
                        offer_context=offer,
                        audience_keywords=keywords or None,
                        media_path=media_path,
                        api_key=api_key,
                        model=model,
                    )
                st.subheader("Suggestions")
                st.markdown(out_md)
                st.download_button(
                    "Download suggestions.md",
                    data=out_md.encode("utf-8"),
                    file_name="suggestions.md",
                )
                if save_to_run:
                    (Path(run_dir) / "suggestions.md").write_text(out_md)
            except Exception as e:
                st.error(str(e))
                with st.expander("Details"):
                    st.exception(e)
            finally:
                if media_path is not None:
                    try:
                        media_path.unlink()
                    except Exception:
                        pass


def _render_creative_embeddings(run_dir: Path) -> None:
    with st.expander("Creative similarity & clustering (optional)", expanded=False):
        st.info(
            "Compute image/video embeddings for similar-creative search and clustering. "
            "This is optional and may be slow on CPU. For videos, a small number of frames are sampled and mean-pooled."
        )

        # Check optional dependency presence early (friendly UX).
        try:
            import open_clip  # noqa: F401
            import torch  # noqa: F401
            import cv2  # noqa: F401
            from PIL import Image  # noqa: F401
        except Exception:
            st.warning(
                "Creative embeddings dependencies are not installed in this environment. "
                "Install locally with: `uv sync --extra embeddings`."
            )
            return

        st.caption(
            "Upload a `creative_media.csv` mapping `ad_id` -> `media_path` (filename) and upload the referenced files."
        )
        media_map_up = st.file_uploader(
            "creative_media.csv",
            type=["csv"],
            accept_multiple_files=False,
            help="Required columns: ad_id, media_path. media_path should match the uploaded filename.",
            key="creative_media_map",
        )
        media_files = st.file_uploader(
            "Creative files (images/videos)",
            type=["png", "jpg", "jpeg", "mp4", "mov", "webm"],
            accept_multiple_files=True,
            key="creative_media_files",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            clusters = st.number_input("clusters", min_value=2, value=10, step=1)
        with c2:
            neighbors = st.number_input("neighbors", min_value=1, value=10, step=1)
        with c3:
            frames = st.number_input("num_video_frames", min_value=1, value=4, step=1)

        if st.button("Run creative analysis", type="secondary", key="run_creative_analysis"):
            if media_map_up is None:
                st.error("Upload creative_media.csv first.")
                st.stop()
            if not media_files:
                st.error("Upload at least one media file.")
                st.stop()

            try:
                from tempfile import TemporaryDirectory

                from meta_elv.creatives import run_creative_analysis

                out_dir = Path(run_dir) / "creative_analysis"
                out_dir.mkdir(parents=True, exist_ok=True)

                with TemporaryDirectory() as td:
                    td_p = Path(td)
                    # Write uploaded media files to a temp dir (not persisted in runs/).
                    for f in media_files:
                        (td_p / f.name).write_bytes(f.getvalue())
                    map_p = td_p / "creative_media.csv"
                    map_p.write_bytes(media_map_up.getvalue())

                    with st.spinner("Embedding creatives and clustering..."):
                        outputs = run_creative_analysis(
                            creative_map_path=map_p,
                            media_dir=td_p,
                            out_dir=out_dir,
                            run_dir=Path(run_dir),
                            neighbors=int(neighbors),
                            clusters=int(clusters),
                            num_video_frames=int(frames),
                        )

                st.success(f"Wrote creative artifacts to: {out_dir}")
                st.download_button(
                    "Download creative_assets.csv",
                    data=outputs.assets_csv.read_bytes(),
                    file_name="creative_assets.csv",
                )
                st.download_button(
                    "Download creative_neighbors.csv",
                    data=outputs.neighbors_csv.read_bytes(),
                    file_name="creative_neighbors.csv",
                )
                st.download_button(
                    "Download creative_clusters.csv",
                    data=outputs.clusters_csv.read_bytes(),
                    file_name="creative_clusters.csv",
                )

                if outputs.cluster_summary_csv is not None and outputs.cluster_summary_csv.exists():
                    st.subheader("Cluster Summary (Top 20)")
                    st.dataframe(__import__("pandas").read_csv(outputs.cluster_summary_csv).head(20), use_container_width=True)
                st.subheader("Neighbors (Sample)")
                st.dataframe(__import__("pandas").read_csv(outputs.neighbors_csv).head(20), use_container_width=True)
                if outputs.warnings:
                    st.warning("Warnings:\n" + "\n".join([f"- {w}" for w in outputs.warnings]))
            except Exception as e:
                st.error(str(e))
                with st.expander("Details"):
                    st.exception(e)


def _has_lightgbm() -> bool:
    try:
        import lightgbm  # noqa: F401

        return True
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def _read_json(path_s: str) -> dict[str, Any] | None:
    p = Path(path_s)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _read_csv_head(path_s: str, *, n: int = 50) -> pd.DataFrame | None:
    p = Path(path_s)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p).head(int(n))
    except Exception:
        return None


def _fmt_pct(x: Any) -> str:
    try:
        return f"{float(x):.1%}"
    except Exception:
        return ""


def _fmt_num(x: Any) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
        if abs(xf) >= 1_000_000:
            return f"{xf:,.0f}"
        if abs(xf) >= 1_000:
            return f"{xf:,.1f}"
        return f"{xf:.3f}" if xf < 10 else f"{xf:.2f}"
    except Exception:
        return str(x)


def _redact_cfg_paths_for_ui(cfg: RunConfig, *, filenames: dict[str, str]) -> RunConfig:
    """
    Streamlit BYO mode can optionally avoid persisting raw uploaded CSVs to disk.
    In that case we still write a config artifact, but redact paths to a placeholder namespace.
    """
    base = Path("__not_persisted__")

    def _p(key: str, default_name: str) -> Path:
        name = filenames.get(key) or default_name
        return base / name

    return replace(
        cfg,
        paths=replace(
            cfg.paths,
            ads_path=_p("ads", "ads.csv"),
            leads_path=_p("leads", "leads.csv"),
            outcomes_path=_p("outcomes", "outcomes.csv") if cfg.paths.outcomes_path is not None else None,
            lead_to_ad_map_path=_p("map", "lead_to_ad_map.csv") if cfg.paths.lead_to_ad_map_path is not None else None,
            ads_placement_path=_p("placement", "ads_placement.csv") if cfg.paths.ads_placement_path is not None else None,
            ads_geo_path=_p("geo", "ads_geo.csv") if cfg.paths.ads_geo_path is not None else None,
            adset_targeting_path=_p("targeting", "adset_targeting.csv") if cfg.paths.adset_targeting_path is not None else None,
            ad_creatives_path=_p("creatives", "ad_creatives.csv") if cfg.paths.ad_creatives_path is not None else None,
        ),
    )


def _default_model_cfg(model_type: str) -> ModelConfig:
    return ModelConfig(
        model_type=model_type,
        calibration_method="sigmoid",
        random_seed=7,
        lgbm_params={
            "n_estimators": 300,
            "learning_rate": 0.06,
            "num_leaves": 31,
            "min_child_samples": 50,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
        },
        logreg_params={"C": 1.0, "max_iter": 2000},
    )


def _build_cfg(
    ads_path: Path,
    leads_path: Path,
    outcomes_path: Path | None,
    lead_to_ad_map_path: Path | None,
    ads_placement_path: Path | None,
    ads_geo_path: Path | None,
    adset_targeting_path: Path | None,
    ad_creatives_path: Path | None,
    value_per_qualified: float,
    model_type: str,
) -> RunConfig:
    return RunConfig(
        schema_version=1,
        paths=PathsConfig(
            ads_path=ads_path,
            leads_path=leads_path,
            outcomes_path=outcomes_path,
            lead_to_ad_map_path=lead_to_ad_map_path,
            ads_placement_path=ads_placement_path,
            ads_geo_path=ads_geo_path,
            adset_targeting_path=adset_targeting_path,
            ad_creatives_path=ad_creatives_path,
        ),
        label=LabelConfig(label_window_days=14, as_of_date=None, require_label_maturity=True),
        features=FeaturesConfig(ads_granularity="daily", feature_window_days=7, feature_lag_days=1),
        business=BusinessConfig(value_per_qualified=float(value_per_qualified)),
        splits=SplitsConfig(train_frac=0.6, calib_frac=0.2, test_frac=0.2),
        model=_default_model_cfg(model_type),
        reporting=ReportingConfig(topk_frac=0.10, ece_bins=10, min_segment_leads=30),
    )


def _load_predictions_sample(run_dir: Path, *, max_rows: int = 20000) -> pd.DataFrame | None:
    pred_p = Path(run_dir) / "predictions.csv"
    if not pred_p.exists():
        return None
    try:
        cols0 = pd.read_csv(pred_p, nrows=0).columns.tolist()
        keep = [
            c
            for c in [
                "lead_id",
                "created_time",
                "campaign_name",
                "adset_name",
                "ad_name",
                "label_status",
                "label",
                "p_qualified_14d",
                "value_per_qualified",
                "elv",
                "score_rank",
            ]
            if c in cols0
        ]
        d = pd.read_csv(pred_p, usecols=keep)
        if len(d) > int(max_rows):
            d = d.head(int(max_rows)).copy()
        return d
    except Exception:
        try:
            d = pd.read_csv(pred_p)
            if len(d) > int(max_rows):
                d = d.head(int(max_rows)).copy()
            return d
        except Exception:
            return None


def _render_chart_or_df(chart: Any | None, fallback_df: pd.DataFrame | None = None) -> None:
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)
        return
    if fallback_df is not None and len(fallback_df):
        st.dataframe(fallback_df, use_container_width=True)


def _chart_label_counts(pos: int, neg: int, unk: int) -> Any | None:
    alt = _altair()
    if alt is None:
        return None
    df = pd.DataFrame(
        [
            {"label_status": "positive", "count": int(pos)},
            {"label_status": "negative", "count": int(neg)},
            {"label_status": "unknown", "count": int(unk)},
        ]
    )
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("label_status:N", title="Label status"),
            y=alt.Y("count:Q", title="Count"),
            color=alt.Color(
                "label_status:N",
                scale=alt.Scale(domain=["positive", "negative", "unknown"], range=["#2ca02c", "#d62728", "#8c8c8c"]),
                legend=None,
            ),
            tooltip=["label_status:N", "count:Q"],
        )
        .properties(height=260)
    )


def _chart_prob_hist(pred: pd.DataFrame) -> Any | None:
    alt = _altair()
    if alt is None:
        return None
    if pred is None or "p_qualified_14d" not in pred.columns:
        return None
    d = pred[["p_qualified_14d"]].dropna().copy()
    if len(d) == 0:
        return None
    d = d.rename(columns={"p_qualified_14d": "p"})
    return (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X("p:Q", bin=alt.Bin(maxbins=30), title="Predicted P(qualified_within_14d)"),
            y=alt.Y("count():Q", title="Leads"),
            tooltip=[alt.Tooltip("count():Q", title="Leads")],
        )
        .properties(height=260)
    )


def _chart_elv_hist(pred: pd.DataFrame) -> Any | None:
    alt = _altair()
    if alt is None:
        return None
    if pred is None or "elv" not in pred.columns:
        return None
    d = pred[["elv"]].dropna().copy()
    if len(d) == 0:
        return None
    return (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X("elv:Q", bin=alt.Bin(maxbins=30), title="Expected $/lead (ELV)"),
            y=alt.Y("count():Q", title="Leads"),
            tooltip=[alt.Tooltip("count():Q", title="Leads")],
        )
        .properties(height=260)
    )


def _chart_lift(metrics: dict[str, Any]) -> Any | None:
    alt = _altair()
    if alt is None:
        return None
    lift = ((metrics or {}).get("model") or {}).get("lift_curve") or {}
    xs = lift.get("population_frac") or []
    ys = lift.get("positive_capture_frac") or []
    if not xs or not ys:
        return None
    df = pd.DataFrame({"population_frac": xs, "capture": ys})
    rand = pd.DataFrame({"population_frac": [0, 1], "capture": [0, 1]})
    c1 = (
        alt.Chart(df)
        .mark_line(color="#1f77b4")
        .encode(
            x=alt.X("population_frac:Q", title="Fraction of leads contacted", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("capture:Q", title="Fraction of qualified captured", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("population_frac:Q", title="Pop frac", format=".2f"),
                alt.Tooltip("capture:Q", title="Capture", format=".2f"),
            ],
        )
    )
    c2 = (
        alt.Chart(rand)
        .mark_line(strokeDash=[6, 4], color="#999")
        .encode(x="population_frac:Q", y="capture:Q")
    )
    return alt.layer(c2, c1).properties(height=280, title="Lift Curve (Cumulative Capture)")


def _chart_calibration(metrics: dict[str, Any]) -> Any | None:
    alt = _altair()
    if alt is None:
        return None
    cal = ((metrics or {}).get("model") or {}).get("calibration") or {}
    bins = cal.get("bins") or []
    rows = []
    for b in bins:
        if not isinstance(b, dict):
            continue
        if b.get("count") and b.get("p_mean") is not None and b.get("y_rate") is not None:
            rows.append({"p_mean": float(b["p_mean"]), "y_rate": float(b["y_rate"]), "count": int(b["count"])})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    perfect = pd.DataFrame({"p_mean": [0, 1], "y_rate": [0, 1]})
    ece = cal.get("ece")
    title = "Calibration"
    if ece is not None:
        try:
            title += f" (ECE={float(ece):.3f})"
        except Exception:
            pass
    c1 = (
        alt.Chart(df)
        .mark_line(color="#ff7f0e")
        .encode(
            x=alt.X("p_mean:Q", title="Mean predicted probability", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("y_rate:Q", title="Empirical positive rate", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("p_mean:Q", title="p_mean", format=".3f"),
                alt.Tooltip("y_rate:Q", title="y_rate", format=".3f"),
                alt.Tooltip("count:Q", title="count"),
            ],
        )
    )
    pts = alt.Chart(df).mark_point(color="#ff7f0e", filled=True, size=60).encode(x="p_mean:Q", y="y_rate:Q")
    c2 = (
        alt.Chart(perfect)
        .mark_line(strokeDash=[6, 4], color="#999")
        .encode(x="p_mean:Q", y="y_rate:Q")
    )
    return alt.layer(c2, c1, pts).properties(height=280, title=title)


def _chart_top_segments(lb: pd.DataFrame, *, metric: str, top_n: int = 20) -> Any | None:
    alt = _altair()
    if alt is None or lb is None or metric not in lb.columns:
        return None
    d = lb.copy()
    if "campaign_name" in d.columns:
        d["segment"] = d["campaign_name"].fillna(d.get("campaign_id"))
    else:
        d["segment"] = d.get("campaign_id")
    d = d.dropna(subset=["segment"]).copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=[metric]).copy()
    if len(d) == 0:
        return None
    d = d.sort_values(metric, ascending=False).head(int(top_n)).copy()
    d["segment"] = d["segment"].astype(str)
    d = d.sort_values(metric, ascending=True)
    color = "#1f77b4" if metric != "spend" else "#9467bd"
    return (
        alt.Chart(d)
        .mark_bar(color=color)
        .encode(
            x=alt.X(f"{metric}:Q", title=metric),
            y=alt.Y("segment:N", sort=None, title=""),
            tooltip=["segment:N", alt.Tooltip(f"{metric}:Q", title=metric)],
        )
        .properties(height=min(520, 18 * int(top_n) + 60), title=f"Top {int(top_n)} segments by {metric}")
    )


def _chart_top_adsets(lb: pd.DataFrame, *, metric: str, top_n: int = 20) -> Any | None:
    alt = _altair()
    if alt is None or lb is None or metric not in lb.columns:
        return None
    d = lb.copy()
    if "adset_name" in d.columns:
        d["segment"] = d["adset_name"].fillna(d.get("adset_id"))
    else:
        d["segment"] = d.get("adset_id")
    d = d.dropna(subset=["segment"]).copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=[metric]).copy()
    if len(d) == 0:
        return None
    d = d.sort_values(metric, ascending=False).head(int(top_n)).copy()
    d["segment"] = d["segment"].astype(str)
    d = d.sort_values(metric, ascending=True)
    color = "#1f77b4" if metric != "spend" else "#9467bd"
    return (
        alt.Chart(d)
        .mark_bar(color=color)
        .encode(
            x=alt.X(f"{metric}:Q", title=metric),
            y=alt.Y("segment:N", sort=None, title=""),
            tooltip=["segment:N", alt.Tooltip(f"{metric}:Q", title=metric)],
        )
        .properties(height=min(520, 18 * int(top_n) + 60), title=f"Top {int(top_n)} adsets by {metric}")
    )


def _chart_spend_vs_elv(lb: pd.DataFrame, *, level: str = "campaign") -> Any | None:
    alt = _altair()
    if alt is None or lb is None:
        return None
    if "spend" not in lb.columns or "predicted_elv" not in lb.columns:
        return None
    d = lb.copy()
    d["spend"] = pd.to_numeric(d["spend"], errors="coerce")
    d["predicted_elv"] = pd.to_numeric(d["predicted_elv"], errors="coerce")
    d = d.dropna(subset=["spend", "predicted_elv"]).copy()
    if len(d) == 0:
        return None

    if level == "adset":
        name = d.get("adset_name") if "adset_name" in d.columns else None
        d["segment"] = name.fillna(d.get("adset_id")) if name is not None else d.get("adset_id")
    else:
        name = d.get("campaign_name") if "campaign_name" in d.columns else None
        d["segment"] = name.fillna(d.get("campaign_id")) if name is not None else d.get("campaign_id")
    d["segment"] = d["segment"].astype(str)

    # Avoid rendering thousands of points in the browser.
    if len(d) > 800:
        d = d.sort_values("spend", ascending=False).head(800).copy()

    tooltips: list[Any] = [
        "segment:N",
        alt.Tooltip("spend:Q", title="spend", format=",.2f"),
        alt.Tooltip("predicted_elv:Q", title="predicted_elv", format=",.2f"),
    ]
    if "lead_count" in d.columns:
        tooltips.append(alt.Tooltip("lead_count:Q", title="lead_count"))
    if "elv_per_spend" in d.columns:
        tooltips.append(alt.Tooltip("elv_per_spend:Q", title="elv_per_spend", format=",.3f"))

    color_field: Any = "low_volume:N" if "low_volume" in d.columns else alt.value("#1f77b4")
    chart = (
        alt.Chart(d)
        .mark_circle(size=70, opacity=0.55)
        .encode(
            x=alt.X("spend:Q", title="Spend"),
            y=alt.Y("predicted_elv:Q", title="Predicted ELV"),
            color=color_field,
            tooltip=tooltips,
        )
        .properties(height=340, title=f"Spend vs predicted ELV ({level})")
    )
    return chart


def _chart_drift(drift: dict[str, Any]) -> Any | None:
    alt = _altair()
    if alt is None or not drift:
        return None

    psi = drift.get("psi_train_vs_test") or drift.get("psi_scoring_vs_train") or {}
    cols = (psi or {}).get("columns") or {}
    rows = []
    for feat, r in cols.items():
        if not isinstance(r, dict):
            continue
        if r.get("psi") is None:
            continue
        rows.append({"feature": str(feat), "psi": float(r["psi"])})
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("psi", ascending=False).head(20)
    thr = drift.get("psi_threshold")
    df["flagged"] = False
    if thr is not None:
        try:
            df["flagged"] = df["psi"] >= float(thr)
        except Exception:
            pass

    base = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("psi:Q", title="PSI"),
            y=alt.Y("feature:N", sort=None, title=""),
            color=alt.Color("flagged:N", scale=alt.Scale(domain=[False, True], range=["#1f77b4", "#d62728"]), legend=None),
            tooltip=["feature:N", alt.Tooltip("psi:Q", format=".3f")],
        )
        .properties(height=420, title="Top drifted features (PSI)")
    )
    if thr is None:
        return base
    try:
        thr_f = float(thr)
    except Exception:
        return base
    rule = alt.Chart(pd.DataFrame({"thr": [thr_f]})).mark_rule(color="#333", strokeDash=[6, 4]).encode(x="thr:Q")
    return alt.layer(base, rule)


def _render_run_dashboard(run_dir: Path) -> None:
    run_dir = Path(run_dir)

    meta = _read_json(str(run_dir / "metadata.json")) or {}
    profile = _read_json(str(run_dir / "data_profile.json")) or {}
    metrics = _read_json(str(run_dir / "metrics.json"))
    drift_json = _read_json(str(run_dir / "drift.json"))

    join = (meta.get("join") or {}) if isinstance(meta, dict) else {}
    labeling = (profile.get("labeling") or {}) if isinstance(profile, dict) else {}
    counts = (labeling.get("counts") or {}) if isinstance(labeling, dict) else {}

    join_strategy = str(join.get("strategy") or "")
    match_rate = join.get("match_rate")
    pos = int(counts.get("positive", 0) or 0)
    neg = int(counts.get("negative", 0) or 0)
    unk = int(counts.get("unknown", 0) or 0)

    st.subheader("Run Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Join strategy", join_strategy or "n/a")
    c2.metric("Join match", _fmt_pct(match_rate) if match_rate is not None else "n/a")
    c3.metric("Labeled", str(pos + neg))
    c4.metric("Unknown", str(unk))
    pr = labeling.get("positive_rate_labeled")
    c5.metric("Pos rate (labeled)", _fmt_pct(pr) if pr is not None else "n/a")

    if isinstance(match_rate, (int, float)) and float(match_rate) < 0.90:
        st.warning(
            "Low join match rate. Results may be unreliable; prefer ID-based joins or provide lead_to_ad_map.csv."
        )

    pred_p = run_dir / "predictions.csv"
    lb_c_p = run_dir / "leaderboard_campaign.csv"
    lb_a_p = run_dir / "leaderboard_adset.csv"
    rep_p = run_dir / "report.html"

    preds = _load_predictions_sample(run_dir, max_rows=30000)
    lb_c = pd.read_csv(lb_c_p) if lb_c_p.exists() else None
    lb_a = pd.read_csv(lb_a_p) if lb_a_p.exists() else None

    # Value / ROI cues (lead-level).
    if preds is not None and len(preds):
        avg_p = float(pd.to_numeric(preds.get("p_qualified_14d"), errors="coerce").mean()) if "p_qualified_14d" in preds.columns else None
        avg_elv = float(pd.to_numeric(preds.get("elv"), errors="coerce").mean()) if "elv" in preds.columns else None
        total_elv = float(pd.to_numeric(preds.get("elv"), errors="coerce").sum()) if "elv" in preds.columns else None

        vpq = None
        if "value_per_qualified" in preds.columns:
            vals = pd.to_numeric(preds["value_per_qualified"], errors="coerce").dropna().unique()
            if len(vals) == 1:
                vpq = float(vals[0])
        is_dollars = vpq is not None and vpq >= 10

        st.subheader("Value Summary")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Value per qualified", f"${vpq:,.0f}" if is_dollars and vpq is not None else (_fmt_num(vpq) if vpq is not None else "n/a"))
        v2.metric("Avg P(qualified)", _fmt_pct(avg_p) if avg_p is not None else "n/a")
        v3.metric("Avg value / lead (ELV)", f"${avg_elv:,.2f}" if is_dollars and avg_elv is not None else (_fmt_num(avg_elv) if avg_elv is not None else "n/a"))
        v4.metric("Total predicted value", f"${total_elv:,.0f}" if is_dollars and total_elv is not None else (_fmt_num(total_elv) if total_elv is not None else "n/a"))

    tabs = st.tabs(["Overview", "Leaderboards", "Model", "Drift", "Creative", "Advisor", "Report", "Downloads"])

    with tabs[0]:
        cL, cR = st.columns([1, 1])
        with cL:
            st.markdown("**Labels (pos/neg/unknown)**")
            _render_chart_or_df(_chart_label_counts(pos, neg, unk))
        with cR:
            st.markdown("**Lead-level predictions**")
            if preds is None:
                st.info("predictions.csv not found yet for this run.")
            else:
                t1, t2 = st.tabs(["P(qualified)", "Expected $/lead (ELV)"])
                with t1:
                    _render_chart_or_df(_chart_prob_hist(preds))
                with t2:
                    _render_chart_or_df(_chart_elv_hist(preds))

        if lb_c is not None and len(lb_c):
            st.markdown("**Top campaigns by ELV per spend**")
            _render_chart_or_df(_chart_top_segments(lb_c, metric="elv_per_spend", top_n=20))

        if preds is not None and len(preds):
            with st.expander("Top leads (by predicted value)", expanded=False):
                cols = [c for c in ["lead_id", "created_time", "campaign_name", "adset_name", "ad_name", "p_qualified_14d", "value_per_qualified", "elv", "label_status"] if c in preds.columns]
                d = preds.copy()
                if "score_rank" in d.columns:
                    d["score_rank"] = pd.to_numeric(d["score_rank"], errors="coerce")
                    d = d.dropna(subset=["score_rank"]).sort_values("score_rank", ascending=True).head(25)
                elif "elv" in d.columns:
                    d["elv"] = pd.to_numeric(d["elv"], errors="coerce")
                    d = d.sort_values("elv", ascending=False).head(25)
                st.dataframe(d[cols], use_container_width=True)

        with st.expander("Data profile details", expanded=False):
            dp_md = run_dir / "data_profile.md"
            if dp_md.exists():
                st.markdown(dp_md.read_text())
            else:
                st.json(profile)

        if preds is not None:
            with st.expander("Predictions preview (first 50 rows)", expanded=False):
                st.dataframe(preds.head(50), use_container_width=True)

    with tabs[1]:
        if lb_c is None or not len(lb_c):
            st.info("leaderboard_campaign.csv not found yet for this run.")
        else:
            st.subheader("Campaign leaderboard")
            top_n = st.slider("Top N", min_value=5, max_value=50, value=20, step=5, key="lb_topn_campaign")
            metric_candidates = [
                "elv_per_spend",
                "roi",
                "expected_profit",
                "predicted_elv",
                "avg_elv",
                "cpl",
                "expected_profit_per_lead",
                "spend",
                "lead_count",
                "avg_p_qualified_14d",
            ]
            metric_options = [m for m in metric_candidates if lb_c is not None and m in lb_c.columns]
            if not metric_options:
                metric_options = ["predicted_elv"]
            default_metric = "elv_per_spend" if "elv_per_spend" in metric_options else metric_options[0]
            metric = st.selectbox(
                "Sort metric",
                metric_options,
                index=metric_options.index(default_metric),
                key="lb_metric_campaign",
            )
            show_low = st.checkbox("Include low-volume segments", value=False, key="lb_show_low_campaign")
            d_all = lb_c.copy()
            if not show_low and "low_volume" in d_all.columns:
                d_all = d_all[~d_all["low_volume"].fillna(False)].copy()
            d_top = d_all.sort_values(metric, ascending=False).head(int(top_n))
            st.dataframe(d_top, use_container_width=True)

            cc1, cc2 = st.columns([1, 1])
            with cc1:
                _render_chart_or_df(_chart_top_segments(d_top, metric=metric, top_n=int(min(top_n, 25))))
            with cc2:
                _render_chart_or_df(_chart_spend_vs_elv(d_all, level="campaign"))

        if lb_a is not None and len(lb_a):
            st.subheader("Adset leaderboard")
            top_n = st.slider("Top N (adset)", min_value=5, max_value=50, value=20, step=5, key="lb_topn_adset")
            metric_candidates = [
                "elv_per_spend",
                "roi",
                "expected_profit",
                "predicted_elv",
                "avg_elv",
                "cpl",
                "expected_profit_per_lead",
                "spend",
                "lead_count",
                "avg_p_qualified_14d",
            ]
            metric_options = [m for m in metric_candidates if lb_a is not None and m in lb_a.columns]
            if not metric_options:
                metric_options = ["predicted_elv"]
            default_metric = "elv_per_spend" if "elv_per_spend" in metric_options else metric_options[0]
            metric = st.selectbox(
                "Sort metric (adset)",
                metric_options,
                index=metric_options.index(default_metric),
                key="lb_metric_adset",
            )
            show_low = st.checkbox("Include low-volume segments (adset)", value=False, key="lb_show_low_adset")
            d_all = lb_a.copy()
            if not show_low and "low_volume" in d_all.columns:
                d_all = d_all[~d_all["low_volume"].fillna(False)].copy()
            d_top = d_all.sort_values(metric, ascending=False).head(int(top_n))
            st.dataframe(d_top, use_container_width=True)

            cc1, cc2 = st.columns([1, 1])
            with cc1:
                _render_chart_or_df(_chart_top_adsets(d_top, metric=metric, top_n=int(min(top_n, 25))))
            with cc2:
                _render_chart_or_df(_chart_spend_vs_elv(d_all, level="adset"))

    with tabs[2]:
        if metrics is None:
            st.info("metrics.json not found (this is likely a score-only run).")
        else:
            st.subheader("Model evaluation")
            m = metrics.get("model") or {}
            base = metrics.get("baseline_campaign_rate") or {}

            # Summary table
            rows = [
                {
                    "model": f"calibrated {metrics.get('model_type', 'model')}",
                    "pr_auc": m.get("pr_auc"),
                    "brier": m.get("brier"),
                    "lift@k": (m.get("lift_at_k") or {}).get("lift"),
                    "capture@k": (m.get("lift_at_k") or {}).get("capture"),
                },
                {
                    "model": "baseline: campaign rate",
                    "pr_auc": base.get("pr_auc"),
                    "brier": base.get("brier"),
                    "lift@k": (base.get("lift_at_k") or {}).get("lift"),
                    "capture@k": (base.get("lift_at_k") or {}).get("capture"),
                },
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                _render_chart_or_df(_chart_lift(metrics))
            with c2:
                _render_chart_or_df(_chart_calibration(metrics))

    with tabs[3]:
        drift = None
        if metrics is not None:
            drift = metrics.get("drift")
        if drift is None:
            drift = drift_json
        if not drift:
            st.info("No drift data available for this run.")
        else:
            st.subheader("Drift (PSI)")
            thr = drift.get("psi_threshold")
            st.caption(f"Threshold: {thr}" if thr is not None else "Threshold: n/a")
            _render_chart_or_df(_chart_drift(drift))

            psi = drift.get("psi_train_vs_test") or drift.get("psi_scoring_vs_train") or {}
            cols = (psi or {}).get("columns") or {}
            rows = []
            for feat, r in cols.items():
                if isinstance(r, dict) and r.get("psi") is not None:
                    rows.append({"feature": str(feat), "psi": float(r["psi"])})
            if rows:
                st.dataframe(pd.DataFrame(rows).sort_values("psi", ascending=False), use_container_width=True)

    with tabs[4]:
        st.subheader("Creative analysis")
        # View any existing artifacts first.
        ca_dirs = [run_dir / "creative_analysis", run_dir]
        shown = False
        for cd in ca_dirs:
            summ = cd / "creative_cluster_summary.csv"
            neigh = cd / "creative_neighbors.csv"
            if summ.exists():
                st.markdown(f"**Cluster summary** (`{cd.name}`)")
                summ_df = pd.read_csv(summ)
                st.dataframe(summ_df.head(50), use_container_width=True)
                alt = _altair()
                if alt is not None and len(summ_df):
                    metric = None
                    for cand in ["predicted_elv", "avg_elv", "avg_p_qualified_14d", "lead_count"]:
                        if cand in summ_df.columns:
                            metric = cand
                            break
                    if metric is not None:
                        d = summ_df.copy()
                        d[metric] = pd.to_numeric(d[metric], errors="coerce")
                        d = d.dropna(subset=[metric]).sort_values(metric, ascending=False).head(20)
                        if len(d):
                            chart = (
                                alt.Chart(d)
                                .mark_bar(color="#1f77b4")
                                .encode(
                                    x=alt.X(f"{metric}:Q", title=metric),
                                    y=alt.Y("cluster_id:N", sort="-x", title="cluster_id"),
                                    tooltip=["cluster_id:N", alt.Tooltip(f"{metric}:Q", title=metric)],
                                )
                                .properties(height=360, title=f"Top clusters by {metric}")
                            )
                            st.altair_chart(chart, use_container_width=True)
                shown = True
            if neigh.exists():
                st.markdown(f"**Neighbors (sample)** (`{cd.name}`)")
                st.dataframe(pd.read_csv(neigh).head(50), use_container_width=True)
                shown = True
        if not shown:
            st.info("No creative analysis artifacts found yet for this run.")

        st.markdown("---")
        _render_creative_embeddings(run_dir)

    with tabs[5]:
        _render_gemini_advisor(run_dir)

    with tabs[6]:
        st.subheader("HTML report")
        if not rep_p.exists():
            st.info("report.html not found yet for this run.")
        else:
            try:
                import streamlit.components.v1 as components

                components.html(rep_p.read_text(), height=980, scrolling=True)
            except Exception:
                st.download_button("Download report.html", data=rep_p.read_bytes(), file_name="report.html")

    with tabs[7]:
        st.subheader("Downloads")
        c1, c2 = st.columns(2)
        with c1:
            if pred_p.exists():
                st.download_button("predictions.csv", data=pred_p.read_bytes(), file_name="predictions.csv")
            if lb_c_p.exists():
                st.download_button(
                    "leaderboard_campaign.csv",
                    data=lb_c_p.read_bytes(),
                    file_name="leaderboard_campaign.csv",
                )
            if lb_a_p.exists():
                st.download_button("leaderboard_adset.csv", data=lb_a_p.read_bytes(), file_name="leaderboard_adset.csv")
        with c2:
            if rep_p.exists():
                st.download_button("report.html", data=rep_p.read_bytes(), file_name="report.html")
            cfg_p = run_dir / "config.yaml"
            if cfg_p.exists():
                st.download_button("config.yaml", data=cfg_p.read_bytes(), file_name="config.yaml")
            meta_p = run_dir / "metadata.json"
            if meta_p.exists():
                st.download_button("metadata.json", data=meta_p.read_bytes(), file_name="metadata.json")


def _render_run_outputs(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    st.success(f"Run directory: {run_dir}")
    _render_run_dashboard(run_dir)


st.set_page_config(page_title="Meta ELV Kit", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.8rem; padding-bottom: 2.2rem; }
div[data-testid="stMetric"] { background: rgba(250,250,250,0.35); border: 1px solid rgba(0,0,0,0.06); padding: 10px 12px; border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Meta Lead Ads ELV Kit")
st.caption("Score-first demo. Do not upload sensitive PII to public deployments.")

with st.expander("Privacy warning", expanded=True):
    st.warning(
        "Treat `leads.csv` as sensitive input. This app is designed to be demoable using synthetic data. "
        "Do not upload real client exports or PII to public deployments."
    )

mode = st.sidebar.radio("Mode", ["Demo data (recommended)", "Bring your own CSVs"], index=0)
value_per_qualified = st.sidebar.number_input(
    "Value per qualified lead (USD)",
    min_value=0.0,
    value=1.0,
    step=1.0,
    help="ELV (expected $/lead) = P(qualified_within_14d) × value_per_qualified. "
    "If you don't know the dollar value yet, keep 1.0 for relative scoring.",
)

runs_base = Path.cwd() / "runs"
run_ids: list[str] = []
try:
    if runs_base.exists():
        run_ids = sorted([p.name for p in runs_base.iterdir() if p.is_dir()], reverse=True)
except Exception:
    run_ids = []
selected_run = st.sidebar.selectbox("Open an existing run (optional)", ["(none)"] + run_ids, index=0)
if selected_run != "(none)":
    st.header(f"Run: {selected_run}")
    _render_run_outputs(runs_base / selected_run)
    st.stop()


if mode.startswith("Demo"):
    st.header("Demo (Synthetic Data)")
    st.write("Generates synthetic CSVs and produces predictions + leaderboards + report.")

    model_choice = st.selectbox(
        "Demo run behavior",
        [
            "Score-first (requires a demo model bundle)",
            "Train locally (slower, but no pre-bundled model needed)",
        ],
        index=0,
    )

    if st.button("Run demo", type="primary"):
        ctx = create_run_context(Path.cwd())
        demo_dir = ctx.run_dir / "demo_data"
        demo_paths = generate_demo_data(demo_dir, DemoDataSpec())

        cfg = _build_cfg(
            ads_path=demo_paths["ads"],
            leads_path=demo_paths["leads"],
            outcomes_path=demo_paths["outcomes"],
            lead_to_ad_map_path=None,
            ads_placement_path=demo_paths.get("ads_placement"),
            ads_geo_path=demo_paths.get("ads_geo"),
            adset_targeting_path=demo_paths.get("adset_targeting"),
            ad_creatives_path=demo_paths.get("ad_creatives"),
            value_per_qualified=value_per_qualified,
            model_type="logreg",
        )
        save_config(cfg, ctx.run_dir / "config.yaml")

        vr = validate_from_config(cfg)
        if not vr.ok:
            st.error("Validation failed (unexpected for demo).")
            st.code(render_validation_summary(vr))
            st.stop()

        with st.spinner("Running pipeline..."):
            try:
                if model_choice.startswith("Score-first"):
                    import importlib.resources as ir

                    demo_model = ir.files("meta_elv.assets").joinpath("demo_model.joblib")
                    if not demo_model.is_file():
                        raise FileNotFoundError(
                            "Bundled demo model not found. Use 'Train locally' or add meta_elv/assets/demo_model.joblib."
                        )
                    with ir.as_file(demo_model) as mp:
                        run_score(cfg, ctx, model_path=mp)
                else:
                    run_train(cfg, ctx)
            except Exception as e:
                st.error(str(e))
                with st.expander("Details"):
                    st.exception(e)
                st.stop()

        _render_run_outputs(ctx.run_dir)

else:
    st.header("Bring Your Own CSVs")
    st.write("Upload CSV exports, validate, then score with an existing model (or train locally).")

    ads_up = st.file_uploader("ads.csv", type=["csv"], accept_multiple_files=False)
    leads_up = st.file_uploader("leads.csv", type=["csv"], accept_multiple_files=False)
    outcomes_up = st.file_uploader("outcomes.csv (optional, required for training)", type=["csv"], accept_multiple_files=False)
    map_up = st.file_uploader("lead_to_ad_map.csv (optional)", type=["csv"], accept_multiple_files=False)
    persist_uploads = st.checkbox(
        "Persist uploaded CSVs to runs/<run_id>/uploaded (not recommended for sensitive data)",
        value=False,
        help="If disabled, uploads are written only to a temporary directory for the run execution. "
        "Derived artifacts (table/predictions/report) are still written under runs/.",
    )
    with st.expander("Optional enrichment CSVs", expanded=False):
        st.caption("If you train a model using these, you must also provide them when scoring with that model.")
        placement_up = st.file_uploader("ads_placement.csv (optional)", type=["csv"], accept_multiple_files=False)
        geo_up = st.file_uploader("ads_geo.csv (optional)", type=["csv"], accept_multiple_files=False)
        targeting_up = st.file_uploader("adset_targeting.csv (optional)", type=["csv"], accept_multiple_files=False)
        creatives_up = st.file_uploader("ad_creatives.csv (optional)", type=["csv"], accept_multiple_files=False)
    model_up = st.file_uploader("model.joblib (optional, for scoring)", type=["joblib", "pkl", "bin"], accept_multiple_files=False)

    allow_train = st.checkbox("Allow training in the app (capped; local recommended)", value=False)
    model_options = ["logreg"]
    if _has_lightgbm():
        model_options.append("lgbm")
    model_type = st.selectbox("Model type", model_options, index=0)
    if "lgbm" not in model_options:
        st.info("LightGBM is not installed in this environment. Only logistic regression is available.")

    max_train_rows = (
        st.number_input("max_train_rows (labeled leads)", min_value=200, value=5000, step=200)
        if allow_train
        else None
    )

    if st.button("Validate"):
        if not ads_up or not leads_up:
            st.error("ads.csv and leads.csv are required.")
            st.stop()

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ads_p = td / "ads.csv"
            leads_p = td / "leads.csv"
            outcomes_p = td / "outcomes.csv" if outcomes_up else None
            map_p = td / "lead_to_ad_map.csv" if map_up else None
            placement_p = td / "ads_placement.csv" if placement_up else None
            geo_p = td / "ads_geo.csv" if geo_up else None
            targeting_p = td / "adset_targeting.csv" if targeting_up else None
            creatives_p = td / "ad_creatives.csv" if creatives_up else None
            _write_upload(ads_up, ads_p)
            _write_upload(leads_up, leads_p)
            if outcomes_up and outcomes_p is not None:
                _write_upload(outcomes_up, outcomes_p)
            if map_up and map_p is not None:
                _write_upload(map_up, map_p)
            if placement_up and placement_p is not None:
                _write_upload(placement_up, placement_p)
            if geo_up and geo_p is not None:
                _write_upload(geo_up, geo_p)
            if targeting_up and targeting_p is not None:
                _write_upload(targeting_up, targeting_p)
            if creatives_up and creatives_p is not None:
                _write_upload(creatives_up, creatives_p)

            cfg = _build_cfg(
                ads_path=ads_p,
                leads_path=leads_p,
                outcomes_path=outcomes_p,
                lead_to_ad_map_path=map_p,
                ads_placement_path=placement_p,
                ads_geo_path=geo_p,
                adset_targeting_path=targeting_p,
                ad_creatives_path=creatives_p,
                value_per_qualified=value_per_qualified,
                model_type=model_type,
            )
            vr = validate_from_config(cfg)
            st.code(render_validation_summary(vr))

    if st.button("Run (train or score)", type="primary"):
        if not ads_up or not leads_up:
            st.error("ads.csv and leads.csv are required.")
            st.stop()

        ctx = create_run_context(Path.cwd())
        upload_names = {
            "ads": getattr(ads_up, "name", "ads.csv") if ads_up else "ads.csv",
            "leads": getattr(leads_up, "name", "leads.csv") if leads_up else "leads.csv",
            "outcomes": getattr(outcomes_up, "name", "outcomes.csv") if outcomes_up else "outcomes.csv",
            "map": getattr(map_up, "name", "lead_to_ad_map.csv") if map_up else "lead_to_ad_map.csv",
            "placement": getattr(placement_up, "name", "ads_placement.csv") if placement_up else "ads_placement.csv",
            "geo": getattr(geo_up, "name", "ads_geo.csv") if geo_up else "ads_geo.csv",
            "targeting": getattr(targeting_up, "name", "adset_targeting.csv") if targeting_up else "adset_targeting.csv",
            "creatives": getattr(creatives_up, "name", "ad_creatives.csv") if creatives_up else "ad_creatives.csv",
        }

        temp_ctx: tempfile.TemporaryDirectory[str] | None = None
        if persist_uploads:
            data_dir = ctx.run_dir / "uploaded"
            data_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_ctx = tempfile.TemporaryDirectory()
            data_dir = Path(temp_ctx.name)

        try:
            ads_p = data_dir / "ads.csv"
            leads_p = data_dir / "leads.csv"
            outcomes_p = data_dir / "outcomes.csv" if outcomes_up else None
            map_p = data_dir / "lead_to_ad_map.csv" if map_up else None
            placement_p = data_dir / "ads_placement.csv" if placement_up else None
            geo_p = data_dir / "ads_geo.csv" if geo_up else None
            targeting_p = data_dir / "adset_targeting.csv" if targeting_up else None
            creatives_p = data_dir / "ad_creatives.csv" if creatives_up else None

            _write_upload(ads_up, ads_p)
            _write_upload(leads_up, leads_p)
            if outcomes_up and outcomes_p is not None:
                _write_upload(outcomes_up, outcomes_p)
            if map_up and map_p is not None:
                _write_upload(map_up, map_p)
            if placement_up and placement_p is not None:
                _write_upload(placement_up, placement_p)
            if geo_up and geo_p is not None:
                _write_upload(geo_up, geo_p)
            if targeting_up and targeting_p is not None:
                _write_upload(targeting_up, targeting_p)
            if creatives_up and creatives_p is not None:
                _write_upload(creatives_up, creatives_p)

            cfg = _build_cfg(
                ads_path=ads_p,
                leads_path=leads_p,
                outcomes_path=outcomes_p,
                lead_to_ad_map_path=map_p,
                ads_placement_path=placement_p,
                ads_geo_path=geo_p,
                adset_targeting_path=targeting_p,
                ad_creatives_path=creatives_p,
                value_per_qualified=value_per_qualified,
                model_type=model_type,
            )
            cfg_artifact = cfg if persist_uploads else _redact_cfg_paths_for_ui(cfg, filenames=upload_names)
            save_config(cfg_artifact, ctx.run_dir / "config.yaml")

            vr = validate_from_config(cfg)
            if not vr.ok:
                st.error("Validation failed.")
                st.code(render_validation_summary(vr))
                st.stop()

            with st.spinner("Running pipeline..."):
                try:
                    if allow_train:
                        if outcomes_p is None:
                            st.error("outcomes.csv is required to train in the app.")
                            st.stop()
                        run_train(cfg, ctx, max_labeled_rows=int(max_train_rows) if max_train_rows else None)
                    else:
                        if model_up is None:
                            st.error("Upload model.joblib to score, or enable training.")
                            st.stop()
                        model_p = ctx.run_dir / "uploaded_model.joblib"
                        model_p.write_bytes(model_up.getvalue())
                        run_score(cfg, ctx, model_path=model_p)
                except Exception as e:
                    st.error(str(e))
                    with st.expander("Details"):
                        st.exception(e)
                    st.stop()

            # Annotate run metadata with UI-specific settings (no lead rows).
            try:
                meta_p = ctx.run_dir / "metadata.json"
                meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
                if not isinstance(meta, dict):
                    meta = {}
                meta["ui"] = {
                    "inputs_persisted": bool(persist_uploads),
                    "uploaded_filenames": {k: v for k, v in upload_names.items() if v},
                }
                meta_p.write_text(json.dumps(meta, indent=2))
            except Exception:
                pass

            _render_run_outputs(ctx.run_dir)
        finally:
            if temp_ctx is not None:
                try:
                    temp_ctx.cleanup()
                except Exception:
                    pass
