from __future__ import annotations

import json
import tempfile
from pathlib import Path

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


def _write_upload(upload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(upload.getvalue())


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
        ),
        label=LabelConfig(label_window_days=14, as_of_date=None, require_label_maturity=True),
        features=FeaturesConfig(ads_granularity="daily", feature_window_days=7, feature_lag_days=1),
        business=BusinessConfig(value_per_qualified=float(value_per_qualified)),
        splits=SplitsConfig(train_frac=0.6, calib_frac=0.2, test_frac=0.2),
        model=_default_model_cfg(model_type),
        reporting=ReportingConfig(topk_frac=0.10, ece_bins=10, min_segment_leads=30),
    )


def _render_run_outputs(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    st.success(f"Run written to: {run_dir}")

    meta_p = run_dir / "metadata.json"
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text())
            join = meta.get("join") or {}
            labeling = meta.get("labeling") or {}
            counts = labeling.get("counts") or {}

            st.subheader("Join / Label Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Join strategy", str(join.get("strategy") or ""))
            mr = join.get("match_rate")
            c2.metric("Join match rate", f"{float(mr):.1%}" if mr is not None else "")
            c3.metric("Labeled (pos+neg)", str(int(counts.get("positive", 0)) + int(counts.get("negative", 0))))
            c4.metric("Unknown (immature)", str(int(counts.get("unknown", 0))))

            if isinstance(mr, (int, float)) and float(mr) < 0.90:
                st.warning(
                    "Low join match rate. Results may be unreliable; prefer ID-based joins or provide lead_to_ad_map.csv."
                )
        except Exception:
            # Avoid breaking the UI if metadata is malformed.
            pass

    drift_p = run_dir / "drift.json"
    if drift_p.exists():
        try:
            drift = json.loads(drift_p.read_text())
            st.subheader("Drift Summary (PSI)")
            c1, c2, c3 = st.columns(3)
            thr = drift.get("psi_threshold")
            c1.metric("PSI threshold", str(thr) if thr is not None else "")
            c2.metric("Flagged features", str(drift.get("n_flagged", 0)))
            flagged = drift.get("flagged_columns") or []
            c3.metric("Flagged list", ", ".join(flagged[:3]) + ("..." if len(flagged) > 3 else ""))
        except Exception:
            pass

    pred_p = run_dir / "predictions.csv"
    lb_c = run_dir / "leaderboard_campaign.csv"
    lb_a = run_dir / "leaderboard_adset.csv"
    rep = run_dir / "report.html"

    cols = st.columns(2)
    with cols[0]:
        if pred_p.exists():
            st.download_button("Download predictions.csv", data=pred_p.read_bytes(), file_name="predictions.csv")
        if lb_c.exists():
            st.download_button(
                "Download leaderboard_campaign.csv", data=lb_c.read_bytes(), file_name="leaderboard_campaign.csv"
            )
        if lb_a.exists():
            st.download_button("Download leaderboard_adset.csv", data=lb_a.read_bytes(), file_name="leaderboard_adset.csv")
    with cols[1]:
        if rep.exists():
            st.download_button("Download report.html", data=rep.read_bytes(), file_name="report.html")

    if lb_c.exists():
        st.subheader("Campaign Leaderboard (Top 20)")
        st.dataframe(
            # Avoid huge rendering; read small head.
            __import__("pandas").read_csv(lb_c).head(20),
            use_container_width=True,
        )
        try:
            top = __import__("pandas").read_csv(lb_c).head(20)
            if "low_volume" in top.columns and bool(top["low_volume"].fillna(False).any()):
                st.warning("Top campaign leaderboard includes low-volume segments. Interpret cautiously.")
        except Exception:
            pass
    if lb_a.exists():
        st.subheader("Adset Leaderboard (Top 20)")
        st.dataframe(__import__("pandas").read_csv(lb_a).head(20), use_container_width=True)


st.set_page_config(page_title="Meta ELV Kit", layout="wide")

st.title("Meta Lead Ads ELV Kit")
st.caption("Score-first demo. Do not upload sensitive PII to public deployments.")

with st.expander("Privacy warning", expanded=True):
    st.warning(
        "Treat `leads.csv` as sensitive input. This app is designed to be demoable using synthetic data. "
        "Do not upload real client exports or PII to public deployments."
    )

mode = st.sidebar.radio("Mode", ["Demo data (recommended)", "Bring your own CSVs"], index=0)
value_per_qualified = st.sidebar.number_input("value_per_qualified", min_value=0.0, value=1.0, step=1.0)


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
    model_up = st.file_uploader("model.joblib (optional, for scoring)", type=["joblib", "pkl", "bin"], accept_multiple_files=False)

    allow_train = st.checkbox("Allow training in the app (capped; local recommended)", value=False)
    model_type = st.selectbox("Model type", ["logreg", "lgbm"], index=0)

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
            _write_upload(ads_up, ads_p)
            _write_upload(leads_up, leads_p)
            if outcomes_up and outcomes_p is not None:
                _write_upload(outcomes_up, outcomes_p)
            if map_up and map_p is not None:
                _write_upload(map_up, map_p)

            cfg = _build_cfg(
                ads_path=ads_p,
                leads_path=leads_p,
                outcomes_path=outcomes_p,
                lead_to_ad_map_path=map_p,
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
        data_dir = ctx.run_dir / "uploaded"
        data_dir.mkdir(parents=True, exist_ok=True)

        ads_p = data_dir / "ads.csv"
        leads_p = data_dir / "leads.csv"
        outcomes_p = data_dir / "outcomes.csv" if outcomes_up else None
        map_p = data_dir / "lead_to_ad_map.csv" if map_up else None
        _write_upload(ads_up, ads_p)
        _write_upload(leads_up, leads_p)
        if outcomes_up and outcomes_p is not None:
            _write_upload(outcomes_up, outcomes_p)
        if map_up and map_p is not None:
            _write_upload(map_up, map_p)

        cfg = _build_cfg(
            ads_path=ads_p,
            leads_path=leads_p,
            outcomes_path=outcomes_p,
            lead_to_ad_map_path=map_p,
            value_per_qualified=value_per_qualified,
            model_type=model_type,
        )
        save_config(cfg, ctx.run_dir / "config.yaml")

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
                st.exception(e)
                st.stop()

        _render_run_outputs(ctx.run_dir)
