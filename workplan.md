# Workplan (Milestones)

This project is designed to read like a small **ML platform**: CSV contracts, self-serve CLI, reproducible runs, and a score-first demo UI.

## Current Status (As Of 2026-02-09)

- Week 1: Done (scaffolding, contracts, demo data, validate)
- Week 2: Done (joins + maturity-safe labels + daily-ads leakage lag + rolling features)
- Week 3: Mostly done (time split, calibration, metrics, baseline campaign-rate, report). LightGBM is optional and may require system OpenMP on macOS.
- Week 4: Done (ELV outputs, leaderboards with min-volume guardrails, run metadata, model/data cards)
- Week 5: Done (Streamlit score-first demo + bundled synthetic demo model)
- Week 6: Done (CI, drift PSI, batch scoring, and docs under docs/)
- Week 7: Done (optional enrichment signals: placement, geo, targeting, creatives)
- Week 8: Done (optional GenAI advisor via Gemini; opt-in)
- Week 9: Done (optional creative media embeddings: similarity + clustering)

## Week 1 — Scaffolding + Contracts + Demo Data

Goal: A stranger can run `elv demo` and see believable outputs (no real data).

Work:
- Create Python package scaffolding (`pyproject.toml`, `src/`, CLI entrypoint).
- Define canonical schemas (v1) and implement a friendly `validate` report.
- Implement synthetic demo generator that emits `ads.csv`, `leads.csv`, `outcomes.csv`.
- Add a score-first demo flow that writes a full `runs/<run_id>/` directory.

Completion standard:
- `elv demo` produces `runs/<run_id>/predictions.csv`, `leaderboard_campaign.csv`, `data_profile.md`.
- `elv validate` fails on missing required columns and reports null % and duplicates.
- Tests exist for schema-required columns and timestamp parsing.

## Week 2 — Table Builder (Joins + Labels + Leakage Guard)

Goal: A reproducible training/scoring table with the 3 biggest ads-ML gotchas handled.

Work:
- Implement join strategy order: IDs > mapping file > name-fallback (with warnings + match rate).
- Implement **label maturity**:
  - positives within window
  - negatives only when mature
  - recent leads labeled `unknown`
- Implement daily-ads leakage guard via `feature_lag_days=1` default.
- Build rolling 7d features (CTR/CPC/CPM, spend/clicks/impressions, activity days, simple trend).

Completion standard:
- `elv build-table` writes a table artifact plus join stats and label counts.
- Table builder refuses to train/eval if there are too few labeled rows (configurable threshold).
- Tests cover: maturity logic, negative labeling rule, lag exclusion.

## Week 3 — Modeling + Calibration + Baselines + Report (Early)

Goal: Honest evaluation that is interpretable for non-ML readers.

Work:
- Time-based split after maturity filtering (train/calib/test).
- Train logistic regression baseline and LightGBM (optional).
- Calibrate probabilities on calibration split (sigmoid default).
- Implement metrics: PR-AUC, Brier, lift@k, top-decile capture, simple ECE bins.
- Generate `report.html` containing join summary, maturity summary, lift, calibration, metrics table.
- Add business baselines:
  - campaign historical qualification rate (train window)
  - optional campaign historical CPL (if spend + lead volume available)

Completion standard:
- `elv train` writes `model.joblib`, `metrics.json`, and `report.html`.
- Report explicitly includes `as_of_date`, split boundaries, and `feature_lag_days`.
- Demo data run finishes in a reasonable time (target under ~60–90s on a laptop).

## Week 4 — ELV Outputs + Leaderboards + Run Metadata

Goal: Turn calibrated probabilities into dollars and actionable rankings.

Work:
- Produce lead-level outputs: `p_qualified_14d`, `elv`, rank/decile.
- Produce campaign + adset leaderboards: spend, predicted ELV, ELV/spend, lead volume.
- Add guardrails:
  - disable adset leaderboard when keys are missing
  - minimum volume thresholds (warn on tiny segments)
- Write `metadata.json` (git sha, schema version, windows, join strategy, metric summary).
- Write `MODEL_CARD.md` and `DATA_CARD.md`.

Completion standard:
- `elv score` produces stable `predictions.csv` and leaderboards.
- `runs/<run_id>/metadata.json` contains enough to reproduce/compare runs.
- Model/data cards are coherent and accurate for the demo.

## Week 5 — Streamlit UI + Hugging Face Space (Score-First)

Goal: A public demo that is reliable (no flaky training) but still feels real.

Work:
- Streamlit app:
  - demo vs BYO mode
  - upload + validate + show join/maturity stats
  - score + leaderboard views + downloads
  - optional train button with strict caps and warnings
- HF Space packaging (`requirements.txt` / docs).
- Explicit warnings about PII: do not upload sensitive data.

Completion standard:
- `streamlit run app.py` works locally and uses the same core pipeline code as the CLI.
- Space can run demo end-to-end without training.
- Training is bounded (row caps / estimator caps) and fails gracefully with messaging.

## Week 6 — Drift Checks + Batch Scoring + CI + Docs

Goal: “Platform polish” that interviewers recognize.

Work:
- Add minimal drift checks (PSI) for 5–10 key features; write to artifacts.
- Add batch scoring mode for “new leads” with consistent artifacts.
- Add CI (lint + unit tests).
- Expand docs:
  - 2-minute quickstart
  - BYO exports guide (schemas + join keys)
  - explanation of maturity/leakage/calibration decisions

Completion standard:
- Scoring can optionally emit drift results and flag high PSI.
- CI passes on PRs (tests include maturity + leakage + join strategy).
- A stranger can succeed without asking repo-specific questions.

## Week 7 — Optional Enrichment Signals (Placement / Geo / Targeting / Creative)

Goal: Support richer ads context signals without breaking the CSV-first, leakage-safe pipeline.

Work:
- Extend config + validation for optional inputs:
  - `paths.ads_placement_path` (`ads_placement.csv`)
  - `paths.ads_geo_path` (`ads_geo.csv`)
  - `paths.adset_targeting_path` (`adset_targeting.csv`)
  - `paths.ad_creatives_path` (`ad_creatives.csv`)
- Implement connector loaders that normalize these inputs to canonical columns.
- Add leakage-safe, rolling-window summary features:
  - spend-share entropy and top-1 share for placement and geo
  - count of active placements/geos in the window
- Add audience keyword features as deterministic numeric hashes (to avoid giant one-hot vectors).
- Add creative-type features (hashed or low-cardinality one-hot) without exploding feature space.
- Update report/profile to show which enrichments were present and any join/coverage stats.

Completion standard:
- `elv validate` reports required columns + coverage summaries for each provided enrichment CSV.
- `elv build-table` produces enrichment-derived features that respect `feature_lag_days` (no same-day leakage).
- Tests cover: lag exclusion for enrichment breakdowns, targeting join behavior, and stable hashing.

## Week 8 — GenAI Advisor (Optional; Gemini)

Goal: Provide credible, opt-in suggestions for targeting and creative improvements without contaminating the ML pipeline or leaking PII.

Work:
- Add a small `meta_elv.genai` module with a single public entrypoint (Gemini-backed) for:
  - “targeting suggestions” from audience keywords + performance summaries
  - “creative suggestions” from optional image/video uploads + performance summaries
- Streamlit UI integration:
  - gated behind an API key (no key => feature disabled)
  - strong privacy warning (“uploads may be sent to a third-party API”)
  - never send lead rows; only aggregate metrics + user-provided context
- Add docs:
  - how to set `GEMINI_API_KEY` locally
  - how to set Space secrets on Hugging Face

Completion standard:
- Streamlit demo still works without any API key.
- With `GEMINI_API_KEY` set, the UI can generate suggestions reliably for demo data.
- Errors are friendly (missing key, missing dependency, API failure, oversized media).

## Week 9 — Creative Media Embeddings (Optional; Similarity + Clustering)

Goal: Provide “platform-y” creative analysis surfaces (neighbors + clusters) using image/video embeddings, without changing the ELV predictive pipeline.

Work:
- Add an optional `embeddings` extra (PyTorch + OpenCLIP + OpenCV).
- Implement a CLI command: `elv creative-analyze`.
- CLI reads `creative_media.csv` (ad_id -> media_path).
- CLI embeds images and sample+embeds video frames (mean-pooled).
- CLI computes top-K cosine neighbors per creative.
- CLI computes KMeans clusters.
- If `--run-dir` is provided and contains `predictions.csv`, produce a cluster-level performance summary (e.g. predicted ELV by cluster).
- Streamlit integration supports uploading `creative_media.csv` + media files.
- Streamlit performs dependency checks and shows friendly “install extra” messaging.
- Streamlit never persists uploaded media to `runs/` by default.
- Report integration includes creative cluster summary + sample neighbors when available.

Completion standard:
- `elv creative-analyze --creative-map creative_media.csv` writes: `creative_neighbors.csv`, `creative_clusters.csv`, `creative_embeddings.npz`.
- When `--run-dir` is provided and has `predictions.csv`, it also writes `creative_cluster_summary.csv`.
- Tests cover the neighbor table shape/constraints and KMeans cluster output shape (no torch required).
