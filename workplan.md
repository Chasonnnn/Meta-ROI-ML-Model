# Workplan (6 Weeks)

This project is designed to read like a small **ML platform**: CSV contracts, self-serve CLI, reproducible runs, and a score-first demo UI.

## Current Status (As Of 2026-02-08)

- Week 1: Done (scaffolding, contracts, demo data, validate)
- Week 2: Done (joins + maturity-safe labels + daily-ads leakage lag + rolling features)
- Week 3: Mostly done (time split, calibration, metrics, baseline campaign-rate, report). LightGBM is optional and may require system OpenMP on macOS.
- Week 4: Mostly done (ELV outputs, leaderboards, run metadata, model/data cards). Guardrails like min-volume warnings are still TODO.
- Week 5: Done (Streamlit score-first demo + bundled synthetic demo model)
- Week 6: Partial (CI exists; drift + batch scoring + expanded docs still TODO)

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
