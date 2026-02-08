# Model Card: ELV (Qualified Within 14 Days)

## Overview
This repository trains a binary classifier to predict:

- `P(qualified_within_14d)`

It is designed for **lead triage** and **budget guidance** via Expected Lead Value (ELV):

- `ELV = P(qualified_within_14d) * value_per_qualified`

## Intended Use

Primary:
- Rank new leads for follow-up (triage).
- Compare campaigns/adsets by predicted ELV per spend.

Not intended for:
- Automated acceptance/rejection decisions.
- Any use where false positives/negatives have high-stakes consequences without human review.

## Inputs

The model is trained from 3 CSVs:
- `ads.csv` daily performance
- `leads.csv` lead-level timestamps + join keys
- `outcomes.csv` lead_id -> qualified_time (optional for scoring-only runs)

Feature sources (v1):
- Lead time features (weekday, hour)
- Rolling daily ad metrics in a pre-lead window (CTR/CPC/CPM, spend/clicks/impressions, simple trends)

## Label Definition

Positive:
- `qualified_time <= created_time + 14 days`

Negative (mature only):
- `qualified_time is null` AND `created_time <= as_of_date - 14 days`

Unknown:
- `created_time > as_of_date - 14 days`

Unknowns are excluded from supervised train/eval by default.

## Leakage Controls

Meta `ads.csv` is assumed to be **daily aggregates**, so “same day” totals may include post-lead activity.
To reduce leakage, v1 uses:
- `feature_lag_days=1` by default (features use metrics through `lead_date - 1 day`)

## Training / Evaluation

Splitting:
- Time-based split on `created_time` after maturity filtering: train -> calibration -> test

Metrics (reported on test):
- PR-AUC (average precision)
- Brier score
- Lift@k and top-decile capture
- Calibration bins + ECE-style summary

Calibration:
- Platt scaling (“sigmoid”) by default, fit on the calibration split.

## Limitations
- Outcomes are delayed and sometimes incomplete; label maturity rules are required to avoid biased negatives.
- Name-based joins are best-effort and may be unreliable; ID joins are the contract.
- The v1 feature set is intentionally simple and avoids using lead form fields (often PII).
- Performance on synthetic demo data is not representative of real-world accounts.

## Ethical / Privacy Considerations
- Treat `leads.csv` as sensitive input. Do not upload or commit PII.
- The OSS demo is designed to run using **synthetic data only**.

