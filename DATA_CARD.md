# Data Card: Meta Lead Ads ELV Kit (CSV Contracts)

## What Data This Project Uses

This project is CSV-first. It expects three inputs:

1. `ads.csv` (daily aggregates)
2. `leads.csv` (lead-level)
3. `outcomes.csv` (lead_id -> qualified_time)

It also optionally accepts:
- `lead_to_ad_map.csv` (when leads export does not include IDs)

## Required Fields (v1 Canonical Schema)

`ads.csv`:
- `date` (YYYY-MM-DD)
- `campaign_id`, `campaign_name`
- `adset_id`, `adset_name`
- `ad_id`, `ad_name`
- `impressions` (number)
- `clicks` (number)
- `spend` (number)

`leads.csv`:
- `lead_id`
- `created_time` (timestamp)
- join keys: `ad_id` preferred, else `adset_id` or `campaign_id`

`outcomes.csv`:
- `lead_id`
- `qualified_time` (timestamp; null if never qualified)

`lead_to_ad_map.csv` (optional):
- `lead_id`
- one of: `ad_id` / `adset_id` / `campaign_id`

## What Gets Dropped (Privacy Default)

By default, connector loaders drop non-canonical columns.
This is intentional to avoid accidentally propagating lead form fields that can contain PII (name, email, phone, etc.).

If you want to add non-PII features, add them explicitly to the canonical schema and update validation + reporting accordingly.

## Known Data Quality Failure Modes
- Duplicated or changing names (campaign/adset/ad) make name-based joins brittle.
- Missing IDs in `leads.csv` requires a mapping file; otherwise joins may be unreliable.
- Daily aggregates can leak same-day signal; v1 uses `feature_lag_days=1` by default.
- Outcomes are delayed; leads inside the label window must be treated as `unknown` for training/eval.

## Synthetic Demo Data

This repo includes a synthetic data generator to allow public demos without real account data.
Synthetic data is not representative of production distributions.

