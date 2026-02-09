# Meta Lead Ads ELV Kit

CSV-first, self-serve pipeline to predict **P(qualified within 14 days)** and convert it to **Expected Lead Value (ELV)**:

`ELV = P(qualified_within_14d) * value_per_qualified`

**Status:** actively being built. See `workplan.md` for the week-by-week build plan.

## Quickstart (Demo)

```bash
uv venv
uv sync --extra dev --extra ui

. .venv/bin/activate
elv demo
```

Artifacts are written to `runs/<run_id>/` (predictions, leaderboards, report, metrics, profiles).

`elv demo` is **score-first** by default using a bundled synthetic demo model (`src/meta_elv/assets/demo_model.joblib`).
To rebuild the bundled demo model:

```bash
uv run -- python scripts/build_demo_model.py
```

Alternative install (no `uv`):

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev,ui,lgbm]"
```

## LightGBM (Optional)
LightGBM is supported (`model.model_type: lgbm`), but on some macOS setups you may need OpenMP:
- `brew install libomp`

## Bring Your Own CSVs

1. Copy `configs/run.yaml` and update file paths.
2. Validate:

```bash
elv validate --config configs/run.yaml
```

3. Build training table:

```bash
elv build-table --config configs/run.yaml
```

4. Train + report:

```bash
elv train --config configs/run.yaml
```

5. Score:

```bash
elv score --config configs/run.yaml --model runs/<run_id>/model.joblib
```

### Batch Scoring (New Leads)
For daily scoring of new leads (typically no outcomes yet):

```bash
elv batch-score --config configs/run.yaml --model runs/<run_id>/model.joblib --leads-path data/new_leads.csv
```

This forces `outcomes_path=null` for the run so the pipeline does not fabricate labels.

### Inference-Only Scoring (No outcomes.csv)
For batch scoring of new leads, set `paths.outcomes_path: null` in your config and run `elv score`.
The pipeline will keep all rows `label_status=unknown` (it will not fabricate negatives).

## What Makes This "Ads ML Realistic"
- **Label maturity**: Leads newer than `label_window_days` are labeled `unknown` and excluded from supervised train/eval.
- **Daily ads.csv leakage guard**: With daily aggregates, same-day metrics can include post-lead activity. Default `feature_lag_days=1` uses metrics only through `lead_date - 1 day`.
- **Join robustness**: ID join is the contract. Name-based joins are best-effort and emit loud warnings + match-rate stats.

## Data Contracts (v1)
Required files:
- `ads.csv` (daily): `date`, `campaign_id`, `campaign_name`, `adset_id`, `adset_name`, `ad_id`, `ad_name`, `impressions`, `clicks`, `spend`
- `leads.csv`: `lead_id`, `created_time` plus join keys (`ad_id` preferred; else `adset_id`/`campaign_id`)
- `outcomes.csv` (optional for scoring, required for training): `lead_id`, `qualified_time` (timestamp; null if never qualified)

If `leads.csv` has no IDs, provide a mapping file:
- `lead_to_ad_map.csv`: `lead_id`, plus one of `ad_id` / `adset_id` / `campaign_id`

## Optional Enrichment CSVs
These inputs are optional, but they let the model use more “real ads context” than spend/clicks alone.

If you **train** a model using any of these, you must also provide them at **score** time (or scoring will fail with a missing-features error).

- `ads_placement.csv` (daily breakdown): `date`, `ad_id`, `placement`, `impressions`, `clicks`, `spend`
- `ads_geo.csv` (daily breakdown): `date`, `ad_id`, `geo`, `impressions`, `clicks`, `spend`
- `adset_targeting.csv` (static): `adset_id`, `audience_keywords` (free-text or delimited list)
- `ad_creatives.csv` (static): `ad_id`, `creative_type` (e.g. image/video)

Configure these under `paths.*` in `configs/run.yaml` (defaults to `null`).

Notes:
- Features remain leakage-safe (respect `feature_lag_days` for daily aggregates).
- High-cardinality inputs are summarized/hashed into stable numeric features.
- placement/geo features: entropy, top-1 share, active-count, rolling totals
- targeting keyword features: deterministic hashed multi-hot vector + counts

## GenAI Advisor (Optional)
For more qualitative suggestions (audiences + creative improvements), the Streamlit app can optionally call Gemini to analyze:
- your leaderboards and run metadata (aggregate only, no lead rows)
- optional creative media you upload (image/video)

This is **off by default** and requires your own API key.

## Run Artifacts
Each run writes to `runs/<run_id>/`:
- `config.yaml`, `metadata.json`
- `data_profile.json`, `data_profile.md`
- `metrics.json`, `report.html`
- `model.joblib`
- `predictions.csv`
- `leaderboard_campaign.csv`, `leaderboard_adset.csv` (when keys exist)
- `drift.json` (score-only runs, when the model bundle contains a drift reference)

## Drift (PSI)
- Train runs compute PSI between train vs test for a small set of numeric features and include it in `metrics.json` and `report.html`.
- Score-only runs compute PSI of scoring data vs the training reference stored in the model bundle and write `drift.json` (also shown in `report.html`).

## Leaderboard Guardrails
Leaderboards include:
- `lead_count` and `low_volume` (defaults to `reporting.min_segment_leads: 30`)
Interpret low-volume segments cautiously.

## UI (Streamlit / Hugging Face Spaces)
Local:

```bash
streamlit run app.py
```

The public demo is intended to be **score-first** with synthetic demo data and a bundled model. Training is optional and should be run locally.

### GenAI Advisor (Gemini, optional)
The Streamlit app includes an optional “GenAI Advisor” panel.

Enable locally:
```bash
uv sync --extra ui --extra genai
export GEMINI_API_KEY="..."
streamlit run app.py
```

Enable on Hugging Face Spaces:
- if the Space is private, add a Space Secret `GEMINI_API_KEY`
- if the Space is public, prefer requiring each user to paste their own key in the UI (do not ship a shared paid key)
- optional: set `GEMINI_MODEL` (defaults to `gemini-2.0-flash`)

Privacy: any media you upload in the advisor panel may be sent to a third-party API. Do not use with sensitive client data on public deployments.

## Creative Embeddings (Similarity + Clustering)
Optional workflow to compute image/video embeddings for:
- similar-creative search
- creative clustering

This is **not installed by default** (it adds PyTorch + OpenCLIP + OpenCV). Install explicitly:

```bash
uv sync --extra embeddings
```

Note: the Hugging Face Space `requirements.txt` intentionally does **not** include this extra (it's heavy). Run locally or customize your Space if you want creative embeddings there.

Run:

```bash
elv creative-analyze --creative-map creative_media.csv --media-dir /path/to/media --run-dir runs/<run_id>
```

See `docs/CREATIVE_EMBEDDINGS.md` for input schema + outputs.

## Safety / Privacy
Do not commit or upload real client exports to the repo. This project is designed to be demoable using synthetic data only.

## Docs
- `docs/QUICKSTART.md`
- `docs/BYO_CSVS.md`
- `docs/BATCH_SCORING.md`
- `docs/CALIBRATION_AND_ELV.md`
- `docs/CONNECTORS.md`
- `docs/HUGGINGFACE_SPACES.md`
- `docs/CREATIVE_EMBEDDINGS.md`
