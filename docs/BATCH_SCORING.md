# Batch Scoring

Batch scoring is intended for "new leads daily" use cases.

## Why It Is Separate

When leads are recent, downstream outcomes are not mature yet. The batch flow is inference-only:
- forces `outcomes_path=null`
- keeps all `label_status=unknown`

## Command

```bash
elv batch-score --config configs/run.yaml --model runs/<run_id>/model.joblib --leads-path data/new_leads.csv
```

Optionally write to a stable output directory:

```bash
elv batch-score --config configs/run.yaml --model runs/<run_id>/model.joblib --leads-path data/new_leads.csv --out-dir runs/batch_latest --overwrite
```

## Outputs

Written under the run directory:
- `predictions.csv`
- `leaderboard_campaign.csv`, `leaderboard_adset.csv` (if keys exist)
- `drift.json` (when the model bundle includes a drift reference)
- `report.html`

