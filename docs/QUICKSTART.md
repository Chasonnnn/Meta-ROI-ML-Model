# Quickstart

This repo is designed to be demoable without any real client data.

## Demo (Score-First)

```bash
uv venv
uv sync --extra dev --extra ui
. .venv/bin/activate

elv demo
```

Outputs are written under `runs/<run_id>/`:
- `predictions.csv` (lead-level scores + ELV)
- `leaderboard_campaign.csv`, `leaderboard_adset.csv`
- `report.html`

## Train Locally (Synthetic Demo Data)

```bash
elv demo --train
```

## UI (Streamlit)

```bash
streamlit run app.py
```

