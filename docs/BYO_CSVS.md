# Bring Your Own CSVs

The core contract is 3 CSVs:
- `ads.csv` (daily performance by ad/adset/campaign)
- `leads.csv` (lead events, plus join keys)
- `outcomes.csv` (optional for scoring, required for training)

If `leads.csv` does not include IDs, provide a mapping file:
- `lead_to_ad_map.csv`

## Canonical Schemas (v1)

### ads.csv (daily)
Required columns:
- `date` (YYYY-MM-DD)
- `campaign_id`, `campaign_name`
- `adset_id`, `adset_name`
- `ad_id`, `ad_name`
- `impressions` (int)
- `clicks` (int)
- `spend` (float)

### leads.csv
Required columns:
- `lead_id`
- `created_time` (timestamp; ISO-8601 recommended)

Join keys (at least one):
- preferred: `ad_id`
- or: `adset_id`
- or: `campaign_id`

### outcomes.csv (required for training)
Required columns:
- `lead_id`
- `qualified_time` (timestamp; null/empty if never qualified)

## Optional Enrichment CSVs

These inputs are optional, but can improve model quality and enable richer suggestions.

If you train a model using any enrichment file, you must provide the same file(s) when scoring with that model.

### ads_placement.csv (daily)
Required columns:
- `date` (YYYY-MM-DD)
- `ad_id`
- `placement`
- `impressions` (int)
- `clicks` (int)
- `spend` (float)

### ads_geo.csv (daily)
Required columns:
- `date` (YYYY-MM-DD)
- `ad_id`
- `geo` (e.g. country/region/state code)
- `impressions` (int)
- `clicks` (int)
- `spend` (float)

### adset_targeting.csv (static)
Required columns:
- `adset_id`
- `audience_keywords` (free text or delimited list)

### ad_creatives.csv (static)
Required columns:
- `ad_id`
- `creative_type` (e.g. image/video)

## Join Rules

Join strategy order:
1. ID join (`ad_id` / `adset_id` / `campaign_id`)
2. Mapping file (`lead_to_ad_map.csv`)
3. Name fallback (best-effort, with loud warnings)

The pipeline writes join match-rate statistics to:
- `runs/<run_id>/metadata.json`
- `runs/<run_id>/data_profile.json` and `data_profile.md`

## Training vs Scoring

- Training requires `outcomes.csv`.
- Scoring does not; if `outcomes.csv` is missing, the pipeline keeps all labels `unknown` (no fabricated negatives).

## Example Config

Start from `configs/run.yaml` and edit file paths:

```yaml
schema_version: 1

paths:
  ads_path: data/byo/ads.csv
  leads_path: data/byo/leads.csv
  outcomes_path: data/byo/outcomes.csv
  lead_to_ad_map_path: null
  ads_placement_path: null
  ads_geo_path: null
  adset_targeting_path: null
  ad_creatives_path: null
```
