# Connectors

Connectors are responsible for loading user data and normalizing it into the canonical v1 schemas.

## Where They Live

- Base interface/constants: `src/meta_elv/connectors/base.py`
- Meta CSV connector (v1): `src/meta_elv/connectors/meta_csv/loader.py`

## Contract

A connector load function returns:
- a pandas DataFrame with canonical columns
- a list of warnings (string messages) describing any normalization actions

Example signature (see `LoadResult`):
- `load_ads(path) -> LoadResult`
- `load_leads(path) -> LoadResult`
- `load_outcomes(path) -> LoadResult`

## Privacy Default

Connectors should drop non-canonical columns by default.

Reason: `leads.csv` can contain PII (names, emails, phone numbers). By keeping only canonical columns, the rest of the pipeline and artifacts stay safer by default.

## Adding A New Connector

1. Create a package under `src/meta_elv/connectors/<your_connector>/`
2. Implement the 3 loaders + optional mapping loader.
3. Normalize:
   - column names (case-insensitive matching is recommended)
   - timestamps (parse to UTC where possible)
   - numeric columns (coerce errors to NaN so validation can report)
4. Ensure `validate` and `table_builder` can consume the canonical outputs.

