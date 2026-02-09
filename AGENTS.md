# AGENTS.md — OSS Ticketing System

> Single source of truth for building this project. Every contributor (human or AI) follows these rules.

## 0) Documentation First (Non-Negotiable)

Before implementing or changing behavior that depends on external systems/frameworks (Typer, Streamlit, LightGBM, scikit-learn calibration APIs, pandas CSV parsing quirks, Hugging Face Spaces, Docker, etc.), read the official documentation or upstream release notes first.
- Prefer official domains/repos as sources of truth.
- If docs are missing/unclear, inspect upstream source or create a minimal reproduction; do not guess.

## 1) Production-Quality Standard (Non-Negotiable)

Build FULLY FUNCTIONAL, POLISHED features, not “toy scripts”.

✅ Required:
- Friendly error handling (actionable messages, no raw stack traces as UX)
- Validation & edge cases covered (schemas, joins, date parsing, empty inputs)
- Reproducible runs (artifacts and metadata under `runs/`)
- UI loading/error states (Streamlit)

❌ Forbidden:
- “Basic” or “minimal” implementations that skip validation or quality checks
- Silent failure modes (e.g., joins that drop most rows without reporting match rate)
- “TODO: add X later” comments in production code
- Placeholder UX copy instead of real behavior
- Downgrading dependencies without explicit user approval

## 1.1) No Backward Compatibility

This project is still under active development. Breaking changes are acceptable.
- Prioritize clean design over compatibility (config format, CLI, run artifacts can change).
- It’s fine to regenerate demo data/models and rewrite run artifacts as needed.

## 2) Git Rules

### Commit Prefix Rule
All commits must start with: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, or `chore:`.

### Commit Message Format
```text
feat: Add maturity-safe label builder
fix: Prevent same-day daily-ads leakage in rolling features
docs: Update README with BYO join strategy rules
refactor: Centralize run metadata writing
test: Add coverage for label maturity unknowns
chore: Add CI for pytest and ruff
```

## 3) TDD Rule

Write or update tests FIRST.
- Start with a failing test capturing the behavior/bug.
- Implement until it passes.
- If behavior changes, update tests in the same PR.

## 4) Security / Privacy Boundaries (Zero Tolerance)

- Never commit secrets (tokens, keys). Use `.env` locally if needed and keep `.env.example` updated.
- Never commit real exports (Meta ads/leads/outcomes) to the repo. Demo data must be synthetic.
- Never log raw PII. Treat uploaded `leads.csv` as potentially sensitive:
  - Do not print raw rows.
  - In logs/reports, show only aggregate counts and rates.
  - If IDs are logged, prefer hashed/truncated IDs.
- Never persist uploaded BYO data from the Streamlit app except explicitly requested by the user.

## 5) Centralized Core Logic (Zero Tolerance)

All business logic must live in reusable Python modules under `src/meta_elv/`.
- CLI should be a thin wrapper around core functions.
- Streamlit UI should call the same core functions as the CLI.
- Connectors must normalize to canonical schema in `src/meta_elv/connectors/`.

## 5.1) Leakage-Safe Feature Rules (Non-Negotiable)

- Any features derived from daily aggregates (ads performance, placement breakdowns, geo breakdowns) must respect `features.feature_lag_days` so same-day post-lead activity never leaks into training or scoring.
- If a model is trained with optional enrichment inputs (placement/geo/targeting/creative), scoring runs must either:
  - provide the same inputs, or
  - fail with a friendly error that explains what is missing and how to fix it.

## 5.2) GenAI Integrations (Opt-In Only)

- Any GenAI feature (Gemini, Hugging Face Inference, etc.) must be **disabled by default** and enabled only when a user provides an API key via environment variables or Streamlit secrets.
- Never send lead-level rows (or any PII-like fields) to GenAI providers. Only send aggregate summaries (leaderboards, counts, rates) plus user-authored context.
- Never write API keys to disk, to `runs/`, or to logs.
- In the Streamlit UI, clearly warn users that uploaded media may be sent to a third-party API.

## 5.3) Creative Media (Embeddings) Handling

- Treat uploaded ad creatives (images/videos) as potentially sensitive. Do not commit them and do not persist them to `runs/` by default.
- Streamlit should store uploaded media in a temporary directory for analysis and delete it after the session step completes (only derived artifacts may be saved).
- The creative-embeddings workflow must be optional and gated behind the `embeddings` extra. Core CLI/UI flows must still run without PyTorch/OpenCLIP/OpenCV installed.
- Any optional-dependency import must fail with a friendly error explaining how to install (`uv sync --extra embeddings`).

## 6) Tech Stack

- Language: Python
- Packaging: `pyproject.toml`
- Data: pandas, numpy
- Modeling: scikit-learn + LightGBM
- CLI: Typer
- UI: Streamlit (score-first Hugging Face Space)
- Reporting: matplotlib + HTML report generation
- Quality: pytest, ruff
- Reproducibility: run artifacts under `runs/<run_id>/` with metadata
