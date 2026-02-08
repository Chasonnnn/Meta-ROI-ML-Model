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

