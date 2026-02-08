.PHONY: sync test lint demo demo-model ui

sync:
\tuv sync --extra dev --extra ui --extra lgbm

test:
\tuv run -- pytest -q

lint:
\tuv run -- ruff check .

demo:
\tuv run -- elv demo

demo-model:
\tuv run -- python scripts/build_demo_model.py

ui:
\tuv run -- streamlit run app.py
