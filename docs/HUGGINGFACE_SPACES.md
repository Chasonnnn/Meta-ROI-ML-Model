# Hugging Face Spaces Deployment (Streamlit)

This repo includes a Streamlit app at `app.py` intended for a score-first public demo.

## Recommended Setup

1. Create a new Space using the Streamlit SDK.
2. Push this repo to the Space.
3. Ensure `requirements.txt` is present (this repo uses:
   - `-e .[ui,genai]`
   - which installs `streamlit` (and the optional Gemini advisor dependency) via extras).

The Space should run `app.py` automatically.

Note: creative embeddings (OpenCLIP/PyTorch/OpenCV) are intentionally not installed in `requirements.txt` because they are heavy. The Streamlit UI will show that feature as unavailable unless you customize the Space dependencies.

## Demo Behavior

Default mode is demo data, score-first:
- synthetic CSVs are generated inside the run directory
- scoring uses the bundled synthetic demo model: `src/meta_elv/assets/demo_model.joblib`

Training is optional and is intended to be done locally.

## Optional GenAI Advisor (Gemini)

The Streamlit app includes an optional “GenAI Advisor” panel that can call Gemini for qualitative suggestions
(targeting + creative improvements).

How to enable:
- If the Space is **private**, you can add a Space Secret named `GEMINI_API_KEY`.
- If the Space is **public**, do **not** put a paid/shared API key in Secrets unless you also add authentication/rate limits.
  Prefer requiring each user to paste their own key in the UI for that session.
- Optional: set `GEMINI_MODEL` (defaults to `gemini-2.0-flash`).

Notes:
- The app sends only aggregate run summaries (leaderboards/metrics) plus any media you upload.
- Do not upload sensitive client exports or PII to public Spaces.

## Safety

Do not upload real client exports or PII to public Spaces.
