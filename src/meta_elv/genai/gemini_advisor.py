from __future__ import annotations

import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd


class GeminiDependencyError(RuntimeError):
    pass


class GeminiConfigError(ValueError):
    pass


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text())
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _read_csv_head_optional(path: Path, n: int) -> list[dict[str, Any]] | None:
    try:
        if not path.exists():
            return None
        df = pd.read_csv(path).head(n)
        return df.to_dict(orient="records")
    except Exception:
        return None


def _build_run_brief(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    meta = _read_json_optional(run_dir / "metadata.json") or {}
    metrics = _read_json_optional(run_dir / "metrics.json") or {}
    profile = _read_json_optional(run_dir / "data_profile.json") or {}
    drift = _read_json_optional(run_dir / "drift.json") or {}

    brief: dict[str, Any] = {
        "join": meta.get("join") or (profile.get("join") or {}),
        "labeling": meta.get("labeling") or (profile.get("labeling") or {}),
        "features": (meta.get("features") or {}) or (((profile.get("details") or {}).get("features")) or {}),
        "warnings": list(meta.get("warnings") or (profile.get("warnings") or [])),
        "metrics": {
            "model_type": metrics.get("model_type"),
            "calibration_method": metrics.get("calibration_method"),
            "model": metrics.get("model") or {},
            "baseline_campaign_rate": metrics.get("baseline_campaign_rate") or {},
            "drift": metrics.get("drift") or drift or {},
        },
        "top_campaigns": _read_csv_head_optional(run_dir / "leaderboard_campaign.csv", 10),
        "top_adsets": _read_csv_head_optional(run_dir / "leaderboard_adset.csv", 10),
    }
    return brief


def _guess_mime(p: Path) -> str | None:
    mime, _ = mimetypes.guess_type(str(p))
    return mime


def _state_name(obj: Any) -> str | None:
    try:
        st = getattr(obj, "state", None)
        if st is None:
            return None
        if isinstance(st, str):
            return st
        name = getattr(st, "name", None)
        if isinstance(name, str):
            return name
        return str(st)
    except Exception:
        return None


def generate_gemini_suggestions(
    *,
    run_dir: Path,
    offer_context: str,
    audience_keywords: str | None = None,
    media_path: Path | None = None,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1024,
    file_upload_timeout_s: int = 120,
) -> str:
    """
    Generate qualitative suggestions using Gemini based on a run directory's aggregate artifacts.

    Privacy guardrail:
    - Only aggregate run artifacts are sent (leaderboards, counts, warnings, metrics).
    - Do not include lead rows or PII-like fields in the prompt.

    If media_path is provided:
    - images are sent as inline bytes
    - videos are uploaded and referenced as a file (may take time)
    """
    if not offer_context or not offer_context.strip():
        raise GeminiConfigError("offer_context is required (describe the offer and target customer).")

    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except Exception as e:  # pragma: no cover
        raise GeminiDependencyError(
            "Gemini advisor requires the `google-genai` package. Install with: uv sync --extra genai"
        ) from e

    # Prefer explicit api_key; otherwise rely on env var behavior.
    # `google-genai` supports GEMINI_API_KEY or GOOGLE_API_KEY env vars.
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise GeminiConfigError(
            "Missing GEMINI_API_KEY. Set it in your environment (or Streamlit secrets) to enable Gemini suggestions."
        )

    model = (model or os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash").strip()

    brief = _build_run_brief(Path(run_dir))
    # Keep prompt compact and structured.
    brief_json = json.dumps(brief, indent=2, sort_keys=True)

    prompt = "\n".join(
        [
            "You are an ads performance advisor for Meta Lead Ads. Your job is to suggest improvements to targeting and creative.",
            "",
            "Constraints:",
            "- Do NOT request or rely on lead-level data or PII. Use only the provided aggregate run summary.",
            "- Be explicit about uncertainty and data limitations.",
            "- Provide concrete, testable actions (A/B tests, budget shifts, placement/geo changes, creative edits).",
            "",
            "Output format (Markdown):",
            "1. Quick diagnosis (3-6 bullets)",
            "2. Targeting / audience suggestions (5-10 bullets)",
            "3. Creative suggestions (5-10 bullets)",
            "4. Next 7-day experiment plan (3-6 bullets)",
            "",
            "Offer context:",
            offer_context.strip(),
            "",
            "Audience keywords / targeting notes (optional):",
            (audience_keywords or "").strip() or "(none provided)",
            "",
            "Aggregate run summary (JSON):",
            "```json",
            brief_json,
            "```",
        ]
    )

    client = genai.Client(api_key=api_key)

    # Media handling
    contents: Any
    if media_path is None:
        contents = prompt
    else:
        media_path = Path(media_path)
        if not media_path.exists():
            raise GeminiConfigError(f"media_path does not exist: {media_path}")

        mime = _guess_mime(media_path) or "application/octet-stream"
        size_mb = media_path.stat().st_size / (1024 * 1024)
        if size_mb > 50:
            raise GeminiConfigError(
                f"Media file is too large ({size_mb:.1f} MB). Please upload a smaller file (<= 50 MB)."
            )

        if mime.startswith("image/"):
            data = media_path.read_bytes()
            part = types.Part.from_bytes(data=data, mime_type=mime)
            # Match the official examples: media part first, then the text prompt.
            contents = [part, prompt]
        elif mime.startswith("video/"):
            uploaded = client.files.upload(file=media_path)
            start = time.time()
            while _state_name(uploaded) == "PROCESSING":
                if (time.time() - start) > float(file_upload_timeout_s):
                    raise GeminiConfigError(
                        f"Video upload is still PROCESSING after {file_upload_timeout_s}s. Try a shorter/smaller clip."
                    )
                time.sleep(2)
                uploaded = client.files.get(name=uploaded.name)
            if _state_name(uploaded) != "ACTIVE":
                raise GeminiConfigError(f"Video upload failed (state={_state_name(uploaded)}).")
            contents = [uploaded, prompt]
        else:
            # Treat any other file type as a generic upload.
            uploaded = client.files.upload(file=media_path)
            start = time.time()
            while _state_name(uploaded) == "PROCESSING":
                if (time.time() - start) > float(file_upload_timeout_s):
                    raise GeminiConfigError(
                        f"File upload is still PROCESSING after {file_upload_timeout_s}s. Try a smaller file."
                    )
                time.sleep(2)
                uploaded = client.files.get(name=uploaded.name)
            if _state_name(uploaded) != "ACTIVE":
                raise GeminiConfigError(f"File upload failed (state={_state_name(uploaded)}).")
            contents = [uploaded, prompt]

    cfg = types.GenerateContentConfig(
        temperature=float(temperature),
        max_output_tokens=int(max_output_tokens),
    )
    resp = client.models.generate_content(model=model, contents=contents, config=cfg)

    out = getattr(resp, "text", None)
    if not out or not str(out).strip():
        raise RuntimeError("Gemini returned an empty response.")
    return str(out).strip() + "\n"
