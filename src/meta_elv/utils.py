from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_compact() -> str:
    # YYYYMMDD_HHMMSS in UTC for stable, sortable run ids
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def try_get_git_sha_short(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "nogit"
    except Exception:
        return "nogit"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def as_date_str(dt: datetime) -> str:
    return dt.date().isoformat()

