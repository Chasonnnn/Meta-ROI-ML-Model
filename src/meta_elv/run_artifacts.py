from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import ensure_dir, try_get_git_sha_short, utc_now_compact, write_json


@dataclass(frozen=True)
class RunContext:
    repo_root: Path
    run_dir: Path
    run_id: str


def create_run_context(repo_root: Path, run_base_dir: Path | None = None) -> RunContext:
    repo_root = repo_root.resolve()
    if run_base_dir is None:
        run_base_dir = repo_root / "runs"

    sha = try_get_git_sha_short(repo_root)
    run_id = f"{utc_now_compact()}_{sha}"
    run_dir = ensure_dir(Path(run_base_dir) / run_id)
    return RunContext(repo_root=repo_root, run_dir=run_dir, run_id=run_id)


def snapshot_config(run_dir: Path, config_path: Path) -> None:
    dst = run_dir / "config.yaml"
    shutil.copyfile(config_path, dst)


def write_metadata(run_dir: Path, metadata: dict[str, Any]) -> None:
    write_json(run_dir / "metadata.json", metadata)

