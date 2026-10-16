"""Reproducibility manifests for DeepSC publication runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import torch


def sha256_file(path: str | Path) -> str:
    file_path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Required artifact does not exist: {file_path}")
    return {
        "path": str(file_path),
        "sha256": sha256_file(file_path),
        "size_bytes": file_path.stat().st_size,
    }


def git_identity(repo_dir: str | Path = ".") -> dict[str, Optional[str]]:
    def run(*args: str) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo_dir, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def runtime_identity() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def create_run_directory(base_dir: str | Path, experiment_id: str) -> Path:
    identifier = Path(experiment_id)
    if (
        not experiment_id
        or identifier.is_absolute()
        or len(identifier.parts) != 1
        or any(part in {"", ".", ".."} for part in identifier.parts)
    ):
        raise ValueError("experiment_id must be a non-empty safe path component")
    run_dir = Path(base_dir).expanduser().resolve() / experiment_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {output}")
    with output.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return output


def base_training_manifest(
    *, experiment_id: str, model_type: str, cli_args: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "model_type": model_type,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_identity(),
        "cli_arguments": dict(cli_args),
        "runtime": runtime_identity(),
        "selected_checkpoint": None,
    }


def update_selected_checkpoint(
    manifest_path: str | Path, checkpoint_path: str | Path, epoch: int, metric: str, value: float
) -> None:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selected_checkpoint"] = {
        **file_identity(checkpoint_path),
        "epoch": epoch,
        "selection_metric": metric,
        "selection_value": float(value),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
