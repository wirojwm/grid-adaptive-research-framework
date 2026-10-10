"""Reproducibility metadata capture helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def capture_git_commit(repo_dir: str | Path = ".") -> tuple[str, bool]:
    """Return (commit_hash, is_dirty) for the repo at repo_dir.

    Raises RuntimeError if repo_dir is not inside a git repository, or has no
    commits yet — an experiment record cannot cite a commit that doesn't exist.
    """
    repo_dir = str(repo_dir)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(f"could not capture git commit at {repo_dir!r}: {exc}") from exc

    return commit, bool(status.strip())
