from __future__ import annotations

import platform
import subprocess
import sys


def _command(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            args,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def collect_provenance() -> dict:
    git_commit = _command("git", "rev-parse", "HEAD")
    git_status = _command("git", "status", "--porcelain")

    return {
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "python": sys.version,
        "platform": platform.platform(),
        "command": sys.argv,
    }