from __future__ import annotations

import platform
import socket
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
    git_commit = _command(
        "git",
        "rev-parse",
        "HEAD",
    )

    git_branch = _command(
        "git",
        "rev-parse",
        "--abbrev-ref",
        "HEAD",
    )

    git_status = _command(
        "git",
        "status",
        "--porcelain",
    )

    return {
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": bool(git_status),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "command": sys.argv,
    }