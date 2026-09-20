from __future__ import annotations

import platform
import socket
import subprocess
import sys
from importlib.metadata import distributions


def _command(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            args,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def collect_packages() -> str:
    packages: list[str] = []

    for dist in distributions():
        name = dist.metadata.get("Name")

        if name:
            packages.append(
                f"{name}=={dist.version}"
            )

    packages.sort(key=str.lower)

    return "\n".join(packages)


def collect_git_status() -> str | None:
    return _command(
        "git",
        "status",
        "--porcelain",
    )


def collect_git_diff() -> str | None:
    return _command(
        "git",
        "diff",
        "HEAD",
        "--binary",
    )


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

    git_status = collect_git_status()

    return {
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": bool(git_status),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "command": sys.argv,
    }