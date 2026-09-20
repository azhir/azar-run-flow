from __future__ import annotations

import os
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import mlflow
import numpy as np

from .config import (
    Config,
    config_yaml,
    flatten_config,
    hydra_overrides,
    resolve_config,
)
from .provenance import (
    collect_git_diff,
    collect_git_status,
    collect_packages,
    collect_provenance,
)
from .seed import set_seed


class Run:
    def __init__(self, run_id: str) -> None:
        self.id = run_id

    def metric(
        self,
        name: str,
        value: float,
        *,
        step: int | None = None,
    ) -> None:
        mlflow.log_metric(
            name,
            float(value),
            step=step,
        )

    def metrics(
        self,
        values: dict[str, float],
        *,
        step: int | None = None,
    ) -> None:
        mlflow.log_metrics(
            {
                key: float(value)
                for key, value in values.items()
            },
            step=step,
        )

    def tag(
        self,
        name: str,
        value: Any,
    ) -> None:
        mlflow.set_tag(name, value)

    def figure(
        self,
        name: str,
        figure: Any,
    ) -> None:
        mlflow.log_figure(
            figure,
            f"figures/{name}",
        )

    def array(
        self,
        name: str,
        array: np.ndarray,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name

            if path.suffix != ".npy":
                path = path.with_suffix(".npy")

            np.save(path, array)

            mlflow.log_artifact(
                str(path),
                artifact_path="arrays",
            )

    def artifact(
        self,
        path: str | Path,
        *,
        folder: str | None = None,
    ) -> None:
        mlflow.log_artifact(
            str(path),
            artifact_path=folder,
        )

    def text(
        self,
        name: str,
        text: str,
        *,
        folder: str = "text",
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name

            path.write_text(
                text,
                encoding="utf-8",
            )

            mlflow.log_artifact(
                str(path),
                artifact_path=folder,
            )

    def json(
        self,
        name: str,
        value: dict,
    ) -> None:
        mlflow.log_dict(
            value,
            f"json/{name}",
        )

    def checkpoint(
        self,
        path: str | Path,
    ) -> None:
        mlflow.log_artifact(
            str(path),
            artifact_path="checkpoints",
        )


def _log_config(
    config: Config,
) -> dict[str, Any]:
    resolved = resolve_config(config)

    # Complete authoritative config.
    mlflow.log_text(
        config_yaml(config),
        "config/resolved.yaml",
    )

    mlflow.log_dict(
        resolved,
        "config/resolved.json",
    )

    # Exact CLI deltas applied by Hydra.
    overrides = hydra_overrides()

    if overrides:
        mlflow.log_text(
            "\n".join(overrides),
            "config/overrides.txt",
        )

    # Searchable fields in MLflow.
    params = flatten_config(resolved)

    if params:
        mlflow.log_params(params)

    return resolved


def _log_provenance(
    provenance: dict[str, Any],
) -> None:
    mlflow.log_dict(
        provenance,
        "provenance/provenance.json",
    )

    mlflow.log_text(
        collect_packages(),
        "provenance/packages.txt",
    )

    git_status = collect_git_status()

    if git_status:
        mlflow.log_text(
            git_status,
            "provenance/git_status.txt",
        )

    if provenance.get("git_dirty"):
        git_diff = collect_git_diff()

        if git_diff:
            mlflow.log_text(
                git_diff,
                "provenance/git_diff.patch",
            )


@contextmanager
def azar_run(
    *,
    config: Config,
    experiment_name: str = "default",
    run_name: str | None = None,
    seed: int | None = None,
    dataset: dict | None = None,
    log_system_metrics: bool = True,
) -> Iterator[Run]:
    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "sqlite:///mlflow.db",
        )
    )

    resolved = resolve_config(config)

    if seed is None:
        config_seed = resolved.get("seed")

        if isinstance(config_seed, int):
            seed = config_seed

    set_seed(seed)

    mlflow.set_experiment(
        experiment_name,
    )

    provenance = collect_provenance()

    provenance.update(
        {
            "seed": seed,
            "dataset": dataset,
        }
    )

    with mlflow.start_run(
        run_name=run_name,
        log_system_metrics=log_system_metrics,
    ) as active_run:
        run = Run(
            active_run.info.run_id,
        )

        _log_config(config)
        _log_provenance(provenance)

        if provenance["git_commit"]:
            run.tag(
                "git_commit",
                provenance["git_commit"],
            )

        if provenance["git_branch"]:
            run.tag(
                "git_branch",
                provenance["git_branch"],
            )

        run.tag(
            "git_dirty",
            provenance["git_dirty"],
        )

        start = time.perf_counter()

        try:
            yield run

        except Exception:
            runtime = (
                time.perf_counter()
                - start
            )

            run.metric(
                "runtime_seconds",
                runtime,
            )

            mlflow.log_text(
                traceback.format_exc(),
                "failure/traceback.txt",
            )

            run.tag(
                "azar.status",
                "failed",
            )

            raise

        else:
            runtime = (
                time.perf_counter()
                - start
            )

            run.metric(
                "runtime_seconds",
                runtime,
            )

            run.tag(
                "azar.status",
                "completed",
            )

            print(
                f"MLflow run: {run.id}"
            )


def run_experiment(
    experiment: Callable[
        [Run, dict[str, Any]],
        Any,
    ],
    *,
    config: Config,
    experiment_name: str = "default",
    run_name: str | None = None,
    seed: int | None = None,
    dataset: dict | None = None,
    log_system_metrics: bool = True,
) -> Any:
    resolved = resolve_config(config)

    with azar_run(
        config=config,
        experiment_name=experiment_name,
        run_name=run_name,
        seed=seed,
        dataset=dataset,
        log_system_metrics=log_system_metrics,
    ) as run:
        return experiment(
            run,
            resolved,
        )