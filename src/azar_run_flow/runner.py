from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import mlflow
import numpy as np

from .provenance import collect_provenance


class Run:
    def metric(
        self,
        name: str,
        value: float,
        *,
        step: int | None = None,
    ) -> None:
        mlflow.log_metric(name, float(value), step=step)

    def metrics(
        self,
        values: dict[str, float],
        *,
        step: int | None = None,
    ) -> None:
        mlflow.log_metrics(
            {k: float(v) for k, v in values.items()},
            step=step,
        )

    def figure(self, name: str, figure: Any) -> None:
        mlflow.log_figure(
            figure,
            f"figures/{name}",
        )

    def array(self, name: str, array: np.ndarray) -> None:
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

    def text(self, name: str, text: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_text(text)

            mlflow.log_artifact(
                str(path),
                artifact_path="text",
            )


def run_experiment(
    experiment: Callable[[Run, dict], Any],
    *,
    config: dict,
    experiment_name: str = "default",
    run_name: str | None = None,
    seed: int | None = None,
    dataset: dict | None = None,
) -> Any:
    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "sqlite:///mlflow.db",
        )
    )

    mlflow.set_experiment(experiment_name)

    provenance = collect_provenance()
    provenance.update(
        {
            "seed": seed,
            "dataset": dataset,
        }
    )

    with mlflow.start_run(
        run_name=run_name,
        log_system_metrics=True,
    ) as active_run:
        run = Run()

        mlflow.log_dict(
            config,
            "config.json",
        )

        mlflow.log_dict(
            provenance,
            "provenance.json",
        )

        if provenance["git_commit"]:
            mlflow.set_tag(
                "git_commit",
                provenance["git_commit"],
            )

        mlflow.set_tag(
            "git_dirty",
            provenance["git_dirty"],
        )

        if seed is not None:
            mlflow.log_param(
                "seed",
                seed,
            )

        start = time.perf_counter()

        result = experiment(
            run,
            config,
        )

        runtime = time.perf_counter() - start

        run.metric(
            "runtime_seconds",
            runtime,
        )

        print(
            f"MLflow run: {active_run.info.run_id}"
        )

        return result