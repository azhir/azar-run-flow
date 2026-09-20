from __future__ import annotations

from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


Config = DictConfig | dict[str, Any]


def resolve_config(config: Config) -> dict[str, Any]:
    """
    Convert a Hydra/OmegaConf config into the final plain-Python
    configuration that actually ran.
    """

    if isinstance(config, DictConfig):
        resolved = OmegaConf.to_container(
            config,
            resolve=True,
            enum_to_str=True,
        )

        if not isinstance(resolved, dict):
            raise TypeError(
                "Top-level experiment config must resolve to a dictionary."
            )

        return resolved

    return dict(config)


def config_yaml(config: Config) -> str:
    """
    Return the fully resolved config as YAML.
    """

    omega = (
        config
        if isinstance(config, DictConfig)
        else OmegaConf.create(config)
    )

    return OmegaConf.to_yaml(
        omega,
        resolve=True,
        sort_keys=False,
    )


def hydra_overrides() -> list[str]:
    """
    Return CLI overrides applied by Hydra for the current job.
    """

    if not HydraConfig.initialized():
        return []

    return list(
        HydraConfig.get().overrides.task
    )


def flatten_config(
    config: dict[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """
    Flatten scalar config values for MLflow parameters.

    Example:
        {"loss": {"alpha": 0.2}}
    becomes:
        {"loss.alpha": 0.2}
    """

    result: dict[str, Any] = {}

    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            result.update(
                flatten_config(
                    value,
                    prefix=name,
                )
            )

        elif isinstance(
            value,
            (str, int, float, bool),
        ) or value is None:
            result[name] = value

    return result