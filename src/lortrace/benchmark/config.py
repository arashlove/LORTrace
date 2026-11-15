"""Configuration loading and path resolution.

The pipeline is driven by a single ``config.yaml``. This module loads it into a
nested, attribute-accessible object and centralises artifact path resolution so
every stage agrees on where files live.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

_REQUIRED_TOP_LEVEL = (
    "data",
    "artifacts",
    "splits",
    "windowing",
    "features",
    "model",
    "training",
    "scoring",
    "evaluation",
)


class Config(SimpleNamespace):
    """Attribute-accessible view over the parsed YAML mapping.

    The original mapping is preserved under :attr:`raw`, and the resolved repo
    root under :attr:`project_root`.
    """

    project_root: Path
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.raw


def _namespacify(value: Any) -> Any:
    if isinstance(value, dict):
        return Config(**{k: _namespacify(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_namespacify(v) for v in value]
    return value


def _find_project_root(start: Path) -> Path:
    """Return the nearest ancestor of ``start`` holding ``pyproject.toml``.

    Relative paths in a config resolve against the repository root, so configs
    can live anywhere (``configs/``, ``outputs/<run>/``). Falls back to the
    config's own directory when no ``pyproject.toml`` is found.
    """
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start


def load_config(path: str | Path) -> Config:
    """Load ``config.yaml`` and return a validated :class:`Config`.

    Args:
        path: Path to the YAML config file.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If a required top-level section is missing.
    """
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    missing = [key for key in _REQUIRED_TOP_LEVEL if key not in raw]
    if missing:
        raise KeyError(f"Config is missing required section(s): {', '.join(missing)}")

    cfg: Config = _namespacify(raw)  # type: ignore[assignment]
    cfg.raw = raw
    cfg.project_root = _find_project_root(config_path.parent)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.windowing.size <= 0:
        raise ValueError("windowing.size must be positive")
    if cfg.windowing.stride <= 0:
        raise ValueError("windowing.stride must be positive")

    per_step = set(cfg.features.per_step)
    raw_used = set(cfg.features.raw_used)
    derived = {"reserve_margin", "curtailment_gap"}
    unknown = per_step - (raw_used | derived)
    if unknown:
        raise ValueError(
            "features.per_step references unknown columns: "
            f"{sorted(unknown)}. Add them to features.raw_used or use a "
            "supported derived feature (reserve_margin, curtailment_gap)."
        )

    descriptions = getattr(cfg.features, "descriptions", None)
    if descriptions is not None:
        if not isinstance(descriptions, dict) and not hasattr(descriptions, "__dict__"):
            raise ValueError("features.descriptions must be a mapping of name -> text")
        # SimpleNamespace from YAML becomes Config with attributes.
        if hasattr(descriptions, "__dict__") and not isinstance(descriptions, dict):
            desc_keys = set(vars(descriptions).keys())
        else:
            desc_keys = set(descriptions.keys())
        needed = set(cfg.features.per_step) | set(cfg.features.window_level)
        missing = needed - desc_keys
        if missing:
            raise ValueError(
                "features.descriptions is missing entries for: "
                f"{sorted(missing)}. Every per_step and window_level feature "
                "needs a short description."
            )

    valid_settings = {
        "semi_supervised",
        "semi_supervised_forecast",
        "semi_supervised_both",
        "unsupervised",
    }
    if cfg.training.setting not in valid_settings:
        raise ValueError(f"training.setting must be one of {sorted(valid_settings)}")

    quantiles = getattr(cfg.evaluation, "threshold_quantiles", None)
    if quantiles is None:
        raise ValueError("evaluation.threshold_quantiles is required (e.g. [0.90, 0.95, 0.99])")
    for q in quantiles:
        if not (0.0 < float(q) < 1.0):
            raise ValueError(f"Each threshold_quantile must be in (0, 1), got {q}")

    sig = getattr(cfg, "signature", None)
    if sig is not None and bool(getattr(sig, "enabled", False)):
        depth = int(getattr(sig, "depth", 2))
        if depth < 1:
            raise ValueError("signature.depth must be >= 1")
        try:
            import iisignature  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "signature.enabled is true but iisignature is not installed. "
                "Run: pip install iisignature"
            ) from exc


def resolve_path(cfg: Config, path: str | Path) -> Path:
    """Resolve ``path`` relative to the project root (absolute paths pass through)."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (cfg.project_root / candidate).resolve()


def processed_dir(cfg: Config) -> Path:
    return resolve_path(cfg, cfg.artifacts.processed_dir)


def model_dir(cfg: Config) -> Path:
    return resolve_path(cfg, cfg.artifacts.model_dir)


def scores_dir(cfg: Config) -> Path:
    return resolve_path(cfg, cfg.artifacts.scores_dir)


def evaluation_dir(cfg: Config) -> Path:
    return resolve_path(cfg, cfg.artifacts.evaluation_dir)


def ensure_dirs(*dirs: Path) -> None:
    """Create the given directories (and parents) if they do not exist."""
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
