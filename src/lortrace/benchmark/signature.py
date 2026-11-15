"""Path log-signature features for LORTrace windows.

Computes a truncated **log-signature** of each multivariate window (optional
time augmentation), then train-only z-scores the coefficients. These become
extra window-level features serialized once per window.

Requires the ``iisignature`` package when ``signature.enabled`` is true.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def logsig_length(dim: int, depth: int) -> int:
    """Return the length of the log-signature for a ``dim``-path of given depth."""
    import iisignature

    return int(iisignature.logsiglength(dim, depth))


def logsig_feature_names(n_coeffs: int) -> list[str]:
    return [f"logsig_{k}" for k in range(n_coeffs)]


def logsig_descriptions(n_coeffs: int, depth: int, time_augment: bool) -> dict[str, str]:
    aug = "time-augmented " if time_augment else ""
    return {
        f"logsig_{k}": (
            f"{aug}log-signature depth-{depth} coefficient {k} (train z-scored)"
        )
        for k in range(n_coeffs)
    }


def _prepare(dim: int, depth: int):
    import iisignature

    return iisignature.prepare(dim, depth)


def compute_log_signatures(
    windows: np.ndarray,
    depth: int = 2,
    time_augment: bool = True,
) -> np.ndarray:
    """Compute log-signatures for a batch of windows.

    Args:
        windows: Array of shape ``(n, W, d)`` (already channel-scaled if desired).
        depth: Truncation depth (default 2).
        time_augment: If true, append normalized time in ``[0, 1]`` as an extra
            channel before computing the signature.

    Returns:
        Array of shape ``(n, L)`` where ``L = logsiglength(d[+1], depth)``.
    """
    import iisignature

    if windows.ndim != 3:
        raise ValueError(f"Expected windows (n, W, d), got shape {windows.shape}")
    n, length, d = windows.shape
    if length < 2:
        raise ValueError("Signature requires at least 2 timesteps per window")
    if n == 0:
        dim = d + (1 if time_augment else 0)
        return np.zeros((0, logsig_length(dim, depth)), dtype=np.float32)

    if time_augment:
        t = np.linspace(0.0, 1.0, length, dtype=np.float64)
        t = np.broadcast_to(t, (n, length))
        paths = np.concatenate(
            [windows.astype(np.float64, copy=False), t[..., None]], axis=-1
        )
    else:
        paths = windows.astype(np.float64, copy=False)

    dim = paths.shape[-1]
    prepared = _prepare(dim, depth)
    out = np.empty((n, logsig_length(dim, depth)), dtype=np.float64)
    for i in range(n):
        out[i] = iisignature.logsig(paths[i], prepared)
    return out.astype(np.float32)


def fit_logsig_scaling(train_logsig: np.ndarray) -> dict[str, list[float]]:
    """Per-coefficient mean/std on the training log-signatures."""
    scaling: dict[str, list[float]] = {}
    if train_logsig.size == 0:
        return scaling
    means = train_logsig.mean(axis=0)
    stds = train_logsig.std(axis=0)
    for k, (mean, std) in enumerate(zip(means, stds)):
        s = float(std) if float(std) >= 1e-8 else 1.0
        scaling[f"logsig_{k}"] = [float(mean), s]
    return scaling


def apply_logsig_scaling(
    logsig: np.ndarray, scaling: dict[str, list[float]]
) -> np.ndarray:
    if logsig.size == 0:
        return logsig
    out = logsig.copy()
    for k in range(out.shape[1]):
        key = f"logsig_{k}"
        mean, std = scaling[key]
        out[:, k] = (out[:, k] - mean) / std
    return out.astype(np.float32)


def signature_enabled(cfg: Any) -> bool:
    sig = getattr(cfg, "signature", None)
    if sig is None:
        return False
    return bool(getattr(sig, "enabled", False))


def signature_settings(cfg: Any) -> dict[str, Any]:
    sig = getattr(cfg, "signature", None)
    if sig is None:
        return {
            "enabled": False,
            "depth": 2,
            "time_augment": True,
            "replace_path": False,
        }
    return {
        "enabled": bool(getattr(sig, "enabled", False)),
        "depth": int(getattr(sig, "depth", 2)),
        "time_augment": bool(getattr(sig, "time_augment", True)),
        "replace_path": bool(getattr(sig, "replace_path", False)),
    }
