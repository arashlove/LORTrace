"""Serialize numeric time-series windows into text / tokens for the LLM.

Tabular LLM anomaly methods (e.g. AnoLLM) convert a row into ``"col is val, col2 is val2, ..."`` and
fine-tunes an LLM with next-token prediction. We adapt this to time series: a
window of ``W`` timesteps is serialized in chronological order, with each
per-step feature named ``t{step}_{feature}``. Window-level context features
(calendar, battery availability) are emitted once, first.

When ``descriptions`` are provided, a compact ``Schema: ...`` block is
prepended once per window (identical at train and score). Schema tokens are
never included in the value-token NLL mask.

Two representations are produced from the same scheme:

* :meth:`Serializer.to_text` - a plain string used for training via TRL's
  ``SFTTrainer`` (loss over the whole sequence).
* :meth:`Serializer.to_tokens` - token ids plus a boolean value-mask used for
  scoring, so negative log-likelihood is summed over *feature-value* tokens
  only (never the column names, schema, or separators).

Feature order is permuted *within* each timestep to discourage the model from
relying on position (as in AnoLLM-style tabular serialization). The chronological order of timesteps is
preserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

UNKNOWN_TOKEN = "Unknown"


def resolve_text_tokenizer(tokenizer_or_processor):
    """Return an object with HuggingFace ``encode`` / special-token ids.

    Vision-language checkpoints (e.g. Cosmos Reason / Qwen2.5-VL) load a
    ``Processor`` that wraps a text tokenizer. Scoring and value masking need
    the inner tokenizer's ``encode`` API.
    """
    tok = tokenizer_or_processor
    if tok is None:
        raise TypeError("tokenizer is None")
    if callable(getattr(tok, "encode", None)):
        return tok
    for attr in ("tokenizer", "text_tokenizer", "sp_model"):
        inner = getattr(tok, attr, None)
        if inner is not None and callable(getattr(inner, "encode", None)):
            return inner
    raise TypeError(
        f"Expected a text tokenizer with encode(), got {type(tok).__name__}. "
        "Vision-language processors must expose .tokenizer."
    )


def format_scaled_value(value: float) -> str:
    """Format a standard-rescaled numeric value to a single decimal place."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return UNKNOWN_TOKEN
    return f"{float(value):.1f}"


def format_int_value(value: float) -> str:
    """Format a window-level integer feature (calendar / flags)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return UNKNOWN_TOKEN
    return str(int(round(float(value))))


def _as_desc_map(descriptions: Mapping[str, str] | object | None) -> dict[str, str]:
    if descriptions is None:
        return {}
    if isinstance(descriptions, dict):
        return {str(k): str(v) for k, v in descriptions.items()}
    # SimpleNamespace / Config from YAML
    if hasattr(descriptions, "__dict__"):
        return {str(k): str(v) for k, v in vars(descriptions).items()}
    raise TypeError(f"Unsupported descriptions type: {type(descriptions)!r}")


@dataclass
class Serializer:
    """Builds text/token representations of a window.

    Args:
        per_step_features: Names of the rescaled features serialized per step.
        window_features: Names of the window-level context features.
        window_size: Number of timesteps per window (``W``).
        descriptions: Optional map of feature name -> short description used to
            build a leading Schema block.
        continuous_window_features: Window-level fields serialized with one
            decimal (z-scored log-signature coeffs); others use integers.
    """

    per_step_features: Sequence[str]
    window_features: Sequence[str]
    window_size: int
    descriptions: Mapping[str, str] | None = None
    continuous_window_features: Sequence[str] = ()
    _desc_map: dict[str, str] = field(init=False, repr=False, default_factory=dict)
    _continuous_window: set[str] = field(init=False, repr=False, default_factory=set)

    def __post_init__(self) -> None:
        self._desc_map = _as_desc_map(self.descriptions)
        self._continuous_window = {
            str(n) for n in self.continuous_window_features
        } | {str(n) for n in self.window_features if str(n).startswith("logsig_")}

    def default_permutation(self) -> list[int]:
        return list(range(len(self.per_step_features)))

    def random_permutation(self, rng: np.random.Generator) -> list[int]:
        perm = self.default_permutation()
        rng.shuffle(perm)
        return perm

    def schema_text(self) -> str:
        """Compact schema header prepended to every window (or empty)."""
        if not self._desc_map:
            return ""
        parts: list[str] = []
        for name in list(self.window_features) + list(self.per_step_features):
            desc = self._desc_map.get(name)
            if desc:
                parts.append(f"{name}={desc}")
        if not parts:
            return ""
        return (
            "Schema: "
            + "; ".join(parts)
            + ". Per-step values are train-split z-scores; "
            "logsig_* are train z-scored log-signature coeffs; "
            "other window fields are raw integers. "
        )

    def _segments(
        self,
        x_window: np.ndarray,
        window_row: np.ndarray,
        perm: Sequence[int],
    ) -> list[tuple[str, str]]:
        """Return ``(prefix, value)`` segments for a single window.

        ``prefix`` is ``" {name} is "`` and ``value`` is the formatted number.
        """
        if x_window.shape[0] != self.window_size:
            raise ValueError(
                f"Expected window of {self.window_size} steps, got {x_window.shape[0]}"
            )
        expected_f = len(self.per_step_features)
        if x_window.shape[1] != expected_f:
            raise ValueError(
                f"Expected {expected_f} per-step features, got {x_window.shape[1]}"
            )
        if len(window_row) != len(self.window_features):
            raise ValueError(
                f"Expected {len(self.window_features)} window features, "
                f"got {len(window_row)}"
            )

        segments: list[tuple[str, str]] = []
        for name, value in zip(self.window_features, window_row):
            if name in self._continuous_window:
                fmt = format_scaled_value(value)
            else:
                fmt = format_int_value(value)
            segments.append((f" {name} is ", fmt))

        for step in range(self.window_size):
            for j in perm:
                name = f"t{step}_{self.per_step_features[j]}"
                segments.append((f" {name} is ", format_scaled_value(x_window[step, j])))
        return segments

    def to_text(
        self,
        x_window: np.ndarray,
        window_row: np.ndarray,
        perm: Sequence[int],
        eos_token: str,
    ) -> str:
        """Serialize a window to a training string terminated by ``eos_token``."""
        schema = self.schema_text()
        segments = self._segments(x_window, window_row, perm)
        body = ",".join(f"{prefix}{value} " for prefix, value in segments)
        return schema + body + eos_token

    def to_tokens(
        self,
        x_window: np.ndarray,
        window_row: np.ndarray,
        perm: Sequence[int],
        tokenizer,
    ) -> dict[str, list[int]]:
        """Serialize a window to token ids with a per-token value mask.

        Returns a dict with:
            input_ids:   list[int]
            value_mask:  list[int]  (1 where the token is part of a feature value)
            col_ids:     list[int]  (per-step feature index for value tokens, else -1;
                                     window-level values use -1)
        The value mask/col ids allow value-only and per-feature NLL aggregation.
        Schema tokens (if any) are always masked out of the NLL.
        """
        tokenizer = resolve_text_tokenizer(tokenizer)
        segments = self._segments(x_window, window_row, perm)
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        comma_ids = tokenizer.encode(",", add_special_tokens=False)

        input_ids: list[int] = []
        value_mask: list[int] = []
        col_ids: list[int] = []

        if bos_id is not None:
            input_ids.append(bos_id)
            value_mask.append(0)
            col_ids.append(-1)

        schema = self.schema_text()
        if schema:
            schema_ids = tokenizer.encode(schema, add_special_tokens=False)
            input_ids.extend(schema_ids)
            value_mask.extend([0] * len(schema_ids))
            col_ids.extend([-1] * len(schema_ids))

        n_window_level = len(self.window_features)
        # Reconstruct which per-step feature each value segment belongs to so we
        # can attribute NLL. Segments after the window-level block cycle through
        # ``perm`` for each step.
        n_segments = len(segments)
        for seg_idx, (prefix, value) in enumerate(segments):
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            value_ids = tokenizer.encode(value, add_special_tokens=False)

            input_ids.extend(prefix_ids)
            value_mask.extend([0] * len(prefix_ids))
            col_ids.extend([-1] * len(prefix_ids))

            if seg_idx < n_window_level or not perm:
                feature_idx = -1
            else:
                within = seg_idx - n_window_level
                feature_idx = int(perm[within % len(perm)])

            input_ids.extend(value_ids)
            value_mask.extend([1] * len(value_ids))
            col_ids.extend([feature_idx] * len(value_ids))

            is_last = seg_idx == n_segments - 1
            sep_ids = [eos_id] if is_last and eos_id is not None else comma_ids
            input_ids.extend(sep_ids)
            value_mask.extend([0] * len(sep_ids))
            col_ids.extend([-1] * len(sep_ids))

        return {"input_ids": input_ids, "value_mask": value_mask, "col_ids": col_ids}


def serializer_from_manifest(manifest: dict) -> Serializer:
    """Build a :class:`Serializer` from a preprocessing ``manifest.json`` dict."""
    return Serializer(
        per_step_features=manifest["per_step_features"],
        window_features=manifest["window_features"],
        window_size=manifest["window_size"],
        descriptions=manifest.get("descriptions"),
        continuous_window_features=manifest.get("continuous_window_features") or (),
    )
