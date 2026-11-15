"""Tests for deterministic serialization, permutations, and value masking."""

from __future__ import annotations

import numpy as np

from lortrace.benchmark.serialization import Serializer, format_scaled_value, resolve_text_tokenizer


class FakeTokenizer:
    """Minimal char-level tokenizer sufficient for masking tests."""

    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        ids = [ord(c) % 90 + 10 for c in text if not c.isspace()]
        return ids or [9]


class FakeVLProcessor:
    """Mimics Qwen2.5-VL / Cosmos processor wrapping a text tokenizer."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def decode(self, *args, **kwargs):
        return ""


def test_resolve_text_tokenizer_unwraps_vl_processor():
    inner = FakeTokenizer()
    proc = FakeVLProcessor(inner)
    assert resolve_text_tokenizer(proc) is inner
    assert resolve_text_tokenizer(inner) is inner


def test_to_tokens_accepts_vl_processor():
    s = _serializer()
    tok = FakeVLProcessor(FakeTokenizer())
    x = np.zeros((3, 3), dtype=np.float32)
    wf = np.array([0, 0], dtype=np.float32)
    out = s.to_tokens(x, wf, [0, 1, 2], tok)
    assert sum(out["value_mask"]) > 0



def _serializer():
    return Serializer(
        per_step_features=["RRP", "TOTALDEMAND", "reserve_margin"],
        window_features=["hour", "battery_active"],
        window_size=3,
    )


def test_format_scaled_value():
    assert format_scaled_value(1.23) == "1.2"
    assert format_scaled_value(-0.25) == "-0.2" or format_scaled_value(-0.25) == "-0.3"
    assert format_scaled_value(float("nan")) == "Unknown"


def test_text_is_deterministic_and_terminated():
    s = _serializer()
    x = np.arange(9, dtype=np.float32).reshape(3, 3)
    wf = np.array([12, 1], dtype=np.float32)
    perm = [0, 1, 2]
    a = s.to_text(x, wf, perm, "<eos>")
    b = s.to_text(x, wf, perm, "<eos>")
    assert a == b
    assert a.endswith("<eos>")
    assert "t0_RRP is 0.0" in a
    assert "hour is 12" in a


def test_schema_block_prepended_when_descriptions_set():
    s = Serializer(
        per_step_features=["RRP", "TOTALDEMAND", "reserve_margin"],
        window_features=["hour", "battery_active"],
        window_size=3,
        descriptions={
            "RRP": "price AUD/MWh (z-scored)",
            "TOTALDEMAND": "demand MW (z-scored)",
            "reserve_margin": "margin (z-scored)",
            "hour": "hour 0-23",
            "battery_active": "battery flag",
        },
    )
    x = np.zeros((3, 3), dtype=np.float32)
    wf = np.array([12, 1], dtype=np.float32)
    text = s.to_text(x, wf, [0, 1, 2], "<eos>")
    assert text.startswith("Schema:")
    assert "RRP=price AUD/MWh (z-scored)" in text
    assert "hour=hour 0-23" in text
    assert text.index("Schema:") < text.index("hour is 12")

    tok = FakeTokenizer()
    out = s.to_tokens(x, wf, [0, 1, 2], tok)
    # Schema tokens must not contribute to value-token NLL.
    assert sum(out["value_mask"]) > 0
    # BOS + schema + body: first value tokens still exist after schema.
    assert out["value_mask"][0] == 0  # BOS
    # At least some leading tokens after BOS are schema (mask 0) before values.
    first_val = next(i for i, m in enumerate(out["value_mask"]) if m == 1)
    assert first_val > 1


def test_schema_empty_without_descriptions():
    s = _serializer()
    assert s.schema_text() == ""


def test_permutation_reorders_within_step_only():
    s = _serializer()
    x = np.arange(9, dtype=np.float32).reshape(3, 3)
    wf = np.array([12, 1], dtype=np.float32)
    text = s.to_text(x, wf, [2, 0, 1], "<eos>")
    # Window-level features always precede step features and keep their order.
    assert text.index("hour is") < text.index("battery_active is")
    assert text.index("battery_active is") < text.index("t0_")
    # Chronological order of steps preserved.
    assert text.index("t0_") < text.index("t1_") < text.index("t2_")


def test_value_mask_counts_and_col_ids():
    s = _serializer()
    tok = FakeTokenizer()
    x = np.zeros((3, 3), dtype=np.float32)
    wf = np.array([0, 0], dtype=np.float32)
    perm = [0, 1, 2]
    out = s.to_tokens(x, wf, perm, tok)

    input_ids = out["input_ids"]
    value_mask = out["value_mask"]
    col_ids = out["col_ids"]
    assert len(input_ids) == len(value_mask) == len(col_ids)
    assert input_ids[0] == tok.bos_token_id
    assert input_ids[-1] == tok.eos_token_id

    # Every value token is masked; value tokens for step features carry a
    # per-step feature index in [0, 2]; window-level values carry -1.
    for tok_id, is_val, col in zip(input_ids, value_mask, col_ids):
        if is_val:
            assert col in (-1, 0, 1, 2)
    assert sum(value_mask) > 0
    # At least one per-step feature index appears among value tokens.
    assert any(c in (0, 1, 2) for c in col_ids)


def test_col_ids_follow_permutation():
    s = _serializer()
    tok = FakeTokenizer()
    x = np.array([[10.0, 20.0, 30.0]], dtype=np.float32).repeat(3, axis=0)
    wf = np.array([0, 0], dtype=np.float32)
    perm = [2, 0, 1]
    out = s.to_tokens(x, wf, perm, tok)
    # The first per-step value token (after window-level block) should map to
    # feature index perm[0] == 2.
    step_feature_cols = [c for c in out["col_ids"] if c in (0, 1, 2)]
    assert step_feature_cols[0] == 2
