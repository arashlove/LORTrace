"""Benchmark pipeline: LLMs as unsupervised detectors of AEMO NSW1 market stress.

Modules:
    config        Load and validate the YAML configuration.
    preprocessing Clean the CSV, align LOR labels, split, and build windows.
    serialization Turn numeric windows into text / tokens for the LLM.
    training      LoRA fine-tuning of a causal-LM backbone with Unsloth.
    scoring       Value-token negative log-likelihood anomaly scores.
    evaluation    LOR recall metrics, forecast/actual matrices, and plots.
    signature     Optional path log-signature window features.
"""

__all__ = [
    "config",
    "preprocessing",
    "serialization",
    "training",
    "scoring",
    "evaluation",
    "signature",
]
