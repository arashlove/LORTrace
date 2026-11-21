"""Download the benchmark data and LoRA adapters from the Hugging Face Hub.

Repo ids and revisions are set in configs/assets.yaml.

Examples:
    uv run python scripts/download_assets.py                 # data + all adapters
    uv run python scripts/download_assets.py --data
    uv run python scripts/download_assets.py --weights --models gemma-3-270m mistral-7b-v0.3
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_repo_id(section: dict, name: str) -> None:
    if section["repo_id"].startswith("YOUR_HF_USERNAME"):
        sys.exit(f"configs/assets.yaml: set {name}.repo_id to the released Hugging Face repo first.")


def download_data(spec: dict) -> None:
    check_repo_id(spec, "data")
    dest = ROOT / spec["local_dir"]
    print(f"Downloading data from {spec['repo_id']}@{spec['revision']} -> {dest.relative_to(ROOT)}")
    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type=spec["repo_type"],
        revision=spec["revision"],
        local_dir=dest,
        allow_patterns=["*.csv.gz"],
    )
    bad = []
    for name, expected in (spec.get("sha256") or {}).items():
        path = dest / name
        if not path.is_file():
            bad.append(f"{name}: missing")
        elif sha256(path) != expected:
            bad.append(f"{name}: checksum mismatch")
    if bad:
        sys.exit("Data verification failed:\n  " + "\n  ".join(bad))
    print("Data verified (sha256).")


def download_weights(spec: dict, models: list[str] | None) -> None:
    check_repo_id(spec, "weights")
    dest = ROOT / spec["local_dir"]
    patterns = [f"{m}/*" for m in models] if models else None
    print(f"Downloading adapters from {spec['repo_id']}@{spec['revision']} -> {dest.relative_to(ROOT)}")
    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type=spec["repo_type"],
        revision=spec["revision"],
        local_dir=dest,
        allow_patterns=patterns,
    )
    print("Adapters downloaded.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch data and adapters from the Hugging Face Hub.")
    parser.add_argument("--data", action="store_true", help="Download only the data.")
    parser.add_argument("--weights", action="store_true", help="Download only the adapters.")
    parser.add_argument("--models", nargs="+", metavar="NAME", help="Adapter subset (default: all).")
    args = parser.parse_args()

    spec = yaml.safe_load((ROOT / "configs" / "assets.yaml").read_text(encoding="utf-8"))
    both = not (args.data or args.weights)
    if args.data or both:
        download_data(spec["data"])
    if args.weights or both:
        download_weights(spec["weights"], args.models)


if __name__ == "__main__":
    main()
