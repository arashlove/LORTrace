"""Maintainer helper: publish data/raw/ and weights/ to the Hugging Face Hub.

Run once (after `hf auth login`) to create the dataset and model repos named
in configs/assets.yaml. data/raw/README.md and weights/README.md become the
dataset and model cards.

Example:
    uv run python scripts/upload_assets.py --private     # review first, then make public on the Hub
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private", action="store_true", help="Create the repos as private.")
    parser.add_argument("--only", choices=["data", "weights"], help="Upload just one of the two.")
    args = parser.parse_args()

    spec = yaml.safe_load((ROOT / "configs" / "assets.yaml").read_text(encoding="utf-8"))
    api = HfApi()
    for key in ("data", "weights"):
        if args.only and args.only != key:
            continue
        s = spec[key]
        if s["repo_id"].startswith("YOUR_HF_USERNAME"):
            raise SystemExit(f"Set {key}.repo_id in configs/assets.yaml first.")
        folder = ROOT / s["local_dir"]
        api.create_repo(s["repo_id"], repo_type=s["repo_type"], private=args.private, exist_ok=True)
        print(f"Uploading {folder.relative_to(ROOT)} -> {s['repo_id']} ({s['repo_type']})")
        api.upload_folder(
            repo_id=s["repo_id"],
            repo_type=s["repo_type"],
            folder_path=folder,
            commit_message=f"Upload {key} for the AEMO LLM stress benchmark",
        )
    print("Done. Pin `revision` in configs/assets.yaml to the resulting commit hashes.")


if __name__ == "__main__":
    main()
