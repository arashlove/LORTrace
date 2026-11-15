# Data

The raw files (~80 MB, gzip-compressed CSV) are hosted on the Hugging Face Hub
and are not stored in git. Fetch and checksum-verify them with:

```bash
uv run python scripts/download_assets.py --data
```

They land in `data/raw/`. See [`raw/README.md`](raw/README.md) for the file
descriptions, provenance (AEMO NEMWEB) and terms of use.

Preprocessed windows are regenerated deterministically into
`outputs/processed/` by `scripts/preprocess.py` (run automatically by
`scripts/reproduce.py`).
