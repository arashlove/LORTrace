---
license: cc-by-4.0
pretty_name: AEMO NSW1 market windows and Lack-of-Reserve labels (2015–2026)
tags:
  - time-series
  - anomaly-detection
  - electricity-market
  - aemo
size_categories:
  - 1M<n<10M
---

# AEMO NSW1 market series and Lack-of-Reserve labels

Data from the LORTrace project (Karimadini et al.). Code:
[github.com/arashlove/LORTrace](https://github.com/arashlove/LORTrace).

> **If you use this dataset, you must cite it** (see
> [License and attribution](#license-and-attribution)).

## Files

| File | Rows | Description |
|---|---|---|
| `causality_framework_2015-2026_NSW1.csv.gz` | 1,222,676 | NSW1 5-minute dispatch series, 2015-01-01 00:05 → 2026-02-02 09:00 (interval-ending timestamps). Contains duplicated timestamps from price re-runs; the pipeline keeps the last record per timestamp (1,166,508 unique). |
| `Actual_LOR_Final.csv.gz` | 5,038,560 | Actual Lack-of-Reserve flags (`ACTUAL_LOR1..3`) on the 5-minute grid for all NEM regions. **Ground truth.** |
| `FORECAST_LOR_Final.csv.gz` | 5,038,560 | Forecast LOR flags (`FORECAST_LOR1..3`) on the same grid. **Institutional baseline.** |

LOR labels were reconstructed from AEMO `MARKETNOTICEDATA` notices; the reconstruction notebooks are in `notebooks/lor_labels/` of
the code repository. Actual LOR coverage ends 2024-07-30.

### Main series columns

`date, REGIONID, RRP, ROP, R_INTERVENTION, DISPATCHABLEGENERATION, TOTALDEMAND,
NETINTERCHANGE, DISPATCHABLELOAD, AVAILABLEGENERATION, DEMANDFORECAST,
INTERVENTION, TOTALINTERMITTENTGENERATION, UIGF, SEMISCHEDULE_CLEAREDMW,
DEMAND_AND_NONSCHEDGEN, BDU_CLEAREDMW_GEN, BDU_CLEAREDMW_LOAD, TEMPERATURE`

### Integrity (sha256)

```
c86fa788389e487f489d5da2d765cd9f44f883b14afa259797a8e288bffd71af  causality_framework_2015-2026_NSW1.csv.gz
3e06e9a58ee59686b7e204ad503885a6c2ce8c6370b1774c0a4838b594884e1d  Actual_LOR_Final.csv.gz
bca82dba403ca1cce11e087bdcab2b74b0f089bc1c4a2fe6dc3760f7cfd81401  FORECAST_LOR_Final.csv.gz
```

## License and attribution

The curation of this dataset (the reconstructed Actual and Forecast LOR
labels, the cleaning, alignment and assembly of the series) is © 2026
Arash Karimadini and Kaveh Karimadini, and is released under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
license.

You are free to use, share and adapt the data for any purpose, **provided you
give appropriate credit**. For this dataset, appropriate credit means:

1. **Citing this dataset** (below) in any publication, report, thesis, model card or
   software that uses the data or anything derived from it, and
2. **Acknowledging AEMO** as the source of the underlying market data.

Suggested acknowledgement: *"This work uses the AEMO NEM Lack-of-Reserve
dataset of Karimadini et al. (2026), derived from public data published by the
Australian Energy Market Operator (AEMO)."*

```bibtex
@misc{karimadini2026aemonemlor,
  title     = {{AEMO NEM Lack-of-Reserve Dataset}},
  author    = {Karimadini, Arash and Karimadini, Kaveh},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/arashlove/aemo-nem-lor}
}
```

### Underlying AEMO data

The source data was published by the Australian Energy Market Operator (AEMO)
via NEMWEB. © AEMO. It is used and redistributed under AEMO's
[copyright permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions),
which require accurate attribution to AEMO as the author of the source
material, and those terms continue to apply to the AEMO material. The CC BY 4.0
license above covers the authors' contributions only. AEMO does not endorse
this derived dataset.
