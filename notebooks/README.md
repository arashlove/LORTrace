# Notebooks

## `lor_labels/` — Lack-of-Reserve label reconstruction

These notebooks document how `Actual_LOR_Final.csv` and
`FORECAST_LOR_Final.csv` were built. You do **not** need to run them to
reproduce the results — the resulting label files are released with the data
(`scripts/download_assets.py --data`).

| Notebook | Step |
|---|---|
| `01_scrape_market_notices.ipynb` | Collect AEMO `MARKETNOTICEDATA` archives from NEMWEB (Selenium) into `NEM_market_notice_CSV_Data/`. |
| `02_build_lor_labels.ipynb` | Extract LOR notices, normalise regions, pair start/cancel notices into episodes, and expand them onto the 5-minute grid for the Actual and Forecast branches. |

Extra dependencies: `selenium`, `requests`, `beautifulsoup4` and a Chrome
driver for step 1. Outputs are written to the working directory.
