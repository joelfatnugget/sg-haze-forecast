# Singapore Atmospheric & Haze Intelligence (`sg-haze-forecast`)

An end-to-end environmental data pipeline, machine learning forecasting engine, and interactive meteorological workbench for Singapore. It ingests live and historical air quality telemetry (`PSI` 24-hr, `PM2.5` 1-hr and 24-hr, `PM10`, `O3`, `CO`, `SO2`, `NO2`) directly from Singapore's National Environment Agency (NEA) via `data.gov.sg` across all five monitoring regions (`North`, `South`, `East`, `West`, `Central`), and produces both **72-hour hourly quantile forecasts** and **12-month seasonal transboundary haze risk projections**.

## Architecture & Core Capabilities

1. **Live & Historical NEA Telemetry Pipeline ([`src/data_fetcher.py`](file:///Users/joel/Documents/Github/sg-haze-forecast/src/data_fetcher.py)):**
   - Queries `https://api.data.gov.sg/v1/environment/psi` and `https://api.data.gov.sg/v1/environment/pm25` concurrently across multi-day windows.
   - Persists readings in a local WAL-enabled SQLite database (`data/sg_haze_telemetry.sqlite`) and performs continuous hourly spatial-diurnal interpolation so autoregressive lag features remain unbroken.
   - Includes a calibrated 2014–2026 monthly Singapore haze and macro-climate dataset anchored to documented historical episodes (2015 Super El Niño crisis where PSI reached 341, 2019 Extreme Positive Indian Ocean Dipole episode, 2023 October transboundary spike, and 2020–2022 La Niña clean-air baselines).

2. **Dual-Horizon Machine Learning Forecasting Engine ([`src/forecaster.py`](file:///Users/joel/Documents/Github/sg-haze-forecast/src/forecaster.py)):**
   - **Short-Term (1 to 72 Hours Ahead):** Trains `HistGradientBoostingRegressor` models (squared-error point forecast + 10th and 90th percentile quantile regressors) using 22 engineered temporal, diurnal harmonic, cross-regional spatial gradient, and particulate ratio features. Supports synoptic wind regime simulation (`Prevailing Flow`, `SW Transboundary Plume`, and `Monsoon Rain Washout`).
   - **Seasonal Outlook (12 Months Ahead):** Models monthly mean PSI, expected peak PSI, and probability of Unhealthy haze days (`PSI > 100`) conditioned on El Niño-Southern Oscillation (`ENSO ONI`), Indian Ocean Dipole (`IOD DMI`), regional rainfall anomalies, and Southwest/Northeast Monsoon phases.

3. **Interactive Telemetry & Forecasting Workbench ([`src/api.py`](file:///Users/joel/Documents/Github/sg-haze-forecast/src/api.py), [`static/index.html`](file:///Users/joel/Documents/Github/sg-haze-forecast/static/index.html)):**
   - **Spatial 5-Region Singapore Map & Plume Scrubber:** Interactive SVG map with live station nodes and a `+0h` to `+72h` horizon slider that scrubs across predicted regional plumes.
   - **72-Hour Quantile Fan Chart & Multi-Horizon Matrix:** Visualizes the past 48 hours of observed NEA readings alongside the next 72 hours of forecasted trajectories and 10th–90th percentile uncertainty bands.
   - **Seasonal ENSO & IOD Simulator:** Interactive sliders and historical presets (`1997/2015 Super El Niño`, `2019 Extreme +IOD`, `La Niña Wet Suppression`) to stress-test Singapore's 12-month haze risk.
   - **Walk-Forward Backtest & Feature Sensitivity Diagnostics:** Out-of-sample holdout verification (`MAE`, `RMSE`, $R^2$, interval coverage) and permutation feature importance.

## Quickstart

### 1. Run the Interactive Web Workbench
```bash
cd /Users/joel/Documents/Github/sg-haze-forecast
.venv/bin/python run.py --mode serve --port 8050
```
Open `http://127.0.0.1:8050` in your browser.

### 2. Run the Command-Line Forecast Report
```bash
.venv/bin/python run.py --mode cli --regime baseline
# Or simulate a Southwest Sumatra transboundary plume:
.venv/bin/python run.py --mode cli --regime sumatra_sw
```

### 3. Run the Automated Test Suite
```bash
.venv/bin/pytest -v
```
