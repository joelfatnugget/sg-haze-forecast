# PRODUCT.md — Singapore Atmospheric & Haze Intelligence (SG-Haze-Forecast)

## Product Overview
**SG-Haze-Forecast** is an environmental monitoring and predictive analytics system for Singapore. It ingests live and historical pollutant readings from Singapore's National Environment Agency (NEA) via `data.gov.sg` (`PSI` 24-hr, `PM2.5` 1-hr and 24-hr, `PM10`, `O3`, `CO`, `SO2`, `NO2` across all 5 regions: North, South, East, West, Central) alongside macro-climatic indices (El Niño-Southern Oscillation ONI, Indian Ocean Dipole DMI, and Southwest/Inter-Monsoon wind regimes).

## Primary Surfaces & Mode
- **Mode:** Operate (Precision Meteorological & Environmental Telemetry Station)
- **Primary Users:** Environmental analysts, researchers, and residents in Singapore tracking hourly air quality trajectories and seasonal transboundary haze probabilities.

## Core Capabilities
1. **Live & Historical Data Ingestion (`data.gov.sg` API):**
   - Concurrent multi-day fetcher with local SQLite/Parquet caching and multi-year historical backfill.
   - Real-time synchronization with NEA's 5 regional monitoring zones (`north`, `south`, `east`, `west`, `central`).
2. **Short-Term Forecasting Engine (24–72 Hours Hourly):**
   - Gradient Boosting (`HistGradientBoostingRegressor` / Quantile Regression) with temporal lags ($t-1 \dots t-48$), rolling statistics (4h, 12h, 24h mean/std/momentum), diurnal harmonic encoding, and spatial cross-region differentials.
   - Outputs point predictions along with 10th–90th percentile uncertainty bands for all 5 regions and the national maximum.
3. **Seasonal & Macro-Climatic Risk Engine (12-Month Outlook):**
   - Models the physical drivers of Southeast Asian transboundary peatland haze: ENSO (El Niño Southern Oscillation), IOD (Indian Ocean Dipole), and the Southwest Monsoon dry phase (June–October).
   - Provides interactive scenario simulation (e.g., adjusting ENSO anomaly and IOD index to stress-test Singapore's monthly severe haze probability and expected peak PSI).
4. **Interactive Telemetry & Forecasting Workbench:**
   - Spatial SVG map of Singapore's 5 regions with live and forecasted scrubbing across the +72h horizon.
   - Multi-horizon trajectory fan charts, regional comparison matrices, model backtest diagnostics (MAE, RMSE, $R^2$, feature importance), and seasonal climate risk heatmaps.
