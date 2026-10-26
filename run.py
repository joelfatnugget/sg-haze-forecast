#!/usr/bin/env python3
"""CLI entrypoint for Singapore Atmospheric & Haze Intelligence."""

from __future__ import annotations

import argparse
import sys

import uvicorn

from src.data_fetcher import sync_live_nea_telemetry
from src.forecaster import forecast_engine


def run_cli_report(horizon_hours: int = 72, wind_regime: str = "baseline") -> None:
    """Sync live data.gov.sg telemetry, train models, and print a terminal forecast summary."""
    print("Synchronizing live Singapore NEA telemetry from data.gov.sg...")
    sync_stats = sync_live_nea_telemetry(days_back=12)
    print(
        f"Cached rows: {sync_stats['total_cached_rows']} | Latest reading: {sync_stats['latest_timestamp']}"
    )

    metrics = forecast_engine.train_all()
    print(
        f"Holdout Backtest -> MAE: {metrics['mae_psi']:.2f} PSI | "
        f"RMSE: {metrics['rmse_psi']:.2f} PSI | R2: {metrics['r2_score']:.3f} | "
        f"80% Quantile Coverage: {metrics['interval_coverage_pct']:.1f}%"
    )

    fc = forecast_engine.forecast_short_term(
        horizon_hours=horizon_hours, wind_regime=wind_regime
    )
    print("\n--- Multi-Horizon Regional Forecast (24-Hr PSI / 1-Hr PM2.5) ---")
    header = f"{'Region':<10} {'Live':<14} {'+6h':<14} {'+12h':<14} {'+24h':<14} {'+48h':<14} {'+72h':<14}"
    print(header)
    print("-" * len(header))

    for reg, live_info in fc["current_regions"].items():
        steps = fc["regional_forecasts"][reg]
        live_str = f"{live_info['psi_24h']:.1f} ({live_info['pm25_1h']:.0f})"
        h6 = f"{steps[5]['psi_24h']:.1f} ({steps[5]['pm25_1h']:.0f})"
        h12 = f"{steps[11]['psi_24h']:.1f} ({steps[11]['pm25_1h']:.0f})"
        h24 = f"{steps[23]['psi_24h']:.1f} ({steps[23]['pm25_1h']:.0f})"
        h48 = f"{steps[47]['psi_24h']:.1f} ({steps[47]['pm25_1h']:.0f})"
        h72 = f"{steps[71]['psi_24h']:.1f} ({steps[71]['pm25_1h']:.0f})"
        print(f"{live_info['label']:<10} {live_str:<14} {h6:<14} {h12:<14} {h24:<14} {h48:<14} {h72:<14}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Singapore Atmospheric & Haze Intelligence (Data Pipeline + ML Forecaster + Web Workbench)"
    )
    parser.add_argument(
        "--mode",
        choices=["serve", "cli"],
        default="serve",
        help="Run interactive FastAPI web workbench ('serve') or print CLI forecast summary ('cli').",
    )
    parser.add_argument("--port", type=int, default=8050, help="Port for web server.")
    parser.add_argument(
        "--regime",
        choices=["baseline", "sumatra_sw", "monsoon_rain"],
        default="baseline",
        help="Atmospheric wind regime for CLI forecast.",
    )
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli_report(horizon_hours=72, wind_regime=args.regime)
        return 0

    uvicorn.run("src.api:app", host="127.0.0.1", port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
