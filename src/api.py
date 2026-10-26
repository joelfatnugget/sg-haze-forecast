"""FastAPI server for Singapore Atmospheric & Haze Intelligence with background continuous learning."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import BASE_DIR, REGION_METADATA
from src.data_fetcher import sync_live_nea_telemetry
from src.forecaster import forecast_engine

logger = logging.getLogger(__name__)
SGT = timezone(timedelta(hours=8))

STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

CONTINUOUS_STATE: Dict[str, Any] = {
    "auto_sync_enabled": True,
    "interval_seconds": 1800,
    "schedule_mode": "smart-hourly-aligned (30m check / retrain on new NEA hour)",
    "last_sync_at": None,
    "next_sync_at": None,
    "last_trained_nea_timestamp": None,
    "last_sync_stats": {},
    "task": None,
}


def _seconds_until_next_nea_slot(now_sgt: datetime) -> Tuple[int, datetime]:
    """Calculate seconds until the next :05 or :35 past the hour (aligned to NEA's hourly publication schedule)."""
    minute = now_sgt.minute
    if minute < 5:
        target = now_sgt.replace(minute=5, second=0, microsecond=0)
    elif minute < 35:
        target = now_sgt.replace(minute=35, second=0, microsecond=0)
    else:
        target = (now_sgt + timedelta(hours=1)).replace(minute=5, second=0, microsecond=0)
    wait_sec = max(60, int((target - now_sgt).total_seconds()))
    return wait_sec, target


async def _background_learning_loop() -> None:
    """Poll data.gov.sg at :05 and :35 past each hour, and retrain only when a new hourly NEA reading arrives."""
    while True:
        now_sgt = datetime.now(SGT)
        wait_sec, next_target = _seconds_until_next_nea_slot(now_sgt)
        CONTINUOUS_STATE["interval_seconds"] = wait_sec
        CONTINUOUS_STATE["next_sync_at"] = next_target.isoformat()

        await asyncio.sleep(wait_sec)
        if not CONTINUOUS_STATE["auto_sync_enabled"]:
            continue
        try:
            sync_stats = await asyncio.to_thread(sync_live_nea_telemetry, 2, True)
            latest_nea_ts = sync_stats.get("latest_timestamp")
            now_after = datetime.now(SGT)
            CONTINUOUS_STATE["last_sync_at"] = now_after.isoformat()
            CONTINUOUS_STATE["last_sync_stats"] = sync_stats

            # Only retrain the ML models if NEA has published a newer hourly timestamp
            if latest_nea_ts and latest_nea_ts != CONTINUOUS_STATE["last_trained_nea_timestamp"]:
                await asyncio.to_thread(
                    forecast_engine.train_all, f"hourly-nea-update ({latest_nea_ts[11:16]} SGT)"
                )
                CONTINUOUS_STATE["last_trained_nea_timestamp"] = latest_nea_ts
        except Exception as exc:
            logger.warning("Background learning loop error: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Train models immediately on cached SQLite data for sub-second startup, then sync live NEA data asynchronously."""
    forecast_engine.train_all(trigger_source="startup-cached")
    now_sgt = datetime.now(SGT)
    wait_sec, next_target = _seconds_until_next_nea_slot(now_sgt)
    CONTINUOUS_STATE["interval_seconds"] = wait_sec
    CONTINUOUS_STATE["last_sync_at"] = now_sgt.isoformat()
    CONTINUOUS_STATE["next_sync_at"] = next_target.isoformat()

    async def _initial_and_periodic_loop() -> None:
        try:
            sync_stats = await asyncio.to_thread(sync_live_nea_telemetry, 12, False)
            latest_ts = sync_stats.get("latest_timestamp")
            CONTINUOUS_STATE["last_sync_stats"] = sync_stats
            if latest_ts != CONTINUOUS_STATE["last_trained_nea_timestamp"]:
                await asyncio.to_thread(forecast_engine.train_all, "startup-live-sync")
                CONTINUOUS_STATE["last_trained_nea_timestamp"] = latest_ts
        except Exception as exc:
            logger.warning("Initial background sync warning: %s", exc)
        await _background_learning_loop()

    bg_task = asyncio.create_task(_initial_and_periodic_loop())
    CONTINUOUS_STATE["task"] = bg_task
    try:
        yield
    finally:
        bg_task.cancel()


app = FastAPI(
    title="Singapore Atmospheric & Haze Intelligence API",
    description="Live NEA PSI/PM2.5/Wind telemetry ingestion, 72-hour wind-vector quantile forecasting, counterfactual sandbox, and continuous learning.",
    version="2.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_dashboard() -> FileResponse:
    """Serve the primary interactive telemetry and forecasting workbench."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/overview")
async def get_overview(
    horizon_hours: int = Query(default=72, ge=12, le=72),
    wind_regime: str = Query(default="baseline", pattern="^(baseline|sumatra_sw|monsoon_rain)$"),
    sim_wind_speed_kn: Optional[float] = Query(default=None, ge=0.5, le=28.0),
    sim_wind_dir_deg: Optional[float] = Query(default=None, ge=0.0, le=360.0),
    sim_hotspot_intensity: float = Query(default=0.0, ge=0.0, le=3.0),
    enso_oni: float = Query(default=0.35, ge=-2.0, le=2.5),
    iod_dmi: float = Query(default=0.22, ge=-1.5, le=2.0),
    rainfall_anomaly_pct: float = Query(default=-10.0, ge=-80.0, le=60.0),
) -> JSONResponse:
    """Return short-term 72h forecast, counterfactual wind simulation, seasonal 12m outlook, and continuous learning diagnostics."""
    short_term = forecast_engine.forecast_short_term(
        horizon_hours=horizon_hours,
        wind_regime=wind_regime,
        sim_wind_speed_kn=sim_wind_speed_kn,
        sim_wind_dir_deg=sim_wind_dir_deg,
        sim_hotspot_intensity=sim_hotspot_intensity,
    )
    seasonal = forecast_engine.forecast_seasonal_outlook(
        enso_oni=enso_oni,
        iod_dmi=iod_dmi,
        rainfall_anomaly_pct=rainfall_anomaly_pct,
    )
    return JSONResponse(
        {
            "status": "ok",
            "region_metadata": REGION_METADATA,
            "short_term": short_term,
            "seasonal": seasonal,
            "continuous_learning": {
                "auto_sync_enabled": CONTINUOUS_STATE["auto_sync_enabled"],
                "interval_seconds": CONTINUOUS_STATE["interval_seconds"],
                "schedule_mode": CONTINUOUS_STATE["schedule_mode"],
                "last_sync_at": CONTINUOUS_STATE["last_sync_at"],
                "next_sync_at": CONTINUOUS_STATE["next_sync_at"],
                "last_trained_nea_timestamp": CONTINUOUS_STATE["last_trained_nea_timestamp"],
                "last_sync_stats": CONTINUOUS_STATE["last_sync_stats"],
            },
            "diagnostics": {
                "metrics": forecast_engine.backtest_metrics,
                "feature_importances": forecast_engine.feature_importances,
                "last_trained_at": forecast_engine.last_trained_at,
                "total_retrain_cycles": forecast_engine.total_retrain_cycles,
            },
        }
    )


@app.post("/api/sync")
async def trigger_live_sync(
    days_back: int = Query(default=2, ge=1, le=25),
    trigger_label: str = Query(default="manual-live-retrain"),
    force_retrain: bool = Query(default=False),
) -> Dict[str, Any]:
    """Re-synchronize with data.gov.sg live endpoints; retrain only if new hourly data arrived or forced."""
    sync_stats = await asyncio.to_thread(sync_live_nea_telemetry, days_back, True)
    latest_ts = sync_stats.get("latest_timestamp")
    now_sgt = datetime.now(SGT)
    wait_sec, next_target = _seconds_until_next_nea_slot(now_sgt)
    CONTINUOUS_STATE["interval_seconds"] = wait_sec
    CONTINUOUS_STATE["last_sync_at"] = now_sgt.isoformat()
    CONTINUOUS_STATE["next_sync_at"] = next_target.isoformat()
    CONTINUOUS_STATE["last_sync_stats"] = sync_stats

    retrained = False
    if force_retrain or latest_ts != CONTINUOUS_STATE["last_trained_nea_timestamp"]:
        metrics = await asyncio.to_thread(forecast_engine.train_all, trigger_label)
        CONTINUOUS_STATE["last_trained_nea_timestamp"] = latest_ts
        retrained = True
    else:
        metrics = forecast_engine.backtest_metrics

    return {
        "status": "retrained" if retrained else "up-to-date (no new NEA hour)",
        "retrained": retrained,
        "sync_stats": sync_stats,
        "metrics": {k: v for k, v in metrics.items() if k != "validation_series"},
    }
