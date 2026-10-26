"""Configuration and NEA Air Quality & Wind Telemetry constants for Singapore."""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
IS_SERVERLESS = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

if IS_SERVERLESS:
    DATA_DIR = Path("/tmp/sg_haze_data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DB_PATH = DATA_DIR / "sg_haze_telemetry.sqlite"
    BUNDLED_DB_PATH = BASE_DIR / "data" / "sg_haze_telemetry.sqlite"
    if BUNDLED_DB_PATH.exists() and not CACHE_DB_PATH.exists():
        shutil.copy2(BUNDLED_DB_PATH, CACHE_DB_PATH)
else:
    DATA_DIR = BASE_DIR / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DB_PATH = DATA_DIR / "sg_haze_telemetry.sqlite"

MODEL_ARTIFACTS_DIR = DATA_DIR / "models"
MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

NEA_PSI_URL = "https://api.data.gov.sg/v1/environment/psi"
NEA_PM25_URL = "https://api.data.gov.sg/v1/environment/pm25"
NEA_WIND_SPEED_URL = "https://api.data.gov.sg/v1/environment/wind-speed"
NEA_WIND_DIR_URL = "https://api.data.gov.sg/v1/environment/wind-direction"

REGIONS = ["north", "south", "east", "west", "central"]

REGION_METADATA: Dict[str, Dict[str, Any]] = {
    "north": {
        "name": "North",
        "station_area": "Woodlands / Yishun / Sembawang",
        "latitude": 1.41803,
        "longitude": 103.8200,
        "wind_stations": ["S104", "S109"],
    },
    "south": {
        "name": "South",
        "station_area": "Sentosa / Pasir Panjang / Marina Barrage",
        "latitude": 1.29587,
        "longitude": 103.8200,
        "wind_stations": ["S60", "S108", "S116", "S102"],
    },
    "east": {
        "name": "East",
        "station_area": "East Coast / Changi / Paya Lebar / Ubin",
        "latitude": 1.35735,
        "longitude": 103.9400,
        "wind_stations": ["S107", "S106", "S06", "S199"],
    },
    "west": {
        "name": "West",
        "station_area": "Jurong / Tuas / Tengah / Nanyang",
        "latitude": 1.35735,
        "longitude": 103.7000,
        "wind_stations": ["S44", "S115", "S117", "S23", "S121"],
    },
    "central": {
        "name": "Central",
        "station_area": "Scotts / Clementi / Kim Chuan / AMK",
        "latitude": 1.35735,
        "longitude": 103.8200,
        "wind_stations": ["S111", "S50", "S43", "S109"],
    },
}

# Mapping from individual MSS/NEA wind station IDs to one or more Singapore regions
STATION_TO_REGIONS: Dict[str, List[str]] = {}
for _reg, _meta in REGION_METADATA.items():
    for _st in _meta["wind_stations"]:
        STATION_TO_REGIONS.setdefault(_st, []).append(_reg)


def degrees_to_cardinal(deg: float) -> str:
    """Convert meteorological wind direction in degrees (0-360) to 16-point compass cardinal."""
    dirs = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW",
    ]
    idx = int(round((deg % 360.0) / 22.5)) % 16
    return dirs[idx]


def compute_sumatra_advection_index(wind_speed_kn: float, wind_dir_deg: float) -> float:
    """Calculate Sumatra/Kalimantan transboundary smoke transport index.

    Riau/Jambi/South Sumatra peatlands lie at bearings ~180° (S) to ~285° (WNW) relative to Singapore,
    with peak plume corridor centered around ~230° (SW). Positive values indicate active onshore
    transboundary transport toward Singapore; negative values indicate offshore/Northeast clean sea breeze.
    """
    rad = math.radians(wind_dir_deg - 230.0)
    alignment = math.cos(rad)  # +1.0 when blowing from 230° SW, -1.0 when blowing from 50° NE
    return round(float(wind_speed_kn * alignment), 2)


def classify_psi_band(psi_value: float) -> Dict[str, str]:
    """Return NEA official PSI descriptor, health advisory, and semantic token."""
    if psi_value <= 50:
        return {
            "band": "Good",
            "severity": "good",
            "advisory": "Normal activities across all population groups.",
        }
    if psi_value <= 100:
        return {
            "band": "Moderate",
            "severity": "moderate",
            "advisory": "Normal activities. Sensitive individuals may monitor prolonged outdoor exertion.",
        }
    if psi_value <= 200:
        return {
            "band": "Unhealthy",
            "severity": "unhealthy",
            "advisory": "Reduce prolonged or strenuous outdoor physical exertion. Vulnerable persons should minimize outdoor activity.",
        }
    if psi_value <= 300:
        return {
            "band": "Very Unhealthy",
            "severity": "very-unhealthy",
            "advisory": "Avoid prolonged or strenuous outdoor physical exertion. Vulnerable individuals should stay indoors.",
        }
    return {
        "band": "Hazardous",
        "severity": "hazardous",
        "advisory": "Minimize all outdoor activity. Use N95 respirators if outdoor travel is unavoidable.",
    }


def classify_pm25_band(pm25_value: float) -> Dict[str, str]:
    """Return NEA 1-hour PM2.5 concentration band (Normal, Elevated, High, Very High)."""
    if pm25_value <= 55:
        return {"band": "Normal", "severity": "good"}
    if pm25_value <= 150:
        return {"band": "Elevated", "severity": "moderate"}
    if pm25_value <= 250:
        return {"band": "High", "severity": "unhealthy"}
    return {"band": "Very High", "severity": "hazardous"}
