"""Automated test suite for Singapore Atmospheric, Wind Vector & Haze Intelligence."""

from fastapi.testclient import TestClient

from src.api import app
from src.config import (
    REGIONS,
    classify_pm25_band,
    classify_psi_band,
    compute_sumatra_advection_index,
    degrees_to_cardinal,
)
from src.data_fetcher import load_hourly_dataframe, load_monthly_climate_dataframe
from src.forecaster import forecast_engine


def test_nea_band_and_wind_helpers() -> None:
    assert classify_psi_band(42.0)["band"] == "Good"
    assert classify_psi_band(75.0)["band"] == "Moderate"
    assert classify_psi_band(135.0)["band"] == "Unhealthy"
    assert classify_psi_band(245.0)["band"] == "Very Unhealthy"
    assert classify_psi_band(320.0)["band"] == "Hazardous"
    assert classify_pm25_band(30.0)["band"] == "Normal"
    assert classify_pm25_band(110.0)["band"] == "Elevated"

    assert degrees_to_cardinal(0.0) == "N"
    assert degrees_to_cardinal(225.0) == "SW"
    # SW wind (230 deg) must produce positive Sumatra advection; NE wind (50 deg) must produce negative
    assert compute_sumatra_advection_index(12.0, 230.0) > 10.0
    assert compute_sumatra_advection_index(12.0, 50.0) < -10.0


def test_hourly_wind_and_seasonal_dataframes() -> None:
    df_hourly = load_hourly_dataframe(limit_hours=72)
    assert not df_hourly.empty
    assert set(df_hourly["region"].unique()) == set(REGIONS)
    assert "wind_speed_kn" in df_hourly.columns
    assert "wind_dir_deg" in df_hourly.columns

    df_monthly = load_monthly_climate_dataframe()
    assert len(df_monthly) >= 120
    sep_2015 = df_monthly[(df_monthly["year"] == 2015) & (df_monthly["month"] == 9)]
    assert not sep_2015.empty
    assert float(sep_2015.iloc[0]["max_psi"]) > 300.0


def test_wind_aware_forecaster_and_counterfactual_predictions() -> None:
    metrics = forecast_engine.train_all(trigger_source="pytest-verification")
    assert metrics["mae_psi"] >= 0.0
    assert metrics["wind_gain_pct"] > 0.0
    assert len(metrics["training_history"]) >= 1

    # Compare counterfactual SW Sumatra gale (230 deg, 16 kn, hotspot 2.0) vs NE sea breeze flush (40 deg, 16 kn, hotspot 0.0)
    fc_sw = forecast_engine.forecast_short_term(
        horizon_hours=72,
        wind_regime="baseline",
        sim_wind_speed_kn=16.0,
        sim_wind_dir_deg=230.0,
        sim_hotspot_intensity=2.0,
    )
    fc_ne = forecast_engine.forecast_short_term(
        horizon_hours=72,
        wind_regime="baseline",
        sim_wind_speed_kn=16.0,
        sim_wind_dir_deg=40.0,
        sim_hotspot_intensity=0.0,
    )

    sim_sw = fc_sw["counterfactual_simulation"]
    sim_ne = fc_ne["counterfactual_simulation"]
    assert sim_sw["simulated_72h_peak_psi"] > sim_ne["simulated_72h_peak_psi"]
    assert sim_sw["delta_24h_psi"] > sim_ne["delta_24h_psi"]
    assert len(sim_sw["directional_sensitivity_rose"]) == 8


def test_fastapi_endpoints_and_continuous_sync() -> None:
    with TestClient(app) as client:
        resp_root = client.get("/")
        assert resp_root.status_code == 200
        assert "Calculated Predictions on Predictions" in resp_root.text

        resp_api = client.get(
            "/api/overview?horizon_hours=72&wind_regime=baseline&sim_wind_speed_kn=14.0&sim_wind_dir_deg=225&sim_hotspot_intensity=1.5"
        )
        assert resp_api.status_code == 200
        data = resp_api.json()
        assert data["status"] == "ok"
        assert "counterfactual_simulation" in data["short_term"]
        assert "continuous_learning" in data
        assert len(data["diagnostics"]["metrics"]["training_history"]) >= 1
