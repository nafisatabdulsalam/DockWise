from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


PVGIS_ENDPOINT = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
SITE = {
    "name": "Ilorin, Nigeria",
    "latitude": 8.4966,
    "longitude": 4.5421,
    "timezone": "Africa/Lagos",
    "coordinate_status": "Assumed city-centre point; fixed before comparative evaluation",
}


def download_pvgis(
    raw_path: str | Path,
    processed_path: str | Path,
    *,
    start_year: int = 2019,
    end_year: int = 2023,
    peak_power_kw: float = 0.03,
    database: str = "PVGIS-SARAH3",
) -> dict[str, Any]:
    """Download once, preserve the exact response, and derive an hourly local-time cache."""
    raw_path = Path(raw_path)
    processed_path = Path(processed_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    params: dict[str, Any] = {
        "lat": SITE["latitude"],
        "lon": SITE["longitude"],
        "startyear": start_year,
        "endyear": end_year,
        "pvcalculation": 1,
        "peakpower": peak_power_kw,
        "loss": 0,
        "angle": 10,
        "aspect": 0,
        "raddatabase": database,
        "components": 1,
        "outputformat": "json",
    }
    response = requests.get(PVGIS_ENDPOINT, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    metadata = process_pvgis_payload(payload, processed_path, endpoint=response.url, database=database)
    metadata["raw_path"] = str(raw_path)
    processed_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    return metadata


def process_pvgis_payload(
    payload: dict[str, Any],
    processed_path: str | Path,
    *,
    endpoint: str = PVGIS_ENDPOINT,
    database: str = "PVGIS-SARAH3",
) -> dict[str, Any]:
    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    hourly = pd.DataFrame(payload["outputs"]["hourly"])
    timestamp_utc = pd.to_datetime(hourly["time"], format="%Y%m%d:%H%M", utc=True)
    timestamp_local_native = timestamp_utc.dt.tz_convert(SITE["timezone"])
    # SARAH timestamps may have a longitude-dependent minute offset.
    # Keep it, while using floored local clock-hour labels for causal block indexing.
    timestamp_local = timestamp_local_native.dt.floor("h")
    frame = pd.DataFrame(
        {
            "timestamp_pvgis_utc": timestamp_utc,
            "timestamp_pvgis_local": timestamp_local_native,
            "timestamp_local": timestamp_local,
            "pv_power_w": pd.to_numeric(hourly["P"], errors="coerce").clip(lower=0.0),
            "global_irradiance_wm2": pd.to_numeric(hourly.get("G(i)", np.nan), errors="coerce"),
            "air_temperature_c": pd.to_numeric(hourly.get("T2m", np.nan), errors="coerce"),
            "wind_speed_mps": pd.to_numeric(hourly.get("WS10m", np.nan), errors="coerce"),
        }
    )
    if frame["pv_power_w"].isna().any():
        raise ValueError("PVGIS response contains missing PV power values")
    # Objective label: 48-hour energy below the training-period 10th percentile.
    training = frame[frame["timestamp_local"].dt.year.isin([2019, 2020])]
    threshold = float(training["pv_power_w"].rolling(48, min_periods=48).sum().quantile(0.10))
    rolling = frame["pv_power_w"].rolling(48, min_periods=48).sum()
    frame["low_solar"] = rolling.le(threshold)
    frame.to_parquet(processed_path, index=False)
    metadata = {
        "endpoint": endpoint,
        "coordinates": SITE,
        "requested_database": database,
        "api_version": "5.3",
        "pv_power_units": "W for a 0.03 kWp array at the stated conversion boundary",
        "irradiance_units": "W/m^2",
        "temperature_units": "degC",
        "wind_speed_units": "m/s",
        "processed_path": str(processed_path),
        "rows": len(frame),
        "start": str(frame["timestamp_local"].min()),
        "end": str(frame["timestamp_local"].max()),
        "low_solar_48h_training_p10_wh": threshold,
        "preprocessing": "Preserved native PVGIS UTC/local timestamps; floored local timestamp to clock hour for indexing; no interpolation; clipped negative P to zero; labelled low solar at the training-period 10th percentile of trailing 48-hour energy.",
        "response_meta": payload.get("meta", {}),
        "response_inputs": payload.get("inputs", {}),
    }
    return metadata


def process_cached_raw(raw_path: str | Path, processed_path: str | Path) -> dict[str, Any]:
    raw_path = Path(raw_path)
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    metadata = process_pvgis_payload(payload, processed_path)
    metadata["raw_path"] = str(raw_path)
    Path(processed_path).with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    return metadata


def inspect_cache(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    frame = pd.read_parquet(path)
    frame["timestamp_local"] = pd.to_datetime(frame["timestamp_local"])
    return {
        "rows": len(frame),
        "years": sorted(int(value) for value in frame["timestamp_local"].dt.year.unique()),
        "missing_power": int(frame["pv_power_w"].isna().sum()),
        "duplicate_timestamps": int(frame["timestamp_local"].duplicated().sum()),
        "maximum_power_w": float(frame["pv_power_w"].max()),
    }

