#!/usr/bin/env python3
"""Generate an interactive full-race F1 tire and lap-consistency dashboard.

Example:
    python f1_full_race_dashboard.py --year 2026 \
        --event "Hungarian Grand Prix" --drivers NOR VER

The output is a self-contained Plotly HTML file plus a CSV summary. FastF1
caching is enabled and network operations use retry/timeout protection.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import time
from pathlib import Path
from typing import Callable, TypeVar

import fastf1
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots

T = TypeVar("T")
LOGGER = logging.getLogger("f1-full-race")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
CACHE = Path(os.environ.get("FASTF1_CACHE_DIR", "~/.cache/fastf1")).expanduser()
TIMEOUT = int(os.environ.get("FASTF1_TIMEOUT", "180"))
RETRIES = int(os.environ.get("FASTF1_RETRIES", "3"))


def timed(operation: Callable[[], T], label: str) -> T:
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            LOGGER.info("%s (attempt %d/%d)", label, attempt, RETRIES)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(operation).result(timeout=TIMEOUT)
        except concurrent.futures.TimeoutError:
            last = TimeoutError(f"{label} exceeded {TIMEOUT} seconds")
        except (requests.RequestException, ConnectionError, OSError) as exc:
            last = exc
        except Exception as exc:
            last = exc
        LOGGER.warning("%s failed: %s", label, last)
        if attempt < RETRIES:
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Unable to complete {label}") from last


def prepare_race_laps(session: fastf1.core.Session, drivers: list[str]) -> pd.DataFrame:
    rows = []
    for driver in drivers:
        laps = session.laps.pick_drivers(driver).copy()
        if laps.empty:
            raise ValueError(f"No race laps found for {driver}")
        laps["Driver"] = driver
        rows.append(laps)
    data = pd.concat(rows, ignore_index=True)
    data["LapTimeSeconds"] = data["LapTime"].dt.total_seconds()
    data["PositionNumeric"] = pd.to_numeric(data["Position"], errors="coerce")
    data["Stint"] = pd.to_numeric(data["Stint"], errors="coerce")
    data["TyreLife"] = pd.to_numeric(data["TyreLife"], errors="coerce")
    # Exclude pit/in-laps, out-laps, safety-car anomalies, and incomplete laps.
    valid = data[
        data["LapTimeSeconds"].between(60, 180)
        & data["Compound"].notna()
        & data["Stint"].notna()
        & data["LapNumber"].notna()
    ].copy()
    valid["LapNumber"] = pd.to_numeric(valid["LapNumber"], errors="coerce")
    return valid.sort_values(["Driver", "LapNumber"])


def summarize_stints(laps: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (driver, stint, compound), group in laps.groupby(["Driver", "Stint", "Compound"], dropna=False):
        group = group.sort_values("LapNumber").copy()
        if len(group) < 3:
            continue
        x = group["TyreLife"].fillna(group["LapNumber"] - group["LapNumber"].min()).to_numpy()
        y = group["LapTimeSeconds"].to_numpy()
        slope = float(np.polyfit(x, y, 1)[0]) if len(np.unique(x)) > 1 else np.nan
        median = float(np.median(y))
        mad = float(np.median(np.abs(y - median)))
        summaries.append({
            "Driver": driver,
            "Stint": int(stint),
            "Compound": str(compound),
            "Laps": len(group),
            "StartLap": int(group["LapNumber"].min()),
            "EndLap": int(group["LapNumber"].max()),
            "MedianLapTimeSeconds": median,
            "MeanLapTimeSeconds": float(y.mean()),
            "LapTimeStdSeconds": float(y.std(ddof=1)),
            "LapTimeMADSeconds": mad,
            "DegradationSecondsPerLap": slope,
        })
    return pd.DataFrame(summaries)


def build_dashboard(session, laps: pd.DataFrame, output: Path) -> pd.DataFrame:
    summary = summarize_stints(laps)
    summary.to_csv(output.with_name("f1_full_race_stint_summary.csv"), index=False)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.09,
        subplot_titles=("Race lap time by compound and stint", "Stint-normalized pace and degradation", "Lap-time consistency: rolling variability"),
    )
    colors = {"NOR": "#1f77b4", "VER": "#ff7f0e"}
    symbols = {"SOFT": "circle", "MEDIUM": "diamond", "HARD": "square", "INTERMEDIATE": "triangle-up", "WET": "x"}

    for driver, group in laps.groupby("Driver"):
        color = colors.get(driver, None)
        custom = np.column_stack([group["LapNumber"], group["Compound"], group["Stint"], group["TyreLife"], group["LapTimeSeconds"]])
        fig.add_trace(go.Scatter(
            x=group["LapNumber"], y=group["LapTimeSeconds"], mode="markers+lines",
            name=f"{driver} lap time", line={"color": color},
            marker={"color": color, "size": 7},
            customdata=custom,
            hovertemplate="Lap %{customdata[0]:.0f}<br>Time: %{customdata[4]:.3f}s<br>Compound: %{customdata[1]}<br>Stint: %{customdata[2]:.0f}<br>Tyre age: %{customdata[3]:.0f}<extra>" + driver + "</extra>",
        ), row=1, col=1)
        rolling = group.set_index("LapNumber")["LapTimeSeconds"].rolling(5, min_periods=3).std()
        fig.add_trace(go.Scatter(x=rolling.index, y=rolling.values, mode="lines", name=f"{driver} rolling σ", line={"color": color}, hovertemplate="Lap %{x:.0f}<br>5-lap σ: %{y:.3f}s<extra>" + driver + "</extra>"), row=3, col=1)

    for _, stint in summary.iterrows():
        driver = stint["Driver"]
        group = laps[(laps["Driver"] == driver) & (laps["Stint"] == stint["Stint"])]
        base = group["LapTimeSeconds"].median()
        normalized = group["LapTimeSeconds"] - base
        fig.add_trace(go.Scatter(
            x=group["TyreLife"], y=normalized, mode="lines+markers",
            name=f"{driver} stint {int(stint['Stint'])} {stint['Compound']}",
            line={"color": colors.get(driver), "dash": "solid" if driver == "NOR" else "dash"},
            marker={"symbol": symbols.get(stint["Compound"], "circle")},
            hovertemplate="Tyre age %{x:.0f}<br>Δ from stint median: %{y:.3f}s<extra>" + driver + " " + str(stint["Compound"]) + "</extra>",
        ), row=2, col=1)

    event = str(session.event["EventName"])
    fig.update_yaxes(title_text="Lap time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Δ from stint median (s)", row=2, col=1)
    fig.update_yaxes(title_text="Rolling 5-lap σ (s)", row=3, col=1)
    fig.update_xaxes(title_text="Race lap", row=1, col=1)
    fig.update_xaxes(title_text="Tyre age (laps)", row=2, col=1)
    fig.update_xaxes(title_text="Race lap", row=3, col=1)
    fig.update_layout(title=f"Full-Race Tire and Lap Consistency — {event}", height=1050, template="plotly_white", hovermode="closest", legend={"orientation": "h", "y": -0.05})
    fig.write_html(output, include_plotlyjs=True, full_html=True)
    LOGGER.info("Wrote %s", output.resolve())
    LOGGER.info("Wrote %s", output.with_name("f1_full_race_stint_summary.csv").resolve())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--event", default="Hungarian Grand Prix")
    parser.add_argument("--drivers", nargs=2, default=["NOR", "VER"])
    parser.add_argument("--output", type=Path, default=Path("f1_full_race_dashboard.html"))
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))
    session = timed(lambda: fastf1.get_session(args.year, args.event, "R"), "Creating race session")
    timed(lambda: session.load(telemetry=False, weather=False, messages=False), "Loading race timing data")
    drivers = [driver.upper() for driver in args.drivers]
    available = set(session.results["Abbreviation"].dropna().astype(str).str.upper())
    unknown = sorted(set(drivers) - available)
    if unknown:
        raise ValueError(f"Drivers not found: {', '.join(unknown)}")
    laps = prepare_race_laps(session, drivers)
    summary = build_dashboard(session, laps, args.output)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
