#!/usr/bin/env python3
"""Interactive F1 telemetry dashboard with corner-level speed analysis.

Default example uses the 2026 Hungarian Grand Prix and Norris/Verstappen.
Change the event and drivers with command-line arguments:

    python f1_interactive_dashboard.py \
        --year 2024 --event "Monaco Grand Prix" --drivers LEC PIA

Install dependencies:
    python -m pip install fastf1 pandas numpy scipy plotly requests
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
from scipy.signal import find_peaks

T = TypeVar("T")
CACHE_DIR = Path(os.environ.get("FASTF1_CACHE_DIR", "~/.cache/fastf1")).expanduser()
TIMEOUT = int(os.environ.get("FASTF1_TIMEOUT", "180"))
RETRIES = int(os.environ.get("FASTF1_RETRIES", "3"))
LOGGER = logging.getLogger("f1-interactive")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


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


def load_telemetry(session: fastf1.core.Session, driver: str) -> tuple[pd.Series, pd.DataFrame]:
    lap = session.laps.pick_drivers(driver).pick_fastest()
    if lap is None or pd.isna(lap.get("LapTime")):
        raise RuntimeError(f"No valid fastest lap found for {driver}")

    # get_telemetry() combines car and position channels, which lets us map
    # circuit corner coordinates to distance along the lap when available.
    telemetry = lap.get_telemetry().add_distance()
    required = ["Distance", "Speed", "Throttle", "Brake", "nGear"]
    missing = [col for col in required if col not in telemetry.columns]
    if missing:
        raise RuntimeError(f"Telemetry for {driver} is missing {missing}")
    keep = required + [col for col in ["X", "Y"] if col in telemetry.columns]
    telemetry = telemetry[keep].copy()
    telemetry = telemetry.dropna(subset=["Distance", "Speed"])
    telemetry = telemetry[telemetry["Distance"].diff().fillna(1) >= 0]
    telemetry = telemetry.drop_duplicates("Distance")
    telemetry["Throttle"] = pd.to_numeric(telemetry["Throttle"], errors="coerce").clip(0, 100)
    brake = pd.to_numeric(telemetry["Brake"], errors="coerce")
    if brake.dropna().max() <= 1:
        brake = brake * 100
    telemetry["Brake"] = brake.fillna(0).clip(0, 100)
    telemetry["nGear"] = pd.to_numeric(telemetry["nGear"], errors="coerce")
    return lap, telemetry.dropna(subset=["Throttle", "nGear"])


def corner_distances(session: fastf1.core.Session, reference: pd.DataFrame) -> list[tuple[str, float]]:
    """Estimate corner apex distances from speed minima and label them by turn order.

    FastF1 circuit-corner coordinates and telemetry GPS coordinates can use
    different transforms. An ordered local-minimum method is therefore more
    stable for comparing two laps: it identifies apex-like points directly on
    the telemetry distance axis, then assigns the circuit's official turn
    numbers in track order when available.
    """
    try:
        expected = len(session.get_circuit_info().corners)
    except Exception:
        expected = 14
    expected = max(14, min(expected, 24))
    raw_distance = reference["Distance"].to_numpy()
    raw_speed = reference["Speed"].to_numpy()
    distance = np.arange(raw_distance.min(), raw_distance.max(), 1.0)
    speed = np.interp(distance, raw_distance, raw_speed)
    minima, _ = find_peaks(-speed, distance=100, prominence=1, plateau_size=(1, None))
    candidates = sorted(
        [(int(index), float(speed[index])) for index in minima],
        key=lambda item: item[1],
    )
    chosen: list[int] = []
    for index, _ in candidates:
        if all(abs(distance[index] - distance[other]) >= 180 for other in chosen):
            chosen.append(index)
        if len(chosen) == expected:
            break
    chosen = sorted(chosen, key=lambda index: distance[index])
    return [(f"T{i}", float(distance[index])) for i, index in enumerate(chosen, start=1)]


def analyze_corners(common: pd.DataFrame, corners: list[tuple[str, float]]) -> pd.DataFrame:
    rows = []
    for name, center in corners:
        start, end = center - 90, center + 120
        segment = common[(common["Distance"] >= start) & (common["Distance"] <= end)]
        if segment.empty:
            continue
        # Positive time_gain means Norris is faster over the segment.
        nor_speed = segment["NOR_Speed"].clip(lower=1) / 3.6
        ver_speed = segment["VER_Speed"].clip(lower=1) / 3.6
        dt_gain = float(np.trapezoid(1 / ver_speed - 1 / nor_speed, segment["Distance"]))
        speed_delta = float((segment["NOR_Speed"] - segment["VER_Speed"]).mean())
        rows.append({
            "corner": name,
            "distance_m": round(center, 1),
            "norris_mean_speed_kmh": round(float(segment["NOR_Speed"].mean()), 2),
            "verstappen_mean_speed_kmh": round(float(segment["VER_Speed"].mean()), 2),
            "norris_speed_delta_kmh": round(speed_delta, 2),
            "estimated_norris_time_gain_s": round(dt_gain, 4),
        })
    return pd.DataFrame(rows).sort_values("estimated_norris_time_gain_s", ascending=False)


def build_dashboard(session, telemetry: dict[str, pd.DataFrame], corners: list[tuple[str, float]], output: Path) -> None:
    ref = telemetry["NOR"]
    end = min(float(df["Distance"].max()) for df in telemetry.values())
    distance = np.linspace(0, end, 1800)
    common = pd.DataFrame({"Distance": distance})
    for driver, df in telemetry.items():
        for channel in ["Speed", "Throttle", "Brake", "nGear"]:
            common[f"{driver}_{channel}"] = np.interp(distance, df["Distance"], df[channel])

    corner_table = analyze_corners(common, corners)
    corner_table.to_csv(output.with_name("f1_corner_analysis.csv"), index=False)
    top_corners = corner_table.head(5)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045,
        specs=[[{}], [{}], [{}], [{"secondary_y": True}]],
        subplot_titles=("Speed", "Throttle", "Brake", "Throttle delta and gear selection"),
    )
    colors = {"NOR": "#1f77b4", "VER": "#ff7f0e"}
    custom_template = np.column_stack([
        distance,
        common["NOR_Speed"], common["NOR_Throttle"], common["NOR_Brake"], common["NOR_nGear"],
        common["VER_Speed"], common["VER_Throttle"], common["VER_Brake"], common["VER_nGear"],
    ])
    hover = (
        "Distance: %{customdata[0]:.0f} m<br>"
        "Speed: %{customdata[1]:.1f} km/h<br>"
        "Throttle: %{customdata[2]:.0f}%<br>"
        "Brake: %{customdata[3]:.0f}%<br>Gear: %{customdata[4]:.0f}<extra>Norris</extra>"
    )
    hover_ver = hover.replace("<extra>Norris</extra>", "<extra>Verstappen</extra>")
    for row, channel in zip([1, 2, 3], ["Speed", "Throttle", "Brake"]):
        fig.add_trace(go.Scatter(x=distance, y=common[f"NOR_{channel}"], name=f"NOR — {channel}", line={"color": colors["NOR"]}, customdata=custom_template[:, [0, 1, 2, 3, 4]], hovertemplate=hover), row=row, col=1)
        fig.add_trace(go.Scatter(x=distance, y=common[f"VER_{channel}"], name=f"VER — {channel}", line={"color": colors["VER"]}, customdata=custom_template[:, [0, 5, 6, 7, 8]], hovertemplate=hover_ver), row=row, col=1)

    throttle_delta = common["NOR_Throttle"] - common["VER_Throttle"]
    fig.add_trace(go.Scatter(x=distance, y=throttle_delta, name="NOR − VER throttle delta", line={"color": "#2ca02c"}, hovertemplate="Distance: %{x:.0f} m<br>Throttle delta: %{y:.1f} percentage points<extra></extra>"), row=4, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=distance, y=common["NOR_nGear"], name="NOR gear", line={"color": colors["NOR"], "dash": "dot"}, hovertemplate="Distance: %{x:.0f} m<br>NOR gear: %{y:.0f}<extra></extra>"), row=4, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=distance, y=common["VER_nGear"], name="VER gear", line={"color": colors["VER"], "dash": "dot"}, hovertemplate="Distance: %{x:.0f} m<br>VER gear: %{y:.0f}<extra></extra>"), row=4, col=1, secondary_y=True)

    for name, center in corners:
        fig.add_vline(x=center, line_width=0.6, line_dash="dot", line_color="gray", row=1, col=1)
    for _, row in top_corners.iterrows():
        fig.add_annotation(x=row["distance_m"], y=1.02, yref="paper", text=row["corner"], showarrow=False, font={"size": 9}, bgcolor="rgba(255,255,255,0.7)")

    event = str(session.event["EventName"])
    fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
    fig.update_yaxes(title_text="Throttle (%)", row=2, col=1, range=[0, 105])
    fig.update_yaxes(title_text="Brake (%)", row=3, col=1, range=[0, 105])
    fig.update_yaxes(title_text="Throttle Δ (pp)", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Gear", row=4, col=1, secondary_y=True, dtick=1)
    fig.update_xaxes(title_text="Distance (m)", row=4, col=1)
    fig.update_layout(title=f"Interactive F1 Telemetry — {event} — Norris vs Verstappen", height=1150, hovermode="x unified", template="plotly_white", legend={"orientation": "h", "y": -0.06})
    fig.write_html(output, include_plotlyjs=True, full_html=True)
    LOGGER.info("Wrote interactive dashboard: %s", output.resolve())
    LOGGER.info("Wrote corner analysis: %s", output.with_name("f1_corner_analysis.csv").resolve())
    print("\nTop Norris gains by corner (positive estimated time gain):")
    print(top_corners.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--event", default="Hungarian Grand Prix")
    parser.add_argument("--drivers", nargs=2, default=["NOR", "VER"])
    parser.add_argument("--output", type=Path, default=Path("f1_interactive_telemetry_dashboard.html"))
    args = parser.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    session = timed(lambda: fastf1.get_session(args.year, args.event, "R"), "Creating race session")
    timed(lambda: session.load(telemetry=True, weather=False, messages=False), "Loading race telemetry")
    telemetry = {}
    laps = {}
    for driver in [d.upper() for d in args.drivers]:
        lap, df = timed(lambda driver=driver: load_telemetry(session, driver), f"Loading {driver} fastest lap")
        laps[driver] = lap
        telemetry[driver] = df
    if set(telemetry) != {"NOR", "VER"}:
        raise ValueError("Corner analysis currently expects the two drivers NOR and VER.")
    corners = corner_distances(session, telemetry["NOR"])
    build_dashboard(session, telemetry, corners, args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        LOGGER.error("Dashboard failed: %s", exc)
        raise SystemExit(1) from exc
