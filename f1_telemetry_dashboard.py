#!/usr/bin/env python3
"""Compare speed, throttle, and braking inputs across F1 fastest laps.

Examples
--------
Latest completed Grand Prix, top two finishers (default):
    python f1_telemetry_dashboard.py

A specific Grand Prix and two drivers:
    python f1_telemetry_dashboard.py --year 2024 --event "Monaco Grand Prix" \
        --drivers LEC PIA

Three or more drivers from a specific race:
    python f1_telemetry_dashboard.py --year 2023 --event "Belgian Grand Prix" \
        --drivers VER HAM LEC NOR

If --drivers is omitted, the script compares the top two classified finishers.
The event name must match FastF1's schedule, such as "Monaco Grand Prix".

Setup
-----
    python -m venv .venv
    source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
    python -m pip install --upgrade pip
    python -m pip install fastf1 pandas matplotlib numpy requests

Environment variables
---------------------
FASTF1_CACHE_DIR  Cache location (default: ~/.cache/fastf1)
FASTF1_TIMEOUT    Timeout per FastF1 operation in seconds (default: 180)
FASTF1_RETRIES    Number of attempts for transient failures (default: 3)
FASTF1_OUTPUT     Output PNG path (default: latest_f1_telemetry_comparison.png)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

T = TypeVar("T")
CACHE_DIR = Path(os.environ.get("FASTF1_CACHE_DIR", "~/.cache/fastf1")).expanduser()
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("FASTF1_TIMEOUT", "180"))
RETRIES = int(os.environ.get("FASTF1_RETRIES", "3"))
DEFAULT_OUTPUT = Path(
    os.environ.get("FASTF1_OUTPUT", "latest_f1_telemetry_comparison.png")
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def run_with_timeout(operation: Callable[[], T], description: str) -> T:
    """Run an operation with retries and a wall-clock timeout."""
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            LOGGER.info("%s (attempt %d/%d)", description, attempt, RETRIES)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(operation).result(timeout=REQUEST_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            last_error = TimeoutError(
                f"{description} exceeded {REQUEST_TIMEOUT_SECONDS} seconds"
            )
            LOGGER.warning("Network timeout: %s", last_error)
        except (requests.RequestException, ConnectionError, OSError) as exc:
            last_error = exc
            LOGGER.warning("Network-related failure: %s", exc)
        except Exception as exc:
            last_error = exc
            LOGGER.warning("FastF1 operation failed: %s", exc)
        if attempt < RETRIES:
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Unable to complete: {description}") from last_error


def latest_completed_race(year: int | None = None) -> tuple[int, str]:
    """Find the most recent completed Grand Prix in the requested season range."""
    current_year = year or datetime.now(timezone.utc).year
    now = pd.Timestamp.now(tz="UTC")
    for season in range(current_year, current_year - 10, -1):
        schedule = run_with_timeout(
            lambda season=season: fastf1.get_event_schedule(
                season, include_testing=False
            ),
            f"Loading {season} event schedule",
        ).copy()
        schedule["EventDate"] = pd.to_datetime(schedule["EventDate"], utc=True)
        completed = schedule[
            (schedule["EventDate"] <= now)
            & schedule["EventName"].astype(str).str.contains("Grand Prix", na=False)
        ]
        if not completed.empty:
            event = completed.sort_values("EventDate").iloc[-1]
            return season, str(event["EventName"])
    raise RuntimeError("No completed Grand Prix was found in the last ten seasons.")


def get_top_two_finishers(session: fastf1.core.Session) -> list[str]:
    """Return abbreviations of the first two classified race finishers."""
    results = session.results.copy()
    results["PositionNumeric"] = pd.to_numeric(results["Position"], errors="coerce")
    finishers = (
        results.dropna(subset=["PositionNumeric", "Abbreviation"])
        .sort_values("PositionNumeric")
        .head(2)
    )
    if len(finishers) < 2:
        raise RuntimeError("Fewer than two classified finishers were returned.")
    return finishers["Abbreviation"].astype(str).tolist()


def fastest_lap_telemetry(
    session: fastf1.core.Session, driver: str
) -> tuple[pd.Series, pd.DataFrame]:
    """Return fastest-lap metadata and distance-indexed speed/input telemetry."""
    laps = session.laps.pick_drivers(driver)
    if laps.empty:
        raise RuntimeError(f"No laps were returned for driver {driver}.")
    lap = laps.pick_fastest()
    if lap is None or pd.isna(lap.get("LapTime")):
        raise RuntimeError(f"No valid fastest lap was returned for driver {driver}.")

    telemetry = lap.get_car_data().add_distance()
    required = ["Distance", "Speed", "Throttle", "Brake"]
    missing = [column for column in required if column not in telemetry.columns]
    if missing:
        raise RuntimeError(f"Telemetry for {driver} is missing: {', '.join(missing)}")

    telemetry = telemetry[required].dropna(subset=["Distance", "Speed"])
    telemetry = telemetry[telemetry["Distance"].diff().fillna(1) >= 0]
    telemetry = telemetry.drop_duplicates(subset="Distance")
    if len(telemetry) < 2:
        raise RuntimeError(f"Insufficient telemetry for driver {driver}.")

    # FastF1's Brake channel is normally boolean. Convert it to a percentage
    # for plotting while preserving numeric brake channels if supplied.
    if telemetry["Brake"].dtype == bool:
        telemetry["Brake"] = telemetry["Brake"].astype(float) * 100.0
    else:
        telemetry["Brake"] = pd.to_numeric(telemetry["Brake"], errors="coerce")
        if telemetry["Brake"].dropna().max() <= 1:
            telemetry["Brake"] *= 100.0
    telemetry["Throttle"] = pd.to_numeric(telemetry["Throttle"], errors="coerce").clip(0, 100)
    telemetry["Brake"] = telemetry["Brake"].fillna(0).clip(0, 100)
    return lap, telemetry


def plot_telemetry_comparison(
    session: fastf1.core.Session,
    driver_data: list[tuple[pd.Series, pd.DataFrame]],
    output_file: Path,
) -> dict[str, dict[str, float]]:
    """Plot synchronized speed, throttle, and brake traces and return summaries."""
    max_distance = min(float(data["Distance"].max()) for _, data in driver_data)
    distance = np.linspace(0, max_distance, 1200)
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    channels = [("Speed", "Speed (km/h)"), ("Throttle", "Throttle (%)"), ("Brake", "Brake (%)")]
    insights: dict[str, dict[str, float]] = {}

    for lap, telemetry in driver_data:
        abbreviation = str(lap["Driver"])
        team = str(lap.get("Team", ""))
        color = None
        try:
            color = fastf1.plotting.get_team_color(team, session=session)
        except Exception:
            pass
        label = f"{abbreviation} — {team}"
        insights[abbreviation] = {}
        for axis, (channel, ylabel) in zip(axes, channels):
            values = np.interp(distance, telemetry["Distance"], telemetry[channel])
            axis.plot(distance, values, color=color, linewidth=1.8, label=label)
            insights[abbreviation][f"{channel.lower()}_mean"] = float(np.nanmean(values))
            insights[abbreviation][f"{channel.lower()}_max"] = float(np.nanmax(values))
        insights[abbreviation]["fastest_lap_seconds"] = float(lap["LapTime"].total_seconds())

    for axis, (_, ylabel) in zip(axes, channels):
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Distance (m)")
    event_name = str(session.event["EventName"])
    year = int(session.event["EventDate"].year)
    fig.suptitle(f"Fastest-Lap Telemetry Comparison — {event_name} ({year})", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.show()
    LOGGER.info("Saved chart to %s", output_file.resolve())
    return insights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, help="Season year, e.g. 2024")
    parser.add_argument(
        "--event", help='Exact FastF1 event name, e.g. "Monaco Grand Prix"'
    )
    parser.add_argument(
        "--drivers", nargs="+", help="Driver abbreviations, e.g. VER HAM LEC"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    if args.event and not args.year:
        raise ValueError("--year is required when --event is specified.")
    if args.event:
        season, event_name = args.year, args.event
    else:
        season, event_name = latest_completed_race(args.year)
    LOGGER.info("Selected race: %s %s", season, event_name)

    session = run_with_timeout(
        lambda: fastf1.get_session(season, event_name, "R"),
        "Creating race session",
    )
    run_with_timeout(
        lambda: session.load(telemetry=True, weather=False, messages=False),
        f"Loading telemetry for {event_name}",
    )

    drivers = [driver.upper() for driver in args.drivers] if args.drivers else get_top_two_finishers(session)
    if len(drivers) < 2:
        raise ValueError("Select at least two drivers.")
    if len(set(drivers)) != len(drivers):
        raise ValueError("Driver abbreviations must be unique.")

    available = set(session.results["Abbreviation"].dropna().astype(str).str.upper())
    unknown = sorted(set(drivers) - available)
    if unknown:
        raise ValueError(f"Driver(s) not found in {event_name}: {', '.join(unknown)}")

    driver_data = []
    for driver in drivers:
        lap, telemetry = fastest_lap_telemetry(session, driver)
        LOGGER.info("%s: fastest lap %s on lap %s", driver, lap["LapTime"], lap["LapNumber"])
        driver_data.append((lap, telemetry))

    insights = plot_telemetry_comparison(session, driver_data, args.output)
    print("\nSummary")
    for driver, values in insights.items():
        print(
            f"{driver}: lap={values['fastest_lap_seconds']:.3f}s, "
            f"mean speed={values['speed_mean']:.1f} km/h, "
            f"mean throttle={values['throttle_mean']:.1f}%, "
            f"peak brake={values['brake_max']:.0f}%"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.error("Dashboard failed: %s", exc)
        raise SystemExit(1) from exc

# References:
# FastF1 Quickstart: https://theoehrly-fast-f1.mintlify.app/quickstart
# FastF1 project: https://github.com/theOehrly/Fast-F1
# pandas documentation: https://pandas.pydata.org/docs/
# matplotlib documentation: https://matplotlib.org/stable/
