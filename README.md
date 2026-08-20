# F1 Telemetry Dashboard

A Python and Plotly dashboard for comparing Formula 1 fastest-lap telemetry. The project supports a specific Grand Prix and selected drivers, or the latest completed race through the companion Matplotlib script.

## Features

The interactive dashboard compares speed, throttle, braking, throttle delta, and gear selection across the lap distance. Plotly hover tooltips expose the synchronized telemetry values. The analysis also exports a CSV ranking detected apex-like segments by estimated Norris time gain over Verstappen.

The repository includes the generated dashboard for the 2026 Hungarian Grand Prix, comparing Norris and Verstappen, along with the source scripts and corner-analysis CSV.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install fastf1 pandas numpy scipy matplotlib plotly requests
```

FastF1 downloads are cached locally. The default cache location is `~/.cache/fastf1`; override it with `FASTF1_CACHE_DIR`.

## Open the included dashboard

Open `f1_interactive_telemetry_dashboard.html` directly in a browser. The file is self-contained and includes the Plotly JavaScript bundle.

## Regenerate the interactive dashboard

```bash
python f1_interactive_dashboard.py \
  --year 2026 \
  --event "Hungarian Grand Prix" \
  --drivers NOR VER \
  --output f1_interactive_telemetry_dashboard.html
```

The command also writes `f1_corner_analysis.csv`. Driver abbreviations and event names must match FastF1's data. The current corner detector identifies ordered apex-like speed minima; the labels are analytical markers and should not be interpreted as official FIA corner metadata without additional circuit-coordinate validation.

## Generate the static dashboard

```bash
python f1_telemetry_dashboard.py \
  --year 2024 \
  --event "Monaco Grand Prix" \
  --drivers LEC PIA \
  --output monaco_telemetry.png
```

If `--event` and `--drivers` are omitted from the static script, it selects the latest completed race and the top two classified finishers.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `FASTF1_CACHE_DIR` | FastF1 cache directory | `~/.cache/fastf1` |
| `FASTF1_TIMEOUT` | Timeout per network/data operation in seconds | `180` |
| `FASTF1_RETRIES` | Retry attempts for transient failures | `3` |
| `FASTF1_OUTPUT` | Default output path for the static chart | `latest_f1_telemetry_comparison.png` |

## Data and references

The project uses FastF1's documented session, lap, and telemetry APIs [1]. The charts are generated with pandas, NumPy, Matplotlib, SciPy, and Plotly.

[1]: https://theoehrly-fast-f1.mintlify.app/quickstart "FastF1 Quickstart Documentation"
[2]: https://github.com/theOehrly/Fast-F1 "FastF1 GitHub Repository"
[3]: https://plotly.com/python/ "Plotly Python Documentation"
