#!/usr/bin/env bash
# Everything the Week 5 writeup contains, regenerated from committed data.
#
# Separate from bench/run_week5.sh so that the seven-hour measurement and the
# thirty-second rendering of it are different commands: a chart is fixed by
# rerunning this, never by rerunning the sweep. CI runs it and diffs the
# result, so a number in the README cannot drift from the parquet it came
# from.
set -euo pipefail
cd "$(dirname "$0")/.."

SLO="${SLO:-4.0}"
LADDER="${LADDER:-results/w5_ladder}"

uv run bench/w5_report.py "$LADDER" --slo "$SLO" --outdir docs \
    --w2 results/w2_ladder --w4 results/w4_admission --fit results/w4_fit \
    --traces "$LADDER/traces" --drain results/w5_drain.json
uv run bench/w5_charts.py "$LADDER" --slo "$SLO" --outdir docs/figs/w5 \
    --fit results/w4_fit
uv run bench/embed_tables.py
