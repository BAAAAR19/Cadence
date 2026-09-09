#!/usr/bin/env bash
# The ablation ladder, rate-major and rung-rotated so that thermal drift on a
# laptop cannot be mistaken for a scheduling result.
#
# Rates come from bench/calibrate.py, not from a guessed grid: on this machine
# a request is ~540 prompt tokens and ~136 output tokens against a 0.5B model
# at ~78 tok/s single-stream, which puts saturation below 1 rps.
set -euo pipefail
cd "$(dirname "$0")/.."

RATES="${RATES:-0.2,0.4,0.6,0.8,1.0,1.4}"
DURATION="${DURATION:-180}"
WARMUP="${WARMUP:-25}"
SLO="${SLO:-4.0}"
SEEDS="${SEEDS:-0}"
CONFIGS="${CONFIGS:-fifo,static,continuous,continuous+cache}"
OUT="${OUT:-results/w2_ladder}"

uv run bench/calibrate.py --out results/calibration.json

uv run bench/run_ladder.py \
    --configs "$CONFIGS" --rates "$RATES" \
    --duration "$DURATION" --warmup "$WARMUP" --cooldown 5 \
    --slo "$SLO" --seeds "$SEEDS" --outdir "$OUT"

uv run bench/analyze.py "$OUT" --slo "$SLO" --out docs/ablation.csv
uv run bench/charts.py "$OUT" --slo "$SLO" --outdir docs/figs
