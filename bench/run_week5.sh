#!/usr/bin/env bash
# Week 5: the full ablation ladder, three seeds, then the tables and figures.
#
# This is the run the writeup's headline table comes from. It is long -- about
# seven and a half hours on the reference machine -- and every part of that
# length is a decision:
#
#   five rungs        each adds exactly one thing to the one above it, so each
#                     row of the table is attributable to a single change
#   six offered loads 0.6 to 4.0 rps, spanning underload to three times the
#                     load at which rung 4's goodput peaks -- the range over
#                     which "holds p99 under overload" is a claim at all
#   three seeds       a single run of a latency benchmark on a laptop is
#                     noise. The table reports a spread and the charts shade
#                     min-max across seeds
#   180 s each        the same window Weeks 2 to 4 used, so the shared rungs
#                     stay comparable across sessions
#
# Two properties of the *order* matter as much as the grid, and both are
# implemented in bench/run_ladder.py: the loop is rate-major with the rung
# order rotated between rates, so no configuration is systematically measured
# on a hotter machine than another; and the gateway is restarted for every
# (rung, rate, seed), so no run inherits a warm prefix cache from another.
#
# Before each measured run the harness times a fixed single-stream generation
# on the idle engine and records its decode rate (bench/run_sweep.py:canary).
# That is this project's substitute for running `powermetrics` alongside the
# sweep: it needs no root, and it is a direct reading of how fast the machine
# was at that moment. The spread of that probe across the whole ladder is
# reported in the writeup.
#
# Interruptions are survivable. Every (rung, rate, seed) is written to its own
# parquet under results/w5_ladder/parts as soon as it finishes, and
#
#     RESUME=1 bash bench/run_week5.sh
#
# picks up at the first run that is missing.
#
# Before starting: close everything else, plug the laptop in, disable sleep
#   caffeinate -dimsu bash bench/run_week5.sh 2>&1 | tee results/w5_ladder.log
set -euo pipefail
cd "$(dirname "$0")/.."

SLO="${SLO:-4.0}"
RATES="${RATES:-0.6,1.0,1.4,1.9,2.6,4.0}"
DURATION="${DURATION:-180}"
WARMUP="${WARMUP:-25}"
SEEDS="${SEEDS:-0,1,2}"
CONFIGS="${CONFIGS:-fifo,static,continuous,continuous+cache,continuous+cache+admission}"
OUT="${OUT:-results/w5_ladder}"
PORT="${PORT:-8200}"

RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then
    RESUME_FLAG="--resume"
fi

# The engine source these numbers came from, recorded before they are taken.
uv run bench/srchash.py > results/src.w5.hash

# shellcheck disable=SC2086
uv run bench/run_ladder.py \
    --configs "$CONFIGS" --rates "$RATES" \
    --duration "$DURATION" --warmup "$WARMUP" --cooldown 5 \
    --slo "$SLO" --seeds "$SEEDS" --port "$PORT" \
    --outdir "$OUT" --trace-dir "$OUT/traces" $RESUME_FLAG

bash bench/report_week5.sh
