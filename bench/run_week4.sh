#!/usr/bin/env bash
# Week 4, end to end: collect, fit, sweep, report.
#
# The order is forced by the method and not by convenience. The predictor has
# to be fitted on traces collected with admission *off* -- otherwise the
# training set is a sample of what the previous controller allowed rather than
# of the latency distribution -- and the calibration set has to be held out
# from the fit, so the model cannot be refreshed from the sweep it is then
# evaluated on.
#
# Two properties of the collection step are load-bearing and are argued for in
# bench/collect_traces.py: the rates are visited in a rotated order across
# several rounds, and whole rounds become whole folds. Both exist so that the
# calibration and test folds are samples of the same distribution, which is the
# exchangeability the conformal guarantee is made of.
#
# The sweep runs the two arms rate-major with the rung order rotated, through
# run_ladder.py, for the same reason Week 2 and Week 3 do: on a laptop, running
# all of one arm and then all of the other measures the thermal state as much
# as the policy.
set -euo pipefail
cd "$(dirname "$0")/.."

SLO="${SLO:-4.0}"
ALPHA="${ALPHA:-0.01}"

# Collection: up to ~1.5x capacity. Deeper overload is deliberately not
# collected -- the queue there never reaches steady state, so the rows are a
# sample of a transient, and a tree model asked to predict past the deepest
# queue it was fitted with is extrapolating.
C_RATES="${C_RATES:-0.4,0.8,1.2,1.6,2.0,2.6,3.4}"
ROUNDS="${ROUNDS:-4}"
C_DURATION="${C_DURATION:-75}"

# The sweep keeps three of Week 2's offered loads so that the shared rung stays
# comparable across the two sessions, and extends to 4.0 and 6.0 rps -- four and
# six times the load at which rung 4's goodput peaks, which is where the claim
# "holds p99 under overload" has to be defended.
RATES="${RATES:-0.6,1.0,1.9,4.0,6.0}"
DURATION="${DURATION:-180}"
WARMUP="${WARMUP:-25}"
SEEDS="${SEEDS:-0}"
# Four arms: the control, and the same controller at three guarantee levels.
# Alpha is the policy's real knob -- it is the strength of the promise made
# about each admitted request -- so a single admission arm would report a
# choice rather than the trade-off.
CONFIGS="${CONFIGS:-continuous+cache,continuous+cache+admission,continuous+cache+admission-a05,continuous+cache+admission-a20}"

uv run bench/srchash.py > results/src.w4.hash

if [ "${SKIP_COLLECT:-0}" != "1" ]; then
    uv run bench/collect_traces.py --config continuous+cache \
        --rates "$C_RATES" --rounds "$ROUNDS" --duration "$C_DURATION" \
        --warmup 20 --cooldown 5 --slo "$SLO" --outdir results/w4_traces
fi

uv run bench/fit_predictor.py --traces results/w4_traces --alpha "$ALPHA" \
    --slo "$SLO" --drop-warmup 20 --split round \
    --out models/admission.pkl --outdir results/w4_fit

uv run bench/run_ladder.py --configs "$CONFIGS" --rates "$RATES" \
    --duration "$DURATION" --warmup "$WARMUP" --cooldown 5 \
    --slo "$SLO" --seeds "$SEEDS" --outdir results/w4_admission \
    --trace-dir results/w4_admission/traces

uv run bench/w4_report.py results/w4_admission --fit results/w4_fit \
    --traces results/w4_admission/traces --slo "$SLO" --outdir docs \
    --w2 results/w2_ladder
# Into docs/figs/w4, not docs/figs: the Week 2 section's figures are of the
# four-rung ladder and are generated from results/w2_ladder. Writing this
# sweep's two-arm versions over them would leave the Week 2 narrative pointing
# at charts of a different experiment.
uv run bench/charts.py results/w4_admission --slo "$SLO" --outdir docs/figs/w4
uv run bench/w4_charts.py results/w4_admission --fit results/w4_fit \
    --slo "$SLO" --outdir docs/figs/w4
uv run bench/embed_tables.py
