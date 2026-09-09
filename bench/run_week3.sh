#!/usr/bin/env bash
# Week 3's three measurements. The writeup presents them in the order the
# argument has to be made -- profile, then microbenchmark, then end-to-end --
# but they run in the order below, with the end-to-end A/B first while the
# machine is coldest, because it is the measurement whose absolute numbers are
# compared against the Week 2 ladder. The profile and the microbenchmark are
# ratios and are not sensitive to that.
#
# The A/B goes through run_ladder.py rather than two separate sweeps, because
# rate-major order with the arms rotated is what keeps thermal drift on a
# laptop out of the comparison.
set -euo pipefail
cd "$(dirname "$0")/.."

RATE="${RATE:-1.9}"           # nearest rung 4's knee: the scheduler is busy
RATES="${RATES:-1.0,1.4,1.9}"
DURATION="${DURATION:-180}"
SLO="${SLO:-4.0}"
HZ="${HZ:-500}"

uv run bench/srchash.py > results/src.hash

uv run bench/run_ladder.py --configs kv-core-python,kv-core-cpp \
    --rates "$RATES" --duration "$DURATION" --warmup 25 --cooldown 5 \
    --slo "$SLO" --seeds 0 --outdir results/w3_ab

for core in python cpp; do
    uv run bench/profile_core.py --backends llamacpp --rate "$RATE" \
        --duration 150 --warmup 20 --cooldown 5 --slo "$SLO" --hz "$HZ" \
        --kv-core "$core" --outdir "results/w3_profile/$core"
done

uv run bench/bench_core.py --out results/w3_bench/bench.json \
    --profile results/w3_profile/python/meta.json

uv run bench/core_report.py --slo "$SLO" --outdir docs
uv run bench/embed_tables.py
