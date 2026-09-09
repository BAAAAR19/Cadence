"""Validate the arrival process in the regime the sweep actually runs in.

The load generator's KS p-value is reported with every run, and the seed used
for the sweep is fixed in advance so that every rung sees the *same* arrival
sequence -- that is what makes the rungs comparable to each other rather than
to six different realisations of an arrival process.

The cost of fixing the seed in advance is that the seed might land in the tail,
and seed 0 does: its realisation is rejected at alpha=0.05 at every rate in the
sweep. Those are not six independent failures. The Kolmogorov-Smirnov statistic
for an exponential fit depends only on the underlying uniform draws, so the six
rates are nested prefixes of one sequence -- one draw, counted six times.

The right response is to establish the *process* over many seeds and then
report the chosen seed honestly, not to re-pick the seed until the p-value
looks better. Re-picking after seeing the statistic is exactly the selection
that would make the number meaningless.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from loadgen import ks_exponential  # noqa: E402
from workloads import build_workload  # noqa: E402


def realisation(seed: int, rate: float, duration: float, workload) -> list[float]:
    """Reproduce exactly what ``loadgen.run`` draws: exponential gaps
    interleaved with workload sampling from the same RNG."""
    rng = random.Random(seed)
    t, gaps = 0.0, []
    while t < duration:
        g = rng.expovariate(rate)
        t += g
        gaps.append(g)
        workload.sample(rng)
    return gaps


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rates", default="0.3,0.6,1.0,1.4,1.9,2.6")
    p.add_argument("--duration", type=float, default=180.0)
    p.add_argument("--seed", type=int, default=0, help="the seed the sweep used")
    p.add_argument("--n-seeds", type=int, default=200)
    p.add_argument("--workload", default="mixed")
    p.add_argument("--out", default="results/loadgen_validation.json")
    a = p.parse_args(argv)

    wl = build_workload(a.workload)
    rates = [float(x) for x in a.rates.split(",")]
    report: dict = {
        "duration_s": a.duration,
        "sweep_seed": a.seed,
        "sweep_seed_per_rate": {},
        "process_over_seeds": {},
    }

    for rate in rates:
        g = realisation(a.seed, rate, a.duration, wl)
        d, pv = ks_exponential(g, rate)
        report["sweep_seed_per_rate"][str(rate)] = {"n": len(g), "D": d, "p": pv}

    # The process itself, in the middle of the swept range.
    probe_rate = rates[len(rates) // 2]
    ps, ns = [], []
    for s in range(a.n_seeds):
        g = realisation(s, probe_rate, a.duration, wl)
        ns.append(len(g))
        ps.append(ks_exponential(g, probe_rate)[1])
    ps_arr, ns_arr = np.array(ps), np.array(ns)
    expected_n = probe_rate * a.duration
    report["process_over_seeds"] = {
        "rate_rps": probe_rate,
        "n_seeds": a.n_seeds,
        "reject_rate_at_0.05": float(np.mean(ps_arr < 0.05)),
        "median_p": float(np.median(ps_arr)),
        "mean_arrivals": float(ns_arr.mean()),
        "sd_arrivals": float(ns_arr.std()),
        "poisson_expected_arrivals": expected_n,
        "poisson_expected_sd": float(np.sqrt(expected_n)),
        "sweep_seed_p_percentile": float((ps_arr < ps[a.seed]).mean()),
    }

    text = json.dumps(report, indent=2)
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
