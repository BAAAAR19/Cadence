"""Collect the training set for the admission predictor.

The gateway writes one JSONL row per request -- the features admission saw, and
what became of the request -- so this script's whole job is to drive it across
enough of the state space that the model has seen the conditions it will later
be asked to predict in.

Two decisions about *how* the load is applied, both of which change the answer.

**Interleaved, not ascending.** The obvious loop -- every rate in increasing
order, once -- produces a trace whose time order is also its load order. Split
that 60/20/20 by time, as split-conformal requires, and the calibration set is
the busy end and the test set is the busiest: three folds drawn from three
different distributions, which is precisely the exchangeability the guarantee
rests on, broken by the data collection rather than by the system. So the rates
are visited in a rotated order across several rounds, and every fold spans the
whole range of load.

**Admission off.** The traces are collected with nothing shed, which is what
makes them a sample of the latency distribution rather than a sample of the
latencies the previous controller happened to allow. It is also the reason the
guarantee is only approximate once the controller is switched on: the
controller changes what runs, and the calibration set no longer describes it.
That is measured rather than hand-waved -- see the online coverage column in
the Week 4 writeup, and the rolling and adaptive modes in
``cadence.admission.conformal``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from configs import env_for  # noqa: E402
from loadgen import RunConfig, interarrival_ks, run  # noqa: E402
from run_sweep import Gateway, warm  # noqa: E402
from srchash import source_hash  # noqa: E402
from workloads import build_workload  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="continuous+cache",
                   help="the rung to collect under; rung 4 is what rung 5 becomes")
    p.add_argument("--rates", default="0.4,0.8,1.2,1.6,2.0,2.6,3.2,4.0")
    p.add_argument("--rounds", type=int, default=2,
                   help="passes over the rate grid, rotated each time")
    p.add_argument("--round-offset", type=int, default=0,
                   help="index of the first round. Exists so that a single "
                        "block can be re-collected into an existing trace "
                        "directory under its original name and its original "
                        "arrival seed -- which is what happened to two blocks "
                        "here, after a test suite was started on the same "
                        "machine mid-collection and contaminated them.")
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--warmup", type=float, default=20.0)
    p.add_argument("--cooldown", type=float, default=5.0)
    p.add_argument("--slo", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=100,
                   help="offset for the per-block arrival seed; kept clear of the "
                        "sweep's seeds so the model is never fitted on the exact "
                        "arrival sequence it is later evaluated against")
    p.add_argument("--workload", default="mixed")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--outdir", default="results/w4_traces")
    p.add_argument("--set", action="append", default=[])
    a = p.parse_args(argv)

    rates = [float(x) for x in a.rates.split(",") if x]
    extra = dict(kv.split("=", 1) for kv in a.set)
    outdir = Path(a.outdir)
    (outdir / "blocks").mkdir(parents=True, exist_ok=True)
    wl = build_workload(a.workload)

    meta_name = "meta.json" if a.round_offset == 0 else f"meta.r{a.round_offset}.json"
    (outdir / meta_name).write_text(
        json.dumps(
            {
                "argv": sys.argv,
                "started": datetime.now(UTC).isoformat(timespec="seconds"),
                "src_hash": source_hash(),
                "config": a.config,
                "env": env_for(a.config, extra),
                "rates": rates,
                "rounds": a.rounds,
                "round_offset": a.round_offset,
                "duration_s": a.duration,
                "warmup_s": a.warmup,
                "slo_s": a.slo,
                "workload": a.workload,
            },
            indent=2,
        )
        + "\n"
    )

    t_start = time.time()
    total = a.rounds * len(rates)
    done = 0
    for rnd in range(a.round_offset, a.round_offset + a.rounds):
        order = rates[rnd % len(rates):] + rates[: rnd % len(rates)]
        for rate in order:
            tag = f"r{rnd}_rate{rate:g}"
            trace = outdir / "blocks" / f"{tag}.jsonl"
            env = env_for(a.config, {**extra, "CADENCE_TRACE_LOG": str(trace),
                                     "CADENCE_SLO_S": str(a.slo)})
            seed = a.seed + rnd * 1000 + int(rate * 10)
            with Gateway(env, a.port, outdir / "collect.server.log") as gw:
                warm(gw.base, "qwen")
                cfg = RunConfig(
                    url=f"{gw.base}/v1/chat/completions",
                    rate_rps=rate, duration_s=a.duration, slo_s=a.slo, seed=seed,
                    warmup_s=a.warmup, cooldown_s=a.cooldown,
                    config_name=a.config, workload=a.workload,
                )
                t0 = time.time()
                df = asyncio.run(run(cfg, wl))
                d, pval = interarrival_ks(df, rate)
                st = gw.stats()
            df["ks_d"], df["ks_p"] = d, pval
            df.to_parquet(outdir / "blocks" / f"{tag}.parquet")
            done += 1
            n_rows = sum(1 for _ in trace.open()) if trace.exists() else 0
            s = df[df.steady]
            ok = s[s.ok.fillna(False)] if len(s) else s
            print(
                f"[{done}/{total}] round={rnd} rate={rate:<4} seed={seed} "
                f"issued={len(df):<4} ok={len(ok):<4} rows={n_rows:<4} "
                f"p50={ok.e2e.quantile(0.5) if len(ok) else float('nan'):6.2f} "
                f"p99={ok.e2e.quantile(0.99) if len(ok) else float('nan'):7.2f} "
                f"hit={st.get('prefix_hit_rate', 0.0):.2f} "
                f"({time.time() - t0:.0f}s, {(time.time() - t_start) / 60:.0f}m total)",
                flush=True,
            )

    files = sorted((outdir / "blocks").glob("*.jsonl"))
    rows = sum(sum(1 for _ in f.open()) for f in files)
    print(f"\n{rows} trace rows in {len(files)} blocks under {outdir} "
          f"in {(time.time() - t_start) / 60:.0f}m")


if __name__ == "__main__":
    main()
