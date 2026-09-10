"""Run several ablation rungs over one sweep of offered loads, rate-major.

Rung-major order -- every rate of rung 1, then every rate of rung 2 -- means the
later rungs are measured on a hotter laptop than the earlier ones. On Apple
silicon that is worth several percent, and it biases in the direction that
flatters exactly the configurations the project is arguing for.

So this runner is rate-major: for each rate it visits every rung, with the rung
order rotated between rates, and it restarts the gateway for every
(rung, rate) pair so no rung inherits a warm prefix cache from another.
Restarting costs a few seconds of model load and buys a comparison that is
actually about scheduling.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from configs import env_for  # noqa: E402
from loadgen import RunConfig, interarrival_ks, run  # noqa: E402
from run_sweep import Gateway, warm  # noqa: E402
from srchash import source_hash  # noqa: E402
from workloads import build_workload  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", required=True, help="comma-separated rungs")
    p.add_argument("--rates", required=True)
    p.add_argument("--duration", type=float, default=180.0)
    p.add_argument("--warmup", type=float, default=25.0)
    p.add_argument("--cooldown", type=float, default=5.0)
    p.add_argument("--slo", type=float, default=4.0)
    p.add_argument("--seeds", default="0")
    p.add_argument("--workload", default="mixed")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--outdir", required=True)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--trace-dir", default=None,
                   help="write one admission trace per (rung, rate, seed) here. "
                        "Rung 5 needs it to report the coverage its bound actually "
                        "achieved while it was deciding; rung 4 with it on is how "
                        "the training set was collected in the first place.")
    a = p.parse_args(argv)

    rungs = [c for c in a.configs.split(",") if c]
    rates = [float(x) for x in a.rates.split(",") if x]
    seeds = [int(s) for s in a.seeds.split(",")]
    extra = dict(kv.split("=", 1) for kv in a.set)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    wl = build_workload(a.workload)

    # Provenance: the exact invocation and the exact environment each rung ran
    # with, next to the parquet it produced. Results whose configuration cannot
    # be reconstructed are not results.
    (outdir / "meta.json").write_text(
        json.dumps(
            {
                "argv": sys.argv,
                "started": datetime.now(UTC).isoformat(timespec="seconds"),
                # A fingerprint of the engine source these numbers came from;
                # see bench/srchash.py. Recorded here rather than only in
                # results/src.hash so that a results directory carries its own
                # provenance.
                "src_hash": source_hash(),
                "rungs": {r: env_for(r, extra) for r in rungs},
                "trace_dir": a.trace_dir,
                "rates": rates,
                "seeds": seeds,
                "duration_s": a.duration,
                "warmup_s": a.warmup,
                "cooldown_s": a.cooldown,
                "slo_s": a.slo,
                "workload": a.workload,
            },
            indent=2,
        )
        + "\n"
    )

    frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    total = len(rates) * len(rungs) * len(seeds)
    done = 0
    t_start = time.time()

    for i, rate in enumerate(rates):
        # Rotate rung order between rates so no rung is systematically last.
        order = rungs[i % len(rungs):] + rungs[: i % len(rungs)]
        for rung in order:
            for seed in seeds:
                env = env_for(rung, extra)
                # The server's own SLO, not just the load generator's. Until
                # Week 4 the two could differ harmlessly -- the gateway used
                # it only for a Prometheus counter and for an EDF ordering
                # that a constant SLO leaves unchanged -- but the admission
                # controller compares its bound against *this* value, so a
                # sweep whose harness measures a 4 s target while the
                # controller enforces the 2 s default would report a
                # controller that sheds nearly everything, for a reason
                # nowhere in the results.
                env.setdefault("CADENCE_SLO_S", str(a.slo))
                if a.trace_dir:
                    tdir = Path(a.trace_dir)
                    tdir.mkdir(parents=True, exist_ok=True)
                    env["CADENCE_TRACE_LOG"] = str(
                        tdir / f"{rung.replace('+', '_')}_rate{rate:g}_seed{seed}.jsonl"
                    )
                log = outdir / f"{rung.replace('+', '_')}.server.log"
                with Gateway(env, a.port, log) as gw:
                    warm(gw.base, "qwen")
                    cfg = RunConfig(
                        url=f"{gw.base}/v1/chat/completions",
                        rate_rps=rate, duration_s=a.duration, slo_s=a.slo,
                        seed=seed, warmup_s=a.warmup, cooldown_s=a.cooldown,
                        config_name=rung, workload=a.workload,
                    )
                    t0 = time.time()
                    df = asyncio.run(run(cfg, wl))
                    d, pval = interarrival_ks(df, rate)
                    st = gw.stats()
                df["ks_d"], df["ks_p"] = d, pval
                df["prefix_hit_rate_server"] = st.get("prefix_hit_rate", float("nan"))
                df["kv_blocks_total"] = st.get("kv_blocks_total", float("nan"))
                frames[rung].append(df)

                done += 1
                s = df[df.steady]
                ok = s[s.ok.fillna(False)] if len(s) else s
                elapsed = time.time() - t_start
                print(
                    f"[{done}/{total}] {rung:<26} rate={rate:<5} seed={seed} "
                    f"n={len(df):<4} ok={len(ok):<4} "
                    f"p50={ok.e2e.quantile(0.5) if len(ok) else float('nan'):6.2f} "
                    f"p99={ok.e2e.quantile(0.99) if len(ok) else float('nan'):7.2f} "
                    f"slo={s.met_slo.mean() if len(s) else float('nan'):.2f} "
                    f"hit={st.get('prefix_hit_rate', 0.0):.2f} "
                    f"({time.time() - t0:.0f}s, {elapsed / 60:.0f}m total)",
                    flush=True,
                )
                # Write incrementally: a two-hour sweep should not lose
                # everything because the last run tripped over something. And
                # a failed write must not take the remaining rungs with it --
                # report it and keep going, so at worst one rung is short a
                # rate rather than the sweep being short two hours.
                try:
                    pd.concat(frames[rung], ignore_index=True).to_parquet(
                        outdir / f"{rung.replace('+', '_')}.parquet"
                    )
                except Exception as exc:
                    print(f"  !! could not write {rung}: {exc}", flush=True)

    print(f"\nwrote {len(frames)} rung files to {outdir} in {(time.time() - t_start) / 60:.0f}m")


if __name__ == "__main__":
    main()
