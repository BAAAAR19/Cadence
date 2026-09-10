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
from run_sweep import Gateway, canary, warm  # noqa: E402
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
    p.add_argument("--resume", action="store_true",
                   help="skip any (rung, rate, seed) whose part file already "
                        "exists under <outdir>/parts. A five-rung, three-seed "
                        "ladder is most of a day; losing it to a laptop that "
                        "went to sleep at hour six is not a measurement "
                        "problem worth having.")
    p.add_argument("--no-canary", action="store_true",
                   help="skip the fixed single-stream probe before each run "
                        "(see bench/run_sweep.py:canary)")
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
    # On a resumed run this becomes meta.resume1.json and so on, rather than
    # overwriting the record of when the sweep actually started and with what.
    meta_path = outdir / "meta.json"
    n_resume = 0
    while meta_path.exists() and a.resume:
        n_resume += 1
        meta_path = outdir / f"meta.resume{n_resume}.json"
    meta_path.write_text(
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
                "resumed": bool(n_resume),
            },
            indent=2,
        )
        + "\n"
    )

    # One parquet per (rung, rate, seed), written the moment it exists. The
    # per-rung files the rest of the pipeline reads are concatenated from
    # these at the end, so an interrupted sweep resumes at the run it was on
    # rather than at the beginning.
    parts = outdir / "parts"
    parts.mkdir(parents=True, exist_ok=True)

    def part_path(rung: str, rate: float, seed: int) -> Path:
        return parts / f"{rung.replace('+', '_')}_rate{rate:g}_seed{seed}.parquet"

    frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    if a.resume:
        for rung in rungs:
            for rate in rates:
                for seed in seeds:
                    f = part_path(rung, rate, seed)
                    if f.exists():
                        frames[rung].append(pd.read_parquet(f))
        n_have = sum(len(v) for v in frames.values())
        if n_have:
            print(f"resuming: {n_have} run(s) already on disk under {parts}", flush=True)

    canaries: list[dict] = []
    canary_log = outdir / "canary.jsonl"
    total = len(rates) * len(rungs) * len(seeds)
    done = 0
    t_start = time.time()

    for i, rate in enumerate(rates):
        # Rotate rung order between rates so no rung is systematically last.
        order = rungs[i % len(rungs):] + rungs[: i % len(rungs)]
        for rung in order:
            for seed in seeds:
                if a.resume and part_path(rung, rate, seed).exists():
                    done += 1
                    print(f"[{done}/{total}] {rung:<26} rate={rate:<5} seed={seed} "
                          f"-- already on disk, skipped", flush=True)
                    continue
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
                    # The thermal probe, on an idle engine, immediately before
                    # the load. Recorded rather than acted on: it is the
                    # covariate that says whether a difference between two
                    # rungs could have been the machine cooling down.
                    cn = {} if a.no_canary else canary(gw.base, "qwen")
                    if cn:
                        rec = {"rung": rung, "rate": rate, "seed": seed,
                               "t": datetime.now(UTC).isoformat(timespec="seconds"), **cn}
                        canaries.append(rec)
                        with canary_log.open("a") as fh:
                            fh.write(json.dumps(rec) + "\n")
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
                # Carried on every row so that any later analysis of these
                # results can condition on the machine's speed without having
                # to join against a second file.
                df["canary_tok_per_s"] = cn.get("tok_per_s", float("nan"))
                frames[rung].append(df)
                try:
                    df.to_parquet(part_path(rung, rate, seed))
                except Exception as exc:
                    print(f"  !! could not write the part file: {exc}", flush=True)

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
                    f"canary={cn.get('tok_per_s', float('nan')):.1f}t/s "
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

    for rung, fs in frames.items():
        if fs:
            pd.concat(fs, ignore_index=True).to_parquet(
                outdir / f"{rung.replace('+', '_')}.parquet"
            )

    if canaries:
        tps = [c["tok_per_s"] for c in canaries if c.get("ok")]
        if tps:
            lo, hi = min(tps), max(tps)
            print(
                f"\nthermal probe over {len(tps)} runs: {lo:.1f}-{hi:.1f} tok/s "
                f"(spread {100 * (hi - lo) / hi:.1f}% of the fastest). "
                f"A large spread means the ladder was measured on a machine "
                f"that changed speed during it; see {canary_log}."
            )

    print(f"\nwrote {len(frames)} rung files to {outdir} in {(time.time() - t_start) / 60:.0f}m")


if __name__ == "__main__":
    main()
