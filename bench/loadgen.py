"""The open-loop load generator.

Built before the thing it measures, on purpose. Tune a scheduler against your
intuition and you will not notice when a "speedup" is a measurement artifact.

The distinction that matters
----------------------------
*Closed loop*: N workers, each sends a request, waits for the response, sends
the next. Offered load is a consequence of server speed -- when the server slows
down the client slows with it, the queue never builds, and overload behaviour,
the entire subject of this project, cannot be observed at all.

*Open loop*: arrivals follow a Poisson process at rate ``lambda``, independent
of the server. Requests are issued on schedule even if earlier ones are still
outstanding, so the queue grows whenever ``lambda`` exceeds capacity.

Coordinated omission is the failure mode of the closed loop: the requests that
would have been slowest are never issued, so the measured tail is optimistic by
an order of magnitude. The fix is to timestamp each request by its *intended*
send time, not its actual one.

Three non-negotiable properties of this generator:

1. arrivals are exponential, not fixed-interval (fixed intervals understate
   queueing);
2. every latency is measured from the intended arrival time (this is what kills
   coordinated omission);
3. a request that is still outstanding never delays the next arrival -- that is
   what makes it open-loop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, field

import httpx
import numpy as np
import pandas as pd


@dataclass
class RunConfig:
    url: str = "http://127.0.0.1:8000/v1/chat/completions"
    rate_rps: float = 4.0
    duration_s: float = 120.0
    slo_s: float = 2.0
    seed: int = 0
    warmup_s: float = 15.0
    cooldown_s: float = 5.0
    timeout_s: float = 300.0
    config_name: str = "unknown"
    workload: str = "mixed"
    extra: dict = field(default_factory=dict)


async def one_request(client, url, payload, t_intended, results, slo_s, timeout_s):
    meta = payload.pop("_meta", {})
    t_send = time.perf_counter()
    rec = {
        "t_intended": t_intended,
        "t_send": t_send,
        "sched_delay": t_send - t_intended,
        "ttft": None,
        "e2e": None,
        "n_tokens": 0,
        "status": None,
        "itl": [],
        "shared_prompt": meta.get("shared"),
        "requested_tokens": meta.get("requested_tokens"),
    }
    try:
        async with client.stream("POST", url, json=payload, timeout=timeout_s) as r:
            rec["status"] = r.status_code
            if r.status_code != 200:
                await r.aread()
                rec["retry_after"] = float(r.headers.get("Retry-After", "nan"))
                rec["e2e"] = time.perf_counter() - t_intended
                results.append(rec)
                return
            last = None
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"].get("content", "")
                if not delta:
                    continue
                now = time.perf_counter()
                if rec["ttft"] is None:
                    rec["ttft"] = now - t_intended  # <-- intended, not send
                else:
                    rec["itl"].append(now - last)
                last = now
                rec["n_tokens"] += 1
    except Exception as e:
        rec["status"] = type(e).__name__
    rec["e2e"] = time.perf_counter() - t_intended  # <-- intended, not send
    rec["ok"] = rec["status"] == 200 and rec["n_tokens"] > 0
    rec["met_slo"] = bool(rec["ok"] and rec["e2e"] <= slo_s)
    results.append(rec)


async def run(cfg: RunConfig, workload) -> pd.DataFrame:
    rng = random.Random(cfg.seed)
    results: list[dict] = []
    tasks: list[asyncio.Task] = []
    # An unbounded connection pool is part of being open-loop: a pool limit is
    # backpressure, and backpressure turns this back into a closed loop.
    limits = httpx.Limits(max_connections=10_000, max_keepalive_connections=10_000)
    async with httpx.AsyncClient(limits=limits) as client:
        t0 = time.perf_counter()
        t_next = t0
        while t_next - t0 < cfg.duration_s:
            t_next += rng.expovariate(cfg.rate_rps)  # exponential inter-arrivals
            delay = t_next - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            payload = workload.sample(rng)
            tasks.append(
                asyncio.create_task(
                    one_request(
                        client, cfg.url, payload, t_next, results, cfg.slo_s, cfg.timeout_s
                    )
                )
            )
        n_issued = len(tasks)
        await asyncio.gather(*tasks)

    df = pd.DataFrame(results)
    df["t_rel"] = df["t_intended"] - t0
    df["rate_rps"] = cfg.rate_rps
    df["config"] = cfg.config_name
    df["workload"] = cfg.workload
    df["seed"] = cfg.seed
    df["slo_s"] = cfg.slo_s
    df["duration_s"] = cfg.duration_s
    df["warmup_s"] = cfg.warmup_s
    df["cooldown_s"] = cfg.cooldown_s
    df["n_issued"] = n_issued
    # Steady state only: the warm-up window contains an empty-queue transient
    # and the cool-down window contains a draining one. Both are kept in the
    # frame and flagged, never silently dropped.
    df["steady"] = (df["t_rel"] > cfg.warmup_s) & (
        df["t_rel"] < cfg.duration_s - cfg.cooldown_s
    )
    df["p50_itl"] = df["itl"].map(lambda v: float(pd.Series(v).median()) if v else float("nan"))
    df["p99_itl"] = df["itl"].map(
        lambda v: float(pd.Series(v).quantile(0.99)) if v else float("nan")
    )
    df["mean_itl"] = df["itl"].map(lambda v: float(pd.Series(v).mean()) if v else float("nan"))
    return df


def ks_exponential(gaps, rate_rps: float) -> tuple[float, float]:
    """One-sample Kolmogorov-Smirnov test of ``gaps`` against Exp(rate).

    Implemented here rather than pulled from scipy: the statistic is four lines
    and the asymptotic p-value is one more, and it keeps the benchmark harness
    free of a dependency it would otherwise use once.

    Returns ``(D, p)``. A small p means the realised arrival process is not the
    one the writeup claims, which invalidates every number in the run.
    """
    x = np.sort(np.asarray([g for g in gaps if g >= 0], dtype=float))
    n = x.size
    if n == 0:
        return float("nan"), float("nan")
    cdf = 1.0 - np.exp(-rate_rps * x)
    i = np.arange(1, n + 1)
    d_plus = np.max(i / n - cdf)
    d_minus = np.max(cdf - (i - 1) / n)
    d = float(max(d_plus, d_minus))
    # Kolmogorov asymptotic survival function, truncated at 100 terms.
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    k = np.arange(1, 101)
    p = float(np.clip(2.0 * np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * k**2 * lam**2)), 0.0, 1.0))
    return d, p


def interarrival_ks(df: pd.DataFrame, rate_rps: float) -> tuple[float, float]:
    """KS test of the realised inter-arrival times against Exp(rate).

    Reported with every run. The *intended* times are tested, not the actual
    send times: the schedule is the arrival process, and a send that slipped
    because the client's own loop was busy is a client defect measured
    separately by ``sched_delay``.
    """
    t = np.sort(df["t_intended"].to_numpy())
    return ks_exponential(np.diff(t), rate_rps)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open-loop Poisson load generator")
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--rate", type=float, required=True, help="offered load, requests/s")
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--slo", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=float, default=15.0)
    p.add_argument("--cooldown", type=float, default=5.0)
    p.add_argument("--workload", default="mixed")
    p.add_argument("--config-name", default="unknown")
    p.add_argument("--out", default=None, help="parquet path")
    return p.parse_args(argv)


def main(argv=None) -> None:  # pragma: no cover - CLI
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from workloads import build_workload

    args = _parse_args(argv)
    cfg = RunConfig(
        url=args.url,
        rate_rps=args.rate,
        duration_s=args.duration,
        slo_s=args.slo,
        seed=args.seed,
        warmup_s=args.warmup,
        cooldown_s=args.cooldown,
        workload=args.workload,
        config_name=args.config_name,
    )
    df = asyncio.run(run(cfg, build_workload(args.workload)))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out)
    d, pval = interarrival_ks(df, args.rate)
    steady = df[df.steady]
    print(
        f"interarrival KS vs Exp({args.rate}): D={d:.4f} p={pval:.3f} "
        f"max_sched_delay={df.sched_delay.max()*1e3:.1f}ms"
    )
    print(
        f"rate={args.rate} n={len(df)} steady={len(steady)} "
        f"ok={steady.ok.mean() if len(steady) else float('nan'):.3f} "
        f"p99_e2e={steady.e2e.quantile(0.99) if len(steady) else float('nan'):.3f}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
