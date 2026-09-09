"""Run one ablation rung across a sweep of offered loads.

Starts the gateway itself rather than assuming one is already up: the rung name
and the server's configuration have to agree, and the surest way to guarantee
that is to make the same command responsible for both.

    uv run bench/run_sweep.py --config fifo \\
        --rates 1,2,4,6,8,10,12,16 --duration 120 --slo 2.0 \\
        --out results/w1_fifo.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from configs import env_for  # noqa: E402
from loadgen import RunConfig, interarrival_ks, run  # noqa: E402
from workloads import build_workload  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class Gateway:
    """The server under test, as a context manager."""

    def __init__(self, env: dict[str, str], port: int, log: Path | None = None) -> None:
        self.env = env
        self.port = port
        self.log = log
        self.proc: subprocess.Popen | None = None

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> Gateway:
        env = {**os.environ, **self.env, "CADENCE_PORT": str(self.port)}
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        out = open(self.log, "w") if self.log else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "cadence.api.app"],
            env=env, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT,
        )
        self._await_health()
        return self

    def _await_health(self, timeout_s: float = 240.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise SystemExit(
                    f"gateway exited with {self.proc.returncode}; see {self.log}"
                )
            try:
                r = httpx.get(f"{self.base}/health", timeout=2.0)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise SystemExit("gateway did not become healthy in time")

    def stats(self) -> dict:
        try:
            return httpx.get(f"{self.base}/stats", timeout=5.0).json()
        except Exception:
            return {}

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def warm(base: str, model: str) -> None:
    """One throwaway request so the first measured arrival does not pay for
    Metal shader compilation and the first KV allocation."""
    try:
        httpx.post(
            f"{base}/v1/chat/completions",
            json={
                "model": model, "stream": False, "max_tokens": 8,
                "messages": [{"role": "user", "content": "warm up"}],
            },
            timeout=180.0,
        )
    except Exception as exc:  # pragma: no cover
        print(f"  warm-up request failed: {exc}", file=sys.stderr)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="ablation rung, see bench/configs.py")
    p.add_argument("--rates", required=True, help="comma-separated offered loads in rps")
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--warmup", type=float, default=15.0)
    p.add_argument("--cooldown", type=float, default=5.0)
    p.add_argument("--slo", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default=None, help="comma-separated seeds; overrides --seed")
    p.add_argument("--workload", default="mixed")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--out", required=True)
    p.add_argument("--restart-per-rate", action="store_true",
                   help="restart the gateway between rates, so a warm prefix cache "
                        "from a previous rate cannot flatter the next one")
    p.add_argument("--set", action="append", default=[],
                   help="extra CADENCE_* override, KEY=VALUE")
    p.add_argument("--label", default=None,
                   help="name recorded in the results instead of --config; use it "
                        "when sweeping a knob within one rung, so the two runs are "
                        "distinguishable in the same frame")
    a = p.parse_args(argv)

    rates = [float(x) for x in a.rates.split(",") if x]
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else [a.seed]
    extra = dict(kv.split("=", 1) for kv in a.set)
    label = a.label or a.config
    env = env_for(a.config, extra)
    env["CADENCE_CONFIG_NAME"] = label
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".server.log")
    wl = build_workload(a.workload)

    frames: list[pd.DataFrame] = []

    def sweep(gw: Gateway, rate_subset: list[float]) -> None:
        for rate in rate_subset:
            for seed in seeds:
                cfg = RunConfig(
                    url=f"{gw.base}/v1/chat/completions",
                    rate_rps=rate, duration_s=a.duration, slo_s=a.slo, seed=seed,
                    warmup_s=a.warmup, cooldown_s=a.cooldown,
                    config_name=label, workload=a.workload,
                )
                t0 = time.time()
                df = asyncio.run(run(cfg, wl))
                d, pval = interarrival_ks(df, rate)
                df["ks_d"] = d
                df["ks_p"] = pval
                st = gw.stats()
                df["prefix_hit_rate_server"] = st.get("prefix_hit_rate", float("nan"))
                frames.append(df)
                s = df[df.steady]
                ok = s[s.ok.fillna(False)] if len(s) else s
                print(
                    f"  rate={rate:<5} seed={seed} n={len(df):<5} "
                    f"ok={len(ok):<5} shed={(s.status == 503).sum() if len(s) else 0:<4} "
                    f"p50_e2e={ok.e2e.quantile(0.5) if len(ok) else float('nan'):.2f} "
                    f"p99_e2e={ok.e2e.quantile(0.99) if len(ok) else float('nan'):.2f} "
                    f"KS_p={pval:.2f} ({time.time() - t0:.0f}s)",
                    flush=True,
                )

    print(f"[{label}] rates={rates} seeds={seeds} duration={a.duration}s", flush=True)
    if a.restart_per_rate:
        for rate in rates:
            with Gateway(env, a.port, log) as gw:
                warm(gw.base, "qwen")
                sweep(gw, [rate])
    else:
        with Gateway(env, a.port, log) as gw:
            warm(gw.base, "qwen")
            sweep(gw, rates)

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out)
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
