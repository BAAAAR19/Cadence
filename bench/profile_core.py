"""Step 3.0 -- profile before porting.

The rule Week 3 opens with is that the C++ port has to be motivated by a
measurement. This runs the gateway under a real offered load with the
sampling profiler enabled, and reports what fraction of the scheduler thread's
busy time is spent in the two data structures the port targets.

It runs the same load twice, against two backends, because one number would be
misleading:

* ``llamacpp`` -- the real model. This is the number that decides whether the
  port pays for itself *today*.
* ``mock`` -- the deterministic backend with its modelled forward pass turned
  down to zero, as an attempt at "the same scheduler with the model's cost
  removed". It does not work as one: the mock's toy sampler hashes a repr of
  the whole prompt on every token, which costs more than everything the
  scheduler does, so the arm measures the mock rather than the scheduler. It
  is left available behind ``--backends`` and left out of the default, and the
  model-free half of the question is answered by ``bench/bench_core.py``
  instead, where the two structures are measured directly.

    uv run bench/profile_core.py --rate 1.4 --duration 120 --outdir results/w3_profile
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import flamegraph as fg  # noqa: E402
from configs import env_for  # noqa: E402
from loadgen import RunConfig, run  # noqa: E402
from run_sweep import Gateway, warm  # noqa: E402
from srchash import source_hash  # noqa: E402
from workloads import build_workload  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def scrape_steps(base: str) -> dict:
    """Mean engine-step duration, straight out of the Prometheus histogram.

    This is the denominator the profile's percentages are percentages *of*,
    and it is what turns the microbenchmark's per-operation microseconds into
    a share of a step rather than a number with no scale.
    """
    import httpx

    try:
        text = httpx.get(f"{base}/metrics", timeout=5.0).text
    except Exception:  # pragma: no cover - the server is going down anyway
        return {}
    total = count = 0.0
    for line in text.splitlines():
        if line.startswith("cadence_step_seconds_sum"):
            total = float(line.rsplit(" ", 1)[1])
        elif line.startswith("cadence_step_seconds_count"):
            count = float(line.rsplit(" ", 1)[1])
    return {
        "n_steps": int(count),
        "step_total_s": total,
        "mean_step_s": total / count if count else None,
    }


def _rel(p: Path) -> str:
    """Repo-relative when it can be, absolute otherwise: a smoke run into a
    scratch directory should not fail at the point of writing provenance."""
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p.resolve())


def profile_one(backend: str, a, outdir: Path) -> dict:
    folded = outdir / f"{backend}.folded"
    overrides = {
        "CADENCE_BACKEND": backend,
        "CADENCE_PROFILE_OUT": str(folded),
        "CADENCE_PROFILE_HZ": str(a.hz),
        "CADENCE_KV_CORE": a.kv_core,
    }
    if backend == "mock":
        # A modelled forward pass that costs nothing, so that what is left is
        # the scheduler. The mock's defaults sleep for milliseconds per step,
        # which would drown exactly the thing this run exists to see.
        overrides |= {
            "CADENCE_MOCK_STEP_OVERHEAD_S": "0",
            "CADENCE_MOCK_DECODE_S_PER_SEQ": "0",
            "CADENCE_MOCK_PREFILL_S_PER_TOKEN": "0",
        }
    env = env_for("continuous+cache", overrides)
    wl = build_workload(a.workload)
    with Gateway(env, a.port, log=outdir / f"{backend}.server.log") as gw:
        warm(gw.base, "qwen")
        cfg = RunConfig(
            url=f"{gw.base}/v1/chat/completions",
            rate_rps=a.rate,
            duration_s=a.duration,
            slo_s=a.slo,
            seed=a.seed,
            warmup_s=a.warmup,
            cooldown_s=a.cooldown,
            config_name=f"profile-{backend}",
            workload=a.workload,
        )
        df = asyncio.run(run(cfg, wl))
        stats = gw.stats()
        steps = scrape_steps(gw.base)

    sampler = json.loads(folded.with_suffix(".json").read_text())
    rows = fg.read_folded(folded)
    busy = fg.busy_micros(rows)
    total = sum(w for _, w in rows)
    svg = outdir / f"{backend}.svg"
    svg.write_text(
        fg.render_svg(
            fg.build_tree(rows),
            f"cadence scheduler thread — {backend} backend, {a.rate} rps, "
            f"kv_core={a.kv_core}",
        )
    )
    buckets = fg.attribute(rows)
    return {
        "backend": backend,
        "kv_core": a.kv_core,
        "rate_rps": a.rate,
        "duration_s": a.duration,
        "completed": int((df["status"] == 200).sum()),
        "sampled_s": total / 1e6,
        "busy_s": busy / 1e6,
        "busy_frac": busy / total if total else 0.0,
        "prefix_hit_rate": stats.get("prefix_hit_rate"),
        "sampler": sampler,
        "steps": steps,
        "buckets": {
            k: {
                "inclusive_pct": 100.0 * v["inclusive"] / busy if busy else 0.0,
                "self_pct": 100.0 * v["self"] / busy if busy else 0.0,
            }
            for k, v in buckets.items()
        },
        "folded": _rel(folded),
        "svg": _rel(svg),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rate", type=float, default=1.4)
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--warmup", type=float, default=15.0)
    p.add_argument("--cooldown", type=float, default=5.0)
    p.add_argument("--slo", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hz", type=int, default=200)
    p.add_argument("--workload", default="mixed")
    p.add_argument("--kv-core", default="python", choices=["python", "cpp", "auto"])
    p.add_argument("--backends", default="llamacpp")
    p.add_argument("--port", type=int, default=8300)
    p.add_argument("--outdir", default="results/w3_profile")
    a = p.parse_args(argv)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    runs = [profile_one(b, a, outdir) for b in a.backends.split(",") if b]

    meta = {
        "argv": sys.argv[1:],
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "src_hash": source_hash(),
        "sampler": "cadence.obs.profiler (in-process; py-spy needs root on macOS)",
        "hz": a.hz,
        "runs": runs,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    for r in runs:
        print(f"\n{r['backend']}: {r['completed']} completions, "
              f"scheduler busy {100 * r['busy_frac']:.0f}% of {r['sampled_s']:.0f} s")
        for name, v in sorted(r["buckets"].items(), key=lambda kv: -kv[1]["inclusive_pct"]):
            print(f"  {name:22s} inclusive {v['inclusive_pct']:6.2f}%   "
                  f"self {v['self_pct']:6.2f}%")
    print(f"\nwrote {outdir}/meta.json")


if __name__ == "__main__":
    main()
