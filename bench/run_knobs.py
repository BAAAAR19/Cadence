"""The two Week-2 knob experiments, at one offered load.

Both exist because the guide is right that finding a knob and quantifying it is
worth more than picking the right default.

**Chunked prefill.** ``max_prefill_tokens`` bounds how much prompt work one
engine step may do. Set it above the longest prompt and prefill is effectively
unchunked: a long prompt arriving mid-stream stalls every decoding sequence for
the length of one large forward pass, which is a p99 ITL spike. The comparison
is published as a before/after ITL histogram, because a stall shows up as a
second mode that a quantile table hides.

**Prefill/decode ordering.** ``prefill_priority`` decides whether a step's
prefill chunks run before or after its decode. Prefill-first costs ITL;
decode-first costs TTFT. Neither is free.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ARMS = {
    "chunked-prefill-512": ["CADENCE_MAX_PREFILL_TOKENS=512"],
    "unchunked-prefill": ["CADENCE_MAX_PREFILL_TOKENS=8192"],
    "prefill-first": ["CADENCE_PREFILL_PRIORITY=true"],
    "decode-first": ["CADENCE_PREFILL_PRIORITY=false"],
}


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rate", type=float, default=0.8)
    p.add_argument("--duration", type=float, default=180.0)
    p.add_argument("--warmup", type=float, default=25.0)
    p.add_argument("--slo", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=8300)
    p.add_argument("--outdir", default="results/w2_knobs")
    p.add_argument("--arms", default=",".join(ARMS))
    a = p.parse_args(argv)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for arm in a.arms.split(","):
        overrides = ARMS[arm]
        cmd = [
            sys.executable, "bench/run_sweep.py",
            "--config", "continuous+cache", "--label", arm,
            "--rates", str(a.rate), "--duration", str(a.duration),
            "--warmup", str(a.warmup), "--cooldown", "5",
            "--slo", str(a.slo), "--seed", str(a.seed),
            "--port", str(a.port), "--out", str(out / f"{arm}.parquet"),
        ]
        for o in overrides:
            cmd += ["--set", o]
        print(f"\n=== {arm}: {' '.join(overrides)}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
