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

# Both chunked-prefill arms raise n_batch together. Unchunked prefill means
# pushing a whole ~540-token prompt through one llama_decode, which needs a
# batch that large; leaving n_batch at 512 does not make prefill unchunked, it
# just makes the configuration invalid. Raising it for *both* arms keeps the
# scheduler's prefill budget as the only difference between them.
_BIG_BATCH = ["CADENCE_N_BATCH=2048"]

# A chunk budget only chunks something if it is smaller than a prompt. This
# workload's prompts are ~540 tokens, so a 512-token budget splits a prompt
# into one full chunk and a 28-token remainder -- which is not chunking, and
# measuring it against "unchunked" measures nothing. The budget is therefore
# swept down to where it bites.
ARMS = {
    "chunk-128": [*_BIG_BATCH, "CADENCE_MAX_PREFILL_TOKENS=128"],
    "chunk-256": [*_BIG_BATCH, "CADENCE_MAX_PREFILL_TOKENS=256"],
    "chunked-prefill-512": [*_BIG_BATCH, "CADENCE_MAX_PREFILL_TOKENS=512"],
    "unchunked-prefill": [*_BIG_BATCH, "CADENCE_MAX_PREFILL_TOKENS=2048"],
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
    p.add_argument("--replicates", type=int, default=0,
                   help="repeat the unmodified configuration this many times, to "
                        "measure the run-to-run spread the knob differences have "
                        "to be larger than")
    a = p.parse_args(argv)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    arms = {k: ARMS[k] for k in a.arms.split(",") if k}
    for i in range(a.replicates):
        arms[f"replicate-{i + 1}"] = []
    for arm, overrides in arms.items():
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
