"""Measure the machine before choosing a sweep grid.

The build guide's example sweep runs from 1 to 16 rps. That grid belongs to a
particular machine; on this one a request is ~540 prompt tokens and ~160 output
tokens against a 0.5B model at ~80 tok/s single-stream, so saturation is below
1 rps and a 1-16 rps sweep would consist entirely of points past collapse.

This script produces the numbers the grid and the SLO are chosen from, and its
output is committed alongside the results so the choice is auditable.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

from cadence.config import Settings  # noqa: E402
from cadence.engine.backends.base import SeqState  # noqa: E402
from cadence.engine.backends.llamacpp import LlamaCppRunner  # noqa: E402
from workloads import build_workload  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-ctx", type=int, default=8192)
    p.add_argument("--batches", default="1,2,4,8")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    cfg = Settings(n_ctx=a.n_ctx, n_batch=512, n_parallel=32, temperature=0.0)
    runner = LlamaCppRunner(cfg)
    rng = random.Random(0)
    wl = build_workload("mixed")
    samples = [wl.sample(rng) for _ in range(64)]
    prompts = [runner.tokenize(runner.format_chat(s["messages"])) for s in samples]
    plens = sorted(len(p_) for p_ in prompts)
    outs = sorted(s["max_tokens"] for s in samples)

    def qtl(xs, q):
        return xs[min(len(xs) - 1, int(q * len(xs)))]

    report: dict = {
        "n_ctx": a.n_ctx,
        "kv_cells": a.n_ctx,
        "prompt_tokens": {"p50": qtl(plens, 0.5), "p95": qtl(plens, 0.95), "max": plens[-1]},
        "requested_output_tokens": {
            "p50": qtl(outs, 0.5), "mean": statistics.mean(outs),
            "p95": qtl(outs, 0.95), "max": outs[-1],
        },
        "decode": {},
    }

    prompt = prompts[0]
    seq = SeqState(seq_id=0, prompt_ids=list(prompt))
    t = time.perf_counter()
    runner.prefill([seq])
    dt = time.perf_counter() - t
    report["prefill"] = {
        "tokens": len(prompt), "seconds": dt, "tokens_per_s": len(prompt) / dt
    }
    runner.free(0)

    for b in (int(x) for x in a.batches.split(",")):
        if b * (len(prompt) + a.steps) > a.n_ctx:
            report["decode"][b] = {"skipped": "would exceed the KV cache"}
            continue
        seqs = [SeqState(seq_id=i, prompt_ids=list(prompt)) for i in range(b)]
        runner.prefill(seqs)
        t = time.perf_counter()
        for _ in range(a.steps):
            runner.decode_step(seqs)
        dt = time.perf_counter() - t
        report["decode"][b] = {
            "step_ms": dt / a.steps * 1e3,
            "tokens_per_s_aggregate": a.steps * b / dt,
            "tokens_per_s_per_seq": a.steps / dt,
        }
        for i in range(b):
            runner.free(i)

    # Implied capacity: one request costs a prefill plus its share of decode.
    mean_out = report["requested_output_tokens"]["mean"]
    best_b = max(
        (b for b, v in report["decode"].items() if "tokens_per_s_aggregate" in v),
        key=lambda b: report["decode"][b]["tokens_per_s_aggregate"],
    )
    solo = report["decode"][1]["tokens_per_s_aggregate"]
    best = report["decode"][best_b]["tokens_per_s_aggregate"]
    prefill_s = report["prefill"]["seconds"]
    report["implied_capacity_rps"] = {
        "fifo": 1.0 / (prefill_s + mean_out / solo),
        "batched_best_case": 1.0 / (prefill_s + mean_out / best),
        "best_batch_size": best_b,
    }
    report["implied_unloaded_e2e_s"] = {
        "p50": prefill_s + report["requested_output_tokens"]["p50"] / solo,
        "p95": prefill_s + report["requested_output_tokens"]["p95"] / solo,
    }

    runner.close()
    text = json.dumps(report, indent=2)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
