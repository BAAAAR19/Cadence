"""The rungs of the ablation ladder, as environment overrides.

Each rung adds exactly one thing to the rung above it, so every row of the
final table is attributable to a single change.

    1  fifo                       no batching -- the baseline
    2  static                     static batching, batch 8, wait-to-fill
    3  continuous                 iteration-level scheduling
    4  continuous+cache           + paged KV and the radix prefix cache
    5  continuous+cache+admission + conformal admission control (Week 4)

Rung 3 keeps the contiguous allocator and no prefix cache on purpose: without
that, rung 4's "memory efficiency and prompt reuse" would be measured against
nothing.
"""

from __future__ import annotations

RUNGS: dict[str, dict[str, str]] = {
    "fifo": {
        "CADENCE_SCHEDULER": "fifo",
        "CADENCE_ENABLE_PAGED_KV": "false",
        "CADENCE_ENABLE_PREFIX_CACHE": "false",
        "CADENCE_ADMISSION": "none",
    },
    "static": {
        "CADENCE_SCHEDULER": "static",
        "CADENCE_STATIC_BATCH_SIZE": "8",
        "CADENCE_ENABLE_PAGED_KV": "false",
        "CADENCE_ENABLE_PREFIX_CACHE": "false",
        "CADENCE_ADMISSION": "none",
    },
    "continuous": {
        "CADENCE_SCHEDULER": "continuous",
        "CADENCE_ENABLE_PAGED_KV": "false",
        "CADENCE_ENABLE_PREFIX_CACHE": "false",
        "CADENCE_ADMISSION": "none",
    },
    "continuous+cache": {
        "CADENCE_SCHEDULER": "continuous",
        "CADENCE_ENABLE_PAGED_KV": "true",
        "CADENCE_ENABLE_PREFIX_CACHE": "true",
        "CADENCE_ADMISSION": "none",
    },
    # Week 3's A/B. Identical to `continuous+cache` in every respect except
    # which implementation of the block allocator and the radix cache is
    # loaded, so that the end-to-end delta of the C++ port is attributable to
    # the port and to nothing else. Run as two rungs of `run_ladder.py` rather
    # than as two sweeps, so that the rate-major, rotated order protects the
    # comparison from thermal drift the same way the ladder is protected.
    "kv-core-python": {
        "CADENCE_SCHEDULER": "continuous",
        "CADENCE_ENABLE_PAGED_KV": "true",
        "CADENCE_ENABLE_PREFIX_CACHE": "true",
        "CADENCE_ADMISSION": "none",
        "CADENCE_KV_CORE": "python",
    },
    "kv-core-cpp": {
        "CADENCE_SCHEDULER": "continuous",
        "CADENCE_ENABLE_PAGED_KV": "true",
        "CADENCE_ENABLE_PREFIX_CACHE": "true",
        "CADENCE_ADMISSION": "none",
        "CADENCE_KV_CORE": "cpp",
    },
    "continuous+cache+admission": {
        "CADENCE_SCHEDULER": "continuous",
        "CADENCE_ENABLE_PAGED_KV": "true",
        "CADENCE_ENABLE_PREFIX_CACHE": "true",
        "CADENCE_ADMISSION": "conformal",
        "CADENCE_ADMISSION_MODE": "static",
    },
    # The same bound, read at a weaker guarantee. Alpha is the policy's real
    # knob -- it is the strength of the promise made about each admitted
    # request -- and it is a rung rather than a knob sweep because the two
    # points bracket the trade-off the week exists to measure: at 0.01 the
    # controller refuses anything whose 99th percentile does not fit, which on
    # this workload is nearly everything; at 0.20 it refuses anything whose
    # 80th percentile does not.
    "continuous+cache+admission-a05": {
        "CADENCE_SCHEDULER": "continuous",
        "CADENCE_ENABLE_PAGED_KV": "true",
        "CADENCE_ENABLE_PREFIX_CACHE": "true",
        "CADENCE_ADMISSION": "conformal",
        "CADENCE_ADMISSION_MODE": "static",
        "CADENCE_ADMISSION_ALPHA": "0.05",
    },
    "continuous+cache+admission-a20": {
        "CADENCE_SCHEDULER": "continuous",
        "CADENCE_ENABLE_PAGED_KV": "true",
        "CADENCE_ENABLE_PREFIX_CACHE": "true",
        "CADENCE_ADMISSION": "conformal",
        "CADENCE_ADMISSION_MODE": "static",
        "CADENCE_ADMISSION_ALPHA": "0.2",
    },
    # The same controller, recalibrating on its own recent completions instead
    # of holding the offline calibration set fixed. It is a separate rung and
    # not a knob sweep because the two answer different questions: the static
    # arm is the one with the finite-sample guarantee, and this one is the
    # answer to the fact that the controller's own shedding breaks the
    # assumption that guarantee rests on. Both are measured.
    "continuous+cache+admission-rolling": {
        "CADENCE_SCHEDULER": "continuous",
        "CADENCE_ENABLE_PAGED_KV": "true",
        "CADENCE_ENABLE_PREFIX_CACHE": "true",
        "CADENCE_ADMISSION": "conformal",
        "CADENCE_ADMISSION_MODE": "rolling",
    },
}

# Knobs held constant across every rung, so that a difference between rows is a
# difference in scheduling policy and nothing else.
COMMON: dict[str, str] = {
    "CADENCE_BACKEND": "llamacpp",
    # 16384 KV cells, not 8192. At 8192 a batch of ten ~540-token prompts plus
    # their generation occupies the pool on its own, so every admission evicts
    # a cached prefix and the prefix cache measures nothing but its own
    # eviction rate. The pool has to be large enough for the thing being
    # measured to exist. 16384 cells is 192 MiB of KV -- trivial on this
    # machine, and still half the model's 32768-token training context.
    "CADENCE_N_CTX": "16384",
    "CADENCE_N_BATCH": "512",
    # Sequence ids are held by running requests *and* by the cached prefixes
    # whose KV they own, so the pool has to cover both.
    "CADENCE_N_PARALLEL": "64",
    "CADENCE_MAX_BATCH": "24",
    "CADENCE_MAX_PREFILL_TOKENS": "512",
    "CADENCE_BLOCK_SIZE": "16",
    "CADENCE_TEMPERATURE": "0.0",
    "CADENCE_MAX_TOKENS_CAP": "512",
    "CADENCE_METRICS_ENABLED": "true",
    "CADENCE_TRACING_ENABLED": "false",
}


def env_for(rung: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    if rung not in RUNGS:
        raise SystemExit(f"unknown config {rung!r}; have {', '.join(RUNGS)}")
    env = dict(COMMON)
    env.update(RUNGS[rung])
    env["CADENCE_CONFIG_NAME"] = rung
    env.update(extra or {})
    return env
