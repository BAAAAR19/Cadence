"""Step 3.5 -- what the C++ port is worth, measured three ways.

The guide's rule is to report three numbers rather than one, because a 40x
microbenchmark win can be a 0% end-to-end win and saying so is the difference
between a portfolio and a sales pitch:

1. **match latency against prompt length.** Does the data structure behave the
   way its asymptotics say it should, in both implementations?
2. **allocator throughput.** Was the port worth doing at the level of the
   operation?
3. **the cost of a scheduler step.** The per-operation numbers multiplied by
   how many operations a real step performs, against the measured mean step
   duration from ``results/w3_profile/meta.json``. This is the number that
   decides whether the end-to-end delta can be anything but zero -- and it is
   computed here rather than hoped for.

The fourth measurement is the one that settled a design decision: the price of
releasing and re-acquiring the GIL, against the duration of the call it would
be released around.

    uv run bench/bench_core.py --out results/w3_bench/bench.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cadence import _core  # noqa: E402
from cadence.engine.kv.block_manager import BlockManager as PyBlocks  # noqa: E402
from cadence.engine.kv.radix_cache import RadixCache as PyCache  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
B = 16


def timeit(fn, *, reps: int, inner: int = 1) -> float:
    """Best-of-`reps` seconds per operation.

    Best-of rather than mean: on a laptop the distribution's tail is other
    processes, not the code under test, and the minimum is the least
    contaminated estimator of what the code costs.
    """
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) / inner)
    return best


# --- 1. match latency vs prompt length -----------------------------------
def bench_match(lengths: list[int], reps: int) -> list[dict]:
    """A shared system prompt with a short unique tail: the workload's shape,
    and the case the cache exists to serve."""
    out = []
    for n in lengths:
        rows = {"prompt_tokens": n}
        for name, mk in (("python", _py_pair), ("cpp", _cpp_pair)):
            blocks, cache = mk(n_blocks=4 * (n // B) + 64)
            prompt = list(range(1000, 1000 + n))
            held = blocks.alloc(n // B)
            cache.insert(prompt, held, 0)
            blocks.release(held)
            probe = prompt[: n - B] + list(range(9000, 9000 + B))
            calls = 200

            def once(cache=cache, probe=probe, calls=calls):
                for _ in range(calls):
                    cache.match(probe, count=False)

            rows[name] = timeit(once, reps=reps, inner=calls) * 1e6  # microseconds
            rows[f"{name}_hit_tokens"] = cache.match(probe, count=False).n_tokens

        # How much of the C++ call is the pybind11 boundary rather than the
        # tree: the same prompt list, into a function that does nothing but
        # accept it.
        def marshal(probe=probe, calls=calls):
            for _ in range(calls):
                _core._consume_tokens(probe)

        rows["cpp_marshalling"] = timeit(marshal, reps=reps, inner=calls) * 1e6
        rows["cpp_tree_walk"] = rows["cpp"] - rows["cpp_marshalling"]
        rows["speedup"] = rows["python"] / rows["cpp"]
        out.append(rows)
    return out


def _py_pair(n_blocks: int):
    b = PyBlocks(n_blocks, B)
    return b, PyCache(b, B)


def _cpp_pair(n_blocks: int):
    b = _core.BlockAllocator(n_blocks, B)
    return b, _core.RadixCache(b, B)


# --- 2. allocator throughput ---------------------------------------------
def bench_allocator(reps: int) -> dict:
    """The operations a decode step actually performs, in the proportions it
    performs them: one ``can_append`` and one ``append_slot`` per running
    sequence per token, and an alloc/release pair per request."""
    out: dict[str, dict] = {}
    for name, mk in (("python", lambda: PyBlocks(4096, B)), ("cpp", lambda: _core.BlockAllocator(4096, B))):
        blocks = mk()
        table = blocks.alloc(34)  # a ~540-token prompt at 16 tokens a block
        filled = 34 * B - 3
        calls = 2000

        def grow(blocks=blocks, table=table, filled=filled, calls=calls):
            for i in range(calls):
                n = filled + (i % 3)
                if blocks.can_append(table, n):
                    pass

        def alloc_release(blocks=blocks, calls=calls):
            for _ in range(calls):
                blocks.release(blocks.alloc(34))

        out[name] = {
            "can_append_us": timeit(grow, reps=reps, inner=calls) * 1e6,
            "alloc_release_34_us": timeit(alloc_release, reps=reps, inner=calls) * 1e6,
        }
    for k in ("can_append_us", "alloc_release_34_us"):
        out.setdefault("speedup", {})[k] = out["python"][k] / out["cpp"][k]
    return out


# --- 3. what that is as a share of one engine step ------------------------
def step_share(match_us: dict, alloc: dict, batches: list[int],
               mean_step_s: float | None) -> dict:
    """One step: ``can_append`` for every running sequence, and one prefix
    match for the request at the head of the wait queue.

    ``append_slot`` is the same order as ``can_append`` and is folded in as a
    second call; the point of the number is its magnitude, not its third
    significant figure.

    Reported across a range of batch sizes rather than at one, because the
    mean running-batch size is not directly measured and the conclusion should
    not depend on guessing it. The cost is linear in the batch, so bracketing
    it -- 8 to the configured ceiling of 24 -- is a stronger statement than
    picking a number in the middle.
    """
    rows: dict = {"mean_step_s": mean_step_s, "batches": batches, "by_batch": []}
    for batch in batches:
        row = {"batch": batch}
        for name in ("python", "cpp"):
            per_step_us = 2 * batch * alloc[name]["can_append_us"] + match_us[name]
            row[name] = {
                "per_step_us": per_step_us,
                "share_of_step": (per_step_us / 1e6) / mean_step_s if mean_step_s else None,
            }
        rows["by_batch"].append(row)
    return rows


# --- 4. the price of releasing the GIL ------------------------------------
def bench_gil(reps: int) -> dict:
    """The build guide recommends ``py::call_guard<py::gil_scoped_release>``
    around ``match``. This prices it: the same empty function, bound with and
    without the guard. If the difference is a meaningful fraction of a match,
    releasing the GIL is a pessimisation, and it also costs the atomicity that
    holding it provides against the /stats reader on the API thread."""
    calls = 20000

    def held():
        for _ in range(calls):
            _core._noop()

    def released():
        for _ in range(calls):
            _core._noop_gil_released()

    a = timeit(held, reps=reps, inner=calls) * 1e9
    b = timeit(released, reps=reps, inner=calls) * 1e9
    return {"call_ns": a, "call_with_gil_release_ns": b, "release_cost_ns": b - a}


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--lengths", default="128,256,512,1024,2048")
    p.add_argument("--batches", default="8,12,24",
                   help="running sequences per step, for the share-of-a-step arithmetic")
    p.add_argument("--profile", default="results/w3_profile/meta.json",
                   help="where to read the measured mean step duration from")
    p.add_argument("--out", default="results/w3_bench/bench.json")
    a = p.parse_args(argv)

    lengths = [int(x) for x in a.lengths.split(",") if x]
    match = bench_match(lengths, a.reps)
    alloc = bench_allocator(a.reps)
    gil = bench_gil(a.reps)

    mean_step_s = None
    prof = Path(a.profile)
    if prof.exists():
        meta = json.loads(prof.read_text())
        for r in meta.get("runs", []):
            if r.get("backend") == "llamacpp" and (r.get("steps") or {}).get("mean_step_s"):
                mean_step_s = r["steps"]["mean_step_s"]
                break

    # The 512-token row is closest to the workload's own ~540-token prompts,
    # so that is the row the share-of-a-step arithmetic uses.
    at_512 = next((r for r in match if r["prompt_tokens"] == 512), match[len(match) // 2])
    share = step_share(
        {"python": at_512["python"], "cpp": at_512["cpp"]},
        alloc,
        [int(x) for x in a.batches.split(",") if x],
        mean_step_s,
    )

    payload = {
        "machine": f"{platform.machine()} / {platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "reps": a.reps,
        "match_us_by_prompt_tokens": match,
        "allocator": alloc,
        "step_share": share,
        "gil": gil,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print("match latency (us/call): python / cpp / of which marshalling / speedup")
    for r in match:
        print(f"  {r['prompt_tokens']:5d}  {r['python']:8.2f}  {r['cpp']:8.2f}  "
              f"{r['cpp_marshalling']:8.2f}  {r['speedup']:6.1f}x")
    print("\nallocator (us/call)")
    for k in ("can_append_us", "alloc_release_34_us"):
        print(f"  {k:22s} {alloc['python'][k]:8.3f}  {alloc['cpp'][k]:8.3f}  "
              f"{alloc['speedup'][k]:6.1f}x")
    print(f"\nKV work per step (mean step {mean_step_s})")
    for row in share["by_batch"]:
        for name in ("python", "cpp"):
            v = row[name]
            pct = f"{100 * v['share_of_step']:.3f}%" if v["share_of_step"] else "n/a"
            print(f"  batch {row['batch']:2d}  {name:7s} {v['per_step_us']:8.1f} us   "
                  f"{pct} of a step")
    print(f"\nGIL: a bound no-op costs {gil['call_ns']:.0f} ns; releasing and "
          f"re-acquiring adds {gil['release_cost_ns']:.0f} ns")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
