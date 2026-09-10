"""Demonstrate the graceful drain, and fail if it is not graceful.

The deployment checklist asks for a drain that works, and "works" here is
three separate claims that a screenshot of a shutdown cannot distinguish:

1. ``/ready`` goes 503 as soon as SIGTERM lands, so a load balancer stops
   routing within one health-check interval;
2. requests that were already streaming finish, with a real
   ``finish_reason`` -- not a truncated stream, which would put a burst of
   half-written responses into exactly the tail the project claims to
   control;
3. requests that arrive after the signal are refused with a 503 and a
   ``Retry-After``, rather than accepted into a process that is leaving.

So this drives all three against a real server on a real socket and asserts
them. It runs against the mock backend by default -- the claims are about the
API and the lifecycle, not about the model -- and against llama.cpp with
``--backend llamacpp`` when the point is to watch it happen on the real thing.

    uv run bench/demo_drain.py
    uv run bench/demo_drain.py --backend llamacpp --n 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent


async def _stream(client: httpx.AsyncClient, base: str, i: int, max_tokens: int) -> dict:
    """One streaming request. Returns what became of it."""
    t0 = time.perf_counter()
    body = {
        "model": "qwen",
        "stream": True,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": f"tell me about number {i}"}],
    }
    tokens, finish, status = 0, None, None
    try:
        async with client.stream("POST", f"{base}/v1/chat/completions", json=body) as r:
            status = r.status_code
            if r.status_code != 200:
                await r.aread()
                return {"i": i, "status": status, "tokens": 0, "finish": None,
                        "retry_after": r.headers.get("Retry-After"),
                        "e2e": time.perf_counter() - t0}
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk["choices"][0]["delta"].get("content"):
                    tokens += 1
                if chunk["choices"][0].get("finish_reason"):
                    finish = chunk["choices"][0]["finish_reason"]
    except Exception as exc:
        return {"i": i, "status": status, "tokens": tokens, "finish": None,
                "error": type(exc).__name__, "e2e": time.perf_counter() - t0}
    return {"i": i, "status": status, "tokens": tokens, "finish": finish,
            "e2e": time.perf_counter() - t0, "t_done": time.perf_counter()}


async def drive(base: str, proc: subprocess.Popen, n: int, max_tokens: int,
                settle_s: float) -> dict:
    async with httpx.AsyncClient(timeout=180.0) as c:
        # In flight when the signal lands.
        inflight = [asyncio.create_task(_stream(c, base, i, max_tokens)) for i in range(n)]
        # Long enough for every one of them to be past its first token, so
        # the drain is interrupting real work rather than racing the admit --
        # and short enough that none of them has finished, which is checked
        # below rather than assumed. A "drain" that signals an idle server
        # demonstrates nothing.
        await asyncio.sleep(settle_s)

        before = await c.get(f"{base}/ready")
        t_sig = time.perf_counter()
        proc.send_signal(signal.SIGTERM)

        # How long until the process takes itself out of rotation.
        unready_after = None
        deadline = time.perf_counter() + 10.0
        while time.perf_counter() < deadline:
            try:
                r = await c.get(f"{base}/ready", timeout=2.0)
                if r.status_code == 503:
                    unready_after = time.perf_counter() - t_sig
                    ready_body = r.json()
                    break
            except Exception:
                break
            await asyncio.sleep(0.05)
        else:
            ready_body = {}

        # A request that arrives after the signal.
        try:
            late = await c.post(
                f"{base}/v1/chat/completions",
                json={"model": "qwen", "stream": False, "max_tokens": 8,
                      "messages": [{"role": "user", "content": "too late"}]},
                timeout=30.0,
            )
            late_status = late.status_code
            late_retry = late.headers.get("Retry-After")
        except Exception as exc:
            late_status, late_retry = type(exc).__name__, None

        done = await asyncio.gather(*inflight)

    rc = proc.wait(timeout=120)
    return {
        "t_signal": t_sig,
        "ready_before_signal": before.status_code,
        "unready_after_s": unready_after,
        "ready_body": ready_body,
        "late_status": late_status,
        "late_retry_after": late_retry,
        "inflight": done,
        "exit_code": rc,
        "drain_wall_s": time.perf_counter() - t_sig,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="mock", choices=("mock", "llamacpp"))
    p.add_argument("--port", type=int, default=8410)
    p.add_argument("--n", type=int, default=6, help="streams in flight when SIGTERM lands")
    p.add_argument("--max-tokens", type=int, default=320,
                   help="long enough that the streams are still running when "
                        "the signal lands; the run fails if they are not")
    p.add_argument("--settle", type=float, default=1.0,
                   help="seconds between the first arrivals and SIGTERM")
    p.add_argument("--grace", type=float, default=90.0)
    p.add_argument("--json-out", default=None)
    a = p.parse_args(argv)

    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "CADENCE_BACKEND": a.backend,
        "CADENCE_CONFIG_NAME": "drain-demo",
        "CADENCE_SCHEDULER": "continuous",
        "CADENCE_PORT": str(a.port),
        "CADENCE_ADMISSION": "none",
        "CADENCE_DRAIN_GRACE_S": str(a.grace),
        "CADENCE_N_CTX": "16384",
        "CADENCE_MAX_BATCH": "16",
    }
    base = f"http://127.0.0.1:{a.port}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "cadence.api.app"], env=env, cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.time() + 240
        while time.time() < deadline:
            if proc.poll() is not None:
                print("server exited before becoming ready", file=sys.stderr)
                return 2
            try:
                if httpx.get(f"{base}/ready", timeout=2.0).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.25)
        else:
            print("server never became ready", file=sys.stderr)
            return 2

        result = asyncio.run(drive(base, proc, a.n, a.max_tokens, a.settle))
    finally:
        if proc.poll() is None:
            proc.kill()

    finished = [r for r in result["inflight"] if r.get("finish") in {"stop", "length"}]
    truncated = [r for r in result["inflight"] if r.get("finish") not in {"stop", "length"}]

    print(f"\n  backend                {a.backend}")
    print(f"  /ready before SIGTERM  {result['ready_before_signal']}")
    print(f"  /ready 503 after       {result['unready_after_s']:.3f}s"
          if result["unready_after_s"] is not None
          else "  /ready 503 after       never")
    print(f"  request arriving late  {result['late_status']} "
          f"(Retry-After: {result['late_retry_after']})")
    n_after = sum(1 for r in result["inflight"] if r.get("t_done", 0.0) > result["t_signal"])
    print(f"  in-flight streams      {len(finished)}/{len(result['inflight'])} finished "
          f"cleanly, {len(truncated)} truncated, {n_after} of them completing "
          f"after the signal")
    for r in result["inflight"]:
        print(f"    #{r['i']}: {r.get('tokens', 0):>4} tokens, finish={r.get('finish')!r}, "
              f"{r.get('e2e', float('nan')):.2f}s"
              + (f", error={r['error']}" if r.get("error") else ""))
    print(f"  drain wall time        {result['drain_wall_s']:.2f}s")
    print(f"  exit code              {result['exit_code']}")

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(result, indent=2, default=str) + "\n")

    problems = []
    if result["ready_before_signal"] != 200:
        problems.append("the server was not ready before the signal")
    if result["unready_after_s"] is None or result["unready_after_s"] > 1.0:
        problems.append("/ready did not go 503 within a second of SIGTERM")
    if result["late_status"] != 503:
        problems.append(f"a request arriving during the drain got {result['late_status']}")
    if result["late_retry_after"] is None:
        problems.append("the refusal carried no Retry-After")
    if truncated:
        problems.append(f"{len(truncated)} in-flight stream(s) were truncated by the drain")
    still_running = [
        r for r in result["inflight"] if r.get("t_done", 0.0) > result["t_signal"]
    ]
    if not still_running:
        problems.append(
            "every stream had already finished when SIGTERM landed, so nothing "
            "was drained; raise --max-tokens or lower --settle"
        )
    if result["exit_code"] not in (0, -signal.SIGTERM):
        problems.append(f"the process exited with {result['exit_code']}")

    if problems:
        print("\nFAILED:")
        for m in problems:
            print(f"  - {m}")
        return 1
    print("\nOK: drained without dropping or truncating anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
