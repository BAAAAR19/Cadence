"""Machinery shared by all three rungs of the ablation ladder.

The one structural decision worth calling out is threading. ``prefill`` and
``decode_step`` are synchronous and compute-bound; calling them on the API
event loop blocks every other coroutine, including the SSE writers, which
corrupts inter-token-latency measurements at exactly the scale being measured.
So the scheduler owns a dedicated thread, requests cross into it through a
lock-protected deque, and tokens cross back out through
``loop.call_soon_threadsafe``. Everything below runs on the engine thread.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from cadence.engine.backends.base import DetokenizerState, SeqState
from cadence.engine.request import Request, State


@dataclass(frozen=True, slots=True)
class Snapshot:
    """System state as admission sees it, at one instant.

    Week 4's feature vector is half request and half this. It is a frozen
    dataclass rather than the ``stats()`` dict because the predictor is fitted
    on these fields by name: a typo in a dict key would train a model on a
    column of NaN and still run.

    Read from the API thread, written by the engine thread, and deliberately
    not locked. Every field is a scalar whose write is a single bytecode, so a
    reader sees old or new and never a torn value; what it can see is a *set*
    of fields that were never simultaneously true -- ``batch_size`` from after
    a step and ``queue_depth`` from before it. That costs the predictor a
    little accuracy on one feature of one request, which is noise of the same
    kind as the measurement itself. A lock would put the API thread on the
    engine thread's critical path to buy nothing, and admission has to be
    cheap: it runs on every arrival, including the ones that are about to be
    refused.
    """

    queue_depth: int = 0
    batch_size: int = 0
    sum_remaining_tokens: int = 0
    kv_free_frac: float = 1.0
    ewma_step_latency_s: float = 0.0
    ewma_tokens_per_s: float = 0.0
    arrival_rate: float = 0.0


class SeqIdPool:
    """llama.cpp sequence ids are a fixed, scarce resource (``n_seq_max``).

    They are handed out to running requests *and* held by the prefix cache for
    the sequences whose KV backs a cached prefix, so exhaustion is a normal
    condition the scheduler must handle rather than an error.
    """

    def __init__(self, n: int) -> None:
        self.n = n
        self._free = list(range(n - 1, -1, -1))

    def alloc(self) -> int | None:
        return self._free.pop() if self._free else None

    def free(self, seq_id: int) -> None:
        if seq_id < 0:
            return
        self._free.append(seq_id)

    @property
    def available(self) -> int:
        return len(self._free)


@dataclass(eq=False)
class Live:
    """A request that currently owns backend state."""

    rq: Request
    seq: SeqState
    detok: DetokenizerState
    last_token_t: float = 0.0
    prefix_node: Any = None
    """The prefix-cache handle this request pinned, or None.

    Untyped on purpose: it is whatever the cache that produced it hands out --
    a Python ``Node`` or a C++ ``NodeRef`` -- and the scheduler's only contract
    is to give it back to the same cache. See cadence.engine.kv.protocols.
    """
    matched_tokens: int = 0
    shared_blocks: list[int] = field(default_factory=list)


class BaseScheduler:
    name = "base"

    def __init__(self, runner, cfg, metrics) -> None:
        self.runner = runner
        self.cfg = cfg
        self.metrics = metrics
        self.seq_ids = SeqIdPool(getattr(runner, "n_seq_max", cfg.n_parallel))

        self._incoming: deque[Request] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.n_waiting = 0
        self.n_running = 0

        # --- state the admission controller reads (see Snapshot) ----------
        self.st_sum_remaining = 0
        self.st_ewma_step_s = 0.0
        self.st_ewma_tokens_per_s = 0.0
        self._arrival_rate = 0.0
        self._last_arrival_t = 0.0

    # --- lifecycle --------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="cadence-engine", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    # --- state for admission ---------------------------------------------
    ARRIVAL_TAU_S = 10.0
    """Time constant of the arrival-rate estimator. Ten seconds is roughly
    twenty arrivals at the loads this project runs at -- long enough for the
    estimate not to be dominated by one exponential gap, short enough to track
    a step change in offered load within the warm-up window."""

    EWMA_ALPHA = 0.1
    """Weight on the newest engine step. The step rate is 20-80 Hz, so this is
    an average over roughly the last quarter-second of work."""

    def note_arrival(self) -> None:
        """One request arrived at the gateway. Called from the API thread,
        *before* the admission decision, so the rate this estimates is offered
        load rather than admitted load -- a controller that measured only what
        it let in would see its own shedding as the arrival rate falling.

        The estimator is an exponentially-decaying rate rather than an EWMA of
        inter-arrival gaps: 1/gap has infinite mean for a Poisson process, so
        averaging it is dominated by whichever gap happened to be smallest.
        """
        now = time.perf_counter()
        prev, self._last_arrival_t = self._last_arrival_t, now
        if prev == 0.0:
            self._arrival_rate = 1.0 / self.ARRIVAL_TAU_S
            return
        dt = max(0.0, now - prev)
        self._arrival_rate = (
            self._arrival_rate * math.exp(-dt / self.ARRIVAL_TAU_S) + 1.0 / self.ARRIVAL_TAU_S
        )

    def _observe_step(self, dt: float, n_tokens: int) -> None:
        """One engine step took ``dt`` seconds and produced ``n_tokens``.

        Both EWMAs are what makes the predictor's system-state features about
        *this* machine under *this* load rather than about the workload alone:
        the same queue depth means something different at 12 ms per step than
        at 50 ms.
        """
        a = self.EWMA_ALPHA
        self.st_ewma_step_s = (1 - a) * self.st_ewma_step_s + a * dt if self.st_ewma_step_s else dt
        if dt > 0:
            tps = n_tokens / dt
            self.st_ewma_tokens_per_s = (
                (1 - a) * self.st_ewma_tokens_per_s + a * tps
                if self.st_ewma_tokens_per_s
                else tps
            )

    def snapshot(self) -> Snapshot:
        """The system half of the admission feature vector."""
        return Snapshot(
            queue_depth=self.n_waiting,
            batch_size=self.n_running,
            sum_remaining_tokens=self.st_sum_remaining,
            kv_free_frac=self.kv_free_frac(),
            ewma_step_latency_s=self.st_ewma_step_s,
            ewma_tokens_per_s=self.st_ewma_tokens_per_s,
            arrival_rate=self._arrival_rate,
        )

    def kv_free_frac(self) -> float:
        """Fraction of the KV pool not currently allocated. Rungs 1-3 run the
        contiguous allocator through the backend and do not track it, so they
        report a full pool rather than a wrong number."""
        return 1.0

    def probe_prefix(self, prompt_ids: list[int]) -> int:
        """Prompt tokens a prefix cache would serve, if there is one.

        Zero everywhere except the continuous scheduler; see the override
        there for why the answer is best-effort.
        """
        return 0

    def submit(self, rq: Request) -> None:
        """Called from the API event loop. The only cross-thread entry point."""
        if rq.trace is not None:
            rq.trace.phase("queue")
        rq.transition(State.WAITING)
        with self._lock:
            self._incoming.append(rq)
        self._wake.set()

    def _take_incoming(self) -> list[Request]:
        with self._lock:
            out = list(self._incoming)
            self._incoming.clear()
        return out

    def _idle_wait(self, timeout: float = 0.05) -> None:
        self._wake.wait(timeout)
        self._wake.clear()

    def _run(self) -> None:  # pragma: no cover - thin wrapper
        try:
            self.loop()
        except Exception as exc:  # surface, never silently die
            import traceback

            traceback.print_exc()
            self._fail_everything(exc)

    def loop(self) -> None:
        raise NotImplementedError

    def _fail_everything(self, exc: BaseException) -> None:
        for rq in self._take_incoming():
            rq.fail(exc)

    # --- per-request state ------------------------------------------------
    def _begin(self, rq: Request, seq_id: int, n_past: int = 0) -> Live:
        if rq.trace is not None:
            rq.trace.phase(
                "prefill",
                prompt_tokens=len(rq.prompt_ids),
                cached_tokens=n_past,
                seq_id=seq_id,
            )
        seq = SeqState(seq_id=seq_id, prompt_ids=rq.prompt_ids, n_past=n_past)
        rq.seq_id = seq_id
        rq.n_past = n_past
        return Live(rq=rq, seq=seq, detok=DetokenizerState(self.runner))

    def _emit(self, live: Live, token_id: int) -> bool:
        """Push one sampled token to the client. Returns False if the request
        should stop (EOS, budget exhausted, or client gone)."""
        rq, seq = live.rq, live.seq
        now = time.perf_counter()
        if rq.cancelled:
            return False

        stop = token_id in self.runner.eos_ids
        if not stop:
            text = live.detok.push(token_id)
            if text:
                rq.emit(text)
        rq.output_ids.append(token_id)
        rq.n_past = seq.n_past

        if "first_token" not in rq.ts:
            rq.mark("first_token")
            if rq.trace is not None:
                rq.trace.phase("decode")
            rq.transition(State.DECODING)
            self.metrics.ttft.observe(now - rq.arrival)
        elif live.last_token_t:
            self.metrics.itl.observe(now - live.last_token_t)
        live.last_token_t = now

        if stop:
            return False
        return len(rq.output_ids) < rq.max_tokens

    def _finish(self, live: Live, reason: str) -> None:
        rq = live.rq
        # Close the phase span before the terminal transition, so the DONE
        # event lands on the request span rather than inside "decode".
        if rq.trace is not None:
            rq.trace.end_phase()
        if rq.cancelled:
            self.metrics.finished("cancelled")
        else:
            rq.finish(reason)
            self.metrics.e2e.observe(rq.e2e or 0.0)
            if (rq.e2e or 0.0) <= self.cfg.slo_s:
                self.metrics.slo_met_total.inc()
            self.metrics.finished("done")
        self.metrics.tokens("output", len(rq.output_ids))
        if rq.trace is not None:
            rq.trace.end(
                **{
                    "cadence.finish_reason": reason,
                    "cadence.output_tokens": len(rq.output_ids),
                    "cadence.cached_prefix_tokens": rq.cached_prefix_len,
                    "cadence.preemptions": rq.n_preemptions,
                    "cadence.e2e_seconds": rq.e2e or 0.0,
                }
            )

    def _gauges(self) -> None:
        self.metrics.queue_depth.set(self.n_waiting)
        self.metrics.batch_size.set(self.n_running)

    # --- introspection for tests / the /stats endpoint --------------------
    def stats(self) -> dict:
        return {
            "scheduler": self.name,
            "waiting": self.n_waiting,
            "running": self.n_running,
            "seq_ids_free": self.seq_ids.available,
            "sum_remaining_tokens": self.st_sum_remaining,
            "ewma_step_latency_s": self.st_ewma_step_s,
            "ewma_tokens_per_s": self.st_ewma_tokens_per_s,
            "arrival_rate": self._arrival_rate,
        }
