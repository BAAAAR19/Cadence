"""Rung 1 of the ablation ladder: one request in flight, strict FIFO, no
batching.

Its purpose is to be beaten, visibly, on a chart. Under an open-loop arrival
process its latency should track what queueing theory predicts for an M/G/1
queue: fine below saturation, unbounded above it.
"""

from __future__ import annotations

import time
from collections import deque

from cadence.engine.request import Request, State
from cadence.engine.scheduler.base import BaseScheduler


class FifoScheduler(BaseScheduler):
    name = "fifo"

    def __init__(self, runner, cfg, metrics) -> None:
        super().__init__(runner, cfg, metrics)
        self.queue: deque[Request] = deque()

    def loop(self) -> None:
        while not self._stop.is_set():
            self.queue.extend(self._take_incoming())
            self.n_waiting = len(self.queue)
            self._gauges()
            if not self.queue:
                self._idle_wait()
                continue

            rq = self.queue.popleft()
            self.n_waiting = len(self.queue)
            self.st_sum_remaining = rq.max_tokens
            if rq.cancelled:
                continue
            self._serve(rq)

    def _serve(self, rq: Request) -> None:
        seq_id = self.seq_ids.alloc()
        while seq_id is None:  # single-flight, so this only happens at startup
            time.sleep(0.001)
            seq_id = self.seq_ids.alloc()

        live = self._begin(rq, seq_id)
        self.n_running = 1
        self._gauges()
        rq.mark("dequeued")
        rq.transition(State.PREFILL)
        self.metrics.queue_wait.observe(rq.ts["dequeued"] - rq.arrival)

        reason = "stop"
        try:
            t0 = time.perf_counter()
            tok = self.runner.prefill([live.seq])[0]
            self._observe_step(time.perf_counter() - t0, 1)
            self.metrics.step_latency.observe(time.perf_counter() - t0)
            self.metrics.tokens("prefill", len(rq.prompt_ids))
            self.metrics.prefill_batch.observe(1)

            cont = self._emit(live, tok)
            while cont and not self._stop.is_set():
                t0 = time.perf_counter()
                tok = self.runner.decode_step([live.seq])[0]
                dt = time.perf_counter() - t0
                self._observe_step(dt, 1)
                self.metrics.step_latency.observe(dt)
                cont = self._emit(live, tok)
            if len(rq.output_ids) >= rq.max_tokens:
                reason = "length"
        except Exception as exc:
            rq.fail(exc)
            reason = "error"
        finally:
            self.runner.free(seq_id)
            self.seq_ids.free(seq_id)
            self.n_running = 0
            self._gauges()
        if reason != "error":
            self._finish(live, reason)
