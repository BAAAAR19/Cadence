"""Rung 2: static batching.

Requests are collected into a wave of up to ``max_batch``, prefilled together,
then decoded together until *every* member of the wave has finished. Nothing
new joins in the meantime.

That last sentence is the whole point of the rung. Output lengths in the
workload are lognormal, so one long generation holds the entire batch open and
the sequences that finished early keep their KV reserved while contributing
nothing. Continuous batching is precisely the removal of that constraint, and
this rung is what makes the size of that win legible.
"""

from __future__ import annotations

import time
from collections import deque

from cadence.engine.request import State
from cadence.engine.scheduler.base import BaseScheduler, Live


class StaticBatchScheduler(BaseScheduler):
    name = "static"

    def __init__(self, runner, cfg, metrics) -> None:
        super().__init__(runner, cfg, metrics)
        self.queue: deque = deque()

    def loop(self) -> None:
        while not self._stop.is_set():
            self.queue.extend(self._take_incoming())
            self.n_waiting = len(self.queue)
            self._gauges()
            if not self.queue:
                self._idle_wait()
                continue

            # Wait-to-fill: give the wave a short window to reach full size
            # rather than launching a batch of one the instant a request lands.
            target = self.cfg.static_batch_size
            deadline = time.perf_counter() + self.cfg.static_fill_timeout_s
            while len(self.queue) < target and time.perf_counter() < deadline:
                self._idle_wait(0.005)
                self.queue.extend(self._take_incoming())
            self.n_waiting = len(self.queue)

            wave: list[Live] = []
            while self.queue and len(wave) < target:
                rq = self.queue.popleft()
                if rq.cancelled:
                    continue
                seq_id = self.seq_ids.alloc()
                if seq_id is None:
                    self.queue.appendleft(rq)
                    break
                rq.mark("dequeued")
                rq.transition(State.PREFILL)
                self.metrics.queue_wait.observe(rq.ts["dequeued"] - rq.arrival)
                wave.append(self._begin(rq, seq_id))
            self.n_waiting = len(self.queue)
            if not wave:
                self._idle_wait()
                continue
            self._run_wave(wave)

    def _run_wave(self, wave: list[Live]) -> None:
        self.n_running = len(wave)
        self.st_sum_remaining = sum(lv.rq.max_tokens for lv in wave)
        self._gauges()
        active = list(wave)
        try:
            t0 = time.perf_counter()
            toks = self.runner.prefill([lv.seq for lv in active])
            self.metrics.step_latency.observe(time.perf_counter() - t0)
            self.metrics.prefill_batch.observe(len(active))
            self.metrics.tokens("prefill", sum(len(lv.rq.prompt_ids) for lv in active))

            done: list[Live] = []
            still: list[Live] = []
            for lv, tok in zip(active, toks, strict=True):
                (still if self._emit(lv, tok) else done).append(lv)

            # The batch stays open until the slowest member is finished.
            while still and not self._stop.is_set():
                t0 = time.perf_counter()
                toks = self.runner.decode_step([lv.seq for lv in still])
                dt = time.perf_counter() - t0
                self._observe_step(dt, len(still))
                self.metrics.step_latency.observe(dt)
                nxt: list[Live] = []
                for lv, tok in zip(still, toks, strict=True):
                    (nxt if self._emit(lv, tok) else done).append(lv)
                still = nxt
            done.extend(still)
        except Exception as exc:
            for lv in active:
                lv.rq.fail(exc)
            done = []
        finally:
            for lv in active:
                self.runner.free(lv.seq.seq_id)
                self.seq_ids.free(lv.seq.seq_id)
            self.n_running = 0
            self._gauges()
        for lv in done:
            reason = "length" if len(lv.rq.output_ids) >= lv.rq.max_tokens else "stop"
            self._finish(lv, reason)
