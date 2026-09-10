"""The engine: owns the backend, the scheduler and the admission policy.

It is the only object the API layer talks to, and the only place that knows
which rung of the ablation ladder is running.
"""

from __future__ import annotations

import asyncio
import time
import uuid

from cadence.admission import Decision, build_controller
from cadence.admission.features import AdmitContext, extract, to_dict
from cadence.admission.trace import TraceWriter
from cadence.engine.backends import build_runner
from cadence.engine.kv import build_kv
from cadence.engine.request import TERMINAL, Request
from cadence.engine.scheduler.continuous import ContinuousScheduler
from cadence.engine.scheduler.fifo import FifoScheduler
from cadence.engine.scheduler.static_batch import StaticBatchScheduler
from cadence.obs.metrics import Metrics
from cadence.obs.tracing import RequestTrace


def build_scheduler(runner, cfg, metrics):
    if cfg.scheduler == "fifo":
        return FifoScheduler(runner, cfg, metrics)
    if cfg.scheduler == "static":
        return StaticBatchScheduler(runner, cfg, metrics)
    if cfg.scheduler == "continuous":
        blocks, prefix, core = build_kv(cfg)
        sched = ContinuousScheduler(runner, cfg, metrics, blocks=blocks, prefix_cache=prefix)
        sched.kv_core = core
        return sched
    raise ValueError(f"unknown scheduler {cfg.scheduler!r}")


class Engine:
    def __init__(self, cfg, runner=None) -> None:
        self.cfg = cfg
        self.metrics = Metrics(cfg.config_name, cfg.metrics_enabled)
        self.runner = runner if runner is not None else build_runner(cfg)
        self.scheduler = build_scheduler(self.runner, cfg, self.metrics)
        self.controller = build_controller(cfg, self.metrics)
        self.trace_log = TraceWriter(cfg.trace_log) if cfg.trace_log else None
        self._started = False

    # --- lifecycle --------------------------------------------------------
    def start(self, warmup: bool = True) -> None:
        if not self._started:
            self.scheduler.start()
            self._started = True
            if warmup:
                self.warmup()

    def warmup(self, timeout_s: float = 120.0) -> bool:
        """Run one tiny generation through the engine before serving anything.

        Two reasons, and the second one is not obvious until it bites.

        The ordinary one: the first forward pass pays for Metal shader
        compilation and the first KV allocation, and a benchmark's first
        measured arrival should not.

        The one that actually matters here: the admission controller's
        system-state features include EWMAs of step latency and token rate,
        and an engine that has never stepped reports both as zero -- a
        combination that does not occur anywhere in the training set, because
        the collection drops each block's first twenty seconds. The model
        extrapolates, badly, and the bound comes out two to three times too
        large. Which sheds the request. Which leaves the engine idle, with the
        EWMAs still at zero, for the next one.

        That is a closed loop with no exit: measured live, a gateway that
        should have admitted most of a 1 rps load shed 89 requests out of 89,
        and the load generator's own warm-up request -- the thing that would
        have broken the cycle -- was shed along with them, because it arrives
        through the same door as everything else. Warming up *inside* the
        engine, where admission cannot refuse it, is the fix.

        The prompt is shorter than one KV block on purpose, so the request
        donates nothing to the prefix cache and the first real arrival still
        finds a cold one.
        """
        rq = Request(
            rid="warmup",
            prompt_ids=self.runner.tokenize("warm up")[: self.cfg.block_size - 1],
            max_tokens=4,
            deadline=time.perf_counter() + timeout_s,
            trace=None,
        )
        self.scheduler.submit(rq)
        deadline = time.perf_counter() + timeout_s
        while rq.state not in TERMINAL and time.perf_counter() < deadline:
            time.sleep(0.01)
        if rq.state not in TERMINAL:
            print("warm-up request did not complete; serving anyway", flush=True)
            rq.cancel()
            return False
        return True

    def stop(self) -> None:
        if self._started:
            self.scheduler.stop()
            self._started = False
        if self.trace_log is not None:
            self.trace_log.close()
        close = getattr(self.runner, "close", None)
        if close:
            close()

    # --- request path -----------------------------------------------------
    def render_prompt(self, body) -> str:
        fmt = getattr(self.runner, "format_chat", None)
        msgs = [m.model_dump() if hasattr(m, "model_dump") else m for m in body.messages]
        if fmt is not None:
            return fmt(msgs)
        return "\n".join(f"{m['role']}: {m['content']}" for m in msgs) + "\nassistant:"

    async def admit(self, body) -> Decision:
        """Cheap enough to run on the event loop: tokenisation of a 1-2 KB
        prompt is tens of microseconds, and the decision needs the token count.

        Week 4 adds a prefix-cache probe and a model evaluation to this path.
        Both are measured rather than assumed cheap --
        ``cadence_admission_decision_seconds`` is a histogram for that reason --
        because every millisecond spent here is a millisecond of TTFT for a
        request that was going to be admitted anyway.
        """
        prompt_ids = await asyncio.to_thread(self.runner.tokenize, self.render_prompt(body))
        max_tokens = body.max_tokens or self.cfg.max_tokens_default
        # Counted before the decision, so the arrival-rate feature is offered
        # load and not admitted load.
        self.scheduler.note_arrival()
        ctx = AdmitContext(
            prompt_ids=prompt_ids,
            max_tokens=max_tokens,
            n_messages=len(body.messages),
            cached_prefix_tokens=self.scheduler.probe_prefix(prompt_ids),
        )
        snap = self.scheduler.snapshot()
        decision = self.controller.decide(ctx, snap)
        decision.prompt_ids = prompt_ids
        decision.max_tokens = max_tokens
        decision.cached_prefix_tokens = ctx.cached_prefix_tokens
        if decision.features is None and self.trace_log is not None:
            # The pass-through policies do not need a feature vector; the trace
            # does, and it has to be the same vector, extracted at the same
            # instant, that a controller would have seen. This is the line that
            # makes a rung-4 run a training set for rung 5.
            decision.features = extract(ctx, snap)
        if decision.action == "shed":
            self.metrics.shed(decision.reason or "unspecified")
            self.metrics.finished("shed")
            self._trace(decision, rq=None)
        return decision

    async def submit(self, body, decision: Decision) -> Request:
        now = time.perf_counter()
        rid = uuid.uuid4().hex[:16]
        rq = Request(
            rid=rid,
            prompt_ids=list(decision.prompt_ids),
            max_tokens=int(decision.max_tokens),
            deadline=now + self.cfg.slo_s,
            arrival=now,
            loop=asyncio.get_running_loop(),
            trace=RequestTrace(
                rid,
                self.cfg.tracing_enabled,
                **{
                    "cadence.config": self.cfg.config_name,
                    "cadence.prompt_tokens": len(decision.prompt_ids),
                    "cadence.max_tokens": int(decision.max_tokens),
                },
            ),
        )
        rq.features = decision.features
        rq.predicted_e2e_s = decision.predicted_e2e_s
        rq.admit_reason = decision.reason
        rq.cached_prefix_probe = decision.cached_prefix_tokens
        self.metrics.tokens("prompt", len(rq.prompt_ids))
        self.scheduler.submit(rq)
        return rq

    def release(self, rq: Request) -> None:
        """Always called from the SSE generator's ``finally``.

        A client that disconnects mid-stream must not leave KV blocks pinned;
        skip this and throughput decays silently over a long run.
        """
        if rq.state not in ("done", "shed"):
            rq.cancel()
        self.controller.observe(rq)
        self._trace(None, rq=rq)

    # --- the training set -------------------------------------------------
    def _trace(self, decision: Decision | None, rq) -> None:
        """One row: what admission saw, and what became of the request.

        Called twice per request at most -- once here for a shed, which has no
        outcome, and once from :meth:`release` for one that ran. The feature
        half is copied from the vector the decision was taken on rather than
        re-extracted, so a row can never describe a system state that is
        newer than the decision it explains.
        """
        if self.trace_log is None:
            return
        x = decision.features if decision is not None else (rq.features if rq else None)
        if x is None:
            return
        row: dict = {
            "config": self.cfg.config_name,
            "t_wall": time.time(),
            "action": decision.action if decision is not None else "admit",
            "reason": (decision.reason if decision is not None else rq.admit_reason),
            "u_bound_s": (
                decision.predicted_e2e_s if decision is not None else rq.predicted_e2e_s
            ),
            **to_dict(x),
        }
        if rq is not None:
            row.update(
                {
                    "rid": rq.rid,
                    "t_arrival": rq.arrival,
                    "e2e_s": rq.e2e,
                    "ttft_s": rq.ttft,
                    "queue_wait_s": (
                        rq.ts["dequeued"] - rq.arrival if "dequeued" in rq.ts else None
                    ),
                    "n_output_tokens": len(rq.output_ids),
                    "finish_reason": rq.finish_reason,
                    "state": str(rq.state),
                    # A cancelled or failed request is a *censored* observation
                    # of its own latency -- the true value is at least what was
                    # measured -- so it is flagged here and dropped by the
                    # fitting script rather than silently trained on.
                    "censored": bool(rq.cancelled or rq.finish_reason not in ("stop", "length")),
                    "cached_prefix_len": rq.cached_prefix_len,
                    "n_preemptions": rq.n_preemptions,
                }
            )
        self.trace_log.write(row)

    def stats(self) -> dict:
        s = self.scheduler.stats()
        s["config"] = self.cfg.config_name
        s["backend"] = self.cfg.backend
        s["admission"] = self.cfg.admission
        ctl = getattr(self.controller, "stats", None)
        if ctl is not None:
            s["controller"] = ctl()
        return s
