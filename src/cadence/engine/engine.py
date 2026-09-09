"""The engine: owns the backend, the scheduler and the admission policy.

It is the only object the API layer talks to, and the only place that knows
which rung of the ablation ladder is running.
"""

from __future__ import annotations

import asyncio
import time
import uuid

from cadence.admission import Decision, build_controller
from cadence.engine.backends import build_runner
from cadence.engine.kv.block_manager import BlockManager, ContiguousBlockManager
from cadence.engine.kv.radix_cache import RadixCache
from cadence.engine.request import Request
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
        blocks = (
            BlockManager(cfg.n_kv_blocks, cfg.block_size)
            if cfg.enable_paged_kv
            else ContiguousBlockManager(cfg.n_kv_blocks, cfg.block_size, cfg.max_tokens_cap)
        )
        prefix = RadixCache(blocks, cfg.block_size) if cfg.enable_prefix_cache else None
        return ContinuousScheduler(runner, cfg, metrics, blocks=blocks, prefix_cache=prefix)
    raise ValueError(f"unknown scheduler {cfg.scheduler!r}")


class Engine:
    def __init__(self, cfg, runner=None) -> None:
        self.cfg = cfg
        self.metrics = Metrics(cfg.config_name, cfg.metrics_enabled)
        self.runner = runner if runner is not None else build_runner(cfg)
        self.scheduler = build_scheduler(self.runner, cfg, self.metrics)
        self.controller = build_controller(cfg)
        self._started = False

    # --- lifecycle --------------------------------------------------------
    def start(self) -> None:
        if not self._started:
            self.scheduler.start()
            self._started = True

    def stop(self) -> None:
        if self._started:
            self.scheduler.stop()
            self._started = False
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
        prompt is tens of microseconds, and the decision needs the token count."""
        prompt_ids = await asyncio.to_thread(self.runner.tokenize, self.render_prompt(body))
        max_tokens = body.max_tokens or self.cfg.max_tokens_default
        decision = self.controller.decide(
            len(prompt_ids), max_tokens, self.scheduler.stats()
        )
        decision.prompt_ids = prompt_ids
        decision.max_tokens = max_tokens
        if decision.action == "shed":
            self.metrics.shed(decision.reason or "unspecified")
            self.metrics.finished("shed")
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

    def stats(self) -> dict:
        s = self.scheduler.stats()
        s["config"] = self.cfg.config_name
        s["backend"] = self.cfg.backend
        return s
