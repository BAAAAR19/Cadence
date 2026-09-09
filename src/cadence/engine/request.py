"""The request lifecycle.

Every scheduling bug is a state-machine bug and every metric is a timestamp on
a transition, so the lifecycle is explicit rather than implied by control flow:

    ARRIVED -> [admission] -> ADMITTED -> WAITING -> PREFILL -> DECODING -> DONE
                   |                        |          |           |
                   +--> SHED (503)          +--------- +-----------+--> PREEMPTED -> WAITING
                                                                     \\-> CANCELLED (client gone)

The class also carries the plumbing that gets tokens from the engine thread
back to the API event loop. The engine never touches asyncio primitives
directly: it calls :meth:`emit`, which hops threads via ``call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import enum
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # a cycle at runtime, a type at check time
    from cadence.obs.tracing import RequestTrace

_SENTINEL = object()


class State(enum.StrEnum):
    ARRIVED = "arrived"
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODING = "decoding"
    PREEMPTED = "preempted"
    DONE = "done"
    SHED = "shed"
    CANCELLED = "cancelled"


TERMINAL = frozenset({State.DONE, State.SHED, State.CANCELLED})


@dataclass(eq=False)
class Request:
    rid: str
    prompt_ids: list[int]
    max_tokens: int
    deadline: float
    """``perf_counter()`` value: arrival + slo. Ordering the wait queue by this
    is earliest-deadline-first, and Week 4's admission controller reuses it."""
    arrival: float = field(default_factory=time.perf_counter)
    state: State = State.ARRIVED
    output_ids: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    cached_prefix_len: int = 0
    """Prompt tokens served by the radix prefix cache, never re-prefilled."""
    n_past: int = 0
    seq_id: int = -1
    n_preemptions: int = 0
    finish_reason: str | None = None
    ts: dict[str, float] = field(default_factory=dict)

    # --- cross-thread streaming plumbing ---------------------------------
    loop: asyncio.AbstractEventLoop | None = None
    trace: RequestTrace | None = None
    """A :class:`~cadence.obs.tracing.RequestTrace`, or None when tracing is off."""
    _q: asyncio.Queue = field(default_factory=asyncio.Queue, repr=False)
    _cancelled: bool = False
    _done: asyncio.Event | None = None
    text: str = ""

    # --- timestamps -------------------------------------------------------
    def mark(self, name: str) -> None:
        self.ts.setdefault(name, time.perf_counter())

    def transition(self, new: State) -> None:
        # Every metric in this project is a timestamp on one of these
        # transitions, so the transition is the thing that gets recorded --
        # not a log line next to it.
        prev, self.state = self.state, new
        self.mark(str(new))
        if self.trace is not None:
            self.trace.event(
                str(new), **{"from": str(prev), "output_tokens": len(self.output_ids)}
            )

    # --- derived ----------------------------------------------------------
    @property
    def slack(self) -> float:
        """Seconds until the deadline. Negative == already doomed."""
        return self.deadline - time.perf_counter()

    @property
    def n_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def ttft(self) -> float | None:
        t = self.ts.get("first_token")
        return None if t is None else t - self.arrival

    @property
    def e2e(self) -> float | None:
        t = self.ts.get("finished")
        return None if t is None else t - self.arrival

    def __lt__(self, other: Request) -> bool:
        # heapq ordering: earliest deadline first, ties broken by arrival so
        # the queue is stable and starvation-free.
        return (self.deadline, self.arrival) < (other.deadline, other.arrival)

    # --- engine -> API ----------------------------------------------------
    def _put(self, item: object) -> None:
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._q.put_nowait, item)

    def emit(self, text: str) -> None:
        """Called from the engine thread for every decoded chunk of text."""
        if not text or self._cancelled:
            return
        self.text += text
        self._put(text)

    def finish(self, reason: str = "stop") -> None:
        if self.state in TERMINAL:
            return
        self.finish_reason = reason
        self.transition(State.DONE)
        self.mark("finished")
        self._put(_SENTINEL)

    def fail(self, exc: BaseException) -> None:
        self.finish_reason = "error"
        self.transition(State.DONE)
        self.mark("finished")
        self._put(exc)

    def cancel(self) -> None:
        """Client went away. The engine notices on its next step."""
        self._cancelled = True
        if self.state not in TERMINAL:
            self.transition(State.CANCELLED)
            self.mark("finished")
        self._put(_SENTINEL)

    # --- API side ---------------------------------------------------------
    async def stream(self) -> AsyncIterator[str]:
        while True:
            item = await self._q.get()
            if item is _SENTINEL:
                return
            if isinstance(item, BaseException):
                raise item
            yield item  # type: ignore[misc]

    async def result(self) -> Request:
        async for _ in self.stream():
            pass
        return self

    def to_openai(self, cid: str, model: str) -> dict:
        return {
            "id": cid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.text},
                    "finish_reason": self.finish_reason or "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(self.prompt_ids),
                "completion_tokens": len(self.output_ids),
                "total_tokens": self.n_tokens,
            },
        }
