"""Process lifecycle: readiness, draining, and the hard concurrency cap.

Three operational concerns that are not the scheduler's business and not the
admission controller's either, kept in one place because they share one piece
of state.

**Readiness is not liveness.** ``/health`` answers "is this process alive" and
is what a supervisor restarts on. ``/ready`` answers "should traffic be sent
here", which is a different question with three different answers: not yet
(the model is still loading -- a 0.5B GGUF plus the first Metal shader
compilation is tens of seconds, and a load balancer that routes during it
measures a cold start as a latency regression), yes, and no longer (draining).
Collapsing the two is the single most common way a deployment turns a rolling
restart into an SLO violation.

**Draining is a state, not an event.** On SIGTERM the process stops being
ready *immediately* -- so the load balancer takes it out of rotation within one
health-check interval -- while the requests already streaming keep streaming to
completion. Killing in-flight generations at shutdown would put a burst of
truncated responses into exactly the tail this project claims to control.

**The cap is a memory bound, not backpressure.** Every accepted request holds a
tokenised prompt, a queue entry and an asyncio task before it holds any KV, so
a large enough arrival burst can exhaust memory in the API layer while the
scheduler is still perfectly healthy. An OOM kill invalidates every SLO claim
the project makes, so the cap exists to make that failure mode unreachable. It
is deliberately set far above anything the scheduler will admit: if the cap is
what is shedding, the conformal controller has already failed and the number to
look at is ``cadence_shed_total{reason="capacity"}``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Lifecycle:
    """Readiness, drain state and the in-flight count for one process."""

    max_concurrent: int
    """Hard ceiling on requests in the API layer at once. Never a scheduling
    decision -- see the module docstring."""

    ready: bool = False
    """Set once the engine has completed its warm-up generation. Until then
    the process is alive but must not be sent traffic."""

    draining: bool = False
    drain_started_at: float | None = None

    in_flight: int = 0
    peak_in_flight: int = 0
    n_rejected_capacity: int = 0
    n_rejected_draining: int = 0

    _ready_at: float | None = field(default=None, repr=False)

    # --- readiness --------------------------------------------------------
    def mark_ready(self) -> None:
        if not self.ready:
            self.ready = True
            self._ready_at = time.time()

    def begin_drain(self) -> None:
        """Idempotent: SIGTERM followed by an impatient SIGINT must not reset
        the clock the drain is measured against."""
        if not self.draining:
            self.draining = True
            self.drain_started_at = time.time()

    @property
    def serving(self) -> bool:
        return self.ready and not self.draining

    # --- the cap ----------------------------------------------------------
    def acquire(self) -> bool:
        """Take one in-flight slot, or refuse.

        Single-threaded by construction: every caller is on the API event
        loop, and there is no await between the test and the increment, so
        this needs no lock. If that ever stops being true the counter is wrong
        in the direction of admitting one too many, which the cap's margin
        absorbs.
        """
        if self.in_flight >= self.max_concurrent:
            self.n_rejected_capacity += 1
            return False
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        return True

    def release(self) -> None:
        # Clamped at zero rather than asserted: a double release is a bug in
        # the route, and the right behaviour for it in production is a wrong
        # gauge, not a 500 in the middle of someone's stream.
        self.in_flight = max(0, self.in_flight - 1)

    def refuse_draining(self) -> None:
        self.n_rejected_draining += 1

    # --- reporting --------------------------------------------------------
    def status(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "draining": self.draining,
            "in_flight": self.in_flight,
            "peak_in_flight": self.peak_in_flight,
            "max_concurrent": self.max_concurrent,
            "rejected_capacity": self.n_rejected_capacity,
            "rejected_draining": self.n_rejected_draining,
            "drain_elapsed_s": (
                None if self.drain_started_at is None else time.time() - self.drain_started_at
            ),
        }
