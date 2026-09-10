"""Admission control: admit or shed.

Week 4 makes this interesting -- it predicts each request's end-to-end latency
distribution and sheds when the split-conformal *upper* bound exceeds the
deadline, which is what turns a hoped-for p99 into a measured one. Rungs 1-4
run the pass-through policy so that they measure scheduling alone, with nothing
shed.

The interface is one call, taking everything knowable at admission and nothing
else, so that every rung of the ablation ladder goes through the same code path
and only the policy changes.

Two actions, not three
----------------------
The build guide's controller has an ``admit / queue / shed`` action space,
where "queue" means "this fits if it jumps the line". Here the wait queue is
*already* earliest-deadline-first (``Request.__lt__``), so a request's position
in it is decided by its deadline and not by admission; there is no third action
to take. Saying so is more honest than adding a "queue" branch that returns the
same behaviour as "admit" and appears in the metrics as though it were a
distinct policy. What the guide's third action buys -- a request that fits only
if prioritised -- this scheduler gets for free, and the ablation that would be
interesting here is EDF versus SJF, which is a scheduler question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from cadence.admission.features import AdmitContext

Action = Literal["admit", "shed"]


@dataclass(slots=True)
class Decision:
    action: Action = "admit"
    retry_after_s: float = 1.0
    reason: str = ""
    predicted_e2e_s: float | None = None
    """The conformal upper bound this decision was taken on, in seconds.

    Not a point prediction: it is the value ``U(x)`` such that the realised
    latency lands below it at least ``1 - alpha`` of the time. The name is kept
    from the Week 1 interface; ``predicted_p99_s`` would be more accurate at
    the default alpha.
    """
    slack_s: float = 0.0
    """The budget it was compared against: the SLO, less any hysteresis."""
    features: np.ndarray | None = None
    """The admission-time feature vector, carried so the trace writer and the
    online recalibrator use the *same* row the decision was taken on rather
    than recomputing it against a system state that has since moved."""

    # Carried, not decided: the prompt is tokenised once, at admission, because
    # the policy needs the token count and the scheduler needs the ids. Passing
    # them along here avoids tokenising the same prompt twice per request.
    prompt_ids: list[int] = field(default_factory=list)
    max_tokens: int = 0
    cached_prefix_tokens: int = 0


class AdmissionController(Protocol):
    def decide(self, ctx: AdmitContext, snap) -> Decision: ...

    def observe(self, rq) -> None: ...


class PassThroughController:
    """Admit everything. The policy for rungs 1-4 of the ladder."""

    name = "none"

    def decide(self, ctx: AdmitContext, snap) -> Decision:
        return Decision(action="admit", reason="pass_through")

    def observe(self, rq) -> None:
        return None


class QueueCapController:
    """A backstop, not a differentiator: shed when the wait queue is longer
    than ``max_waiting``.

    Included so that an overload run cannot exhaust memory before Week 4's
    controller exists, and so the 503 path is exercised end to end from Week 1.
    At the loads this project runs, with ``max_waiting`` at its default, it
    never fires -- which is the point: rungs 1-4 shed nothing, so their
    collapse under overload is the scheduler's own.
    """

    name = "queue_cap"

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def decide(self, ctx: AdmitContext, snap) -> Decision:
        if snap.queue_depth >= self.cfg.max_waiting:
            return Decision(
                action="shed", retry_after_s=self.cfg.retry_after_s, reason="queue_full"
            )
        return Decision(action="admit", reason="pass_through")

    def observe(self, rq) -> None:
        return None


def build_controller(cfg, metrics=None):
    if cfg.admission == "none":
        return QueueCapController(cfg)
    if cfg.admission == "conformal":
        from cadence.admission.conformal_controller import ConformalController

        return ConformalController(cfg, metrics)
    raise ValueError(f"unknown admission policy {cfg.admission!r}")
