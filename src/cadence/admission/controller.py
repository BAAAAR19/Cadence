"""Admission control: admit / queue / shed.

Week 4 makes this interesting -- it predicts each request's end-to-end latency
distribution and sheds when the split-conformal *upper* bound exceeds the
deadline, which is what turns a hoped-for p99 into a measured one. Until then
the gateway runs the pass-through policy so that Weeks 1-2 measure scheduling
alone, with nothing shed.

The interface is fixed now because every rung of the ablation ladder has to go
through the same code path; only the policy changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

Action = Literal["admit", "shed"]


@dataclass(slots=True)
class Decision:
    action: Action = "admit"
    retry_after_s: float = 1.0
    reason: str = ""
    predicted_e2e_s: float | None = None
    """Filled in by the conformal controller: the upper bound it admitted on."""

    # Carried, not decided: the prompt is tokenised once, at admission, because
    # the policy needs the token count and the scheduler needs the ids. Passing
    # them along here avoids tokenising the same prompt twice per request.
    prompt_ids: list[int] = field(default_factory=list)
    max_tokens: int = 0


class AdmissionController(Protocol):
    def decide(self, n_prompt_tokens: int, max_tokens: int, snapshot: dict) -> Decision: ...

    def observe(self, rq) -> None: ...


class PassThroughController:
    """Admit everything. The Week 1-2 policy, and rung 1-3 of the ladder."""

    def decide(self, n_prompt_tokens: int, max_tokens: int, snapshot: dict) -> Decision:
        return Decision(action="admit")

    def observe(self, rq) -> None:
        return None


class QueueCapController:
    """A backstop, not a differentiator: shed when the wait queue is longer
    than ``max_waiting``.

    Included so that an overload run cannot exhaust memory before Week 4's
    controller exists, and so the 503 path is exercised end to end from Week 1.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def decide(self, n_prompt_tokens: int, max_tokens: int, snapshot: dict) -> Decision:
        if snapshot.get("waiting", 0) >= self.cfg.max_waiting:
            return Decision(
                action="shed", retry_after_s=self.cfg.retry_after_s, reason="queue_full"
            )
        return Decision(action="admit")

    def observe(self, rq) -> None:
        return None


def build_controller(cfg):
    if cfg.admission == "none":
        return QueueCapController(cfg)
    if cfg.admission == "conformal":
        from cadence.admission.conformal_controller import ConformalController

        return ConformalController(cfg)
    raise ValueError(f"unknown admission policy {cfg.admission!r}")
