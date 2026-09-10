"""Prompt + engine snapshot -> feature vector.

The one design rule, because it is the pitfall that invalidates the whole
experiment: the extractor may read only what is knowable *at admission time*.
It must never see anything derived from how the request actually turned out --
not its output length, not its measured latency. That leakage produces a
suspiciously good predictor and a coverage guarantee that means nothing.

The rule is enforced structurally rather than by discipline. :func:`extract`
takes an :class:`AdmitContext` and a :class:`~cadence.engine.scheduler.base.Snapshot`,
and neither of those types has a field that is only known after the fact: the
``Request`` object, which grows ``output_ids`` and a finish timestamp as it
runs, is not reachable from here. The trace writer that records training rows
takes the vector produced by *this* function at admission and stores it
verbatim, so the columns the model is fitted on are, by construction, the
columns the controller will have.

``max_tokens`` deserves a note. It is the client's *cap*, not the realised
output length, so it is legitimately an admission-time feature -- and it is the
single most informative one, because it bounds the decode work. It is also why
the residual uncertainty is irreducible rather than a modelling failure: within
that cap the actual output length is anywhere from one token to all of them,
and which one it is depends on what the model decides to say.

The feature that is *not* here
------------------------------
The build guide's list ends with ``arrival_rate_ewma``, and it is the one
feature that must not be in it. Not because it is unavailable or unpredictive
-- it is both available and, in a trace collected with admission off, strongly
predictive -- but because what it predicts through is queueing, and admission
control is precisely the intervention that severs that path. Offered load
raises latency *by* filling the queue; a controller that sheds keeps the queue
empty while the offered load stays exactly where it was.

A model fitted on observational data cannot tell the difference, and the
consequence was measured rather than reasoned about. With the arrival rate in
the feature set, the deployed controller admitted 57% of a 0.6 rps load and 1%
of a 1.9 rps load, with the queue empty and the batch empty in *both* cases:
the only thing that had changed was the feature encoding how much work was
being offered, and the model dutifully raised its bound for a system that was
sitting idle. Shedding then kept it idle, and the offered rate -- which counts
arrivals, not admissions, by design -- kept the bound high. That run is kept in
``results/w4_admission_confounded/``.

Every other system-state feature here is a *measured consequence* rather than a
cause: queue depth, batch size, remaining tokens, free KV, and the two rate
EWMAs all describe congestion that exists, and they stay true under the
intervention because they are the mechanism the intervention works through.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FEATURES: tuple[str, ...] = (
    # --- request-intrinsic (known at admission) ---
    "n_prompt_tokens",
    "n_cached_prefix_tokens",  # from the radix cache, before admitting
    "n_new_prefill_tokens",  # n_prompt - n_cached: the work actually to be done
    "max_tokens",  # the cap the client requested
    "log_max_tokens",
    "n_messages",
    "has_shared_system_prompt",  # 1.0 if the prefix hit exceeded a threshold
    # --- system state (the reason predictions are context-dependent) ---
    "queue_depth",
    "running_batch_size",
    "sum_remaining_tokens",  # over running seqs, of (max_tokens - produced)
    "kv_blocks_free_frac",
    "ewma_step_latency_s",  # recent decode-step cost; tracks real throughput
    "ewma_tokens_per_s",
)

N_FEATURES = len(FEATURES)

SHARED_PREFIX_TOKENS = 256
"""A hit longer than this means the request landed on one of the workload's
shared system prompts rather than on an incidental few-token overlap. The
workload's shared prompts are 600-900 tokens and its unique ones share no long
prefix, so the threshold separates the two populations cleanly and the feature
is close to binary in practice."""


@dataclass(frozen=True, slots=True)
class AdmitContext:
    """Everything about a request that is knowable before it runs.

    Frozen, and holding no reference to the :class:`~cadence.engine.request.Request`
    it describes, so that a feature cannot be added later that reaches through
    it into the outcome.
    """

    prompt_ids: list[int] = field(default_factory=list)
    max_tokens: int = 0
    n_messages: int = 1
    cached_prefix_tokens: int = 0
    """What ``scheduler.probe_prefix`` said, at arrival."""

    @property
    def n_prompt_tokens(self) -> int:
        return len(self.prompt_ids)


def extract(ctx: AdmitContext, snap) -> np.ndarray:
    """One row, in the order of :data:`FEATURES`.

    ``snap`` is a :class:`~cadence.engine.scheduler.base.Snapshot`; it is not
    annotated as one to keep this module importable without dragging the
    engine in, which is what lets the offline fitting script use it.
    """
    n_prompt = len(ctx.prompt_ids)
    cached = min(ctx.cached_prefix_tokens, n_prompt)
    return np.array(
        [
            n_prompt,
            cached,
            n_prompt - cached,
            ctx.max_tokens,
            np.log1p(ctx.max_tokens),
            ctx.n_messages,
            float(cached > SHARED_PREFIX_TOKENS),
            snap.queue_depth,
            snap.batch_size,
            snap.sum_remaining_tokens,
            snap.kv_free_frac,
            snap.ewma_step_latency_s,
            snap.ewma_tokens_per_s,
        ],
        dtype=np.float64,
    )


def to_dict(x: np.ndarray) -> dict[str, float]:
    """The row as named columns, for the trace log.

    Written by name rather than positionally so that a reordering of
    :data:`FEATURES` cannot silently re-label a committed trace file.
    """
    return {name: float(v) for name, v in zip(FEATURES, x, strict=True)}


def from_frame(df) -> np.ndarray:
    """The design matrix from a trace frame, columns in :data:`FEATURES` order.

    Raises rather than filling: a missing column means the trace was written by
    a different version of this module, and quietly substituting zeros for it
    would fit a model on a feature that is always zero and then serve it one
    that is not.
    """
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"trace is missing feature columns: {missing}")
    return df[list(FEATURES)].to_numpy(dtype=np.float64)
