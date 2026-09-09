"""The one interface both backends satisfy.

The scheduler never learns which model is underneath. The property that makes
continuous batching possible at all is that this layer exposes *step-level*
decoding: :meth:`decode_step` takes a list of sequences and advances each by
exactly one token. A backend that only offers "generate until done for one
prompt" cannot be batched continuously -- you would end up building static
batching with extra steps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class SeqState:
    """Backend-side view of one sequence.

    ``seq_id`` is the KV-cache sequence id owned by the backend; the scheduler
    allocates them and is responsible for calling :meth:`ModelRunner.free`.
    """

    seq_id: int
    prompt_ids: list[int]
    output_ids: list[int] = field(default_factory=list)
    n_past: int = 0
    """Tokens already resident in this sequence's KV cache."""
    finished: bool = False
    stop_reason: str | None = None

    @property
    def n_prompt_remaining(self) -> int:
        """Prompt tokens not yet pushed through a forward pass."""
        return max(0, len(self.prompt_ids) - self.n_past)

    @property
    def prompt_done(self) -> bool:
        return self.n_past >= len(self.prompt_ids)


@runtime_checkable
class ModelRunner(Protocol):
    """Step-level interface. One call == one forward pass over a batch."""

    def tokenize(self, text: str) -> list[int]: ...

    def detokenize(self, ids: Sequence[int]) -> str: ...

    def prefill(self, seqs: Sequence[SeqState]) -> list[int]:
        """Run *all* remaining prompt tokens for each seq; return the first
        sampled token per seq. Convenience wrapper over ``prefill_chunk``."""

    def prefill_chunk(self, seqs: Sequence[SeqState], budget: int) -> list[int | None]:
        """Advance each seq's prompt by at most ``budget`` tokens *in total*
        across the batch. Returns the sampled token for any seq whose prompt
        completed in this chunk, and ``None`` for seqs still prefilling."""

    def decode_step(self, seqs: Sequence[SeqState]) -> list[int]:
        """One token for every sequence in the batch. Returns sampled ids."""

    def free(self, seq_id: int) -> None:
        """Release backend-side KV slots for a finished/evicted sequence."""

    def copy_prefix(self, src_seq_id: int, dst_seq_id: int, n_tokens: int) -> None:
        """Share the first ``n_tokens`` of ``src``'s KV with ``dst``.

        This is what makes the radix prefix cache pay off: the shared tokens
        are never re-run through the model.
        """

    @property
    def kv_capacity_tokens(self) -> int: ...

    @property
    def eos_ids(self) -> frozenset[int]: ...


class DetokenizerState:
    """Incremental UTF-8 safe detokenizer for one sequence.

    Emitting ``detokenize([tok])`` per token corrupts multi-byte characters
    (and multi-token graphemes) because a single token can be half a code
    point. Buffer bytes and only release complete text.
    """

    __slots__ = ("_runner", "_ids", "_emitted")

    def __init__(self, runner: ModelRunner) -> None:
        self._runner = runner
        self._ids: list[int] = []
        self._emitted = 0

    def push(self, token_id: int) -> str:
        self._ids.append(token_id)
        text = self._runner.detokenize(self._ids)
        if text.endswith("�"):
            return ""  # incomplete code point; wait for the next token
        out = text[self._emitted :]
        self._emitted = len(text)
        return out
