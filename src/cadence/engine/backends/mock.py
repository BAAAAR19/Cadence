"""A deterministic, model-free runner.

Its job is to make the scheduler, the KV bookkeeping and the load generator
testable in CI without a 500 MB GGUF and without Metal. It is *not* used for
any number that appears in the README -- those all come from the llama.cpp
backend -- but it is used for correctness tests, where determinism is worth
more than realism.

The cost model is deliberately crude and explicit: a forward pass costs
``prefill_us_per_token`` per prompt token plus ``decode_us_per_seq`` per
sequence in the batch, plus a fixed per-step overhead. That reproduces the one
property the scheduler actually depends on -- a batched step is much cheaper
than the same sequences stepped one at a time.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence

from cadence.engine.backends.base import SeqState

_VOCAB = 4096
_EOS = 2


class MockRunner:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._n_ctx = cfg.n_ctx
        self.step_overhead_s = float(getattr(cfg, "mock_step_overhead_s", 0.004))
        self.decode_s_per_seq = float(getattr(cfg, "mock_decode_s_per_seq", 0.0015))
        self.prefill_s_per_token = float(getattr(cfg, "mock_prefill_s_per_token", 0.00012))
        self.freed: list[int] = []
        self.copied: list[tuple[int, int, int]] = []
        self.steps = 0

    # --- tokenizer: a stable byte-level toy ------------------------------
    def tokenize(self, text: str) -> list[int]:
        return [1] + [b % _VOCAB for b in text.encode("utf-8")]

    def detokenize(self, ids: Sequence[int]) -> str:
        return "".join(f"<{i}>" for i in ids if i != 1)

    # --- sampling: pure function of (prompt, position) -------------------
    @staticmethod
    def _sample(seq: SeqState) -> int:
        h = hashlib.blake2b(
            repr((seq.prompt_ids, len(seq.output_ids))).encode(), digest_size=4
        ).digest()
        return int.from_bytes(h, "big") % _VOCAB

    def _burn(self, seconds: float) -> None:
        """Spend the modelled cost of one forward pass.

        ``time.sleep`` rather than a spin loop, and the distinction matters: a
        Python busy-wait holds the GIL for its whole duration, which starves the
        API event loop and shows up as inter-token latency that no real backend
        would produce. A real forward pass releases the GIL inside the native
        call, and ``sleep`` is the faithful stand-in for that.
        """
        if seconds > 0:
            time.sleep(seconds)

    # --- ModelRunner ------------------------------------------------------
    def prefill_chunk(self, seqs: Sequence[SeqState], budget: int) -> list[int | None]:
        self.steps += 1
        spent = 0
        out: list[int | None] = []
        for s in seqs:
            take = min(s.n_prompt_remaining, max(0, budget - spent))
            s.n_past += take
            spent += take
            out.append(self._sample(s) if s.prompt_done else None)
        self._burn(self.step_overhead_s + spent * self.prefill_s_per_token)
        for s, tok in zip(seqs, out, strict=True):
            if tok is not None:
                s.output_ids.append(tok)
                s.n_past += 1
        return out

    def prefill(self, seqs: Sequence[SeqState]) -> list[int]:
        out: list[int | None] = [None] * len(seqs)
        pending = list(seqs)
        while pending:
            got = self.prefill_chunk(pending, budget=10**9)
            for s, tok in zip(pending, got, strict=True):
                if tok is not None:
                    out[list(seqs).index(s)] = tok
            pending = [s for s in pending if not s.prompt_done]
        return [t for t in out if t is not None]

    def decode_step(self, seqs: Sequence[SeqState]) -> list[int]:
        self.steps += 1
        self._burn(self.step_overhead_s + len(seqs) * self.decode_s_per_seq)
        out = []
        for s in seqs:
            tok = self._sample(s)
            s.output_ids.append(tok)
            s.n_past += 1
            if tok == _EOS:
                s.finished = True
                s.stop_reason = "stop"
            out.append(tok)
        return out

    def free(self, seq_id: int) -> None:
        self.freed.append(seq_id)

    def copy_prefix(self, src_seq_id: int, dst_seq_id: int, n_tokens: int) -> None:
        self.copied.append((src_seq_id, dst_seq_id, n_tokens))

    @property
    def kv_capacity_tokens(self) -> int:
        return self._n_ctx

    @property
    def eos_ids(self) -> frozenset[int]:
        return frozenset({_EOS})
