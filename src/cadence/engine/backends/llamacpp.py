"""In-process llama.cpp backend with real per-sequence KV control.

This is the backend every measured number in the README comes from. It talks
to the low-level ``llama_decode`` API rather than to a completion helper,
because that is the only way to get the two things the scheduler needs:

* a batch whose members are *different sequences* advanced by one token each,
* explicit KV-cache sequence ids, so a prefix can be shared between requests
  (``llama_memory_seq_cp``) and a preempted sequence's KV can be dropped
  (``llama_memory_seq_rm``).

The context is created with ``kv_unified = True``, so ``n_ctx`` is one shared
pool of token slots across all sequences rather than a per-sequence
reservation. That is what makes the paged block manager an honest model of the
real resource: ``n_ctx // block_size`` blocks, competed for by every request in
flight.
"""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Sequence

import numpy as np

try:  # pragma: no cover - import guard
    import llama_cpp
    from llama_cpp import _internals as internals
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "llama-cpp-python is required for the llamacpp backend: uv add llama-cpp-python"
    ) from exc

from cadence.engine.backends.base import SeqState

_LOG_SILENCED = False
_LOG_SINK = None  # see _silence_llama_log


class KVSlotUnavailable(RuntimeError):
    """llama.cpp could not place this batch in the KV cache.

    The unified cache assigns a *contiguous* run of cells per micro-batch, so
    repeatedly freeing and re-allocating sequences of different lengths can
    leave it fragmented enough that a 512-token prefill chunk has nowhere to
    go even though the total free cell count is ample. The scheduler's answer
    is the one it already has for memory pressure: preempt and retry.
    """


def _silence_llama_log() -> None:
    """llama.cpp writes its loader chatter to stderr, which drowns out the
    benchmark output and, worse, costs measurable time inside the decode loop
    on a verbose build."""
    global _LOG_SILENCED, _LOG_SINK
    if _LOG_SILENCED:
        return

    @llama_cpp.llama_log_callback
    def _sink(level, text, user_data):  # noqa: ARG001
        return None

    llama_cpp.llama_log_set(_sink, ctypes.c_void_p(0))
    # Held in a module global, not an attribute on the function: llama.cpp
    # keeps the raw pointer, so if Python collects the ctypes callback the
    # next log line calls into freed memory.
    _LOG_SINK = _sink
    _LOG_SILENCED = True


class LlamaCppRunner:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        if not cfg.verbose_backend:
            _silence_llama_log()

        mparams = llama_cpp.llama_model_default_params()
        mparams.n_gpu_layers = -1  # Metal on Apple silicon
        self._model = internals.LlamaModel(
            path_model=cfg.model_path, params=mparams, verbose=False
        )

        cparams = llama_cpp.llama_context_default_params()
        cparams.n_ctx = cfg.n_ctx
        cparams.n_batch = cfg.n_batch
        cparams.n_ubatch = cfg.n_batch
        cparams.n_seq_max = min(cfg.n_parallel, llama_cpp.llama_max_parallel_sequences())
        cparams.kv_unified = True  # one shared pool of n_ctx token slots
        # Compact the cache once a tenth of it is holes. Without this, a long
        # run fragments until a full-size prefill chunk cannot be placed.
        cparams.defrag_thold = 0.1
        if cfg.n_threads:
            cparams.n_threads = cfg.n_threads
            cparams.n_threads_batch = cfg.n_threads
        self._ctx = internals.LlamaContext(model=self._model, params=cparams, verbose=False)

        self.n_seq_max = int(cparams.n_seq_max)
        self.n_vocab = self._model.n_vocab()
        self._batch = llama_cpp.llama_batch_init(cfg.n_batch, 0, 1)
        self._rng = np.random.default_rng(cfg.seed)
        self._eos = self._discover_eos()

        self.n_forward_passes = 0
        self.n_prefill_tokens = 0
        self.n_decode_tokens = 0

    def close(self) -> None:
        if getattr(self, "_batch", None) is not None:
            llama_cpp.llama_batch_free(self._batch)
            self._batch = None
        self._ctx.close()
        self._model.close()

    # --- vocabulary -------------------------------------------------------
    def _discover_eos(self) -> frozenset[int]:
        ids = {self._model.token_eos()}
        # Qwen's ChatML turn terminator is not the model's canonical EOS.
        for text in ("<|im_end|>", "<|endoftext|>"):
            try:
                toks = self._model.tokenize(text.encode(), add_bos=False, special=True)
            except Exception:
                continue
            if len(toks) == 1:
                ids.add(toks[0])
        return frozenset(i for i in ids if i >= 0)

    @property
    def eos_ids(self) -> frozenset[int]:
        return self._eos

    @property
    def kv_capacity_tokens(self) -> int:
        return int(self.cfg.n_ctx)

    # --- tokenizer --------------------------------------------------------
    def tokenize(self, text: str) -> list[int]:
        return self._model.tokenize(text.encode("utf-8"), add_bos=False, special=True)

    def detokenize(self, ids: Sequence[int]) -> str:
        if not ids:
            return ""
        return self._model.detokenize(list(ids), special=False).decode("utf-8", errors="replace")

    @staticmethod
    def format_chat(messages: Sequence[dict]) -> str:
        """Qwen2.5 uses ChatML. Rendered here rather than by a chat handler so
        that the prompt -- and therefore the prefix-cache hit rate -- is exactly
        reproducible across runs."""
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    # --- batch construction ----------------------------------------------
    def _batch_add(self, i: int, token: int, pos: int, seq_id: int, logits: bool) -> None:
        b = self._batch
        b.token[i] = token
        b.pos[i] = pos
        b.n_seq_id[i] = 1
        b.seq_id[i][0] = seq_id
        b.logits[i] = ctypes.c_int8(1 if logits else 0)

    def _decode(self, n_tokens: int) -> None:
        self._batch.n_tokens = n_tokens
        self.n_forward_passes += 1
        rc = llama_cpp.llama_decode(self._ctx.ctx, self._batch)
        if rc == 1:
            raise KVSlotUnavailable(f"no KV slot for a batch of {n_tokens} tokens")
        if rc != 0:
            raise RuntimeError(f"llama_decode failed with {rc}")

    def _logits(self, i: int) -> np.ndarray:
        ptr = llama_cpp.llama_get_logits_ith(self._ctx.ctx, i)
        if not ptr:
            raise RuntimeError(f"no logits for batch index {i}")
        return np.ctypeslib.as_array(ptr, shape=(self.n_vocab,))

    def _sample(self, logits: np.ndarray) -> int:
        if self.cfg.temperature <= 0.0:
            return int(np.argmax(logits))
        z = logits.astype(np.float64) / self.cfg.temperature
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        if self.cfg.top_p < 1.0:
            order = np.argsort(-p)
            keep = np.searchsorted(np.cumsum(p[order]), self.cfg.top_p) + 1
            mask = np.zeros_like(p)
            mask[order[:keep]] = p[order[:keep]]
            p = mask / mask.sum()
        return int(self._rng.choice(self.n_vocab, p=p))

    # --- ModelRunner ------------------------------------------------------
    def prefill_chunk(self, seqs: Sequence[SeqState], budget: int) -> list[int | None]:
        """Push at most ``budget`` prompt tokens (summed over the batch)
        through one forward pass.

        A sequence that finishes its prompt in this chunk gets logits on its
        last token and returns a sampled id; the others return ``None`` and
        will be continued on a later step. That interleaving is what keeps a
        2000-token prompt from stalling every decoding sequence for the length
        of one enormous forward pass.
        """
        with self._lock:
            # Never build a batch larger than the one that was allocated.
            # Settings validates this at startup; this is the second line of
            # defence, because the failure mode is a memory overrun rather
            # than an exception.
            budget = min(budget, self.cfg.n_batch)
            plan: list[tuple[SeqState, int]] = []
            spent = 0
            for s in seqs:
                take = min(s.n_prompt_remaining, budget - spent)
                if take <= 0:
                    continue
                plan.append((s, take))
                spent += take
            if not plan:
                return [None] * len(seqs)

            i = 0
            sampled_at: dict[int, SeqState] = {}
            for s, take in plan:
                last = s.n_past + take == len(s.prompt_ids)
                for k in range(take):
                    pos = s.n_past + k
                    want = last and k == take - 1
                    self._batch_add(i, s.prompt_ids[pos], pos, s.seq_id, want)
                    if want:
                        sampled_at[i] = s
                    i += 1
            self._decode(i)
            self.n_prefill_tokens += i

            for s, take in plan:
                s.n_past += take

            out: dict[int, int] = {}
            for idx, s in sampled_at.items():
                tok = self._sample(self._logits(idx))
                s.output_ids.append(tok)
                s.n_past += 1
                if tok in self._eos:
                    s.finished = True
                    s.stop_reason = "stop"
                out[id(s)] = tok
            return [out.get(id(s)) for s in seqs]

    def prefill(self, seqs: Sequence[SeqState]) -> list[int]:
        """Run every remaining prompt token. Used by the FIFO and static-batch
        rungs, and by the equivalence test."""
        first: dict[int, int] = {}
        pending = [s for s in seqs if not s.prompt_done]
        while pending:
            got = self.prefill_chunk(pending, budget=self.cfg.n_batch)
            for s, tok in zip(pending, got, strict=True):
                if tok is not None:
                    first[id(s)] = tok
            pending = [s for s in pending if not s.prompt_done]
        return [first[id(s)] for s in seqs if id(s) in first]

    def decode_step(self, seqs: Sequence[SeqState]) -> list[int]:
        """One token for every sequence in the batch, in one forward pass."""
        if not seqs:
            return []
        with self._lock:
            for i, s in enumerate(seqs):
                tok = (s.output_ids or s.prompt_ids)[-1]
                self._batch_add(i, tok, s.n_past - 1, s.seq_id, True)
            self._decode(len(seqs))
            self.n_decode_tokens += len(seqs)
            out = []
            for i, s in enumerate(seqs):
                tok = self._sample(self._logits(i))
                s.output_ids.append(tok)
                s.n_past += 1
                if tok in self._eos:
                    s.finished = True
                    s.stop_reason = "stop"
                out.append(tok)
            return out

    def free(self, seq_id: int) -> None:
        with self._lock:
            llama_cpp.llama_memory_seq_rm(self._ctx.memory, seq_id, -1, -1)

    def copy_prefix(self, src_seq_id: int, dst_seq_id: int, n_tokens: int) -> None:
        """Share ``src``'s first ``n_tokens`` of KV with ``dst``.

        In llama.cpp's unified cache this adds ``dst`` to the existing cells
        rather than duplicating them, which is exactly the reference-counted
        block sharing the radix cache models on the Python side.
        """
        if n_tokens <= 0:
            return
        with self._lock:
            llama_cpp.llama_memory_seq_cp(self._ctx.memory, src_seq_id, dst_seq_id, 0, n_tokens)

    def reset(self) -> None:
        with self._lock:
            llama_cpp.llama_memory_clear(self._ctx.memory, True)
