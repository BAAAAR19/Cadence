"""The Week-1 bootstrap backend: llama-server over HTTP.

Wrapping ``llama-server``'s completion API gets the gateway end-to-end in an
afternoon, and it is honest about what it can and cannot do. It satisfies
:class:`~cadence.engine.backends.base.ModelRunner` for a **batch of one**, which
is exactly what the FIFO rung needs, and it refuses batches larger than that
rather than pretending:

    A backend that can only "generate until done for one prompt" cannot be
    batched continuously. If this class silently accepted a batch and served it
    sequentially, the continuous rung would quietly become static batching with
    extra steps, and the whole ablation would be measuring nothing.

So Week 2 replaces it with the in-process
:class:`~cadence.engine.backends.llamacpp.LlamaCppRunner`, which drives
``llama_decode`` and real KV sequence ids. This class stays for two reasons: it
is the fastest way to bring the gateway up on a machine where the in-process
binding will not build, and it documents the boundary that made the in-process
port necessary.

Start the server it talks to with::

    llama-server -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf \\
        --port 8081 --ctx-size 8192 --parallel 8 --cont-batching
"""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Sequence

import httpx

from cadence.engine.backends.base import SeqState


class BatchTooLarge(RuntimeError):
    """Raised instead of silently serialising a batch. See the module docstring."""


class LlamaServerRunner:
    """Step-level interface over llama-server's streaming completion endpoint.

    Each sequence owns a background thread holding an open SSE stream; a
    ``decode_step`` pulls the next token off that stream. That is enough to make
    single-sequence stepping look identical to the in-process backend, and it is
    the most this API can honestly support.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.base = cfg.llama_server_url.rstrip("/")
        self.client = httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0))
        self._streams: dict[int, _TokenStream] = {}
        self._lock = threading.Lock()
        props = self.client.get(f"{self.base}/props").json()
        self._n_ctx = int(props.get("default_generation_settings", {}).get("n_ctx", cfg.n_ctx))
        self._eos: frozenset[int] = frozenset()

    def close(self) -> None:
        for s in list(self._streams.values()):
            s.stop()
        self._streams.clear()
        self.client.close()

    # --- tokenizer: llama-server exposes the model's own ------------------
    def tokenize(self, text: str) -> list[int]:
        r = self.client.post(f"{self.base}/tokenize", json={"content": text})
        r.raise_for_status()
        return r.json()["tokens"]

    def detokenize(self, ids: Sequence[int]) -> str:
        if not ids:
            return ""
        r = self.client.post(f"{self.base}/detokenize", json={"tokens": list(ids)})
        r.raise_for_status()
        return r.json()["content"]

    @staticmethod
    def format_chat(messages: Sequence[dict]) -> str:
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    @property
    def kv_capacity_tokens(self) -> int:
        return self._n_ctx

    @property
    def eos_ids(self) -> frozenset[int]:
        # Stop is signalled by the stream ending, not by a token id: this
        # backend never sees the sampled EOS token.
        return self._eos

    # --- ModelRunner ------------------------------------------------------
    def _one(self, seqs: Sequence[SeqState]) -> SeqState:
        if len(seqs) != 1:
            raise BatchTooLarge(
                f"llamacpp_http serves one sequence at a time, got {len(seqs)}. "
                "Use the in-process llamacpp backend for anything batched."
            )
        return seqs[0]

    def prefill(self, seqs: Sequence[SeqState]) -> list[int]:
        seq = self._one(seqs)
        prompt = self.detokenize(seq.prompt_ids)
        stream = _TokenStream(
            self.client, self.base, prompt,
            n_predict=self.cfg.max_tokens_cap,
            temperature=self.cfg.temperature,
            seed=self.cfg.seed,
        )
        with self._lock:
            self._streams[seq.seq_id] = stream
        tok = stream.next_token()
        seq.n_past = len(seq.prompt_ids)
        if tok is None:
            seq.finished = True
            seq.stop_reason = "stop"
            return []
        seq.output_ids.append(tok)
        seq.n_past += 1
        return [tok]

    def prefill_chunk(self, seqs: Sequence[SeqState], budget: int) -> list[int | None]:
        # No chunking is possible: llama-server prefills the whole prompt inside
        # one request. Reported honestly rather than faked with a delay.
        return list(self.prefill(seqs)) or [None]

    def decode_step(self, seqs: Sequence[SeqState]) -> list[int]:
        seq = self._one(seqs)
        stream = self._streams.get(seq.seq_id)
        if stream is None:
            raise RuntimeError(f"decode_step before prefill for seq {seq.seq_id}")
        tok = stream.next_token()
        if tok is None:
            seq.finished = True
            seq.stop_reason = "stop"
            return [seq.output_ids[-1] if seq.output_ids else 0]
        seq.output_ids.append(tok)
        seq.n_past += 1
        return [tok]

    def free(self, seq_id: int) -> None:
        with self._lock:
            stream = self._streams.pop(seq_id, None)
        if stream is not None:
            stream.stop()

    def copy_prefix(self, src_seq_id: int, dst_seq_id: int, n_tokens: int) -> None:
        raise NotImplementedError(
            "llama-server manages its own prefix cache internally and exposes no "
            "way to share KV between slots; the radix cache needs the in-process "
            "backend."
        )


class _TokenStream:
    """One open SSE stream, drained by a background thread into a queue."""

    def __init__(self, client, base, prompt, n_predict, temperature, seed) -> None:
        self.q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._client = client
        self._payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "seed": seed,
            "stream": True,
            "return_tokens": True,
            "cache_prompt": True,
        }
        self._base = base
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        try:
            with self._client.stream(
                "POST", f"{self._base}/completion", json=self._payload
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if self._stop.is_set():
                        break
                    if not line.startswith("data: "):
                        continue
                    d = json.loads(line[6:])
                    for tok in d.get("tokens") or []:
                        self.q.put(tok)
                    if d.get("stop"):
                        break
        except Exception as exc:  # surfaced to the caller, never swallowed
            self.q.put(exc)
        finally:
            self.q.put(None)

    def next_token(self) -> int | None:
        item = self.q.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def stop(self) -> None:
        self._stop.set()
