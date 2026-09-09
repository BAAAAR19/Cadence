"""Rung 3: continuous batching, paged KV, radix prefix cache.

The shift is from "run one request to completion" to "on every step, decide who
is in the batch". A request joins the running batch as soon as there is room; a
finished request leaves immediately instead of holding the batch open until its
slowest peer finishes.

Two decisions make or break it:

**Chunked prefill.** A 2000-token prompt prefilled in one shot stalls every
decoding sequence for the duration of that forward pass and shows up as a spike
in p99 ITL. Prefill is split into chunks of ``max_prefill_tokens`` and
interleaved with decode steps.

**Prefill/decode ratio.** Always prioritising prefill starves decoding and
hurts ITL; always prioritising decode hurts TTFT. ``prefill_priority`` is the
knob, and both settings are measured rather than assumed.
"""

from __future__ import annotations

import heapq
import time
from contextlib import contextmanager

from cadence.engine.kv.block_manager import BlockManager, OutOfBlocks
from cadence.engine.kv.protocols import BlockPool, PrefixCache, PrefixMatch
from cadence.engine.kv.radix_cache import Match, RadixCache
from cadence.engine.request import Request, State
from cadence.engine.scheduler.base import BaseScheduler, Live

try:  # the backend is optional in tests that run on the mock runner
    from cadence.engine.backends.llamacpp import KVSlotUnavailable

    _KV_SLOT_ERRORS: tuple[type[BaseException], ...] = (KVSlotUnavailable,)
except ImportError:  # pragma: no cover
    _KV_SLOT_ERRORS = ()


class ContinuousScheduler(BaseScheduler):
    name = "continuous"
    kv_core = "python"
    """Set by ``build_scheduler``; reported in ``/stats`` so that every run
    records which implementation of the KV structures produced it."""

    def __init__(self, runner, cfg, metrics, blocks=None, prefix_cache=None) -> None:
        super().__init__(runner, cfg, metrics)
        self.blocks: BlockPool = blocks or BlockManager(cfg.n_kv_blocks, cfg.block_size)
        self.prefix: PrefixCache | None = prefix_cache
        if self.prefix is None and cfg.enable_prefix_cache:
            self.prefix = RadixCache(self.blocks, cfg.block_size)
        if self.prefix is not None:
            # An owner sequence outlives the request that created it: it is
            # only handed back once the last radix node naming it is evicted.
            self.prefix.on_seq_released = self._release_owner_seq

        self._watermark = max(1, int(self.blocks.n_blocks * cfg.kv_watermark))
        self.waiting: list[Request] = []  # min-heap by deadline (EDF)
        self.running: list[Live] = []
        self.metrics.kv_blocks_total.set(self.blocks.n_blocks)

    # --- owner-sequence bookkeeping ---------------------------------------
    def _release_owner_seq(self, seq_id: int) -> None:
        self.runner.free(seq_id)
        self.seq_ids.free(seq_id)

    # --- admission --------------------------------------------------------
    def _intake(self) -> None:
        for rq in self._take_incoming():
            if rq.cancelled:
                continue
            heapq.heappush(self.waiting, rq)
        self.n_waiting = len(self.waiting)

    def _match_prefix(self, rq: Request) -> PrefixMatch:
        """Look up a prompt without counting the lookup.

        A request at the head of a busy queue is matched on every step until
        it is admitted -- possibly hundreds of times. Counting each of those
        would weight the reported hit rate by how long a request waited, which
        is not what the number is supposed to mean. The counters are advanced
        once, in ``_admit``.
        """
        if self.prefix is None:
            return Match(n_tokens=0, block_ids=[], node=None)
        return self.prefix.match(rq.prompt_ids, count=False)

    @contextmanager
    def _pinned_match(self, rq: Request):
        """Match ``rq``'s prompt and hold the matched path for the length of
        the admission decision.

        Without the pin this is a use-after-free waiting to happen, and it
        happened: admission matches a prefix, then discovers it is short of
        blocks, then calls ``_reclaim`` -- which evicts by LRU and, under
        enough pressure, evicts the very node just matched. ``hit.block_ids``
        then names blocks that are back on the free list, and the ``share()``
        in ``_admit`` raises ``cannot share free block``, killing the engine
        thread. The seq-id branch below evicts too, with the same consequence.

        Pinning is the right fix rather than re-matching after each eviction:
        re-matching leaves the same window open one line further down, and it
        spends the prefix on a request that is about to reuse it. A pinned
        node is simply not an eviction candidate, so the decision sees one
        consistent view of the cache from match to admit.

        The pin is released on the way out; ``_admit`` has by then taken its
        own reference for the life of the request.
        """
        hit = self._match_prefix(rq)
        if hit.node is not None and self.prefix is not None:
            self.prefix.acquire(hit.node)
        try:
            yield hit
        finally:
            if hit.node is not None and self.prefix is not None:
                self.prefix.release(hit.node)

    def _reclaim(self, n_blocks: int) -> None:
        """Blocks pinned by the prefix cache are reclaimable; blocks held by a
        running sequence are not. Evict cached prefixes before preempting."""
        if self.prefix is not None and n_blocks > 0:
            self.prefix.evict(n_blocks)

    def _schedule(self) -> tuple[list[Live], list[Live]]:
        """Choose this step's prefill set and decode set.

        A pure function of scheduler state: it allocates, but it never runs the
        model, which is what makes it testable on its own.
        """
        # 0. Sequences part-way through a chunked prompt finish what they
        #    started before anything new is admitted. A sequence that is still
        #    prefilling must never appear in the decode set as well: feeding it
        #    a decode token would repeat a position the KV cache already holds.
        prefill: list[Live] = [lv for lv in self.running if not lv.seq.prompt_done]
        n_admitted = 0  # `prefill` already contains running sequences; only the
                        # newly admitted ones add to the batch.
        budget = self.cfg.max_prefill_tokens
        budget -= min(budget, sum(lv.seq.n_prompt_remaining for lv in prefill))

        # 1. Admit from the wait queue while KV blocks, sequence ids and the
        #    token budget allow.
        while self.waiting and len(self.running) + n_admitted < self.cfg.max_batch:
            rq = self.waiting[0]
            if rq.cancelled:
                heapq.heappop(self.waiting)
                continue

            # A match names blocks and a node that only stay valid while the
            # path is pinned -- and both gates below can evict. Pinning for the
            # length of the decision is the invariant that makes the match
            # usable at the end of it; see ``_pinned_match``.
            with self._pinned_match(rq) as hit:
                # Gate on what admission will actually take, plus a watermark
                # that leaves room for sequences already running to grow.
                #
                # Gating on the *worst* case instead -- everything the request
                # could eventually occupy -- is the obvious-looking alternative
                # and it is worse in both directions: it admits too few
                # requests, and because the shortfall is made up by evicting
                # cached prefixes, it spends the prefix cache to buy batch
                # slots it does not use. Under paged allocation the honest gate
                # is the near-term one, and over-commit is resolved by
                # preemption, which is what preemption is for.
                need = self.blocks.initial_blocks(len(rq.prompt_ids), cached=hit.n_tokens)
                need += self._watermark
                if need > self.blocks.free_blocks():
                    self._reclaim(need - self.blocks.free_blocks())
                    if need > self.blocks.free_blocks():
                        break  # memory-bound: stop admitting

                new_tokens = len(rq.prompt_ids) - hit.n_tokens
                if budget <= 0 and prefill:
                    break
                if new_tokens > budget and prefill:
                    break  # token-budget-bound; a lone request may still exceed
                           # it and will be chunked across steps instead.

                seq_id = self.seq_ids.alloc()
                if seq_id is None and self.prefix is not None:
                    # Cached prefixes pin backend sequences, so the pool can be
                    # exhausted by the cache rather than by running work. Drop
                    # the coldest entries a few at a time until one frees up --
                    # not the whole cache, which would take the shared prompts
                    # with it.
                    for _ in range(8):
                        if self.prefix.evict_nodes(2) == 0:
                            break
                        seq_id = self.seq_ids.alloc()
                        if seq_id is not None:
                            break
                if seq_id is None:
                    break  # sequence-id-bound

                heapq.heappop(self.waiting)
                live = self._admit(rq, seq_id, hit)
            budget -= min(new_tokens, budget)
            prefill.append(live)
            n_admitted += 1

        self.n_waiting = len(self.waiting)

        # 2. Every running sequence that has finished its prompt decodes one
        #    token, if it still has room to grow.
        decode: list[Live] = []
        for live in list(self.running):
            if live.rq.cancelled:
                self._retire(live, "cancelled")
                continue
            if not live.seq.prompt_done:
                continue  # still prefilling; handled above
            if self.blocks.can_append(live.rq.block_ids, live.rq.n_tokens):
                decode.append(live)
            else:
                self._preempt(live)
        return prefill, decode

    def _admit(self, rq: Request, seq_id: int, hit: Match) -> Live:
        shared = list(hit.block_ids)
        if shared:
            self.blocks.share(shared)
        try:
            # Only the prompt is allocated here. Generation grows the block
            # table a block at a time; see BlockManager.initial_blocks.
            fresh = self.blocks.alloc(
                self.blocks.initial_blocks(len(rq.prompt_ids), cached=hit.n_tokens)
            )
        except OutOfBlocks:
            if shared:
                self.blocks.release(shared)
            raise

        rq.block_ids = shared + fresh
        rq.cached_prefix_len = hit.n_tokens
        rq.mark("dequeued")
        rq.transition(State.PREFILL)
        self.metrics.queue_wait.observe(rq.ts["dequeued"] - rq.arrival)

        live = self._begin(rq, seq_id, n_past=hit.n_tokens)
        live.matched_tokens = hit.n_tokens
        live.shared_blocks = shared
        if hit.n_tokens and hit.node is not None and self.prefix is not None:
            # Materialise the hit: llama.cpp adds this sequence to the cells
            # already holding those tokens, so they are never re-run.
            self.runner.copy_prefix(hit.owner_seq, seq_id, hit.n_tokens)
            self.prefix.acquire(hit.node)  # pin it for the life of the request
            live.prefix_node = hit.node
            self.metrics.prefix_hit_tokens.inc(hit.n_tokens)
        if self.prefix is not None:
            self.prefix.query_tokens += len(rq.prompt_ids)
            self.prefix.hit_tokens += hit.n_tokens
            self.metrics.prefix_query_tokens.inc(len(rq.prompt_ids))
        self.running.append(live)
        return live

    # --- preemption (Step 2.4) --------------------------------------------
    def _preempt(self, live: Live) -> None:
        """Recompute policy: drop this sequence's KV, requeue it, prefill again
        later.

        The victim chosen is the caller's -- the sequence that could not grow --
        but the *selection* below prefers the request with the most slack, so a
        request already close to its deadline is not the one thrown away.
        Swapping blocks to host memory is the standard alternative; it trades
        memory bandwidth for the wasted prefill compute, and it is not built
        here (see README, "what is deliberately not built").
        """
        victim = max(self.running, key=lambda lv: lv.rq.slack)
        if victim.rq.n_preemptions > 8:
            victim = live  # pathological churn: give up on this one instead
        self._release_backend(victim)
        rq = victim.rq
        rq.n_preemptions += 1
        rq.output_ids.clear()
        rq.n_past = 0
        rq.cached_prefix_len = 0
        rq.ts.pop("first_token", None)
        if rq.trace is not None:
            rq.trace.event(
                "preempted",
                policy=self.cfg.preemption_policy,
                discarded_output_tokens=len(rq.output_ids),
                free_blocks=self.blocks.free_blocks(),
            )
        rq.transition(State.PREEMPTED)
        self.running.remove(victim)
        rq.transition(State.WAITING)
        heapq.heappush(self.waiting, rq)
        self.metrics.preempted(self.cfg.preemption_policy)

    def _relieve_backend_pressure(self) -> None:
        if self.prefix is not None and self.prefix.evict_all_unused() > 0:
            return  # cached prefixes were holding cells; that may be enough
        if self.running:
            self._preempt(self.running[0])

    def _release_backend(self, live: Live) -> None:
        """Return everything the request borrowed except its wait-queue slot."""
        rq = live.rq
        if live.prefix_node is not None and self.prefix is not None:
            self.prefix.release(live.prefix_node)
            live.prefix_node = None
        if rq.block_ids:
            self.blocks.release(rq.block_ids)
            rq.block_ids = []
        if rq.seq_id >= 0:
            self.runner.free(rq.seq_id)
            self.seq_ids.free(rq.seq_id)
            rq.seq_id = -1

    def _retire(self, live: Live, reason: str) -> None:
        """A finished request donates its prompt's KV to the prefix cache
        instead of dropping it, which is what makes the cache warm at all."""
        rq = live.rq
        donated = False
        if (
            self.prefix is not None
            and reason != "cancelled"
            and rq.seq_id >= 0
            and len(rq.block_ids) > 0
        ):
            n_prompt_blocks = len(rq.prompt_ids) // self.blocks.block_size
            if n_prompt_blocks > 0:
                node = self.prefix.insert(
                    rq.prompt_ids, rq.block_ids[:n_prompt_blocks], rq.seq_id
                )
                # The cache only takes ownership of the backend sequence when
                # it actually named it. An insert that lands on a path already
                # owned by someone else keeps the existing owner, and this
                # request's sequence must still be freed -- otherwise sequence
                # ids leak until the pool is empty and admission stalls.
                donated = node is not None and node.owner_seq == rq.seq_id
        if live.prefix_node is not None and self.prefix is not None:
            self.prefix.release(live.prefix_node)
            live.prefix_node = None
        if rq.block_ids:
            self.blocks.release(rq.block_ids)
            rq.block_ids = []
        if rq.seq_id >= 0 and not donated:
            self.runner.free(rq.seq_id)
            self.seq_ids.free(rq.seq_id)
        rq.seq_id = -1
        if live in self.running:
            self.running.remove(live)
        self._finish(live, reason)

    # --- the loop ---------------------------------------------------------
    def loop(self) -> None:
        while not self._stop.is_set():
            self._intake()
            prefill, decode = self._schedule()
            self.n_running = len(self.running)
            self._gauges()

            if not prefill and not decode:
                self._idle_wait()
                continue

            t0 = time.perf_counter()
            order = (
                [("prefill", prefill), ("decode", decode)]
                if self.cfg.prefill_priority
                else [("decode", decode), ("prefill", prefill)]
            )
            try:
                for kind, group in order:
                    if not group:
                        continue
                    if kind == "prefill":
                        self._do_prefill(group)
                    else:
                        self._do_decode(group)
            except _KV_SLOT_ERRORS:
                # The backend's cache could not place this batch. Same remedy
                # as running out of blocks: free a sequence and try again on
                # the next step. Counted so it shows up in the metrics rather
                # than as a mysterious latency spike.
                self._relieve_backend_pressure()
                continue
            except Exception as exc:
                self._abort(exc)
                continue
            self.metrics.step_latency.observe(time.perf_counter() - t0)
            self._update_kv_gauges()

    def _do_prefill(self, group: list[Live]) -> None:
        # The prefill set is chosen before step 2 of _schedule, which can
        # preempt or retire a sequence that is in it. Running the model on a
        # sequence whose KV has just been dropped -- and whose sequence id may
        # already have been handed to someone else -- is exactly the kind of
        # cross-request corruption this project is meant not to have.
        group = [lv for lv in group if lv in self.running]
        if not group:
            return
        budget = self.cfg.max_prefill_tokens
        seqs = [lv.seq for lv in group]
        before = sum(lv.seq.n_past for lv in group)
        toks = self.runner.prefill_chunk(seqs, budget)
        self.metrics.tokens("prefill", sum(lv.seq.n_past for lv in group) - before)
        self.metrics.prefill_batch.observe(len(group))
        for lv, tok in zip(group, toks, strict=True):
            if tok is None:
                continue  # still chunking through the prompt
            self._grow(lv)
            if not self._emit(lv, tok):
                self._retire(lv, self._reason(lv))

    def _do_decode(self, group: list[Live]) -> None:
        alive = [lv for lv in group if lv in self.running]  # see _do_prefill
        if not alive:
            return
        toks = self.runner.decode_step([lv.seq for lv in alive])
        self.metrics.tokens("decode", len(alive))
        for lv, tok in zip(alive, toks, strict=True):
            self._grow(lv)
            if not self._emit(lv, tok):
                self._retire(lv, self._reason(lv))

    def _grow(self, live: Live) -> bool:
        """Reserve a slot for the token just produced, privatising a shared
        block first if the write would land in one (copy-on-write).

        Cached prefixes are the reclaimable memory, so they are evicted before
        the request is given up on. If even that is not enough the request
        simply does not grow this step; the next ``_schedule`` sees
        ``can_append() == False`` and preempts it.
        """
        rq = live.rq
        before = self.blocks.n_copy_on_write
        for attempt in (0, 1):
            try:
                self.blocks.append_slot(rq.block_ids, rq.n_tokens)
                break
            except OutOfBlocks:
                if attempt == 0 and self.prefix is not None and self.prefix.evict(1) > 0:
                    continue
                return False
        if self.blocks.n_copy_on_write != before:
            self.metrics.cow_total.inc()
        return True

    @staticmethod
    def _reason(live: Live) -> str:
        rq = live.rq
        if rq.cancelled:
            return "cancelled"
        return "length" if len(rq.output_ids) >= rq.max_tokens else "stop"

    def _abort(self, exc: BaseException) -> None:
        for live in list(self.running):
            self._release_backend(live)
            self.running.remove(live)
            live.rq.fail(exc)

    def _update_kv_gauges(self) -> None:
        self.metrics.kv_blocks_free.set(self.blocks.free_blocks())
        live_tokens = [lv.rq.n_tokens for lv in self.running]
        self.metrics.kv_fragmentation.set(self.blocks.fragmentation_ratio(live_tokens))

    def stats(self) -> dict:
        out = super().stats()
        out.update(
            {
                "kv_blocks_free": self.blocks.free_blocks(),
                "kv_blocks_total": self.blocks.n_blocks,
                "prefix_hit_rate": self.prefix.hit_rate if self.prefix else 0.0,
                "prefix_nodes": self.prefix.n_nodes() if self.prefix else 0,
                "cow": self.blocks.n_copy_on_write,
                "kv_core": self.kv_core,
            }
        )
        return out
