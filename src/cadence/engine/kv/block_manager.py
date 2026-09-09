"""Paged KV cache: the block manager (Python reference).

The alternative -- one contiguous KV region per sequence, sized for
``max_tokens`` -- wastes memory to internal fragmentation on every request that
stops early, and that waste is what caps batch size and therefore throughput.

Instead KV memory is carved into fixed-size blocks of ``block_size`` tokens
(16 is the standard choice). Each sequence holds a *block table*: a list of
physical block ids. Sequences grow by appending blocks, release them on
completion, and share identical prefixes by reference count.

Week 3 replaces the hot path (``alloc``/``release``/``share``) with C++17
behind pybind11; this module stays as the reference the extension is
differentially tested against.
"""

from __future__ import annotations


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class OutOfBlocks(RuntimeError):
    def __init__(self, wanted: int, available: int) -> None:
        super().__init__(f"out of KV blocks: wanted {wanted}, {available} free")
        self.wanted = wanted
        self.available = available


class BlockManager:
    """Fixed-size KV block allocator with reference-counted sharing."""

    def __init__(self, n_blocks: int, block_size: int = 16) -> None:
        if n_blocks <= 0 or block_size <= 0:
            raise ValueError("n_blocks and block_size must be positive")
        self.n_blocks = n_blocks
        self.block_size = block_size
        # LIFO free list: the most recently freed block is handed out first,
        # because it is the one most likely to still be warm.
        self.free: list[int] = list(range(n_blocks - 1, -1, -1))
        self.refcount: list[int] = [0] * n_blocks
        self.n_copy_on_write = 0

    # --- accounting -------------------------------------------------------
    def free_blocks(self) -> int:
        return len(self.free)

    def used_blocks(self) -> int:
        return self.n_blocks - len(self.free)

    def blocks_needed(self, n_tokens: int, max_new_tokens: int, cached: int = 0) -> int:
        """Worst-case footprint: what this request would occupy if it ran to
        its full token budget.

        Used as the *admission gate* -- a request is not let in unless that many
        blocks are currently free. It is deliberately not the number that gets
        allocated (see :meth:`initial_blocks`): gating on the worst case while
        allocating the actual case is what keeps admission from thrashing
        without paying for the worst case in resident memory.
        """
        total = n_tokens + max_new_tokens
        return max(0, _ceil_div(total, self.block_size) - _ceil_div(cached, self.block_size))

    def initial_blocks(self, n_prompt_tokens: int, cached: int = 0) -> int:
        """Blocks to allocate at admission: enough for the prompt, and no more.

        The sequence then grows one block at a time through :meth:`append_slot`
        as it actually generates. This is the whole point of paging -- a request
        that stops after 20 tokens never occupies the 512 it was allowed to ask
        for -- and it is what makes preemption a live path rather than dead
        code: several requests can each pass the admission gate and still
        collectively outgrow the pool.
        """
        return max(
            0, _ceil_div(n_prompt_tokens, self.block_size) - _ceil_div(cached, self.block_size)
        )

    # --- allocation -------------------------------------------------------
    def alloc(self, n: int) -> list[int]:
        """Allocate ``n`` blocks with refcount 1.

        Strong exception guarantee: on failure nothing is consumed.
        """
        if n < 0:
            raise ValueError("n must be >= 0")
        if n > len(self.free):
            raise OutOfBlocks(n, len(self.free))
        out = [self.free.pop() for _ in range(n)]
        for b in out:
            self.refcount[b] = 1
        return out

    def share(self, block_ids: list[int]) -> None:
        for b in block_ids:
            if self.refcount[b] <= 0:
                raise RuntimeError(f"cannot share free block {b}")
            self.refcount[b] += 1

    def release(self, block_ids: list[int]) -> None:
        for b in block_ids:
            if self.refcount[b] <= 0:
                raise RuntimeError(f"double free of block {b}")
            self.refcount[b] -= 1
            if self.refcount[b] == 0:
                self.free.append(b)

    # --- copy-on-write ----------------------------------------------------
    def is_shared(self, block_id: int) -> bool:
        return self.refcount[block_id] > 1

    def fork_for_write(self, block_table: list[int], block_idx: int) -> int:
        """Privatise a shared block that is about to be appended to.

        This is where paged caches actually break. A block that is only
        partially full and shared between two sequences must be copied before
        either appends to it; get it wrong and one user's tokens appear in
        another user's stream.
        """
        old = block_table[block_idx]
        if self.refcount[old] == 1:
            return old
        # Raises OutOfBlocks if there is nowhere to copy to, consuming nothing.
        # Callers gate on can_append, which accounts for this.
        new = self.alloc(1)[0]
        self.n_copy_on_write += 1
        self.release([old])
        block_table[block_idx] = new
        return new

    # --- growth -----------------------------------------------------------
    def slots_in(self, block_table: list[int]) -> int:
        return len(block_table) * self.block_size

    def can_append(self, block_table: list[int], n_filled: int, n: int = 1) -> bool:
        """Can ``n`` more tokens be written, given the free list as it stands?

        Spare room in the last block is not sufficient on its own: if that
        block is shared, writing into it first requires privatising it, and
        that copy needs a free block like any other allocation. Ignoring the
        copy makes this optimistic, which shows up as the scheduler deciding
        one step late that it needed to preempt.
        """
        spare = self.slots_in(block_table) - n_filled
        idx = n_filled // self.block_size
        cow = 1 if (idx < len(block_table) and self.is_shared(block_table[idx])) else 0
        if spare >= n:
            return cow <= len(self.free)
        return _ceil_div(n - spare, self.block_size) + cow <= len(self.free)

    def append_slot(self, block_table: list[int], n_filled: int) -> int | None:
        """Make room for one more token. Returns a newly allocated block id if
        one was needed, else ``None``. Raises :class:`OutOfBlocks` if it cannot."""
        if self.slots_in(block_table) > n_filled:
            # Writing into the last block. If it is shared, privatise first.
            idx = n_filled // self.block_size
            if idx < len(block_table) and self.is_shared(block_table[idx]):
                self.fork_for_write(block_table, idx)
            return None
        new = self.alloc(1)[0]
        block_table.append(new)
        return new

    # --- metrics ----------------------------------------------------------
    def fragmentation_ratio(self, live_token_counts: list[int]) -> float:
        """allocated tokens / (allocated blocks x block_size).

        1.0 means every allocated block is completely full. The contiguous
        allocator this replaces sits at ``mean(n_tokens) / max_tokens``.
        """
        used = self.used_blocks()
        if used == 0:
            return 1.0
        return sum(live_token_counts) / (used * self.block_size)

    def snapshot(self) -> dict[str, int]:
        return {
            "total": self.n_blocks,
            "free": self.free_blocks(),
            "used": self.used_blocks(),
            "cow": self.n_copy_on_write,
        }


class ContiguousBlockManager(BlockManager):
    """The allocator the paged one replaces, kept so the ablation is honest.

    A sequence gets one slab sized for the *maximum* generation length the
    server will ever produce, not for its own request. Every request that stops
    early leaves the tail of its slab reserved and unusable, so the batch size
    is capped by the worst case rather than the typical case -- which is exactly
    the throughput ceiling paged KV exists to lift.

    It shares the free list and refcount machinery, because the point of the
    comparison is the *reservation policy*, not the bookkeeping around it.
    """

    def __init__(self, n_blocks: int, block_size: int = 16, reserve_tokens: int = 512) -> None:
        super().__init__(n_blocks, block_size)
        self.reserve_tokens = reserve_tokens

    def blocks_needed(self, n_tokens: int, max_new_tokens: int, cached: int = 0) -> int:
        # ``max_new_tokens`` is deliberately ignored: the slab is sized for the
        # server-wide cap.
        total = n_tokens + self.reserve_tokens
        return max(0, _ceil_div(total, self.block_size) - _ceil_div(cached, self.block_size))

    def initial_blocks(self, n_prompt_tokens: int, cached: int = 0) -> int:
        # The slab is taken in full, up front. That is the definition of this
        # allocator, and the reason its batch size is capped by the worst case
        # rather than the typical one.
        return self.blocks_needed(n_prompt_tokens, 0, cached)
