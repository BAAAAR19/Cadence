"""The paged KV block manager.

The copy-on-write path gets the most attention here, because it is where paged
caches actually break, and because the failure is silent: one user's tokens
appear in another user's stream.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cadence.engine.kv.block_manager import BlockManager, ContiguousBlockManager, OutOfBlocks


def test_free_list_is_lifo():
    """Recently freed blocks are handed out first: they are the ones most
    likely to still be warm."""
    bm = BlockManager(8, block_size=16)
    a = bm.alloc(3)
    bm.release([a[1]])
    assert bm.alloc(1) == [a[1]]


def test_alloc_is_all_or_nothing():
    bm = BlockManager(4, 16)
    bm.alloc(3)
    with pytest.raises(OutOfBlocks):
        bm.alloc(2)
    assert bm.free_blocks() == 1, "a failed alloc must consume nothing"


def test_share_and_release_refcounts():
    bm = BlockManager(4, 16)
    b = bm.alloc(2)
    bm.share(b)
    bm.release(b)
    assert bm.free_blocks() == 2, "still referenced by the second holder"
    bm.release(b)
    assert bm.free_blocks() == 4


def test_double_free_is_loud():
    bm = BlockManager(2, 16)
    b = bm.alloc(1)
    bm.release(b)
    with pytest.raises(RuntimeError):
        bm.release(b)


def test_copy_on_write_privatises_a_shared_block():
    bm = BlockManager(8, 16)
    shared = bm.alloc(1)
    bm.share(shared)  # two sequences now hold it
    a_table, b_table = list(shared), list(shared)

    new = bm.fork_for_write(a_table, 0)
    assert new != shared[0]
    assert a_table[0] == new
    assert b_table[0] == shared[0], "the other sequence must be untouched"
    assert bm.refcount[shared[0]] == 1
    assert bm.n_copy_on_write == 1


def test_no_copy_when_not_shared():
    bm = BlockManager(8, 16)
    t = bm.alloc(1)
    assert bm.fork_for_write(t, 0) == t[0]
    assert bm.n_copy_on_write == 0


def test_append_slot_grows_and_forks():
    bm = BlockManager(8, block_size=4)
    table = bm.alloc(1)
    # Fill the first block: four tokens, no new allocation.
    for n in range(4):
        assert bm.append_slot(table, n) is None
    assert len(table) == 1
    # The fifth token needs a new block.
    assert bm.append_slot(table, 4) is not None
    assert len(table) == 2


def test_append_into_a_shared_tail_block_forks_it():
    bm = BlockManager(8, block_size=4)
    table = bm.alloc(1)
    bm.share(table)  # a peer holds the same partially-filled block
    peer = list(table)
    bm.append_slot(table, 2)  # writing token index 2, inside block 0
    assert table[0] != peer[0], "wrote into a block another sequence still holds"


def test_can_append_reports_the_memory_ceiling():
    bm = BlockManager(2, block_size=4)
    t = bm.alloc(2)
    assert bm.can_append(t, n_filled=7)
    assert not bm.can_append(t, n_filled=8), "no blocks left to grow into"


def test_contiguous_allocator_reserves_more_than_paged():
    paged = BlockManager(1024, 16)
    contig = ContiguousBlockManager(1024, 16, reserve_tokens=512)
    # A request with a 600-token prompt that only wants 32 tokens out.
    assert paged.blocks_needed(600, 32) < contig.blocks_needed(600, 32)
    # ...and the contiguous slab ignores the request's own budget entirely.
    assert contig.blocks_needed(600, 32) == contig.blocks_needed(600, 512)


def test_fragmentation_ratio():
    bm = BlockManager(16, block_size=16)
    bm.alloc(2)
    assert bm.fragmentation_ratio([32]) == 1.0
    assert bm.fragmentation_ratio([17]) == pytest.approx(17 / 32)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    ops=st.lists(
        st.tuples(
            st.sampled_from(["alloc", "release", "share", "fork"]),
            st.integers(min_value=0, max_value=3),
        ),
        max_size=120,
    )
)
def test_refcounts_and_free_list_stay_consistent(ops):
    """Property: however the allocator is driven, a block is on the free list
    if and only if its refcount is zero, and no block is ever on it twice."""
    bm = BlockManager(24, block_size=16)
    held: list[list[int]] = []
    for op, k in ops:
        if op == "alloc":
            n = k + 1
            if n <= bm.free_blocks():
                held.append(bm.alloc(n))
        elif op == "release" and held:
            bm.release(held.pop(k % len(held)))
        elif op == "share" and held:
            t = held[k % len(held)]
            bm.share(t)
            held.append(list(t))
        elif op == "fork" and held:
            t = held[k % len(held)]
            if t:
                bm.fork_for_write(t, k % len(t))

        free = set(bm.free)
        assert len(free) == len(bm.free), "a block appears twice on the free list"
        for b in range(bm.n_blocks):
            assert (bm.refcount[b] == 0) == (b in free)
        assert all(c >= 0 for c in bm.refcount)

    for t in held:
        bm.release(t)
    assert bm.free_blocks() == bm.n_blocks, "blocks leaked"
