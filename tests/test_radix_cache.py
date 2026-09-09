"""The radix prefix cache.

The three correctness rules are each asserted directly, because each of them
fails silently in a different and confusing way: a match past a block boundary
corrupts the sharing sequence's tail, evicting a referenced node produces
garbage mid-stream, and evicting an interior node orphans its children.

Every test runs twice, against the Python reference and against the C++17
extension (the ``kv`` fixture in conftest). Assertions are therefore written
against the public surface only -- ``paths()`` rather than ``root.children`` --
because the extension's nodes are not Python objects to walk.
"""

from __future__ import annotations

B = 16


def _cache(kv, n_blocks=64):
    bm = kv.BlockManager(n_blocks, B)
    return bm, kv.RadixCache(bm, B)


def _insert(bm, rc, tokens, owner):
    n_blocks = len(tokens) // B
    blocks = bm.alloc(n_blocks)
    node = rc.insert(tokens, blocks, owner)
    bm.release(blocks)  # the inserting request drops its own reference
    return node


def test_miss_on_an_empty_cache(kv):
    bm, rc = _cache(kv)
    assert not rc.match(list(range(100))).hit


def test_exact_prefix_is_matched_and_truncated_to_a_block_boundary(kv):
    bm, rc = _cache(kv)
    toks = list(range(1000, 1000 + 8 * B))
    _insert(bm, rc, toks, owner=3)

    m = rc.match(toks + [7, 7, 7])
    assert m.n_tokens == 8 * B
    assert m.n_tokens % B == 0
    assert len(m.block_ids) == 8
    assert m.owner_seq == 3


def test_a_partial_block_is_never_shared(kv):
    """A block that is only partly full will be written by whichever sequence
    continues first, so the match must stop below it."""
    bm, rc = _cache(kv)
    toks = list(range(2000, 2000 + 5 * B))
    _insert(bm, rc, toks, owner=1)

    probe = toks[: 3 * B + 7] + [999] * 20
    m = rc.match(probe)
    assert m.n_tokens % B == 0
    assert m.n_tokens <= 3 * B


def test_match_always_leaves_a_token_to_prefill(kv):
    """A 100% hit would leave no token for the forward pass to produce logits
    from."""
    bm, rc = _cache(kv)
    toks = list(range(4 * B))
    _insert(bm, rc, toks, owner=0)
    m = rc.match(toks)
    assert m.n_tokens < len(toks)


def test_divergent_branches_share_the_common_prefix(kv):
    bm, rc = _cache(kv)
    common = list(range(500, 500 + 6 * B))
    a = common + list(range(9000, 9000 + 2 * B))
    b = common + list(range(7000, 7000 + 2 * B))
    _insert(bm, rc, a, owner=1)
    _insert(bm, rc, b, owner=2)

    m = rc.match(common + [1, 2, 3])
    assert m.n_tokens >= 6 * B
    # Both branches still resolve to their own full length.
    assert rc.match(a + [0]).n_tokens >= 8 * B
    assert rc.match(b + [0]).n_tokens >= 8 * B


def test_referenced_nodes_are_never_evicted(kv):
    bm, rc = _cache(kv, n_blocks=16)
    toks = list(range(4 * B))
    _insert(bm, rc, toks, owner=5)
    m = rc.match(toks + [1])
    rc.acquire(m.node)  # a running sequence is using it

    freed = rc.evict(16)
    assert freed == 0
    assert rc.match(toks + [1]).hit

    rc.release(m.node)
    assert rc.evict(16) > 0
    assert not rc.match(toks + [1]).hit


def test_eviction_is_lru_over_leaves(kv):
    bm, rc = _cache(kv, n_blocks=32)
    old = list(range(100, 100 + 4 * B))
    new = list(range(9000, 9000 + 4 * B))
    _insert(bm, rc, old, owner=1)
    _insert(bm, rc, new, owner=2)
    rc.match(new + [0])  # touch the newer entry

    rc.evict(1)
    assert not rc.match(old + [0]).hit, "the least recently used entry survived"
    assert rc.match(new + [0]).hit


def test_eviction_frees_blocks_and_releases_owner_sequences(kv):
    bm, rc = _cache(kv, n_blocks=32)
    released: list[int] = []
    rc.on_seq_released = released.append
    toks = list(range(6 * B))
    _insert(bm, rc, toks, owner=11)
    assert bm.free_blocks() == 32 - 6, "the cache holds a reference to the blocks"

    rc.evict(6)
    assert bm.free_blocks() == 32
    assert released == [11], "the backend sequence backing the cached KV was not freed"


def test_split_keeps_both_halves_usable(kv):
    bm, rc = _cache(kv)
    long = list(range(300, 300 + 8 * B))
    _insert(bm, rc, long, owner=1)
    short = long[: 4 * B] + list(range(8000, 8000 + 2 * B))
    _insert(bm, rc, short, owner=2)

    assert rc.match(long + [0]).n_tokens == 8 * B
    assert rc.match(short + [0]).n_tokens == 6 * B
    assert rc.match(long[: 4 * B] + [1, 2]).n_tokens == 4 * B


def test_token_level_hit_rate_is_what_is_reported(kv):
    """Partial hits are the normal case, so a request-level rate would hide
    most of what the cache is doing."""
    bm, rc = _cache(kv)
    toks = list(range(10 * B))
    _insert(bm, rc, toks, owner=1)
    rc.query_tokens = rc.hit_tokens = 0

    probe = toks[: 5 * B] + list(range(5000, 5000 + 5 * B))
    m = rc.match(probe)
    assert 0 < m.n_tokens < len(probe)
    assert rc.hit_rate == m.n_tokens / len(probe)
    assert 0.0 < rc.hit_rate < 1.0


def test_no_block_leak_across_insert_and_evict_cycles(kv):
    bm, rc = _cache(kv, n_blocks=64)
    for i in range(20):
        toks = list(range(i * 1000, i * 1000 + 4 * B))
        if bm.free_blocks() < 4:
            rc.evict(4)
        _insert(bm, rc, toks, owner=i)
    rc.evict_all_unused()
    assert bm.free_blocks() == 64


def test_divergence_inside_a_block_does_not_corrupt_the_tree(kv):
    """Two prompts that share a long prefix and diverge *inside* a block --
    the normal case for a shared system prompt followed by different user
    turns -- must not graft one prompt's tail under the other's.

    Getting this wrong builds a path through the tree that spells a token
    sequence no request ever sent. It is quiet: matches still mostly work, the
    tree just fills with entries nothing can reach.
    """
    bm, rc = _cache(kv, n_blocks=128)
    shared = list(range(1000, 1000 + 6 * B))          # 96 shared tokens
    a = shared + [7] * 10 + list(range(2000, 2000 + 20))  # diverges at +10
    b = shared + [7] * 10 + list(range(3000, 3000 + 20))
    _insert(bm, rc, a, owner=1)
    _insert(bm, rc, b, owner=2)

    # Every root path must be a prefix of a sequence that was actually inserted.
    for path in rc.paths():
        assert list(path) in (a[: len(path)], b[: len(path)]), (
            f"tree contains a path of {len(path)} tokens matching neither input"
        )

    # And both still match the full shared prefix.
    for seq in (a, b):
        assert rc.match(seq).n_tokens >= 6 * B


def test_shared_system_prompt_with_different_tails_hits_most_of_the_prompt(kv):
    """The workload's actual shape: a long shared prefix, a short unique tail.
    Nearly all of the prompt should be reusable."""
    bm, rc = _cache(kv, n_blocks=256)
    system = list(range(50_000, 50_000 + 33 * B))  # 528 tokens
    first = system + list(range(1, 14))
    second = system + list(range(900, 913))
    _insert(bm, rc, first, owner=1)

    m = rc.match(second)
    assert m.n_tokens >= 32 * B
    assert m.n_tokens / len(second) > 0.9
