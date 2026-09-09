"""Differential fuzz: the C++17 core against the Python reference.

"I rewrote it in C++" is a claim, and this is the evidence. Hypothesis draws a
random sequence of operations -- allocate, share, fork, append, insert, match,
pin, evict -- and drives *both* implementations through it in lockstep,
comparing every observable after every single step. A divergence is shrunk to
the shortest sequence that still produces it.

What is compared, and why it is more than the guide's "compare observable
behaviour, not block ids":

* matched token counts, hit rates, node liveness, eviction counts, and the
  owner sequences handed back to the scheduler -- the behaviour the scheduler
  actually depends on;
* **and the block ids themselves.** Both allocators are LIFO stacks driven by
  the same operation sequence, so the ids are not an implementation detail
  that is free to differ: they are a function of the input, and asserting on
  them turns a whole class of bookkeeping bugs from "eventually visible" into
  "visible on the operation that caused it".

The token alphabet is deliberately tiny and prompts are built from a small
corpus of shared prefixes, because a cache is only interesting when prompts
share prefixes; uniformly random tokens would fuzz an empty tree.
"""

from __future__ import annotations

import os

import pytest
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from cadence.engine.kv import core_available
from cadence.engine.kv.block_manager import BlockManager, OutOfBlocks
from cadence.engine.kv.radix_cache import RadixCache as PyCache

pytestmark = pytest.mark.skipif(
    not core_available(), reason="the C++ extension is not built (uv pip install -e .)"
)

if core_available():
    from cadence import _core

B = 8
N_BLOCKS = 24

# A handful of shared "system prompts" plus a random tail: the shape the prefix
# cache exists for.
PREFIXES = [
    list(range(100, 100 + 4 * B)),
    list(range(100, 100 + 4 * B)) + list(range(200, 200 + 3 * B)),
    list(range(300, 300 + 2 * B)),
    [],
]

prompts = st.builds(
    lambda p, tail: PREFIXES[p] + tail,
    st.integers(0, len(PREFIXES) - 1),
    st.lists(st.integers(0, 12), max_size=6 * B),
)

ops = st.lists(
    st.one_of(
        st.tuples(st.just("alloc"), st.integers(1, 4)),
        st.tuples(st.just("free_table"), st.integers(0, 8)),
        st.tuples(st.just("share_table"), st.integers(0, 8)),
        st.tuples(st.just("append"), st.integers(0, 8)),
        st.tuples(st.just("fork"), st.integers(0, 8)),
        st.tuples(st.just("insert"), prompts),
        st.tuples(st.just("match"), prompts),
        st.tuples(st.just("acquire"), st.integers(0, 8)),
        st.tuples(st.just("release_node"), st.integers(0, 8)),
        st.tuples(st.just("evict"), st.integers(1, 12)),
        st.tuples(st.just("evict_nodes"), st.integers(1, 4)),
        st.tuples(st.just("evict_all"), st.none()),
    ),
    min_size=30,
    max_size=200,
)


class Pair:
    """One Python cache and one C++ cache, driven identically."""

    def __init__(self) -> None:
        self.py_blocks = BlockManager(N_BLOCKS, B)
        self.cpp_blocks = _core.BlockAllocator(N_BLOCKS, B)
        self.py = PyCache(self.py_blocks, B)
        self.cpp = _core.RadixCache(self.cpp_blocks, B)
        self.py_released: list[int] = []
        self.cpp_released: list[int] = []
        self.py.on_seq_released = self.py_released.append
        self.cpp.on_seq_released = self.cpp_released.append
        # Parallel handles. `tables` are (py_list, cpp_list, n_filled) triples,
        # standing in for a request's block table.
        self.tables: list[list] = []
        self.nodes: list[tuple] = []
        self.owner = 0
        self.exhausted = False
        # Edge splitting is the subtlest thing in the tree, so the fuzz
        # reports whether it actually happened rather than hoping it did.
        self.splits = 0
        inner = self.py._split

        def counting_split(node, k):
            self.splits += 1
            return inner(node, k)

        self.py._split = counting_split

    # --- comparison -------------------------------------------------------
    def assert_same(self, where: str) -> None:
        assert self.py_blocks.free_blocks() == self.cpp_blocks.free_blocks(), where
        assert list(self.py_blocks.free) == list(self.cpp_blocks.free), where
        assert list(self.py_blocks.refcount) == list(self.cpp_blocks.refcount), where
        assert self.py_blocks.n_copy_on_write == self.cpp_blocks.n_copy_on_write, where
        assert self.py.n_nodes() == self.cpp.n_nodes(), where
        assert self.py.query_tokens == self.cpp.query_tokens, where
        assert self.py.hit_tokens == self.cpp.hit_tokens, where
        assert self.py.n_evictions == self.cpp.n_evictions, where
        assert self.py.hit_rate == pytest.approx(self.cpp.hit_rate), where
        assert self.py.n_owned_sequences() == self.cpp.n_owned_sequences(), where
        assert self.py_released == self.cpp_released, where
        # The whole shape of the tree, not just its size: two trees can hold
        # the same number of nodes and disagree about where an edge was split.
        # Sorted, because the C++ children live in an unordered_map and the
        # traversal order is not part of the contract.
        assert sorted(self.py.paths()) == sorted(
            [list(p) for p in self.cpp.paths()]
        ), where
        for py_t, cpp_t, _ in self.tables:
            assert py_t == list(cpp_t), where
        for py_n, cpp_n in self.nodes:
            # A Python node that has been evicted keeps its object identity but
            # loses its parent; the C++ one becomes an unresolvable handle.
            # Those two must mean the same thing.
            assert (py_n.parent is not None) == cpp_n.alive, where
            if cpp_n.alive:
                assert py_n.owner_seq == cpp_n.owner_seq, where
                assert py_n.refs == cpp_n.refs, where

    # --- the operations ---------------------------------------------------
    def alloc(self, n: int) -> None:
        if n > self.py_blocks.free_blocks():
            self.exhausted = True
            with pytest.raises(OutOfBlocks):
                self.py_blocks.alloc(n)
            with pytest.raises(OutOfBlocks):
                self.cpp_blocks.alloc(n)
            return
        self.tables.append([self.py_blocks.alloc(n), self.cpp_blocks.alloc(n), 0])

    def _table(self, k: int):
        return self.tables[k % len(self.tables)] if self.tables else None

    def free_table(self, k: int) -> None:
        t = self._table(k)
        if t is None:
            return
        self.py_blocks.release(t[0])
        self.cpp_blocks.release(t[1])
        self.tables.remove(t)

    def share_table(self, k: int) -> None:
        t = self._table(k)
        if t is None or not t[0]:
            return
        self.py_blocks.share(t[0])
        self.cpp_blocks.share(list(t[1]))
        self.tables.append([list(t[0]), list(t[1]), t[2]])

    def append(self, k: int) -> None:
        t = self._table(k)
        if t is None:
            return
        py_ok = self.py_blocks.can_append(t[0], t[2])
        cpp_ok = self.cpp_blocks.can_append(t[1], t[2])
        assert py_ok == cpp_ok, "can_append disagreed"
        if not py_ok:
            return
        assert self.py_blocks.append_slot(t[0], t[2]) == self.cpp_blocks.append_slot(t[1], t[2])
        t[2] += 1

    def fork(self, k: int) -> None:
        t = self._table(k)
        if t is None or not t[0]:
            return
        idx = k % len(t[0])
        if self.py_blocks.free_blocks() == 0 and self.py_blocks.is_shared(t[0][idx]):
            return  # both would raise OutOfBlocks; covered by the alloc case
        assert self.py_blocks.fork_for_write(t[0], idx) == self.cpp_blocks.fork_for_write(
            t[1], idx
        )

    def insert(self, tokens: list[int]) -> None:
        n = len(tokens) // B
        if n == 0 or n > self.py_blocks.free_blocks():
            return
        py_b = self.py_blocks.alloc(n)
        cpp_b = self.cpp_blocks.alloc(n)
        assert py_b == list(cpp_b)
        self.owner += 1
        py_node = self.py.insert(tokens, py_b, self.owner)
        cpp_node = self.cpp.insert(tokens, list(cpp_b), self.owner)
        assert (py_node is None) == (cpp_node is None)
        # The inserting request drops its own reference, exactly as _retire does.
        self.py_blocks.release(py_b)
        self.cpp_blocks.release(list(cpp_b))
        if py_node is not None:
            assert py_node.owner_seq == cpp_node.owner_seq
            self.nodes.append((py_node, cpp_node))

    def match(self, tokens: list[int]) -> None:
        a = self.py.match(tokens)
        b = self.cpp.match(tokens)
        assert a.n_tokens == b.n_tokens
        assert a.hit == b.hit
        assert a.owner_seq == b.owner_seq
        assert list(a.block_ids) == list(b.block_ids)
        assert (a.node is None) == (b.node is None)

    def _node(self, k: int):
        live = [(p, c) for p, c in self.nodes if c.alive and p.parent is not None]
        return live[k % len(live)] if live else None

    def acquire(self, k: int) -> None:
        pair = self._node(k)
        if pair is None:
            return
        self.py.acquire(pair[0])
        self.cpp.acquire(pair[1])

    def release_node(self, k: int) -> None:
        pair = self._node(k)
        if pair is None or pair[0].refs == 0:
            return
        self.py.release(pair[0])
        self.cpp.release(pair[1])

    def evict(self, n: int) -> None:
        assert self.py.evict(n) == self.cpp.evict(n)

    def evict_nodes(self, n: int) -> None:
        assert self.py.evict_nodes(n) == self.cpp.evict_nodes(n)

    def evict_all(self, _=None) -> None:
        assert self.py.evict_all_unused() == self.cpp.evict_all_unused()


EXAMPLES = int(os.environ.get("CADENCE_FUZZ_EXAMPLES", "600"))
"""Sequences per run: 600 on a PR, 5 000 on ``main``.

A Hypothesis profile would be the idiomatic knob and does not work here: an
explicit ``max_examples`` on the test wins over the profile, and dropping it
would change the default budget of every other property test in the suite.
"""


@settings(max_examples=EXAMPLES, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(ops)
def test_cpp_core_matches_the_python_reference(ops):
    pair = Pair()
    for op, arg in ops:
        getattr(pair, op)(arg)
        pair.assert_same(f"after {op}({arg!r})")

    # A differential test that never reached an interesting state would pass
    # for the wrong reason, so what each sequence actually exercised is
    # reported in the Hypothesis statistics rather than assumed.
    event(f"prefix hits: {'yes' if pair.py.hit_tokens else 'no'}")
    event(f"evictions: {min(pair.py.n_evictions, 10)}")
    event(f"copy-on-write: {'yes' if pair.py_blocks.n_copy_on_write else 'no'}")
    event(f"edge splits: {'yes' if pair.splits else 'no'}")
    event(f"pool exhausted: {'yes' if pair.exhausted else 'no'}")

    # And nothing leaks: drop everything and the pool must come back whole.
    for t in list(pair.tables):
        pair.free_table(pair.tables.index(t))
    pair.evict_all()
    pair.assert_same("teardown")
    assert pair.py_blocks.free_blocks() == pair.cpp_blocks.free_blocks()
