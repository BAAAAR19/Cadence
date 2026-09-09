"""The radix prefix cache (Python reference).

When 70% of requests share a long system prompt, prefilling it every time is
pure waste. A radix tree over token sequences lets a new request find the
longest cached prefix and reuse those KV blocks instead of recomputing them.

Each node holds a run of token ids and the KV block ids covering them.
Matching walks the tree consuming the incoming prompt; a partial match inside a
node splits that node.

Three rules keep it correct:

1. **Match only at block boundaries.** A partial block cannot be shared,
   because the tail of it will be written by whichever sequence continues
   first. The match is truncated down to a multiple of ``block_size``.
2. **Reference-count nodes, not just blocks.** A node whose ``refs > 0`` must
   never be evicted, or a running sequence loses its history mid-generation.
3. **Evict leaves only, LRU by ``last_used``, bottom-up.** Evicting an
   interior node would orphan its children.

A fourth rule is specific to running on top of llama.cpp: the *blocks* are an
accounting model, but the physical KV lives in a llama.cpp sequence. Every node
therefore names an ``owner_seq`` -- a sequence id whose KV holds exactly the
tokens on the path from the root to that node -- and the scheduler materialises
a hit with ``runner.copy_prefix(owner_seq, new_seq, n_tokens)``. Owner
sequences are reference-counted too, and only released back to the pool when
the last node naming them is evicted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from cadence.engine.kv.block_manager import BlockManager

BlockKey = tuple[int, ...]


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class Node:
    tokens: list[int]
    """The edge label: a run of token ids."""
    blocks: list[int]
    """KV blocks covering exactly these tokens (aligned to block boundaries)."""
    children: dict[int, Node] = field(default_factory=dict)
    parent: Node | None = None
    refs: int = 0
    """Sequences currently using this node. Never evict while > 0."""
    last_used: float = 0.0
    owner_seq: int = -1
    depth_tokens: int = 0
    """Path length from the root through this node, in tokens."""

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass(slots=True)
class Match:
    n_tokens: int
    block_ids: list[int]
    node: Node | None
    owner_seq: int = -1

    @property
    def hit(self) -> bool:
        return self.n_tokens > 0


class RadixCache:
    def __init__(self, blocks: BlockManager, block_size: int = 16) -> None:
        self.blocks = blocks
        self.block_size = block_size
        self.root = Node(tokens=[], blocks=[], depth_tokens=0)
        self.root.refs = 1  # the root is never evictable
        self._seq_refs: dict[int, int] = {}
        self.on_seq_released = lambda seq_id: None
        # token-level counters; a request-level hit rate would hide the fact
        # that partial hits are the normal case
        self.query_tokens = 0
        self.hit_tokens = 0
        self.n_evictions = 0

    def _key(self, tokens: list[int], i: int = 0) -> BlockKey:
        """Children are keyed by the whole first *block* of their edge, not by
        its first token.

        Keying on the first token looks natural and is wrong here. Two prompts
        that share only the four-token ChatML header would land in the same
        child slot, diverge inside the first block -- which cannot be split,
        because a block is the unit of sharing -- and the tree would have
        nowhere to put the second one. Keying on the block means sequences that
        differ anywhere inside their first block are simply siblings, which is
        exactly what they are.
        """
        return tuple(tokens[i : i + self.block_size])

    # --- lookup -----------------------------------------------------------
    def match(self, token_ids: list[int], *, count: bool = True) -> Match:
        """Longest prefix match, truncated to a multiple of ``block_size``.

        Mutating: a prompt that diverges part-way through a node's edge splits
        that node, so the shared head becomes reusable by both. Splitting is
        pure bookkeeping -- no block is allocated, copied or freed.

        At least one token is always left to prefill: a forward pass needs a
        token to produce logits from, so a 100% hit would leave nothing to
        sample.
        """
        if count:
            self.query_tokens += len(token_ids)
        limit = max(0, len(token_ids) - 1)

        node: Node = self.root
        best: Node | None = None
        i = 0
        while i + self.block_size <= limit:
            child = node.children.get(self._key(token_ids, i))
            if child is None:
                break
            n = _common_prefix_len(child.tokens, token_ids[i:limit])
            if n < len(child.tokens):
                # Partial match inside the edge. Split the node at the last
                # block boundary the two sequences agree on, so the shared head
                # becomes a node in its own right and can be handed out.
                k = n - (n % self.block_size)
                if k > 0 and child.owner_seq >= 0:
                    self._split(child, k)
                    head = node.children[self._key(token_ids, i)]
                    head.last_used = time.monotonic()
                    best = head
                break
            i += n
            node = child
            node.last_used = time.monotonic()
            if node.owner_seq >= 0:
                best = node

        if best is None:
            return Match(n_tokens=0, block_ids=[], node=None)

        matched = (best.depth_tokens // self.block_size) * self.block_size
        matched = min(matched, limit)
        matched = (matched // self.block_size) * self.block_size
        if matched == 0:
            return Match(n_tokens=0, block_ids=[], node=None)

        block_ids = self._path_blocks(best)[: matched // self.block_size]
        if count:
            self.hit_tokens += matched
        return Match(
            n_tokens=matched, block_ids=block_ids, node=best, owner_seq=best.owner_seq
        )

    def _path_blocks(self, node: Node) -> list[int]:
        chain: list[Node] = []
        cur: Node | None = node
        while cur is not None and cur is not self.root:
            chain.append(cur)
            cur = cur.parent
        out: list[int] = []
        for n in reversed(chain):
            out.extend(n.blocks)
        return out

    # --- pinning ----------------------------------------------------------
    def acquire(self, node: Node | None) -> None:
        """Pin the path so nothing on it is evicted while a request uses it."""
        cur = node
        while cur is not None:
            cur.refs += 1
            cur.last_used = time.monotonic()
            cur = cur.parent

    def release(self, node: Node | None) -> None:
        cur = node
        while cur is not None:
            cur.refs -= 1
            cur.last_used = time.monotonic()
            cur = cur.parent

    # --- insertion --------------------------------------------------------
    def insert(self, token_ids: list[int], block_ids: list[int], owner_seq: int) -> Node | None:
        """Record that ``owner_seq``'s KV holds ``token_ids``, covered by
        ``block_ids``.

        Only whole blocks are stored -- a partial trailing block cannot be
        shared. Blocks handed in here gain a cache reference; the caller keeps
        its own reference and releases it independently.
        """
        n = min(len(token_ids), len(block_ids) * self.block_size)
        n = (n // self.block_size) * self.block_size
        if n == 0:
            return None
        toks = token_ids[:n]
        blks = block_ids[: n // self.block_size]

        node = self.root
        i = 0
        while i < n:
            child = node.children.get(self._key(toks, i))
            if child is None:
                new = Node(
                    tokens=toks[i:],
                    blocks=blks[i // self.block_size :],
                    parent=node,
                    last_used=time.monotonic(),
                    depth_tokens=n,
                )
                self.blocks.share(new.blocks)
                self._retain_seq(owner_seq)
                new.owner_seq = owner_seq
                node.children[self._key(toks, i)] = new
                return new
            m = _common_prefix_len(child.tokens, toks[i:])
            if m < len(child.tokens):
                k = m - (m % self.block_size)
                if k == 0:
                    # Unreachable while children are block-keyed: a key match
                    # means the first block agrees, so m >= block_size. Kept as
                    # an assertion in code form, because descending here would
                    # graft this prompt's tail under a node it does not follow.
                    return node if node is not self.root else None
                self._split(child, k)
                child = node.children[self._key(toks, i)]
                m = k
            i += m
            node = child
            node.last_used = time.monotonic()

        # Exact path already present. Give it an owner if it has none, so a
        # later request can actually reuse it.
        if node.owner_seq < 0:
            self._retain_seq(owner_seq)
            node.owner_seq = owner_seq
        return node

    def _split(self, node: Node, k: int) -> None:
        """Split ``node``'s edge after ``k`` tokens. ``k`` is a multiple of the
        block size, so the block list splits cleanly too."""
        k -= k % self.block_size
        if k == 0 or k >= len(node.tokens):
            return
        parent = node.parent
        assert parent is not None
        head = Node(
            tokens=node.tokens[:k],
            blocks=node.blocks[: k // self.block_size],
            parent=parent,
            refs=node.refs,
            last_used=node.last_used,
            depth_tokens=node.depth_tokens - (len(node.tokens) - k),
        )
        # The owner's KV covers the whole path, so it also covers any prefix
        # of it: the head can safely name the same sequence.
        if node.owner_seq >= 0:
            self._retain_seq(node.owner_seq)
            head.owner_seq = node.owner_seq

        node.tokens = node.tokens[k:]
        node.blocks = node.blocks[k // self.block_size :]
        node.parent = head
        head.children[self._key(node.tokens)] = node
        # The head's first block is the old node's first block, so this
        # overwrites the entry the old node occupied rather than leaking one.
        parent.children[self._key(head.tokens)] = head

    # --- owner sequences --------------------------------------------------
    def _retain_seq(self, seq_id: int) -> None:
        if seq_id < 0:
            return
        self._seq_refs[seq_id] = self._seq_refs.get(seq_id, 0) + 1

    def _release_seq(self, seq_id: int) -> None:
        if seq_id < 0:
            return
        n = self._seq_refs.get(seq_id, 0) - 1
        if n <= 0:
            self._seq_refs.pop(seq_id, None)
            self.on_seq_released(seq_id)
        else:
            self._seq_refs[seq_id] = n

    # --- eviction ---------------------------------------------------------
    def _leaves(self) -> list[Node]:
        out: list[Node] = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n is not self.root and n.is_leaf and n.refs == 0:
                out.append(n)
            stack.extend(n.children.values())
        return out

    def evict(self, n_blocks_needed: int) -> int:
        """LRU over leaf nodes with ``refs == 0``. Never evicts an in-use node.

        Returns the number of blocks actually returned to the free list, which
        can be fewer than requested when everything is pinned.
        """
        freed = 0
        while freed < n_blocks_needed:
            leaves = self._leaves()
            if not leaves:
                break
            victim = min(leaves, key=lambda n: n.last_used)
            freed += self._evict_node(victim)
            self.n_evictions += 1
        return freed

    def evict_all_unused(self) -> int:
        return self.evict(self.blocks.n_blocks)

    def evict_nodes(self, n: int) -> int:
        """Evict up to ``n`` unreferenced leaves, least recently used first.

        Used when *sequence ids* rather than blocks are the scarce resource:
        every cached prefix pins the backend sequence whose KV holds it, so a
        cache that has grown one entry per unique prompt can exhaust the
        sequence pool while blocks are still plentiful. Dropping the coldest
        few entries is the proportionate response; dropping the whole cache
        (which is what ``evict_all_unused`` does) throws away the shared
        prompts that are the entire reason the cache exists.
        """
        evicted = 0
        for _ in range(max(0, n)):
            leaves = self._leaves()
            if not leaves:
                break
            self._evict_node(min(leaves, key=lambda x: x.last_used))
            self.n_evictions += 1
            evicted += 1
        return evicted

    def _evict_node(self, node: Node) -> int:
        parent = node.parent
        assert parent is not None
        del parent.children[self._key(node.tokens)]
        self.blocks.release(node.blocks)
        n = len(node.blocks)
        self._release_seq(node.owner_seq)
        node.parent = None
        return n

    # --- metrics ----------------------------------------------------------
    @property
    def hit_rate(self) -> float:
        return self.hit_tokens / self.query_tokens if self.query_tokens else 0.0

    def n_nodes(self) -> int:
        n, stack = 0, [self.root]
        while stack:
            cur = stack.pop()
            n += 1
            stack.extend(cur.children.values())
        return n
