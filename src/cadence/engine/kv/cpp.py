"""The C++17 core, assembled the same way the Python one is.

Thin by design: everything interesting is in ``src/cpp``, and everything this
module does is choose between the paged and contiguous allocators exactly as
``build_kv`` does for the Python reference. If this file ever grows logic, the
two implementations have started to diverge somewhere other than in language.
"""

from __future__ import annotations

from cadence import _core
from cadence.engine.kv.protocols import BlockPool, PrefixCache


def build_cpp_kv(cfg) -> tuple[BlockPool, PrefixCache | None]:
    blocks = (
        _core.BlockAllocator(cfg.n_kv_blocks, cfg.block_size)
        if cfg.enable_paged_kv
        else _core.ContiguousBlockAllocator(
            cfg.n_kv_blocks, cfg.block_size, cfg.max_tokens_cap
        )
    )
    prefix = _core.RadixCache(blocks, cfg.block_size) if cfg.enable_prefix_cache else None
    return blocks, prefix
