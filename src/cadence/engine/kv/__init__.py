"""KV memory: the block allocator and the radix prefix cache.

Two implementations of each live here. The Python ones in
``block_manager.py`` and ``radix_cache.py`` are the reference: they are the
readable statement of the semantics, they are what the C++ is differentially
tested against, and they are what runs when the extension is not built. The
C++17 ones behind ``cadence._core`` are what runs by default.

``build_kv`` is the single place that chooses, so that the scheduler never
learns which one it got.
"""

from __future__ import annotations

import importlib

from cadence.engine.kv.protocols import BlockPool, PrefixCache, PrefixMatch, PrefixNode

__all__ = [
    "BlockPool",
    "PrefixCache",
    "PrefixMatch",
    "PrefixNode",
    "build_kv",
    "core_available",
    "core_name",
]


def core_available() -> bool:
    try:
        importlib.import_module("cadence._core")
    except ImportError:
        return False
    return True


def core_name(requested: str) -> str:
    """Resolve ``auto`` against what is actually installed.

    A request for ``cpp`` that cannot be satisfied is an error rather than a
    silent downgrade: a benchmark that thinks it measured the extension and
    quietly measured Python is worse than a benchmark that failed.
    """
    if requested == "cpp":
        if not core_available():
            raise RuntimeError(
                "CADENCE_KV_CORE=cpp but cadence._core is not built; "
                "run `uv pip install -e .` (needs a C++17 compiler and CMake)"
            )
        return "cpp"
    if requested == "python":
        return "python"
    return "cpp" if core_available() else "python"


def build_kv(cfg) -> tuple[BlockPool, PrefixCache | None, str]:
    """Return ``(blocks, prefix_cache_or_None, core_name)`` for this config.

    The annotation is the load-bearing part: both branches below must satisfy
    the same protocols, so a drift between the two implementations is a type
    error here rather than a divergence discovered in a benchmark.
    """
    which = core_name(cfg.kv_core)
    if which == "cpp":
        from cadence.engine.kv.cpp import build_cpp_kv

        return (*build_cpp_kv(cfg), which)

    from cadence.engine.kv.block_manager import BlockManager, ContiguousBlockManager
    from cadence.engine.kv.radix_cache import RadixCache

    blocks = (
        BlockManager(cfg.n_kv_blocks, cfg.block_size)
        if cfg.enable_paged_kv
        else ContiguousBlockManager(cfg.n_kv_blocks, cfg.block_size, cfg.max_tokens_cap)
    )
    prefix = RadixCache(blocks, cfg.block_size) if cfg.enable_prefix_cache else None
    return blocks, prefix, which
