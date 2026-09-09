from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

MODEL = ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs the real model on disk")


@pytest.fixture(scope="session")
def model_path() -> str:
    if not MODEL.exists():
        pytest.skip(f"model not present at {MODEL}")
    return str(MODEL)


@pytest.fixture(scope="session")
def llama_runner(model_path):
    """One shared llama.cpp context for the whole session.

    Loading the model costs a few seconds and 500 MB; the tests that need it
    are careful to reset the KV cache rather than to rebuild the context.
    """
    from cadence.config import Settings
    from cadence.engine.backends.llamacpp import LlamaCppRunner

    cfg = Settings(model_path=model_path, n_ctx=8192, n_batch=512, n_parallel=16,
                   temperature=0.0)
    runner = LlamaCppRunner(cfg)
    yield runner
    runner.close()


@pytest.fixture(params=["python", "cpp"])
def kv(request):
    """The KV core under test: the Python reference, or the C++17 extension.

    Every hand-written test of the block manager and the radix cache runs
    against both. The Python module is the specification, so "the extension is
    correct" means "the extension passes the specification's own tests" --
    not only that it agrees with it on random inputs, which is what
    tests/test_kv_parity.py separately establishes.
    """
    if request.param == "cpp":
        core = pytest.importorskip(
            "cadence._core", reason="the C++ extension is not built (uv pip install -e .)"
        )
        return SimpleNamespace(
            name="cpp",
            BlockManager=core.BlockAllocator,
            ContiguousBlockManager=core.ContiguousBlockAllocator,
            RadixCache=core.RadixCache,
        )
    from cadence.engine.kv.block_manager import BlockManager, ContiguousBlockManager
    from cadence.engine.kv.radix_cache import RadixCache

    return SimpleNamespace(
        name="python",
        BlockManager=BlockManager,
        ContiguousBlockManager=ContiguousBlockManager,
        RadixCache=RadixCache,
    )


@pytest.fixture(params=["python", "cpp"])
def kv_core(request):
    """``CADENCE_KV_CORE`` for a test that drives the whole engine.

    The ``kv`` fixture above checks the two implementations against the KV
    specification directly; this one checks that the scheduler cannot tell
    them apart, which is a different claim and the one the ablation depends
    on.
    """
    if request.param == "cpp":
        pytest.importorskip(
            "cadence._core", reason="the C++ extension is not built (uv pip install -e .)"
        )
    return request.param


@pytest.fixture
def cfg_mock():
    from cadence.config import Settings

    return Settings(backend="mock", scheduler="continuous", config_name="test",
                    n_ctx=4096, block_size=16, max_batch=8, n_parallel=16)
