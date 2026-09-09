from __future__ import annotations

import sys
from pathlib import Path

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


@pytest.fixture
def cfg_mock():
    from cadence.config import Settings

    return Settings(backend="mock", scheduler="continuous", config_name="test",
                    n_ctx=4096, block_size=16, max_batch=8, n_parallel=16)
