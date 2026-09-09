from cadence.engine.backends.base import ModelRunner, SeqState

__all__ = ["ModelRunner", "SeqState", "build_runner"]


def build_runner(cfg):
    """Construct the backend named by ``cfg.backend``.

    Imports are local so that a test run with the mock backend never needs
    llama_cpp installed.
    """
    if cfg.backend == "mock":
        from cadence.engine.backends.mock import MockRunner

        return MockRunner(cfg)
    if cfg.backend == "llamacpp":
        from cadence.engine.backends.llamacpp import LlamaCppRunner

        return LlamaCppRunner(cfg)
    if cfg.backend == "llamacpp_http":
        from cadence.engine.backends.llamacpp_http import LlamaServerRunner

        return LlamaServerRunner(cfg)
    raise ValueError(f"unknown backend {cfg.backend!r}")
