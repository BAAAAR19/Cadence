"""Every knob in the gateway lives here.

Settings are read from the environment with a ``CADENCE_`` prefix, so a load
sweep can flip a scheduler or a budget without editing code:

    CADENCE_SCHEDULER=continuous CADENCE_MAX_PREFILL_TOKENS=512 uv run cadence-serve
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SchedulerName = Literal["fifo", "static", "continuous"]
BackendName = Literal["llamacpp", "llamacpp_http", "mock"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CADENCE_", env_file=".env", extra="ignore"
    )

    # --- identity -------------------------------------------------------
    config_name: str = Field(
        default="dev",
        description="Label attached to every metric and every parquet row. "
        "One value per rung of the ablation ladder.",
    )
    served_model_name: str = "qwen"

    # --- backend --------------------------------------------------------
    backend: BackendName = "llamacpp"
    model_path: str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    llama_server_url: str = "http://localhost:8081"
    n_ctx: int = 8192
    n_batch: int = 512
    n_parallel: int = 16
    n_threads: int | None = None
    seed: int = 1234
    verbose_backend: bool = False

    # Sampling. Greedy by default: two sequences decoded together must be
    # comparable with the same two decoded apart (see tests/test_backend_equivalence).
    temperature: float = 0.0
    top_p: float = 1.0

    # --- scheduler ------------------------------------------------------
    scheduler: SchedulerName = "continuous"
    max_batch: int = 16
    max_prefill_tokens: int = 512
    """Token budget for prefill work per engine step. Chunked prefill splits a
    long prompt across steps so decoding sequences are not stalled by it."""
    prefill_priority: bool = True
    """True: run this step's prefill chunks before decoding. False: decode
    first, prefill with whatever budget is left. The TTFT/ITL trade-off knob."""
    max_tokens_default: int = 128
    max_tokens_cap: int = 512
    """The largest generation the server will produce. The contiguous
    allocator reserves a slab this size for every sequence."""
    static_batch_size: int = 8
    static_fill_timeout_s: float = 0.05
    """Static batching is wait-to-fill: a wave is launched once it reaches
    ``static_batch_size`` or this long has passed, whichever comes first."""
    max_waiting: int = 4096

    # --- KV memory ------------------------------------------------------
    block_size: int = 16
    kv_blocks: int | None = None
    """Number of paged KV blocks. Defaults to n_ctx // block_size, i.e. the
    real capacity of the llama.cpp unified KV cache."""
    kv_watermark: float = 0.02
    """Fraction of the block pool the scheduler refuses to admit into. A small
    reserve absorbs the growth of sequences already running, so admission does
    not immediately create the preemption it will then have to resolve."""
    enable_prefix_cache: bool = True
    enable_paged_kv: bool = True
    preemption_policy: Literal["recompute"] = "recompute"

    # --- SLO / admission (Week 4 makes admission interesting) ------------
    slo_s: float = 2.0
    admission: Literal["none", "conformal"] = "none"
    retry_after_s: float = 1.0

    # --- observability --------------------------------------------------
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    otlp_endpoint: str = "http://localhost:4317"

    # --- server ---------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000

    @model_validator(mode="after")
    def _prefill_budget_fits_the_batch(self) -> Settings:
        """``max_prefill_tokens`` is the number of prompt tokens one engine
        step may push through ``llama_decode``, and llama.cpp's batch is
        allocated for ``n_batch`` tokens. Asking for more than that used to
        write past the end of it -- a silent corruption that showed up as every
        request failing, with no indication of why. Raising unchunked prefill
        is legitimate; it just has to raise ``n_batch`` with it.
        """
        if self.max_prefill_tokens > self.n_batch:
            raise ValueError(
                f"max_prefill_tokens ({self.max_prefill_tokens}) exceeds n_batch "
                f"({self.n_batch}); raise n_batch to at least the prefill budget"
            )
        return self

    @property
    def n_kv_blocks(self) -> int:
        return self.kv_blocks if self.kv_blocks is not None else self.n_ctx // self.block_size


def get_settings() -> Settings:
    return Settings()
