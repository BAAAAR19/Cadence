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

    # The mock backend's cost model, exposed so a measurement can turn it
    # down. ``bench/profile_core.py`` runs the same scheduler with the
    # modelled forward pass set to ~0 to see what the bookkeeping costs on its
    # own; the defaults are the realistic ones the correctness tests use.
    mock_step_overhead_s: float = 0.004
    mock_decode_s_per_seq: float = 0.0015
    mock_prefill_s_per_token: float = 0.00012

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
    kv_core: Literal["auto", "python", "cpp"] = "auto"
    """Which implementation of the block allocator and the radix cache to run.

    ``auto`` prefers the C++17 extension and falls back to the Python
    reference if it was not built. The explicit values exist so that a
    measurement can pin one -- an A/B whose two arms differ in more than the
    thing being compared is not an A/B -- and so that the Python reference
    stays reachable as an executable specification rather than becoming dead
    code the moment the extension lands.
    """

    # --- SLO / admission -------------------------------------------------
    slo_s: float = 2.0
    admission: Literal["none", "conformal"] = "none"
    retry_after_s: float = 1.0
    """Floor for the ``Retry-After`` header on a 503. The conformal controller
    raises it to its estimate of when capacity will exist."""

    admission_model: str = "models/admission.pkl"
    """The fitted quantile regressor and its calibration scores; see
    ``bench/fit_predictor.py`` and ``cadence.admission.artifact``."""
    admission_alpha: float = 0.0
    """Target miscoverage. 0 means "whatever the model was calibrated at",
    which is the honest default: overriding it here recalibrates the bound at a
    level the reported coverage plot was not drawn for, so the override exists
    for the safety-versus-goodput sweep and is recorded in the run's meta."""
    admission_mode: Literal["static", "rolling", "aci"] = "static"
    """static: the offline split-conformal bound, which is the one with the
    finite-sample guarantee. rolling: recalibrate on the most recent
    completions. aci: adaptive conformal inference. The last two exist because
    the controller's own shedding breaks the exchangeability the first one
    assumes; see cadence.admission.conformal."""
    admission_safety: float = 1.0
    """Multiplier on the bound before it is compared with the budget. Kept at 1
    for the headline runs -- a tuned fudge factor with no measurement behind it
    is a red flag -- and swept in the writeup."""
    admission_hysteresis: float = 0.9
    """While shedding, the bound must fit inside this fraction of the SLO
    before admitting resumes. Damps the shed / admit oscillation."""
    admission_window: int = 512
    admission_refresh: int = 16
    admission_aci_gamma: float = 0.005

    trace_log: str | None = None
    """JSONL of one row per request -- admission features, then outcome. This
    is how the training set for the predictor is collected; unset means no
    trace is written and no feature vector is extracted."""

    # --- observability --------------------------------------------------
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    otlp_endpoint: str = "http://localhost:4317"
    profile_out: str | None = None
    """Path for a folded-stack profile of the scheduler thread. Unset means no
    profiler thread is started at all; see ``cadence.obs.profiler``."""
    profile_hz: int = 200

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
