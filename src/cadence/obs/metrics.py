"""Prometheus collectors.

Every claim the writeup makes should be visible on a dashboard while a load
test runs; that also makes debugging the scheduler dramatically faster.

One caveat, stated here and repeated in the README: Prometheus quantiles are
interpolated *within* a bucket, so a histogram with default buckets reports a
p99 that is wrong by a lot at exactly the values that matter. The buckets below
are dense around a 2 s SLO. Even so, the numbers that go in the README are
computed from the load generator's raw parquet; Prometheus is for live
observation only.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

LAT_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
    0.75, 1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0, 8.0, 16.0, 32.0,
)
ITL_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)

_L = ["config"]

ttft = Histogram(
    "cadence_ttft_seconds", "Time to first token, from arrival",
    _L, buckets=LAT_BUCKETS, registry=REGISTRY,
)
itl = Histogram(
    "cadence_itl_seconds", "Inter-token latency",
    _L, buckets=ITL_BUCKETS, registry=REGISTRY,
)
e2e = Histogram(
    "cadence_e2e_seconds", "End-to-end latency, from arrival",
    _L, buckets=LAT_BUCKETS, registry=REGISTRY,
)
queue_wait = Histogram(
    "cadence_queue_wait_seconds", "Time spent in the wait queue",
    _L, buckets=LAT_BUCKETS, registry=REGISTRY,
)
step_latency = Histogram(
    "cadence_step_seconds", "Engine step duration",
    _L, buckets=ITL_BUCKETS, registry=REGISTRY,
)
prefill_batch = Histogram(
    "cadence_prefill_batch_size", "Sequences prefilling per step",
    _L, buckets=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32), registry=REGISTRY,
)

queue_depth = Gauge("cadence_queue_depth", "Requests waiting", _L, registry=REGISTRY)
batch_size = Gauge("cadence_batch_size", "Sequences in the running batch", _L, registry=REGISTRY)
kv_blocks_total = Gauge("cadence_kv_blocks_total", "KV blocks in the pool", _L, registry=REGISTRY)
kv_blocks_free = Gauge("cadence_kv_blocks_free", "Free KV blocks", _L, registry=REGISTRY)
kv_fragmentation = Gauge(
    "cadence_kv_fragmentation_ratio",
    "allocated tokens / (allocated blocks x block_size)",
    _L, registry=REGISTRY,
)
prefix_hit_tokens = Counter(
    "cadence_prefix_cache_hit_tokens_total", "Prompt tokens served from the prefix cache",
    _L, registry=REGISTRY,
)
prefix_query_tokens = Counter(
    "cadence_prefix_cache_query_tokens_total", "Prompt tokens looked up in the prefix cache",
    _L, registry=REGISTRY,
)
preemptions = Counter(
    "cadence_preemptions_total", "Sequences preempted", _L + ["policy"], registry=REGISTRY
)
shed_total = Counter("cadence_shed_total", "Requests shed", _L + ["reason"], registry=REGISTRY)
slo_met_total = Counter(
    "cadence_slo_met_total", "Requests completing within the SLO", _L, registry=REGISTRY
)
requests_total = Counter(
    "cadence_requests_total", "Requests by terminal state", _L + ["state"], registry=REGISTRY
)
tokens_total = Counter(
    "cadence_tokens_total", "Tokens processed", _L + ["kind"], registry=REGISTRY
)
cow_total = Counter(
    "cadence_kv_copy_on_write_total", "Shared KV blocks privatised before a write",
    _L, registry=REGISTRY,
)


class Metrics:
    """Thin façade that binds the ``config`` label once.

    The scheduler calls ``self.metrics.ttft.observe(x)`` without ever knowing
    which ablation rung it is running as.
    """

    def __init__(self, config_name: str, enabled: bool = True) -> None:
        self.config = config_name
        self.enabled = enabled
        lb = {"config": config_name}
        self.ttft = ttft.labels(**lb)
        self.itl = itl.labels(**lb)
        self.e2e = e2e.labels(**lb)
        self.queue_wait = queue_wait.labels(**lb)
        self.step_latency = step_latency.labels(**lb)
        self.prefill_batch = prefill_batch.labels(**lb)
        self.queue_depth = queue_depth.labels(**lb)
        self.batch_size = batch_size.labels(**lb)
        self.kv_blocks_total = kv_blocks_total.labels(**lb)
        self.kv_blocks_free = kv_blocks_free.labels(**lb)
        self.kv_fragmentation = kv_fragmentation.labels(**lb)
        self.prefix_hit_tokens = prefix_hit_tokens.labels(**lb)
        self.prefix_query_tokens = prefix_query_tokens.labels(**lb)
        self.slo_met_total = slo_met_total.labels(**lb)
        self.cow_total = cow_total.labels(**lb)
        self._preemptions = preemptions
        self._shed = shed_total
        self._requests = requests_total
        self._tokens = tokens_total

    def preempted(self, policy: str = "recompute") -> None:
        self._preemptions.labels(config=self.config, policy=policy).inc()

    def shed(self, reason: str) -> None:
        self._shed.labels(config=self.config, reason=reason).inc()

    def finished(self, state: str) -> None:
        self._requests.labels(config=self.config, state=state).inc()

    def tokens(self, kind: str, n: int) -> None:
        if n:
            self._tokens.labels(config=self.config, kind=kind).inc(n)
