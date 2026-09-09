# Cadence

An LLM inference gateway with SLO-aware scheduling: continuous batching, a
paged KV cache and a radix prefix cache in front of a small local model, built
so that a tail-latency target can be *measured* rather than hoped for.

> **Status: Weeks 0–2 of five are complete.** Shipped: the OpenAI-compatible
> streaming gateway, the open-loop measurement harness, and the systems core
> (iteration-level scheduling, paged KV, radix prefix cache, metrics and
> tracing). Not yet built: the C++17 port of the two hot data structures
> (Week 3) and the conformal admission controller (Week 4) that turns the p99
> target into a distribution-free guarantee. Both are stubbed with the
> interfaces they will fill, and the tests for them skip rather than pass
> vacuously. See [Roadmap](#roadmap).

---

## The three claims, and where each is defended

| Claim | Evidence | Status |
|---|---|---|
| **Continuous batching holds latency past the point where FIFO collapses.** | [Latency vs offered load](#results), across four configurations on one interleaved sweep. | Measured |
| **The measurements are sound.** | Open-loop Poisson generator, latency timestamped from *intended* arrival, KS-tested arrival process, goodput reported alongside throughput. | Measured |
| **The C++ is load-bearing, not decorative.** | Profile first, then port the allocator and radix match, then report the end-to-end delta honestly. | Week 3 |

---

## Architecture

```
                 OPEN-LOOP LOAD GENERATOR (Poisson arrivals, fixed rate)
                                  |
                                  v
      +---------------------------------------------------------------+
      | FastAPI /v1/chat/completions (SSE streaming)                    |
      +---------------------------------------------------------------+
                                  |
                     [ 1 ] ADMISSION CONTROLLER            <- Week 4
                     predict latency dist -> conformal upper bound
                     admit / queue / shed (503 + Retry-After)
                                  |
                                  v
                     [ 2 ] WAIT QUEUE (priority = deadline)
                                  |
                                  v
      +---------------------------------------------------------------+
      | [ 3 ] CONTINUOUS-BATCHING SCHEDULER                             |
      |   every step: admit new / chunk prefill / evict / decode        |
      +---------------------------------------------------------------+
           |                        |                       |
           v                        v                       v
  [ 4 ] PAGED KV CACHE     [ 5 ] RADIX PREFIX CACHE   [ 6 ] MODEL RUNNER
  block allocator          shared-prompt reuse         llama.cpp, in-process
  (C++17 in Week 3)        (C++17 in Week 3)           llama_decode
           |                        |                       |
           +------------------------+-----------------------+
                                  |
                     Prometheus / OpenTelemetry
                     TTFT, ITL, E2E, queue depth, goodput
```

The scheduler runs on its own thread. `prefill` and `decode_step` are
synchronous and compute-bound, and calling them on the API event loop blocks
every SSE writer — which corrupts inter-token-latency measurements at exactly
the millisecond scale this project measures at. Requests cross into the engine
through a lock-protected deque; tokens cross back out through
`loop.call_soon_threadsafe`.

---

## Quick start

```bash
git clone <this repo> && cd cadence
uv sync --all-groups
mkdir -p models && hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
    qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir models
uv run cadence-serve
```

Then, with no code changes on the client side:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
for chunk in client.chat.completions.create(
        model="qwen", messages=[{"role": "user", "content": "hello"}], stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```

Observability:

```bash
docker compose -f deploy/docker-compose.yml up --build
# Grafana on :3000 with the Cadence dashboard provisioned, Prometheus on :9090
```

> **Verified so far:** the compose file, Prometheus config, OTel collector
> config, Grafana provisioning and the dashboard JSON are committed and
> schema-checked in CI, and the gateway's `/metrics` and OTLP export are
> covered by tests. The stack has **not** been brought up end to end — Docker
> is not installed on the machine these measurements were taken on. Treating
> "the YAML parses" as "the dashboard works" is exactly the kind of claim this
> project is trying not to make, so it is flagged rather than implied. Note
> also that the container builds `llama-cpp-python` for CPU: there is no Metal
> inside Docker on macOS, so containerised throughput is well below the host
> numbers reported here.

Reproduce the measurements:

```bash
./bench/run_ablation.sh
```

---

## Why a 0.5B model

Not a compromise — the correct choice for what is being benchmarked. Cadence is
a **scheduler**, and a large model would make model time dominate every
scheduling decision, mask the effect of each policy, and turn a five-minute
experiment into an hour-long one. Qwen2.5-0.5B-Instruct at Q4\_K\_M decodes fast
enough that the scheduler is the bottleneck, and it is instruction-tuned, so
response lengths have a realistic distribution.

The consequence is stated rather than hidden: absolute throughput here is a
property of an M-series laptop and a 0.5B model, and only the *relative*
differences between configurations transfer.

---

## Methodology

### Open loop, and why it is not negotiable

A closed-loop generator — N workers, each waiting for its response before
sending the next — makes offered load a *consequence* of server speed. When the
server slows, the client slows with it, the queue never builds, and overload
behaviour cannot be observed at all. Since overload behaviour is the entire
subject of this project, that harness would answer the wrong question.

Cadence's generator (`bench/loadgen.py`) has three properties, each of which is
tested:

1. **Arrivals are exponential**, not fixed-interval. Fixed intervals understate
   queueing. The realised inter-arrival distribution is KS-tested against
   `Exp(λ)` on every run and the p-value is recorded in the parquet.
2. **Every latency is measured from the intended arrival time**, never the
   actual send time. This is what kills coordinated omission: in a closed loop
   the requests that *would* have been slowest are never issued, so the
   measured tail is optimistic by an order of magnitude.
3. **An outstanding request never delays the next arrival.** The connection
   pool is unbounded on purpose — a pool limit is backpressure, and backpressure
   turns the harness back into a closed loop.

#### The seed is fixed in advance, and it landed in the tail

Every rung replays the **same** arrival sequence (seed 0), because a comparison
between rungs is only a comparison if the thing offered to them is identical.
The cost of choosing the seed before seeing the data is that the seed can land
in the tail of the KS distribution, and this one did: seed 0's realisation is
rejected at α=0.05 at every rate in the sweep.

Those are not six independent failures. The KS statistic for an exponential fit
depends only on the underlying uniform draws, so the six rates are nested
prefixes of one sequence — one draw, reported six times.

`bench/validate_loadgen.py` establishes the process itself over 200 seeds in
exactly the regime the sweep runs in (180 s at 1.4 rps, gaps interleaved with
workload sampling from the same RNG), and commits the result to
`results/loadgen_validation.json`:

| Quantity | Measured | Expected |
|---|---|---|
| KS rejection rate at α=0.05 | 7.5% | ~5% |
| Median KS p-value | 0.478 | 0.5 |
| Mean arrivals per run | 253.8 | 252 |
| SD of arrivals per run | 17.2 | 15.9 |
| Percentile of seed 0's p-value | 1.5th | — |

So the arrival process is exponential and seed 0 is a draw from its tail. The
alternative — re-rolling the seed until the p-value looks better — is exactly
the selection that would make the statistic meaningless, so it is reported
rather than fixed.

The generator was also validated against a model-free stub before it was ever
pointed at the gateway (`bench/stub_server.py`):

| Check | Result |
|---|---|
| λ=5 rps for 60 s against a `sleep(0.1)` stub | 341 requests, near-zero queueing |
| Arrival process over 300 seeds | mean 301.5 arrivals (Poisson expects 300, σ≈17), KS rejection rate 7% at α=0.05, p-values ≈ uniform |
| λ=20 rps against a stub that can only do 10 rps | median E2E 8.4 s → 25.4 s → 41.9 s → 57.9 s over four 15 s windows — growing without bound, as an open loop must |

That last row is the one that matters. A closed-loop harness plateaus there.

### Metric definitions

| Metric | Definition | Why this one |
|---|---|---|
| TTFT | intended arrival → first content token | includes queueing; what a user perceives as "did it start" |
| ITL | gap between consecutive content tokens | streaming smoothness; p99 ITL exposes batch-step stalls |
| E2E | intended arrival → last token | the SLO variable |
| Throughput | completed requests/s | capacity |
| **Goodput** | requests/s completing **within** the SLO | the only number that matters under overload |
| Shed rate | fraction returned 503 | cost of admission control |

Goodput is the headline. A gateway serving 40 rps with a 12 s p99 against a 2 s
SLO has a goodput of nearly zero; one serving 22 rps within SLO and shedding
the rest has a goodput of 22. That framing is what makes admission control
obviously correct rather than obviously lossy — and it is the objective Week 4
optimises directly.

### The workload

70% of requests share one of four long, realistic system prompts (~480–650
tokens); 30% carry a unique system prompt of comparable length, so the only
difference between the two populations is *shareability*, not size. Output
lengths are lognormal(4.6, 0.8), clipped to [16, 512].

The heavy tail is the point. With fixed-length outputs every scheduling policy
looks identical, latency prediction is trivial, and there is nothing to build.
A lognormal tail creates head-of-line blocking, makes continuous batching pay
off visibly, and gives the Week 4 predictor something real to be uncertain
about. A `uniform` workload is included precisely to demonstrate that — see
[Negative results](#negative-results-and-limitations).

### The sweep grid and the SLO were measured, not guessed

The build guide's example sweep runs 1–16 rps. On this machine that grid would
consist entirely of points past collapse. `bench/calibrate.py` measures the
machine first and commits its output to `results/calibration.json`:

| Quantity | Measured |
|---|---|
| Prompt length | p50 541, p95 639, max 648 tokens |
| Requested output length | p50 100, mean 136, p95 380 tokens |
| Prefill | 1 641 tok/s |
| Decode, batch 1 | 81.5 tok/s (12.3 ms/step) |
| Decode, batch 4 | 205.3 tok/s aggregate (19.5 ms/step) |
| Decode, batch 8 | 205.8 tok/s aggregate (38.9 ms/step) |
| Decode, batch 16 | 314.6 tok/s aggregate (50.9 ms/step) |
| KV cache | 16 384 cells, unified across sequences |
| Implied FIFO capacity | ≈ 0.50 rps |
| Implied best-case batched capacity | ≈ 1.31 rps |
| Implied unloaded E2E | p50 1.56 s, p95 5.00 s |

Three things follow, and all are load-bearing for reading the results.

**The sweep runs 0.3–2.6 rps**, which brackets the knee of every rung: FIFO
saturates near 0.5, continuous batching near 1.3, and continuous batching with
a warm prefix cache near 2.2.

**The KV pool is 16 384 cells, not 8 192.** This was found the hard way. At
8 192, a batch of ten ~540-token prompts plus their generation occupies the
whole pool, so every admission evicted a cached prefix and the measured prefix
hit rate was 10% — a number that described the cache's eviction rate and
nothing else. The pool has to be large enough for the thing being measured to
exist. At 16 384 cells (192 MiB of KV, still half the model's 32 768-token
training context) the hit rate settles at ~0.61 against a theoretical ceiling
of ~0.64, and p50 end-to-end latency at 1.0 rps falls from 3.95 s to 1.17 s.

**The SLO is 4.0 s**, not the guide's 2.0 s. Implied unloaded end-to-end
latency is p50 1.56 s and p95 5.00 s, so a 2 s target would be unattainable for
most requests at *zero* load: it would measure the output-length distribution,
not the scheduler. Against the measured output-length distribution, 4.0 s is
attainable for roughly 90% of requests on an idle server and is comfortably
violated under overload — which is what a useful SLO looks like: it is the
*scheduler* that decides whether it is met, not the draw of the output length.
Because the choice is a choice, `bench/analyze.py --slo` recomputes goodput at
any target from the same raw records, and an SLO-sensitivity table is published
alongside the headline numbers.

### Guarding against thermal drift

Running every rate of rung 1, then every rate of rung 2, measures the later
rungs on a hotter laptop — and biases in the direction that flatters exactly
the configurations this project argues for. `bench/run_ladder.py` is therefore
**rate-major with rotated rung order**, and it restarts the gateway for every
(rung, rate) pair so no configuration inherits another's warm prefix cache.

---

## Results

*Generated by `bench/make_report.py` from the committed parquet in
`results/w2_ladder/`. Figures by `bench/charts.py`.*

<!-- RESULTS -->

---

## What is in the box

### The scheduler (`src/cadence/engine/scheduler/continuous.py`)

Every step is a decision. `_schedule()` is a pure function of scheduler state —
it allocates but never runs the model, which is what makes it testable alone.
It picks a prefill set and a decode set subject to three simultaneous limits:
free KV blocks, free backend sequence ids, and a per-step prefill token budget.

**Chunked prefill.** A 2 000-token prompt prefilled in one shot stalls every
decoding sequence for the length of that forward pass and shows up as a spike
in p99 ITL. Prefill is split into chunks of `max_prefill_tokens` and
interleaved with decode steps. A sequence part-way through its prompt is never
also placed in the decode set — feeding it a decode token would repeat a
position the KV cache already holds, and llama.cpp rejects the batch outright.

**Prefill/decode ordering** is a knob (`prefill_priority`), not a guess:
prioritising prefill costs ITL, prioritising decode costs TTFT, and both
settings are measured.

**Earliest-deadline-first** ordering of the wait queue, because `deadline` and
`slack` are two lines now and are exactly what Week 4's controller will reuse.

### Paged KV (`src/cadence/engine/kv/block_manager.py`)

Fixed 16-token blocks, a LIFO free list (recently freed blocks are the warm
ones), reference-counted sharing, and copy-on-write when a shared partially
filled block is about to be appended to.

Copy-on-write is where paged caches actually break, and the failure is silent:
one user's tokens appear in another user's stream. There is a test for exactly
that, against the real model — two requests with an identical long system
prompt and different user turns, asserted to produce byte-identical output to
running each alone.

The contiguous allocator it replaces is kept (`ContiguousBlockManager`) so the
ablation compares reservation *policies* rather than two different codebases:
paged reserves for the request's own budget and grows a block at a time;
contiguous reserves a slab sized for the server-wide output cap, wasting the
difference on every request that stops early.

### Radix prefix cache (`src/cadence/engine/kv/radix_cache.py`)

A radix tree over token runs. Matching walks the tree; a prompt that diverges
part-way through an edge splits that node so the shared head becomes reusable
by both. Three rules keep it correct, and each has a test:

1. **Match only at block boundaries.** A partially filled block cannot be
   shared — whichever sequence continues first writes its tail.
2. **Reference-count nodes, not just blocks.** A node with `refs > 0` is never
   evicted, or a running sequence loses its history mid-generation.
3. **Evict unreferenced leaves only, LRU.** Evicting an interior node orphans
   its children.

A fourth rule is specific to running on llama.cpp: blocks are an accounting
model, but the physical KV lives in a llama.cpp *sequence*. Every node names an
`owner_seq` whose KV holds exactly the tokens on its root path, a hit is
materialised with `llama_memory_seq_cp` (which adds the new sequence to the
existing cells rather than copying them), and owner sequences are
reference-counted and returned to the pool only when the last node naming them
is evicted. Sequence-id conservation is asserted in the test suite.

Hit rate is reported **token-level**, not request-level, because partial hits
are the normal case and a request-level rate would hide most of what the cache
does.

### The model runner (`src/cadence/engine/backends/`)

`ModelRunner` exposes step-level decoding: `decode_step` takes a list of
sequences and advances each by exactly one token. A backend that only offers
"generate until done for one prompt" makes continuous batching impossible — you
end up building static batching with extra steps.

The llama.cpp backend drives `llama_decode` directly with a hand-built
`llama_batch`, so it controls per-token positions, per-token sequence ids and
which rows produce logits. The context is created with `kv_unified = True`, so
`n_ctx` is one shared pool of 8 192 cells competed for by every request in
flight — which makes the block manager an honest model of the real resource
rather than a simulation next to it.

A deterministic `MockRunner` mirrors the same interface with an explicit cost
model. It produces no number in this README; it exists so that the scheduler,
the KV bookkeeping, the streaming path and the load generator are all enforced
in CI without a 500 MB GGUF or a GPU.

### Observability (`src/cadence/obs/`)

Prometheus histograms with buckets **dense around the SLO** — the default
buckets interpolate a p99 that is wrong by exactly the amount that matters —
plus gauges for queue depth, batch size, free KV blocks and fragmentation
ratio, and counters for preemptions, copy-on-write, shed reasons and
token-level prefix hits. OpenTelemetry spans per phase (queue → prefill →
decode), off by default because the exporter's own overhead is visible in ITL
at this scale.

The Grafana dashboard is **generated** by `deploy/grafana/make_dashboard.py`,
which validates every PromQL query against the collectors actually registered
in `cadence.obs.metrics`. A renamed metric fails CI instead of silently
emptying a panel three weeks later.

---

## Tests

```bash
uv run pytest -q -m "not slow"   # no model needed; this is what CI runs
uv run pytest -q -m slow         # against the real GGUF
```

The tests worth knowing about:

| Test | What it protects |
|---|---|
| `test_backend_equivalence` | Two sequences decoded in one batch produce what they produce decoded apart (≥99% top-token agreement — batching genuinely changes numerics, so bit-exactness is the wrong assertion). Chunked prefill likewise. |
| `test_kv_isolation` | Cross-request contamination, against the real model. Prefix cache on vs off, and paged vs contiguous, must produce identical text. |
| `test_loadgen` | The harness itself: exponential arrivals, latency from intended arrival, arrival rate unaffected by a slow server, warm-up rows flagged rather than dropped. |
| `test_block_manager` | Hypothesis property: a block is on the free list iff its refcount is zero, never twice, and nothing leaks. Plus copy-on-write. |
| `test_radix_cache` | Block-boundary truncation, referenced nodes never evicted, LRU over leaves, owner-sequence release, no block leak over insert/evict cycles. |
| `test_scheduler` | A late arrival joins the running batch (and, under static batching, provably cannot). KV blocks and sequence ids return to baseline after a run and after a mid-stream client disconnect. |
| `test_api_sse` | An unmodified `openai` Python client streams against the server. Concurrent streams carry only their own tokens. Dashboard queries reference metrics that exist. |

---

## Negative results and limitations

Reported because a ladder where every rung improves every metric is not
credible.

<!-- NEGATIVE -->

**Stated omissions.** Swapping preempted KV to host memory is the standard
alternative to recompute; it trades memory bandwidth for wasted prefill compute
and is not built — recompute is 20 lines and the prompts here are short enough
that it is the right trade. Prefix-cache entries are inserted when a request
*completes*, not when its prefill finishes, so a cold-start burst of
simultaneous identical prompts shares nothing; inserting at prefill completion
would capture that, at the cost of having to invalidate cache entries when a
sequence is preempted.

---

## Roadmap

| Week | Ships | Status |
|---|---|---|
| 0 | Toolchain, repository scaffold | Done |
| 1 | SSE gateway, FIFO baseline, open-loop load generator, metric definitions | Done |
| 2 | Continuous batching, paged KV, radix prefix cache, metrics + tracing stack | Done |
| 3 | Block allocator and radix match in C++17 behind pybind11, fuzz-tested against the Python reference | Not started |
| 4 | Latency predictor + split-conformal admission control | Not started |
| 5 | Full ablation ladder with seeds, CI load gate, deploy, writeup | Partial (ladder and charts exist; seeds and gate are Week 5) |

Week 3 will begin with `py-spy record` under load, and the port happens only if
the profile says the allocator and radix match are on the hot path — and if it
says otherwise, that gets written down too.

---

## Repository layout

```
src/cadence/
  api/          FastAPI app, OpenAI routes, SSE framing
  engine/
    engine.py           owns backend + scheduler + admission policy
    request.py          lifecycle state machine, cross-thread streaming
    scheduler/          fifo.py, static_batch.py, continuous.py
    kv/                 block_manager.py, radix_cache.py  (Python reference)
    backends/           base.py protocol, llamacpp.py, mock.py
  admission/    Week 4: features, predictor, conformal, controller
  obs/          Prometheus collectors, OpenTelemetry setup
src/cpp/        Week 3: block_allocator, radix_cache, pybind11 bindings
bench/          calibrate, loadgen, workloads, run_sweep, run_ladder, analyze,
                charts, make_report, stub_server
deploy/         docker-compose, Prometheus, OTel collector, generated Grafana dashboard
results/        committed parquet + calibration; every number here comes from these
tests/
```
