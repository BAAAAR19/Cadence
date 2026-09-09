# Cadence

An LLM inference gateway with SLO-aware scheduling: continuous batching, a
paged KV cache and a radix prefix cache in front of a small local model, built
so that a tail-latency target can be *measured* rather than hoped for.

> **Status: Weeks 0–3 of five are complete.** Shipped: the OpenAI-compatible
> streaming gateway, the open-loop measurement harness, the systems core
> (iteration-level scheduling, paged KV, radix prefix cache, metrics and
> tracing), and the C++17 port of the two KV data structures behind pybind11,
> fuzz-tested against the Python reference. Not yet built: the conformal
> admission controller (Week 4) that turns the p99 target into a
> distribution-free guarantee. It is stubbed with the interface it will fill,
> and its test skips rather than passing vacuously. See [Roadmap](#roadmap).
>
> Week 3's headline is a negative result, reported as one: the profile says
> the two structures are **0.02% of scheduler-thread time**, the port is a
> 2.1× microbenchmark win and a 0% end-to-end win, and what it actually bought
> was a latent admission bug that had been crashing nothing only because no
> measured configuration reached the pressure it needs. See
> [Week 3](#week-3-the-c17-core-and-what-it-was-actually-worth).

---

## The three claims, and where each is defended

| Claim | Evidence | Status |
|---|---|---|
| **Iteration-level scheduling and prompt reuse move the collapse point 3.2× further out.** | Goodput at a 4 s SLO rises 0.37 → 1.19 rps across the four-rung ladder, on one interleaved sweep with a fixed arrival sequence. | **Measured** |
| **The measurements are sound.** | Open-loop Poisson generator built before the scheduler, latency timestamped from *intended* arrival, arrival process KS-validated over 200 seeds, goodput reported alongside throughput — with the finding that throughput cannot distinguish the four rungs at all. | **Measured** |
| **My scheduler holds p99 under overload.** | Not yet true, and the ladder shows why: past its own knee every rung degrades, because none of them refuses work. This is what Week 4's conformal admission controller is for, and the harness above is what will decide whether it worked. | Week 4 |
| **My C++ is load-bearing, not decorative.** | It is not load-bearing, and that is the measured answer rather than the hoped-for one: the profile puts the allocator and the radix cache at 0.02% of scheduler-thread time, the port is 2.1× on the microbenchmark and 0% end to end, and the arithmetic said so before the run did. The port earned its place on correctness instead — it surfaced a use-after-free in the Python version, and it is held to the reference by a differential fuzz over 5 000 random operation sequences. | **Measured** |

The third row is the one worth reading. The project's headline claim is a Week 4
claim; Weeks 0–3 build the baseline it has to beat and the instrument that can
tell whether it did. The fourth row is the one most likely to be overclaimed in
a portfolio, so it is stated the way the measurement came out.

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
  C++17 via pybind11       C++17 via pybind11          llama_decode
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
uv sync --all-groups          # also builds the C++17 KV core; needs CMake >= 3.26
mkdir -p models && hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
    qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir models
uv run cadence-serve
```

There is no separate build step and no committed binary: the project's build
backend is scikit-build-core, so installing it compiles `cadence._core`. If it
is not built, the gateway falls back to the Python reference and says so in
`/stats`; `CADENCE_KV_CORE=python|cpp` pins one explicitly, and asking for
`cpp` when it is missing is an error rather than a silent downgrade — a
benchmark that thinks it measured the extension and quietly measured Python is
worse than a benchmark that failed.

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

Reproduce Week 3's — profile, microbenchmark, end-to-end A/B, in that order,
because that is the order the argument has to be made in:

```bash
./bench/run_week3.sh
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

### What the source was, when

Every number in the ladder comes from one engine source revision, recorded as
`results/src.hash` and checked before the sweep started. Three defensive
changes landed afterwards, while analysing the results:

* `Settings` now refuses a prefill budget larger than `n_batch` at startup
  (it previously overran the pre-allocated `llama_batch` and failed every
  request with no indication why — found while setting up the chunked-prefill
  experiment);
* `prefill_chunk` caps the per-call token count at `n_batch` as a second line
  of defence;
* `BlockManager.can_append` now accounts for the free block a shared block's
  copy-on-write will need — found by the Hypothesis property test on a new
  draw.

All three are no-ops for the ladder's configuration, but "should be a no-op" is
not a measurement. Re-running `continuous+cache` at 1.0 rps on the changed
source reproduces the recorded point within run-to-run noise:

| | goodput | SLO met | TTFT p50 | ITL p99 | E2E p50 | E2E p99 | prefix hit |
|---|---|---|---|---|---|---|---|
| recorded | 1.098 | 0.890 | 0.067 | 0.336 | 1.223 | 8.313 | 0.621 |
| re-run | 1.078 | 0.874 | 0.071 | 0.361 | 1.317 | 8.465 | 0.626 |
| delta | −1.9% | −1.9% | +5.1% | +7.5% | +7.7% | +1.8% | +0.5% |

Week 3 changed the engine source again, in four places that could in principle
move a number:

* `ContinuousScheduler._pinned_match` holds a matched prefix for the length of
  the admission decision, fixing a crash under memory pressure (described in
  [Week 3](#week-3-the-c17-core-and-what-it-was-actually-worth));
* `RadixCache` uses a logical clock rather than `time.monotonic()` for LRU
  ordering — identical ordering, and the only version of the rule two
  implementations can be asked to agree on;
* `RadixCache.match` no longer slices the prompt on every node it visits, which
  removes the lookup's only super-linear term;
* the C++ core became the default when it is built.

The same protocol applies: the A/B's `kv-core-python` arm at 1.0 rps *is* the
re-verification, on source that differs from the ladder's in all four ways, and
it lands on the recorded point. `results/src.hash` is now produced by
`bench/srchash.py` and stamped into every run's `meta.json`, so this check is a
function rather than a note. The Week 2 value in that file was computed by
hand; the runs it labels are unchanged.

### One run was repeated, and it is marked as such

The `fifo` run at 2.6 rps completed but its parquet write failed: the `status`
column mixed integer HTTP codes with the string `'ReadTimeout'`, which only
happens at an overload severe enough for requests to hit the client's 300 s
timeout, and pyarrow refused the frame. The run was repeated twelve hours later
on byte-identical engine source and reproduced the lost one closely (p50
196.9 s vs 194.9 s, p99 301.5 s vs 301.3 s, 329 vs 337 completions). The merge
is recorded in `results/w2_ladder/meta.json`. The underlying bug is fixed —
`status` is now a nullable integer and transport failures go in a separate
`error` column — and a failed write no longer aborts the remaining rungs.

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

### The ablation ladder

Each rung adds exactly one thing to the rung above it, so each row is
attributable. Same workload, same arrival sequence, same seed, same duration;
rate-major with rotated rung order.

<!-- LADDER -->

| Config                       |   Peak goodput (rps, SLO 4s) |   at offered (rps) |   p50 TTFT (s) |   p99 TTFT (s) | p99 E2E past saturation (s)   |   p99 ITL (s) | Prefix hit rate   |
|:-----------------------------|-----------------------------:|-------------------:|---------------:|---------------:|:------------------------------|--------------:|:------------------|
| 1  FIFO, no batching         |                         0.37 |                0.6 |           3.23 |           9.88 | 155 @ 1.4 rps                 |         0.015 | n/a               |
| 2  Static batching (8)       |                         0.55 |                0.6 |           1.11 |           4.51 | 51 @ 1.4 rps                  |         0.031 | n/a               |
| 3  Continuous batching       |                         0.78 |                1   |           0.44 |           1.68 | 173 @ 2.6 rps                 |         0.378 | n/a               |
| 4  + paged KV + prefix cache |                         1.19 |                1.4 |           0.1  |           0.91 | 33 @ 2.6 rps                  |         0.366 | 65%               |

<!-- /LADDER -->

**Goodput at the SLO rises 3.2× from rung 1 to rung 4, and p50 TTFT falls 32×.**
Every rung earns its place: batching at all (rung 2) buys 1.5×, making that
batching iteration-level (rung 3) buys another 1.4×, and reusing the shared
system prompt (rung 4) buys a further 1.5×.

![Latency vs offered load](docs/figs/latency_vs_load.png)

The knee is where each rung's line leaves the floor: FIFO at ~0.5 rps, static
at ~0.8, continuous at ~1.0, and continuous with caches at ~1.9. Past the knee
every configuration degrades — none of them holds p99 flat, because none of
them yet *refuses* work. That is precisely the gap Week 4's admission
controller exists to close, and it is why this chart is the "before" picture
rather than the result.

![Throughput, SLO attainment and goodput](docs/figs/goodput_vs_load.png)

The left panel is the trap. With nothing shed and a client that waits up to
300 s, completed requests per second is identical across all four rungs to
within 1% — and so is output token throughput (26.3, 41.5, 73.4, 111.0, 132.5,
170.8 tok/s at the six offered loads, varying by under 1.5% between rungs). A
scheduler with no admission control decides *when* work completes, not whether.
Read alone, "throughput" would say these four configurations are the same
system. Goodput says one of them serves 3.2× as many users inside the SLO.

#### The Week 1 exit number: what FIFO actually saturates at

*(The build guide asks for a separate `results/w1_fifo.parquet` here. The FIFO
baseline is `results/w2_ladder/fifo.parquet` instead — a second copy of the
same rows is two things that can drift, and the ladder file is the one the
charts and tables are generated from.)*

Measured from the sustained completion rate in the middle 60% of the three
overloaded FIFO runs: **0.87–0.92 requests/s**, at a mean output of 63–66
tokens, i.e. **55–61 output tokens/s**.

The request-rate figure is the less trustworthy of the two, and it is worth
saying why. The workload's mean requested output is 136 tokens, but the
requests FIFO *completes* under overload average 63 — long generations take
longer, so within the client's 300 s timeout the short ones finish and the long
ones do not. "Requests per second" is therefore inflated by survivorship. Token
throughput is not, and 57 tok/s matches the calibrated single-stream figure
almost exactly (0.33 s of prefill plus 63 tokens at 81.5 tok/s is 1.10 s per
request, or 0.91 rps). Two numbers that should agree, agreeing, for a reason
that can be stated.

### The full sweep

<!-- SWEEP -->

| Config           |   Offered (rps) |   Requests |   Throughput |   Goodput |   SLO met |   TTFT p50 |   TTFT p99 |   ITL p50 |   ITL p99 |   E2E p50 |   E2E p99 |
|:-----------------|----------------:|-----------:|-------------:|----------:|----------:|-----------:|-----------:|----------:|----------:|----------:|----------:|
| fifo             |             0.3 |         57 |        0.392 |     0.351 |     0.895 |      0.575 |      4.667 |     0.012 |     0.015 |     1.715 |     5.183 |
| fifo             |             0.6 |        105 |        0.757 |     0.375 |     0.495 |      3.229 |      9.879 |     0.013 |     0.015 |     4.097 |    10.137 |
| fifo             |             1   |        182 |        1.234 |     0     |     0     |     33.276 |     49.585 |     0.013 |     0.015 |    33.593 |    50.668 |
| fifo             |             1.4 |        247 |        1.67  |     0     |     0     |     77.617 |    153.923 |     0.013 |     0.015 |    78.816 |   154.639 |
| fifo             |             1.9 |        311 |        2.079 |     0     |     0     |    133.941 |    240.352 |     0.013 |     0.015 |   135.411 |   241.048 |
| fifo             |             2.6 |        412 |        2.202 |     0     |     0     |    194.802 |    299.963 |     0.013 |     0.015 |   196.861 |   301.513 |
| static           |             0.3 |         57 |        0.392 |     0.344 |     0.877 |      0.442 |      3.713 |     0.013 |     0.02  |     1.959 |     5.354 |
| static           |             0.6 |        105 |        0.757 |     0.548 |     0.724 |      1.109 |      4.505 |     0.013 |     0.031 |     2.242 |     7.315 |
| static           |             1   |        182 |        1.234 |     0.434 |     0.352 |      2.798 |     10.404 |     0.017 |     0.042 |     4.894 |    11.941 |
| static           |             1.4 |        247 |        1.67  |     0     |     0     |     21.966 |     47.044 |     0.024 |     0.043 |    24.597 |    50.989 |
| static           |             1.9 |        311 |        2.079 |     0     |     0     |     63.776 |    112.019 |     0.025 |     0.043 |    68.498 |   114.001 |
| static           |             2.6 |        412 |        2.757 |     0     |     0     |    129.126 |    194.54  |     0.024 |     0.043 |   131.96  |   196.672 |
| continuous       |             0.3 |         57 |        0.392 |     0.364 |     0.93  |      0.35  |      0.613 |     0.015 |     0.303 |     1.239 |     4.868 |
| continuous       |             0.6 |        105 |        0.757 |     0.649 |     0.857 |      0.365 |      0.907 |     0.016 |     0.318 |     1.328 |     6.227 |
| continuous       |             1   |        182 |        1.234 |     0.78  |     0.632 |      0.441 |      1.682 |     0.022 |     0.378 |     2.95  |    15.581 |
| continuous       |             1.4 |        247 |        1.67  |     0.027 |     0.016 |      7.07  |     20.691 |     0.07  |     0.481 |    14.843 |    48.622 |
| continuous       |             1.9 |        311 |        2.079 |     0     |     0     |     38.466 |     75.989 |     0.069 |     0.49  |    47.613 |    98.39  |
| continuous       |             2.6 |        412 |        2.757 |     0     |     0     |     95.323 |    155.992 |     0.069 |     0.494 |   104.845 |   172.648 |
| continuous+cache |             0.3 |         57 |        0.392 |     0.371 |     0.947 |      0.067 |      0.445 |     0.015 |     0.057 |     0.966 |     4.573 |
| continuous+cache |             0.6 |        105 |        0.757 |     0.721 |     0.952 |      0.066 |      0.59  |     0.017 |     0.071 |     0.844 |     4.378 |
| continuous+cache |             1   |        182 |        1.234 |     1.098 |     0.89  |      0.067 |      0.801 |     0.02  |     0.336 |     1.223 |     8.313 |
| continuous+cache |             1.4 |        247 |        1.67  |     1.19  |     0.713 |      0.099 |      0.91  |     0.038 |     0.366 |     2.017 |    14.405 |
| continuous+cache |             1.9 |        311 |        2.079 |     1.157 |     0.556 |      0.32  |      1.179 |     0.058 |     0.401 |     3.509 |    27.212 |
| continuous+cache |             2.6 |        412 |        2.757 |     1.01  |     0.367 |      0.525 |      3.942 |     0.072 |     0.435 |     5.464 |    32.592 |

<!-- /SWEEP -->

### The SLO is a choice, so here is what it changes

<!-- SLO -->

|   SLO (s) |   FIFO, no batching |   Static batching (8) |   Continuous batching |   + paged KV + prefix cache |
|----------:|--------------------:|----------------------:|----------------------:|----------------------------:|
|         2 |                0.22 |                  0.35 |                  0.52 |                        0.84 |
|         3 |                0.28 |                  0.48 |                  0.62 |                        1.07 |
|         4 |                0.37 |                  0.55 |                  0.78 |                        1.19 |
|         6 |                0.58 |                  0.81 |                  1.01 |                        1.45 |
|         8 |                0.66 |                  0.98 |                  1.11 |                        1.81 |

<!-- /SLO -->

The ordering of the four rungs is identical at every target from 2 s to 8 s,
and rung 4 leads rung 1 by between 2.5× and 3.8× throughout. The conclusion
does not depend on where the line was drawn.

### Where the time goes

![TTFT and ITL](docs/figs/ttft_and_itl.png)

![Latency over one run](docs/figs/latency_over_time.png)

The single run at 1.4 rps is worth reading closely. FIFO climbs linearly and
without bound — the queue never drains, which is the open-loop signature a
closed-loop harness cannot produce. Static batching draws visible sawtooth
waves: each ramp is one batch filling while the previous one finishes, and each
drop is a wave completing. Continuous batching scatters, because a request's
latency stops being a function of who it queued behind.

### Prefix cache

![Prefix cache hit rate](docs/figs/prefix_hit_rate.png)

65% of prompt tokens are served from the radix cache, against a workload where
70% of requests share a system prompt and each shared prompt is ~92% reusable
(the differing user turn is the remaining 8%) — a ceiling of about 64%, which
it reaches. The rate is token-level, not request-level, because partial hits
are the normal case.

### Methodology summary

<!-- METHOD -->

- Runs: 24 (4 configurations x 6 offered loads x 1 seed)
- Requests issued: 6,332; analysed in steady state: 5,256
- Per run: 180s of arrivals, first 25s and last 5s excluded from every quantile
- Inter-arrival KS test against Exp(lambda), one per distinct arrival realisation: 0/6 pass at alpha=0.05 (median p = 0.005). See `results/loadgen_validation.json` and the note below.
- Client-side scheduling slip (actual send minus intended arrival): p99 2.8 ms, max 87 ms

<!-- /METHOD -->

Every number above is regenerated from the committed parquet by
`bench/make_report.py` and substituted into this file by
`bench/embed_tables.py`; `--check` fails CI if the two ever disagree.

---

## Week 3: the C++17 core, and what it was actually worth

The rule this week opens with is that the port has to be motivated by a
measurement, and the measurement has to be reported whichever way it comes out.
It came out against the port, and that is the interesting part.

### The profile came first, and it says these structures are not hot

`py-spy` needs `task_for_pid` and therefore root on macOS, and running the
server under test as root changes the server under test. So the gateway samples
itself: `cadence/obs/profiler.py` is a sampling profiler on the same principle
— snapshot the interpreter's per-thread stacks at a fixed rate — from a thread
that needs no privileges. Two details make its output checkable rather than
impressionistic. Samples are weighted by the interval they actually cover, not
counted, because the sampler competes for the GIL and its wake-ups jitter; and
the realised rate and the number of missed wake-ups are written down next to
the profile rather than assumed — 59 277 samples at an achieved 381 Hz, none
missed, for the run below.

If you would rather have py-spy's own view of the same process, it will attach
to a running gateway with elevated permissions and should agree:

```bash
sudo py-spy record --pid $(pgrep -f cadence.api.app) --duration 60 \
     --threads -o /tmp/cadence.svg
```

`uv run bench/profile_core.py --rate 1.9 --duration 150` runs the gateway under
a real open-loop load with the sampler on, at the offered load nearest rung 4's
knee, and produces this:

<!-- CORE_PROFILE -->

| KV core          | Component             | Inclusive   | Self   |
|:-----------------|:----------------------|:------------|:-------|
| Python reference | scheduler             | 100.00%     | 0.00%  |
| Python reference | model runner          | 100.00%     | 92.57% |
| Python reference | tokenizer (llama.cpp) | 4.20%       | 4.20%  |
| Python reference | sampling (numpy)      | 3.23%       | 3.23%  |
| Python reference | kv: radix cache       | 0.02%       | 0.00%  |
| C++17 core       | scheduler             | 100.00%     | 0.00%  |
| C++17 core       | model runner          | 100.00%     | 92.73% |
| C++17 core       | tokenizer (llama.cpp) | 4.06%       | 4.06%  |
| C++17 core       | sampling (numpy)      | 3.21%       | 3.21%  |

* **Python reference**: 59,277 samples at 381 Hz over 155 s, of which the scheduler thread was busy 99%. 2,180 engine steps, mean 70.3 ms. 324 completed requests at 1.9 rps.
* **C++17 core**: 59,932 samples at 387 Hz over 155 s, of which the scheduler thread was busy 98%. 2,671 engine steps, mean 56.8 ms. 324 completed requests at 1.9 rps.

<!-- /CORE_PROFILE -->

![Scheduler thread, Python KV core](docs/figs/w3_flamegraph_python.svg)

*Rendered from the committed folded stacks by `bench/core_report.py`; the C++
arm is `docs/figs/w3_flamegraph_cpp.svg`. Open either directly for per-frame
tooltips.*

The block manager and the radix cache never appear as the innermost frame at
all. On the Python arm they are on the stack for **0.02%** of the scheduler
thread's busy time and account for none of it to two decimal places; on the
C++ arm they do not appear in 59 932 samples at all. The forward pass is
92.6%, and it is split across `_decode` and `_logits` because
`llama_get_logits_ith` forces the deferred `llama_synchronize` — the compute is
charged where it is waited for, not where it is launched. llama.cpp's own
tokenizer is 4.2% and greedy sampling through numpy is 3.2%.

One difference between the two arms is worth not over-reading: the C++ arm ran
**2 671 steps averaging 56.8 ms** where the Python arm ran **2 180 averaging
70.3 ms**. Same completions, same throughput, same offered load — the loop
divided identical work into more, shorter steps. It is visible in the engine's
own histogram and in nothing a client can see.

So the honest answer to "profile first, then port if the profile says they are
hot" is: **the profile says they are not hot.** They are three orders of
magnitude away from mattering.

### Ported anyway, for the reasons that survive that answer

The guide's own instruction for this outcome is to say which it was and port
them regardless, for the per-step allocation-free guarantee. That is a real
reason and it is not the one that turned out to matter. What the port bought
was **correctness the pure-Python version had been getting away with**, and an
interface written down in a form that can be checked.

**A latent crash, found by pointing a profiler at a configuration the ladder
never reached.** The first thing the profiling run did was kill the engine
thread inside a minute:

```
RuntimeError: cannot share free block 810
  continuous.py:_schedule -> _admit -> block_manager.py:share
```

Admission matches a cached prefix, discovers it is short of blocks, and calls
`_reclaim`, which evicts by LRU. Under enough pressure the node it evicts is
the one that was just matched — so `hit.block_ids` names blocks that are back
on the free list, and `share()` refuses them. In Python that is an exception
that takes down the engine thread. It is also the exact shape of a
use-after-free: had the port handed the scheduler a `Node*`, the same sequence
would have dereferenced freed memory instead of raising.

The fix is `_pinned_match`: a match is held for the length of the admission
decision, so the decision sees one consistent view of the cache from match to
admit. Re-matching after each eviction would have been the smaller change and
it is the wrong one — it leaves the same window open one line further down, and
it spends the cached prefix to buy a batch slot the request may not take. Two
regression tests cover it, and both fail without the pin.

That bug was reachable on the committed Week 2 source. It did not fire during
the Week 2 ladder because that configuration never reached the pressure it
needs; the mock backend at 4 rps reaches it in under a minute.

**The C++ makes the dangerous case loud rather than lethal.** Node identity
across the boundary is a generation-stamped handle looked up in a registry, not
a `Node*` in a Python object. A stale handle raises `radix node N has been
evicted` instead of dereferencing freed memory, and `NodeRef.alive` lets a test
assert the distinction. The Python `Node` keeps its object identity after
eviction and loses its parent; the differential test asserts that those two
states mean the same thing on every operation.

**The GIL was doing load-bearing work.** The build guide recommends
`py::call_guard<py::gil_scoped_release>` around `match`. Measured, a bound
no-op costs 48 ns and releasing and re-acquiring the GIL adds 35 ns to it —
a 73% overhead on the call frame, against a `match` whose own work is a
handful of hash lookups. That is a pessimisation, not an optimisation. The larger cost is correctness: holding the GIL for the whole of
every binding call is what makes each operation atomic against the `/stats`
endpoint, which reads `n_nodes()` from the API thread while the scheduler
thread is splitting nodes. Under Python that concurrency is a torn read;
with the GIL dropped in C++ it would be a data race over a tree another thread
is restructuring. So the port does not take the guide's advice, and the reason
is a number rather than a preference.

### The microbenchmark, and the thing it found

<!-- CORE_BENCH -->

**Longest-prefix match, by prompt length**

|   Prompt tokens |   Python (us) |   C++ (us) |   of which pybind11 marshalling | Speedup   |
|----------------:|--------------:|-----------:|--------------------------------:|:----------|
|             128 |          4.55 |       2.69 |                            2.04 | 1.7x      |
|             256 |          8.96 |       4.87 |                            4.52 | 1.8x      |
|             512 |         18.81 |       8.8  |                            9.16 | 2.1x      |
|            1024 |         39.48 |      19.1  |                           19.6  | 2.1x      |
|            2048 |         86.61 |      37.02 |                           33.79 | 2.3x      |

**Allocator operations**

| Operation                                |   Python (us) |   C++ (us) | Speedup   |
|:-----------------------------------------|--------------:|-----------:|:----------|
| can_append (once per sequence per token) |         0.246 |      0.16  | 1.5x      |
| alloc + release of a 34-block table      |         4.175 |      1.272 | 3.3x      |

**What that is as a share of one engine step** (mean step 70.3 ms, measured)

|   Running batch | Python   | C++     | Python, share of a step   | C++, share of a step   |
|----------------:|:---------|:--------|:--------------------------|:-----------------------|
|               8 | 22.7 us  | 11.4 us | 0.032%                    | 0.016%                 |
|              12 | 24.7 us  | 12.6 us | 0.035%                    | 0.018%                 |
|              24 | 30.6 us  | 16.5 us | 0.044%                    | 0.023%                 |

**The price of releasing the GIL, which is why the port does not**

| Measurement                                  |   ns per call |
|:---------------------------------------------|--------------:|
| a bound no-op, GIL held throughout           |            48 |
| the same no-op, GIL released and re-acquired |            83 |
| cost of the release/re-acquire pair          |            35 |

<!-- /CORE_BENCH -->

Two results worth more than the speedup column.

**Almost all of the C++ `match` call is the pybind11 boundary.** Timing a
function that does nothing but accept the same prompt list gives essentially
the whole cost of the real call: converting a 512-element Python list into a
`std::vector<int32_t>` is the work. The tree walk itself — a handful of hash
lookups over block-sized keys — is below the noise floor of the measurement.
Any further optimisation of this structure would have to change how the prompt
crosses the boundary, not what happens after it arrives.

**The Python reference was quietly quadratic, and fixing it was part of doing
this honestly.** `match` compared each node's edge against `token_ids[i:limit]`
— a fresh slice of up to the whole remaining prompt, on every node visited, so
a 512-token lookup copied ~30 slices to compare a few hundred integers.
Comparing in place removes the only super-linear term. It also cuts the
measured C++ advantage roughly in half, which is exactly why a comparison
against an unoptimised reference is not a comparison.

### End to end: the delta is nothing, and it had to be

Two rungs identical in every respect except `CADENCE_KV_CORE`, run through the
same rate-major, rotated-order runner the ablation ladder uses, so the
comparison is protected from thermal drift the same way:

<!-- CORE_AB -->

| KV core          |   Offered (rps) |   Throughput |   Goodput |   SLO met |   TTFT p50 |   ITL p99 |   E2E p50 |   E2E p99 |
|:-----------------|----------------:|-------------:|----------:|----------:|-----------:|----------:|----------:|----------:|
| C++17 core       |             1   |        1.234 |     1.085 |     0.879 |      0.069 |     0.352 |     1.279 |     8.655 |
| Python reference |             1   |        1.234 |     1.085 |     0.879 |      0.068 |     0.352 |     1.275 |     8.517 |
| C++17 core       |             1.4 |        1.67  |     1.19  |     0.713 |      0.095 |     0.384 |     2.143 |    14.894 |
| Python reference |             1.4 |        1.67  |     1.19  |     0.713 |      0.113 |     0.382 |     2.08  |    14.796 |
| C++17 core       |             1.9 |        2.079 |     1.023 |     0.492 |      0.436 |     0.438 |     4.037 |    30.751 |
| Python reference |             1.9 |        2.079 |     1.05  |     0.505 |      0.42  |     0.42  |     3.954 |    28.584 |

**C++ relative to Python**

|   Offered (rps) | Goodput   | Throughput   | TTFT p50   | ITL p99   | E2E p99   |
|----------------:|:----------|:-------------|:-----------|:----------|:----------|
|             1   | +0.0%     | +0.0%        | +1.7%      | -0.2%     | +1.6%     |
|             1.4 | +0.0%     | +0.0%        | -15.9%     | +0.5%     | +0.7%     |
|             1.9 | -2.5%     | +0.0%        | +3.8%      | +4.3%     | +7.6%     |

<!-- /CORE_AB -->

Throughput is identical to three decimal places at every rate. Goodput is
identical at 1.0 and 1.4 rps and 2.5% *lower* for the C++ arm at 1.9; p50 TTFT
is 16% lower for C++ at 1.4 rps and 4% higher at 1.9. The signs disagree across
rates and across metrics, which is what noise looks like — the Week 2
replicates put run-to-run spread at 1.7%, and 1.9 rps is past rung 4's knee
where the spread is larger still. There is no effect here to attribute.

The arithmetic said so before the run did. The KV bookkeeping is 23–31 µs per
step in Python and 11–17 µs in C++ across the plausible batch range, against a
**measured mean step of 70 ms**: 0.03% of a step becoming 0.02% of a step.
Nothing downstream of that can move a latency percentile.

That is the honest framing and it is the one worth having: **a 2.1× 
microbenchmark win is a 0% end-to-end win here, because the 0.5B model's
forward pass is roughly four orders of magnitude more expensive than the
bookkeeping around it.** The port would start to pay as that ratio closes —
bigger batches, a smaller or quantised-further model, faster hardware, or
speculative decoding where the scheduler runs several times per accepted token
— and that is a hypothesis this repository has not tested rather than a result
it has.

### Proving the port is correct, not just fast

Two independent lines of evidence, because "I rewrote it in C++" is a claim.

**The specification's own tests run against both.** Every hand-written test in
`tests/test_block_manager.py` and `tests/test_radix_cache.py` is parametrised
over the two implementations: thirteen cases in each file, run twice — 52
test executions in all. The Python module *is* the
specification, so correctness means passing the specification's tests, not
only agreeing with it on random inputs. Making that
possible is why the caches grew a `paths()` method: a test that asserted on
`root.children` could only ever run against one of them.

**Differential fuzz.** `tests/test_kv_parity.py` draws random operation
sequences — allocate, share, fork, append, insert, match, pin, evict — and
drives both implementations through them in lockstep, comparing every
observable after every step: matched token counts, hit rates, node liveness,
eviction counts, the owner sequences handed back to the scheduler, **and the
block ids themselves**. The guide suggests not comparing block ids because they
are an implementation detail; here they are not, because both allocators are
LIFO stacks driven by the same operation sequence, so an id is a function of
the input and asserting on it turns a class of bookkeeping bugs from
"eventually visible" into "visible on the operation that caused it".

600 sequences of 30–200 operations run on every PR, 5 000 on `main`. A
differential test that never reaches an interesting state passes for the wrong
reason, so each sequence reports what it exercised, and the distribution is in
the CI log rather than assumed:

| Reached at least once | Share of the 5 000 sequences |
|---|---:|
| a prefix hit | 40% |
| a copy-on-write | 65% |
| an eviction | 90% |
| an edge split | 28% |
| the block pool exhausted | 9% |

### What the two implementations agree on, in the type system

`cadence/engine/kv/protocols.py` is the interface the scheduler depends on, and
`build_kv` is annotated as returning it, so if either implementation drifts
from the other's surface mypy says so before a benchmark does. Writing it down
found the one place where the claim is false: a prefix-cache *node* is not
interchangeable — the Python cache hands out a `Node` and the C++ one a
`NodeRef`, and neither accepts the other's. The contract the scheduler actually
honours is narrower than a shared type would express ("hand the handle back to
the cache that gave it to you"), so the protocol says `Any` and says why. mypy
runs enforced rather than advisory from this week.

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

### The C++17 KV core (`src/cpp/`)

Two headers and a bindings file: `BlockAllocator` (a LIFO free-list stack, a
flat refcount vector, O(1) allocate/share/release/copy-on-write) and
`RadixCache` (block-keyed children, owner sequences, LRU eviction over
unreferenced leaves on a logical clock). Built on install by scikit-build-core;
selected by `CADENCE_KV_CORE`; held to the Python reference by
`tests/test_kv_parity.py`.

Three binding decisions carry the drop-in contract. Block tables stay Python
lists and are mutated in place, because the operations only ever read the
length and one element — converting the table on every decode step would cost
O(len) to save nothing. `OutOfBlocks` is translated back into the *Python*
exception class, because the scheduler catches it by identity and a same-named
C++ exception would sail through `except OutOfBlocks` and kill the engine
thread. And the GIL is held for the whole of every call, which is what makes
each operation atomic against the `/stats` reader on the API thread.

What it was worth is measured in [Week 3](#week-3-the-c17-core-and-what-it-was-actually-worth).

### The model runner (`src/cadence/engine/backends/`)

`ModelRunner` exposes step-level decoding: `decode_step` takes a list of
sequences and advances each by exactly one token. A backend that only offers
"generate until done for one prompt" makes continuous batching impossible — you
end up building static batching with extra steps.

The llama.cpp backend drives `llama_decode` directly with a hand-built
`llama_batch`, so it controls per-token positions, per-token sequence ids and
which rows produce logits. The context is created with `kv_unified = True`, so
`n_ctx` is one shared pool of 16 384 cells competed for by every request in
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

# the differential fuzz on its own, at the depth main runs it
CADENCE_FUZZ_EXAMPLES=5000 uv run pytest -q tests/test_kv_parity.py \
  --hypothesis-show-statistics
```

The KV tests are parametrised over both implementations of the core, so a
green suite means the C++ extension satisfies the Python reference's own
specification and not merely that the two agree on random inputs.

The tests worth knowing about:

| Test | What it protects |
|---|---|
| `test_backend_equivalence` | Two sequences decoded in one batch produce what they produce decoded apart (≥99% top-token agreement — batching genuinely changes numerics, so bit-exactness is the wrong assertion). Chunked prefill likewise. |
| `test_kv_isolation` | Cross-request contamination, against the real model. Prefix cache on vs off, and paged vs contiguous, must produce identical text. |
| `test_loadgen` | The harness itself: exponential arrivals, latency from intended arrival, arrival rate unaffected by a slow server, warm-up rows flagged rather than dropped. |
| `test_block_manager` | Hypothesis property: a block is on the free list iff its refcount is zero, never twice, and nothing leaks. Plus copy-on-write. **Runs against both the Python reference and the C++ core.** |
| `test_radix_cache` | Block-boundary truncation, referenced nodes never evicted, LRU over leaves, owner-sequence release, no block leak over insert/evict cycles. **Runs against both implementations.** |
| `test_kv_parity` | The differential fuzz: random operation sequences driven through the Python reference and the C++17 core in lockstep, comparing every observable — including block ids — after every step. 600 sequences per PR, 5 000 on `main`, with the coverage each sequence reached reported rather than assumed. |
| `test_scheduler` | A late arrival joins the running batch (and, under static batching, provably cannot). KV blocks and sequence ids return to baseline after a run and after a mid-stream client disconnect. A prefix matched during admission is not evicted out from under the admission decision — the Week 3 crash, with both failure modes covered. `CADENCE_KV_CORE=auto` degrades to the Python reference when the extension is missing, and an explicit `cpp` refuses to. |
| `test_api_sse` | An unmodified `openai` Python client streams against the server. Concurrent streams carry only their own tokens. Dashboard queries reference metrics that exist. |

---

## Negative results and limitations

Reported because a ladder where every rung improves every metric is not
credible.

**1. Continuous batching makes p99 inter-token latency 25× worse.** This is
the largest single regression in the ladder and it is a direct consequence of
the thing that makes rung 3 work. FIFO's p99 ITL is 15 ms and never moves;
continuous batching's is 378 ms at its peak-goodput load. Two causes, both
structural: a decode step now carries a whole batch rather than one sequence,
and prefill chunks are interleaved between decode steps, so a decoding
sequence's next token waits behind somebody else's prompt.

The prefix cache is the evidence for the second cause. Rung 4 does strictly
more work per request than rung 3 in every respect except prefill, and its p99
ITL at 0.3 rps is **57 ms against rung 3's 303 ms** — a 5× improvement bought
by not running prefill at all for cached prompts. The regression is prefill
interference, and the fix for it is to have less prefill to interfere with.

**2. Static batching is worse than no batching at low load.** At 0.3 rps rung 2
has a *higher* p50 end-to-end latency than rung 1 (1.96 s vs 1.72 s) and lower
SLO attainment (0.877 vs 0.895). Wait-to-fill is the cause: a request that
arrives into an empty server waits up to 50 ms for companions that never come.
Rung 2 only starts paying at 0.6 rps, where it overtakes FIFO on goodput by
1.5×. A batching scheme that is unconditionally better than not batching would
be a suspicious result, not a good one.

**3. Throughput does not distinguish the four configurations at all.**
Completed requests per second and output tokens per second are identical across
all four rungs to within 1.5% at every offered load. This is not a measurement
artifact — it is what an open loop with no admission control and a patient
client must produce. Every request eventually completes; the scheduler decides
when. Any comparison of these four systems on throughput would conclude,
correctly and uselessly, that they are the same.

**4. Goodput declines past its own peak, and nothing in Weeks 1–2 stops it.**
Rung 4 peaks at 1.19 rps of goodput at 1.4 rps offered, then falls to 1.16 at
1.9 and 1.01 at 2.6 — it does more work and delivers less of it on time. No
rung holds p99 flat past saturation. The project's headline claim is a Week 4
claim, and Weeks 1–2 do not support it yet; what they establish is the baseline
it will have to beat and the harness that can tell whether it did.

**5. The prefix cache's value is entirely contingent on KV headroom, and at the
first sizing it measured nothing.** At 8 192 KV cells the running batch alone
consumed the pool, every admission evicted a cached prefix, and the measured
hit rate was 10% — a number that described the cache's own eviction rate. The
mechanism was correct; the experiment was not. Doubling the pool to 16 384
cells moved the hit rate to 61% and p50 latency at 1.0 rps from 3.95 s to
1.17 s. A cache benchmarked in a configuration where it cannot retain anything
reports a fact about the configuration, not about the cache.

**6. Chunked prefill is the single biggest p99 win in the scheduler — and the
default it shipped with was the wrong size.** The chunk budget was swept at
1.4 rps offered load, on `continuous+cache`, with `n_batch` raised to 2048 for
every arm so that the scheduler's prefill budget is the only thing that varies:

<!-- KNOBS -->

**Chunked prefill** — one run each at 1.4 rps offered load.

| Arm                              |   Goodput |   SLO met |   TTFT p50 |   TTFT p99 |   ITL p50 |   ITL p99 |   E2E p50 |   E2E p99 |
|:---------------------------------|----------:|----------:|-----------:|-----------:|----------:|----------:|----------:|----------:|
| prefill budget 128 tokens        |     1.116 |     0.668 |      0.241 |      1.592 |     0.049 |     0.156 |     2.401 |    17.798 |
| prefill budget 256 tokens        |     0.987 |     0.591 |      0.274 |      2.548 |     0.057 |     0.276 |     3.387 |    26.992 |
| prefill budget 512 tokens        |     0.981 |     0.587 |      0.261 |      1.556 |     0.062 |     0.462 |     3.383 |    27.12  |
| unchunked (budget 2048 > prompt) |     0.981 |     0.587 |      0.36  |      1.113 |     0.049 |     0.503 |     3.249 |    18.543 |

**Prefill/decode order** — one run each at 1.4 rps offered load.

| Arm                   |   Goodput |   SLO met |   TTFT p50 |   TTFT p99 |   ITL p50 |   ITL p99 |   E2E p50 |   E2E p99 |
|:----------------------|----------:|----------:|-----------:|-----------:|----------:|----------:|----------:|----------:|
| prefill before decode |     1.19  |     0.713 |      0.106 |      0.902 |     0.038 |     0.368 |     2.025 |    14.478 |
| decode before prefill |     1.197 |     0.717 |      0.15  |      0.959 |     0.036 |     0.366 |     2.029 |    14.389 |

**Replicates of the unmodified configuration** — identical settings, 3 runs, 1.4 rps offered load. The spread here is the floor any knob difference has to clear.

| Arm         |   Goodput |   SLO met |   TTFT p50 |   TTFT p99 |   ITL p50 |   ITL p99 |   E2E p50 |   E2E p99 |
|:------------|----------:|----------:|-----------:|-----------:|----------:|----------:|----------:|----------:|
| replicate-1 |      1.19 |     0.713 |      0.107 |      0.906 |     0.037 |     0.366 |     2.028 |    14.422 |
| replicate-2 |      1.19 |     0.713 |      0.106 |      0.911 |     0.038 |     0.368 |     2.002 |    14.397 |
| replicate-3 |      1.19 |     0.713 |      0.105 |      0.898 |     0.038 |     0.366 |     2.035 |    14.453 |

<!-- /KNOBS -->

![Chunked prefill, ITL distribution and tail](docs/figs/itl_chunked_prefill.png)

The worst-case stall tracks the budget almost linearly — 128 tokens → 228 ms,
256 → 361 ms, 512 → 585 ms, unchunked → 958 ms — which is the mechanism made
visible: **the longest gap a decoding sequence can see is one prefill chunk's
forward pass.** p99 ITL improves 3.2× from unchunked to a 128-token budget
(503 ms → 156 ms) and p999 improves 4.5× (891 ms → 198 ms).

Three honest qualifications:

* The ladder ran with a 512-token budget, and this sweep says 128 would have
  been better on ITL *and* on goodput (1.116 vs 0.981). That is a tuning result
  found after the ladder rather than folded back into it, and it is reported
  here rather than retrofitted there.
* The first attempt at this experiment measured nothing. With ~540-token
  prompts, a 512-token budget splits a prompt into one full chunk and a
  28-token remainder — comparing that against "unchunked" compares two
  configurations that behave almost identically. The budget had to be swept
  down to where it actually bites.
* Raising `n_batch` from 512 to 2048, which unchunked prefill requires, costs
  about 6% of goodput on its own (1.190 → 1.116 at the best chunk size). Every
  arm above pays it, so the comparison between them is clean, but the absolute
  numbers are not comparable to the ladder's.

**7. The prefill/decode priority knob does nothing measurable here, and the
replicates are what make that statement possible.** Prioritising decode over
prefill costs 42% on p50 TTFT (0.106 s → 0.150 s) and buys nothing on ITL
(p99 0.366 s vs 0.368 s) or goodput (1.197 vs 1.190). Three replicates of the
unmodified configuration agree to within 1.7% on every metric — goodput
identical to three decimal places, p50 TTFT 0.105–0.107 s, p99 ITL
0.366–0.368 s — so the TTFT difference is real and outside the noise while the
ITL difference is not. The knob exists and is worth having; on *this* workload,
where prefill is small relative to decode, prefill-first is simply the right
default and the trade-off curve the guide expects to see is flat.

**8. The C++ port is a 2.1× microbenchmark win and a 0% end-to-end win.** The
profile put the two data structures at 0.02% of scheduler-thread time before
the port started, and the arithmetic — 23–31 µs of bookkeeping against a
measured 70 ms engine step — said the end-to-end delta had to be far smaller
than the 1.7% run-to-run spread. It is: throughput identical to three decimal
places at every rate, and goodput differences that change sign between rates.
The measured speedup is also capped by something other than the algorithm:
nearly all of the C++ `match` call is pybind11 converting the prompt list into
a `std::vector`, and the tree walk is below the noise floor. Reported in full in
[Week 3](#week-3-the-c17-core-and-what-it-was-actually-worth), because a port
justified after the fact by the speedup it did not deliver is the failure mode
this section exists to avoid.

**9. Two of the four correctness bugs found in this project so far were found
by instruments, not by tests.** Hypothesis found the copy-on-write accounting
error in `can_append`; the Week 3 profiling run found the use-after-free in
admission by crashing the engine thread within a minute of being pointed at a
configuration the ladder never reached. Both had been on `main`, both passed
the whole suite. The lesson recorded here is about coverage of *states*, not of
lines: the pressure regime that triggers a bug is a thing a test has to be told
to reach.

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
| 3 | Block allocator and radix match in C++17 behind pybind11, fuzz-tested against the Python reference | Done |
| 4 | Latency predictor + split-conformal admission control | Not started |
| 5 | Full ablation ladder with seeds, CI load gate, deploy, writeup | Partial — the four-rung ladder, the charts and the writeup exist; rung 5, multiple seeds and the CI load gate are Week 5 |

Week 3 began with a profile under load, the profile said the allocator and the
radix match are *not* on the hot path, and that is written down: see
[Week 3](#week-3-the-c17-core-and-what-it-was-actually-worth) for what was
ported anyway and why.

### What is deliberately not here yet

Against the project's own definition of done, these boxes are unticked and it
is worth being explicit about which:

| | |
|---|---|
| Rung 5 (conformal admission) | Week 4. Its absence is why "holds p99 under overload" is not claimed. |
| Three seeds per rung | The ladder is one seed. Run-to-run spread *is* quantified — three replicates of one configuration agree to within 1.7% on every metric — but that is not the same as three arrival realisations, and the Week 5 table will need the latter. |
| CI load-test regression gate | The CI runs correctness, lint and a check that the README's numbers match the committed parquet. It does not yet fail a PR on a goodput regression. |
| An end-to-end win from the C++ core | There isn't one, and the arithmetic says there could not be at this model size. The claim the port supports is correctness and a written-down interface, not speed. |
| Live deployment URL | `docker compose` is committed but unverified (no Docker on this machine — see above). |
| Empirical coverage plot | Week 4; there is no predictor to have coverage yet. |

---

## Repository layout

```
src/cadence/
  api/          FastAPI app, OpenAI routes, SSE framing
  engine/
    engine.py           owns backend + scheduler + admission policy
    request.py          lifecycle state machine, cross-thread streaming
    scheduler/          fifo.py, static_batch.py, continuous.py
    kv/                 block_manager.py, radix_cache.py  (Python reference),
                        protocols.py (the interface both cores satisfy),
                        cpp.py (the C++ core, assembled the same way)
    backends/           base.py protocol, llamacpp.py, mock.py
  admission/    Week 4: features, predictor, conformal, controller
  obs/          Prometheus collectors, OpenTelemetry setup, sampling profiler
  _core.pyi     hand-written stubs for the extension
src/cpp/        the C++17 KV core
  include/cadence/    block_allocator.hpp, radix_cache.hpp
  src/bindings.cpp    pybind11 module, built by scikit-build-core on install
bench/          calibrate, validate_loadgen, loadgen, workloads, stub_server,
                run_sweep, run_ladder, run_knobs, merge_rerun, srchash,
                run_ablation.sh, run_week3.sh,
                profile_core, flamegraph, bench_core,
                analyze, charts, make_report, knob_report, core_report,
                embed_tables
deploy/         docker-compose, Prometheus, OTel collector, generated Grafana dashboard
results/
  calibration.json        machine measurements the sweep grid and SLO were chosen from
  loadgen_validation.json arrival process over 200 seeds
  src.hash                engine source fingerprint (bench/srchash.py), stamped
                          into every run's meta.json from Week 3 on
  w2_ladder/              the four-rung ladder, plus meta.json provenance
  w2_knobs/               chunked-prefill sweep, prefill/decode order, replicates
  w3_profile/             sampling profile of the scheduler thread, per KV core
  w3_bench/               the C++/Python microbenchmark
  w3_ab/                  the end-to-end A/B between the two KV cores
  validation/             open-loop generator checked against a model-free stub
docs/           tables and figures, all generated from the parquet above
tests/
```
