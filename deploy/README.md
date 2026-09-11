# Deploying Cadence

Three ways to run it, and one thing you have to know before turning on
admission control anywhere.

| | What you get | What it costs |
|:--|:--|:--|
| `docker compose up` | gateway + Prometheus + Grafana with the dashboard provisioned | one machine, no URL |
| `fly deploy` | one public URL, CPU-only | a Fly.io account |
| `uv run cadence-serve` | the fastest path, Metal on Apple silicon | your laptop |

Everything below assumes you are in the repository root.

---

## docker compose

```bash
docker compose -f deploy/docker-compose.yml up --build
```

The gateway is on `:8000`, Prometheus on `:9090`, Grafana on `:3000` with the
Cadence dashboard already provisioned and anonymous login on. The first build
compiles `llama-cpp-python` and the C++17 KV core, which takes a few minutes;
after that it is cached.

`./models` is mounted over the image's baked copy, so if you already have the
GGUF the build does not download a second one — and a locally refitted
admission artifact reaches the container the same way.

Compose's health check points at **`/ready`**, not `/health`: the container is
alive within a second and is not ready to serve for the thirty or so seconds
the model takes to load. See below.

**The container is slower than the host, and by a lot.** There is no Metal
inside Docker on macOS, so llama.cpp runs on CPU. None of the numbers in the
README were measured this way; they come from the host, and the writeup says
so.

---

## Fly.io

```bash
fly launch --no-deploy --copy-config --config deploy/fly.toml
fly deploy --config deploy/fly.toml --ha=false
```

`deploy/fly.toml` is commented line by line. The parts that are decisions
rather than boilerplate:

- **The weights are baked into the image**, fetched once at build time and
  verified against a pinned SHA-256. A model pulled on boot is a cold start
  that scales with the network, and a cold start is a latency outlier in
  exactly the tail this project is about.
- **`auto_stop_machines = false`.** A gateway that scales to zero pays a model
  load on the first request after every idle period. If the demo has to be
  cheap, stop the machine; do not let it flap.
- **Two checks, two endpoints.** The readiness check drives routing and points
  at `/ready`. The liveness check restarts a wedged process and points at
  `/health`, which deliberately stays 200 while draining.
- **`kill_timeout = 95s`**, one step above `CADENCE_DRAIN_GRACE_S = 90`, so a
  SIGKILL means a bug rather than a race.
- **`CADENCE_ADMISSION = none`.** Read the next section before changing it.

---

## Calibrating admission for a deployment

**The committed predictor is not valid on your hardware.**

`models/admission.pkl` was fitted on llama.cpp with Metal on an M-series
laptop. A split-conformal bound guarantees coverage on data *exchangeable with
its calibration set* and says nothing at all otherwise. Move it to four shared
vCPUs, where the same request takes several times longer, and it is not a
conservative bound — it is an arbitrary number. The observed behaviour is
unambiguous once you know to look for it: the gateway refuses essentially
every request at zero offered load, which looks exactly like correct overload
behaviour.

The gateway will tell you. Every artifact records the backend it was
calibrated on, and `cadence.admission.conformal_controller` compares that with
the running configuration at start-up:

```
warning: admission model was calibrated on backend 'llamacpp' and is being
served on 'mock'. A conformal bound is only valid on data exchangeable with
its calibration set; on a different backend it is not conservative, it is
arbitrary, and the usual symptom is that every request is shed at zero load.
```

The same fingerprint is on `/stats`, under `controller.calibrated_on` and
`controller.domain_warning`.

To enable admission control on a deployment, calibrate it there:

```bash
# 1. Find where this machine saturates. Everything downstream depends on it.
uv run bench/calibrate.py --out results/calibration.json

# 2. Collect traces with admission OFF, so the training set is a sample of the
#    latency distribution and not of what a previous controller allowed.
#    Rates should span roughly 0.5x to 1.5x the capacity step 1 implies.
uv run bench/collect_traces.py --config continuous+cache \
    --rates <your rates> --rounds 4 --duration 75 --slo 4.0 \
    --outdir results/deploy_traces

# 3. Fit and calibrate. `--split round` puts whole collection rounds into
#    whole folds, which is the exchangeability this trace actually has.
uv run bench/fit_predictor.py --traces results/deploy_traces \
    --alpha 0.01 --slo 4.0 --split round \
    --out models/admission-deploy.pkl --outdir results/deploy_fit

# 4. Point the gateway at it.
CADENCE_ADMISSION=conformal CADENCE_ADMISSION_MODEL=models/admission-deploy.pkl
```

Then check `results/deploy_fit/fit.json` for the held-out coverage before
believing anything. This is the same pipeline the CI load gate uses to produce
`models/admission-ci-mock.pkl` for the mock backend — see `bench/ci_baseline.sh`.

---

## The endpoints

| Path | Question it answers | While draining |
|:--|:--|:--|
| `/health` | is this process alive? | **200** |
| `/ready` | should traffic be sent here? | **503** |
| `/stats` | what is the engine doing? | 200 |
| `/metrics` | Prometheus | 200 |
| `/v1/models`, `/v1/chat/completions` | OpenAI | 503 with `Retry-After` |

Collapsing liveness and readiness is the single most common way a deployment
turns a rolling restart into an SLO violation, so they are separate here and
they disagree on purpose. The reasoning is in `src/cadence/api/lifecycle.py`.

### Shutdown

On SIGTERM, in this order:

1. `/ready` goes 503 immediately, so the load balancer removes the instance
   within one check interval (5 s on Fly);
2. new requests are refused with a 503 and a `Retry-After`;
3. requests already streaming run to completion, up to
   `CADENCE_DRAIN_GRACE_S`;
4. the scheduler stops and the backend is released.

Not a claim — a test. `bench/demo_drain.py` drives all four steps against a
real server on a real socket and exits non-zero if any of them fails,
including if every stream happened to finish before the signal landed (which
would mean nothing was drained):

```bash
uv run bench/demo_drain.py                       # mock backend, seconds
uv run bench/demo_drain.py --backend llamacpp    # the real thing
```

`tests/test_lifecycle.py` asserts the same behaviour at the ASGI level on
every CI run.

### The concurrency cap

`CADENCE_MAX_CONCURRENT_REQUESTS` bounds requests in the API layer, refusing
past it with `cadence_shed_total{reason="capacity"}`. It is a **memory bound,
not backpressure**: every accepted request holds a tokenised prompt and a
queue entry before it holds any KV, so a large enough burst could exhaust
memory while the scheduler is perfectly healthy — and an OOM kill invalidates
every SLO claim the project makes.

It is deliberately set far above anything the scheduler will admit. If this is
what is shedding, the conformal controller has already failed, and the
counters say which one fired.
