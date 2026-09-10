# The five-minute demo

Start a load test, watch the dashboard, push past saturation, and watch p99
stay flat while the shed rate rises. Five minutes, on a laptop, with nothing
pre-baked.

The point of running it live rather than showing the charts is that the charts
are a claim about a system and this is the system. Everything below is the
committed configuration; nothing here is a demo mode.

---

## Before you start

```bash
uv sync --all-groups
hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
    qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir models
docker compose -f deploy/docker-compose.yml up -d prometheus grafana
```

Grafana is on <http://localhost:3000>, dashboard **Cadence**, already
provisioned. Set the time range to *Last 15 minutes* and the refresh to 5 s.

Two panels carry the whole demo: **p99 end-to-end** and **shed rate**. Keep
them side by side.

---

## 1. Start the gateway with admission control (30 s)

```bash
CADENCE_CONFIG_NAME=demo \
CADENCE_SCHEDULER=continuous \
CADENCE_ENABLE_PAGED_KV=true CADENCE_ENABLE_PREFIX_CACHE=true \
CADENCE_ADMISSION=conformal CADENCE_ADMISSION_ALPHA=0.2 \
CADENCE_SLO_S=4.0 CADENCE_N_CTX=16384 CADENCE_MAX_BATCH=24 \
uv run cadence-serve
```

While the model loads, in another terminal:

```bash
curl -s localhost:8000/ready | jq .status    # "loading", then "ready"
```

**Say this while it loads:** liveness and readiness are different questions.
`/health` is 200 within a second; `/ready` stays 503 until the engine has
completed a warm-up generation. A load balancer that polls the first one
routes traffic into a cold start and measures it as a latency regression.

---

## 2. Underload: the controller is invisible (60 s)

```bash
uv run bench/loadgen.py --rate 1.0 --duration 60 --slo 4.0 \
    --config-name demo --workload mixed
```

On the dashboard: p99 settles under the SLO line, shed rate is at or near
zero, batch size sits in the single digits.

**Say this:** at 1 rps this configuration is inside its budget, so a
controller that is working correctly does nothing at all. The interesting
behaviour is the behaviour under load it cannot serve.

---

## 3. Push past saturation (90 s)

Goodput for this configuration peaks near **1.4 rps**. Go to three times it:

```bash
uv run bench/loadgen.py --rate 4.0 --duration 90 --slo 4.0 \
    --config-name demo --workload mixed
```

Watch the two panels together. The shed rate climbs; p99 does not.

**Say this:** the shed rate rising is the p99 staying flat. Those are the same
event seen from two sides. The controller is predicting, per request and
before admitting it, an upper bound on that request's end-to-end latency, and
refusing the ones whose bound does not fit inside the budget. Under overload,
throughput and goodput point in opposite directions — serving 4 rps that all
miss the SLO is worth less than serving 2 rps that all meet it.

Then show what a refusal actually says:

```bash
curl -s -i -X POST localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen","stream":false,"max_tokens":128,
       "messages":[{"role":"user","content":"hello"}]}' | head -3
```

`503`, and a `Retry-After` computed from the queue's remaining decode work
divided by the measured token rate — an estimate of when capacity will exist,
not a constant wearing a header's clothes.

---

## 4. The comparison, if there is time (90 s)

Stop the gateway, restart it with `CADENCE_ADMISSION=none`, and repeat step 3.
Same load, same workload, same seed. p99 goes to tens of seconds and keeps
climbing for as long as the load runs, because nothing is refused and the
queue only grows.

That pair of runs is the whole project in three minutes.

---

## 5. The drain (30 s)

With a load test running, press `Ctrl-C` on the gateway once, and watch:

```bash
watch -n1 'curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/ready'
```

`/ready` flips to 503 immediately; the requests already streaming finish;
new arrivals get a 503 with a `Retry-After`; then the process exits.

Or run the scripted version, which asserts all of it:

```bash
uv run bench/demo_drain.py --backend llamacpp
```

---

## If something goes wrong

**Everything is shed, at every load, including an idle engine.** The
predictor is being used on a machine or backend it was not calibrated on.
Check `curl -s localhost:8000/stats | jq .controller.domain_warning`. A
conformal bound is only valid on data exchangeable with its calibration set;
see [`deploy/README.md`](../deploy/README.md).

**Grafana's p99 disagrees with the parquet.** It will, slightly, and that is
expected: Prometheus interpolates within a histogram bucket. Every number in
the writeup is computed from the load generator's raw records. The dashboard
is for watching, not for reporting.

**p99 is worse than the README's numbers.** Check what else is running. The
figures come from a quiet, plugged-in machine with the sweep's rate-major
rotation and a thermal probe before every run; a laptop with a browser and a
build going is a different computer.
