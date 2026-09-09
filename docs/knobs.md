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
