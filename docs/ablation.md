| Config | Isolates | Peak goodput (rps, SLO 4s) | at offered (rps) | p50 TTFT (s) | p99 TTFT (s) | p99 E2E at 2.6 rps (s) | Prefix hit rate |
|:--|:--|--:|--:|--:|--:|--:|--:|
| 1  FIFO, no batching | the baseline everything is measured against | 0.44 ± 0.10 | 0.6 | 2.41 | 7.44 ± 4.15 | 300.1 ± 3.2 | n/a |
| 2  static batching (8) | the cost of head-of-line blocking | 0.55 ± 0.02 | 0.6 | 1.23 | 5.04 ± 3.59 | 217.8 ± 48.2 | n/a |
| 3  continuous batching | iteration-level scheduling | 0.81 ± 0.06 | 1.0 | 0.42 | 1.41 ± 0.78 | 187.1 ± 26.8 | n/a |
| 4  + paged KV + prefix cache | memory efficiency and prompt reuse | 1.23 ± 0.06 | 1.4 | 0.10 | 0.85 ± 0.10 | 42.3 ± 8.1 | 64% |
| 5  + conformal admission (80%) | tail-latency control under overload | 2.02 ± 0.12 | 4.0 | 0.07 | 0.81 ± 0.18 | 3.8 ± 0.8 | 72% |
| 5  + conformal admission (95%) | the same control, promised harder | 1.62 ± 0.06 | 4.0 | 0.06 | 0.65 ± 0.05 | 2.0 ± 0.1 | 74% |
| 5  + conformal admission (99%) | the same control, promised harder still | 0.00 | — | — | — | — | — |

`±` is the min-max range over three seeds. Peak goodput is the maximum over the offered-load grid, and the column beside it is the load at which that maximum occurred. The last latency column is read at 2.6 rps -- about twice the 1.4 rps at which rung 4's goodput peaks -- so every rung is compared at the same offered load, well past saturation for all five.
