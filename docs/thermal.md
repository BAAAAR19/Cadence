| Measured before | probes | decode (tok/s), mean | min | max |
|:--|--:|--:|--:|--:|
| 1  FIFO, no batching | 18 | 87.4 | 83.8 | 94.7 |
| 2  static batching (8) | 18 | 83.9 | 81.4 | 90.4 |
| 3  continuous batching | 18 | 85.7 | 82.7 | 88.3 |
| 4  + paged KV + prefix cache | 36 | 87.6 | 83.6 | 103.9 |
| 5  + conformal admission (80%) | 18 | 88.5 | 81.1 | 94.4 |
| 5  + conformal admission (95%) | 17 | 89.0 | 82.6 | 93.4 |
| **whole sweep** | 125 | **87.1** | 81.1 | 103.9 |

A fixed 64-token single-stream generation, greedy, on an idle engine, timed immediately before each of the 125 measured runs (`bench/run_sweep.py:canary`). It is this project's substitute for `powermetrics`, which needs root: the number is a direct reading of how fast the machine was at that moment.

Across the whole sweep the probe varied by 21.9%. What matters for the ladder is not that spread but whether it fell *unevenly* on the rungs, and the per-rung means differ by 5.7% -- the rate-major, rung-rotated order is what keeps that small, and this table is how the claim is checked rather than asserted.

19 of the 144 probes returned nothing, all of them on the admission rungs (5  + conformal admission (99%): 18, 5  + conformal admission (95%): 1). The probe goes through the same door as every other request, so a controller that is shedding sheds it too. That is a defect in the instrument rather than in the engine -- the probe should bypass admission, and does not -- and it is left as it ran: the rungs whose thermal coverage is thinner are named here rather than quietly averaged in.
