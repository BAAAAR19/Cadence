| Config                       |   Peak goodput (rps, SLO 4s) |   at offered (rps) |   p50 TTFT (s) |   p99 TTFT (s) | p99 E2E past saturation (s)   |   p99 ITL (s) | Prefix hit rate   |
|:-----------------------------|-----------------------------:|-------------------:|---------------:|---------------:|:------------------------------|--------------:|:------------------|
| 1  FIFO, no batching         |                         0.37 |                0.6 |           3.23 |           9.88 | 155 @ 1.4 rps                 |         0.015 | n/a               |
| 2  Static batching (8)       |                         0.55 |                0.6 |           1.11 |           4.51 | 51 @ 1.4 rps                  |         0.031 | n/a               |
| 3  Continuous batching       |                         0.78 |                1   |           0.44 |           1.68 | 173 @ 2.6 rps                 |         0.378 | n/a               |
| 4  + paged KV + prefix cache |                         1.19 |                1.4 |           0.1  |           0.91 | 33 @ 2.6 rps                  |         0.366 | 65%               |
