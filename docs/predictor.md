| Model                             |   Coverage (target ≥ 99%) |   Q (log-ratio) |   Mean bound U (s) |   Median bound U (s) | Would admit   | of those, met SLO   |
|:----------------------------------|--------------------------:|----------------:|-------------------:|---------------------:|:--------------|:--------------------|
| Conformalised quantile regression |                     0.997 |           0.73  |              21.74 |                18.23 | 3.7%          | 100.0%              |
| Throughput arithmetic (fitted)    |                     0.995 |           0.03  |             105.96 |               105.66 | 3.3%          | 100.0%              |
| Constant quantile (no features)   |                     0.999 |           0.106 |              45.71 |                45.71 | 0.0%          | -                   |

The quantile pair is fitted at 90% and the bound is calibrated to 99%; the nonconformity score is `ratio`. Both were chosen on a held-out slice of the training fold, and neither can affect validity — only width:

| Nonconformity score                            |   Coverage |      Q |   Median bound U (s) | Would admit   |
|:-----------------------------------------------|-----------:|-------:|---------------------:|:--------------|
| absolute — `y − q_hi`, the build guide's       |     0.9986 | 24.697 |                32.96 | 0.0%          |
| ratio — `log1p(y) − log1p(q_hi)`  *(deployed)* |     0.9973 |  0.73  |                18.23 | 3.7%          |
| scaled — `(y − q_hi) / (q_hi − q_lo)`          |     1      |  2.956 |                26.98 | 3.3%          |

What the upper-quantile model leans on:

| Feature              |   Importance |
|:---------------------|-------------:|
| ewma_tokens_per_s    |        0.13  |
| queue_depth          |        0.122 |
| ewma_step_latency_s  |        0.119 |
| max_tokens           |        0.104 |
| log_max_tokens       |        0.101 |
| kv_blocks_free_frac  |        0.1   |
| sum_remaining_tokens |        0.1   |
| n_prompt_tokens      |        0.099 |

Safety factor, read off the same held-out split:

|   Safety factor | Would admit   | of those, met SLO   | Offline goodput (admitted ∧ in SLO)   |
|----------------:|:--------------|:--------------------|:--------------------------------------|
|            0.6  | 11.4%         | 97.6%               | 11.1%                                 |
|            0.7  | 9.1%          | 98.5%               | 8.9%                                  |
|            0.8  | 6.7%          | 98.0%               | 6.6%                                  |
|            0.9  | 5.4%          | 100.0%              | 5.4%                                  |
|            1    | 3.7%          | 100.0%              | 3.7%                                  |
|            1.1  | 2.2%          | 100.0%              | 2.2%                                  |
|            1.25 | 1.2%          | 100.0%              | 1.2%                                  |
|            1.5  | 0.3%          | 100.0%              | 0.3%                                  |
