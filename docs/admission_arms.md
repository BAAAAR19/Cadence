| Config | 0.6 rps | 1 rps | 1.4 rps | 1.9 rps | 2.6 rps | 4 rps |
|:--|--:|--:|--:|--:|--:|--:|
| **Goodput (rps within 4s)** | | | | | | |
| 4  + paged KV + prefix cache | 0.69 | 1.07 | 1.23 | 1.15 | 0.72 | 0.00 |
| 5  + conformal admission (99%) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5  + conformal admission (95%) | 0.37 | 0.62 | 0.77 | 0.93 | 1.20 | 1.62 |
| 5  + conformal admission (80%) | 0.59 | 0.91 | 1.13 | 1.30 | 1.65 | 2.02 |
| **p99 end-to-end (s)** | | | | | | |
| 4  + paged KV + prefix cache | 5.4 | 8.3 | 13.7 | 24.9 | 42.3 | 90.0 |
| 5  + conformal admission (99%) | - | - | - | - | - | - |
| 5  + conformal admission (95%) | 1.6 | 1.7 | 1.9 | 1.8 | 2.0 | 2.3 |
| 5  + conformal admission (80%) | 2.8 | 3.4 | 3.2 | 3.6 | 3.8 | 4.7 |
| **Refused (503)** | | | | | | |
| 4  + paged KV + prefix cache | 0% | 0% | 0% | 0% | 0% | 0% |
| 5  + conformal admission (99%) | 100% | 100% | 100% | 100% | 100% | 100% |
| 5  + conformal admission (95%) | 50% | 47% | 51% | 53% | 54% | 59% |
| 5  + conformal admission (80%) | 18% | 21% | 28% | 33% | 37% | 48% |
| **SLO attainment among admitted** | | | | | | |
| 4  + paged KV + prefix cache | 96% | 91% | 79% | 59% | 28% | 0% |
| 5  + conformal admission (99%) | - | - | - | - | - | - |
| 5  + conformal admission (95%) | 100% | 100% | 100% | 100% | 100% | 100% |
| 5  + conformal admission (80%) | 100% | 100% | 100% | 99% | 99% | 97% |

Mean over three seeds. The last block is the promise the controller actually kept: of the requests it chose to admit, how many finished inside 4s. The row above it is what that cost.
