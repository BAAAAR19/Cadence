| offered (rps) | metric | block 1 (rungs 1-5, alpha 0.01) | block 2 (rung 5 arms) | |
|--:|:--|--:|--:|--:|
| 0.6 | p99 end-to-end (s) | 5.4 | 5.3 | -0% |
| 1 | p99 end-to-end (s) | 8.3 | 8.8 | +7% |
| 1.4 | p99 end-to-end (s) | 13.7 | 14.3 | +4% |
| 1.9 | p99 end-to-end (s) | 24.9 | 27.5 | +11% |
| 2.6 | p99 end-to-end (s) | 42.3 | 38.3 | -9% |
| 4 | p99 end-to-end (s) | 90.0 | 95.6 | +6% |
| 0.6 | goodput (rps) | 0.69 | 0.69 | +0% |
| 1 | goodput (rps) | 1.07 | 1.07 | +0% |
| 1.4 | goodput (rps) | 1.23 | 1.22 | -1% |
| 1.9 | goodput (rps) | 1.15 | 1.11 | -3% |
| 2.6 | goodput (rps) | 0.72 | 1.00 | +40% |
| 4 | goodput (rps) | 0.00 | 0.00 | both zero |

Rung 4, run twice: once interleaved with rungs 1-3, once interleaved with the rung-5 arms hours later, with everything else identical. The largest disagreement is 11% in p99 end-to-end and 40% in goodput.

The goodput figure is the one to take seriously, and it is worse than it looks at first: the large disagreements are at the offered loads just past the collapse point, where goodput is falling steeply and a few percent of extra machine speed moves a lot of requests across the SLO line. That is not noise in the measurement so much as genuine sensitivity in the thing being measured, and it is why the ladder's claims are made about the shape of these curves rather than about individual cells.

Any cross-block comparison -- which means every comparison involving rung 5 -- should be read with this as its floor. The rung-5 differences are one to two orders of magnitude, so they survive it comfortably; a 10% difference between two rungs measured in different blocks would not be a result.
