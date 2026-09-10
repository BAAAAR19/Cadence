| What SIGTERM did | Measured |
|:--|:--|
| `/ready` before the signal | 200 |
| `/ready` after the signal | 503 after 2 ms |
| a request arriving mid-drain | 503 with `Retry-After: 1.000` |
| streams in flight when it landed | 8 |
| of those, still running at the signal | 8 |
| of those, finished with a real `finish_reason` | 8 |
| truncated | 0 |
| wall time from signal to exit | 2.29s |

One run of `uv run bench/demo_drain.py`, which exits non-zero if any row above comes out wrong -- including if every stream had already finished when the signal landed, since that would mean nothing was drained.
