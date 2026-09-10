"""One JSON line per request: the features admission saw, and what happened.

This is the training set. It is written by the server rather than reconstructed
from the load generator's parquet for one reason that decides the whole
experiment: the features must be the ones the *controller* will have, recorded
at the instant it would have had them. A row assembled afterwards from a
client-side log would carry queue depths and cache states sampled at some other
moment, and the model would be fitted on a system that never existed.

The row is written twice-in-one: the feature half is filled in at admission and
held on the request, the outcome half at completion. Nothing derived from the
outcome can reach the feature half -- see :mod:`cadence.admission.features` for
why that is a structural property here and not a convention.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path


class TraceWriter:
    """Append-only JSONL, flushed in batches.

    Called from the API event loop, so the write has to be cheap: a few hundred
    bytes into a buffered file, with an ``fsync``-free flush every
    ``flush_every`` rows so that a run killed with SIGINT loses at most that
    many. At the loads this project runs -- single-digit requests per second --
    the cost is invisible; it is written down because "the measurement
    apparatus perturbs the measurement" is exactly the kind of thing this
    project is meant to notice.
    """

    def __init__(self, path: str | Path, flush_every: int = 32) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1 << 16)
        self._lock = threading.Lock()
        self._since_flush = 0
        self.flush_every = flush_every
        self.n_rows = 0

    def write(self, row: dict) -> None:
        line = json.dumps(row, separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")
            self.n_rows += 1
            self._since_flush += 1
            if self._since_flush >= self.flush_every:
                self._fh.flush()
                self._since_flush = 0

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()
                self._fh.close()
