"""A sampling profiler that runs inside the gateway process.

Week 3 opens with a profile, because the case for porting a data structure to
C++ is a measurement and not a preference. The guide calls for ``py-spy
record``; on macOS py-spy needs ``task_for_pid`` and therefore root, and
running the thing under test as root changes the thing under test. So the
gateway samples itself with the same technique py-spy uses -- periodically
snapshot the interpreter's per-thread stacks and count where they are -- from a
thread that needs no privileges at all.

Two properties make the output trustworthy rather than indicative:

* **Samples are weighted by elapsed time, not counted.** The sampler thread
  competes for the GIL like everything else, so its wake-ups jitter. Weighting
  each sample by the interval it actually covers makes the attribution
  independent of that jitter; counting would over-weight the periods where the
  interpreter happened to be responsive.
* **Time inside a released GIL is attributed to the Python frame that released
  it.** ``llama_decode`` runs with the GIL dropped, and the stack still names
  ``decode_step``. That is the honest attribution for this question: the
  forward pass is charged to the runner, and whatever is left is what a port
  could actually address.

Enabled with ``CADENCE_PROFILE_OUT=<path>``; costs nothing when unset. The
output is a folded-stack file (``a;b;c <micros>``), the format Brendan Gregg's
flamegraph tools read, rendered by ``bench/flamegraph.py``.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import Counter
from pathlib import Path


class SamplingProfiler:
    """Samples one named thread's Python stack at a fixed rate."""

    def __init__(self, out: str | Path, hz: int = 200, thread_name: str = "cadence-engine") -> None:
        self.out = Path(out)
        self.interval = 1.0 / max(1, hz)
        self.thread_name = thread_name
        self.folded: Counter[str] = Counter()
        self.n_samples = 0
        self.n_missed = 0
        """Wake-ups where the target thread did not exist yet, or had gone."""
        self.elapsed_s = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- lifecycle --------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="cadence-profiler", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self.dump()

    # --- sampling ---------------------------------------------------------
    def _target_ident(self) -> int | None:
        for t in threading.enumerate():
            if t.name == self.thread_name:
                return t.ident
        return None

    def _loop(self) -> None:
        ident = None
        t_prev = time.perf_counter()
        while not self._stop.is_set():
            self._stop.wait(self.interval)
            now = time.perf_counter()
            dt, t_prev = now - t_prev, now
            if ident is None or ident not in sys._current_frames():
                ident = self._target_ident()
            if ident is None:
                self.n_missed += 1
                continue
            frame = sys._current_frames().get(ident)
            if frame is None:
                self.n_missed += 1
                continue
            self.folded[self._stack(frame)] += int(dt * 1e6)
            self.n_samples += 1
            self.elapsed_s += dt

    @staticmethod
    def _stack(frame) -> str:
        """Root-first ``module:function`` chain, as folded-stack text.

        The frame object is walked immediately and never retained: holding a
        frame from another thread keeps its locals alive, which for the
        scheduler thread means pinning whole requests.
        """
        parts: list[str] = []
        f = frame
        while f is not None:
            code = f.f_code
            parts.append(f"{Path(code.co_filename).name}:{code.co_name}")
            f = f.f_back
        parts.reverse()
        return ";".join(parts)

    # --- output -----------------------------------------------------------
    def dump(self) -> None:
        """Folded stacks, plus a sidecar recording what the sampler actually
        achieved.

        The requested rate is a request: ``Event.wait`` has millisecond
        granularity and the sampler competes for the GIL. Writing down the
        realised rate and the number of missed wake-ups is the difference
        between a profile that can be argued with and one that has to be
        taken on faith.
        """
        import json

        self.out.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{stack} {micros}" for stack, micros in sorted(self.folded.items())]
        self.out.write_text("\n".join(lines) + "\n")
        self.out.with_suffix(".json").write_text(
            json.dumps(
                {
                    "thread": self.thread_name,
                    "hz_requested": round(1.0 / self.interval, 1),
                    "hz_achieved": round(self.n_samples / self.elapsed_s, 1)
                    if self.elapsed_s
                    else 0.0,
                    "n_samples": self.n_samples,
                    "n_missed": self.n_missed,
                    "elapsed_s": round(self.elapsed_s, 3),
                },
                indent=2,
            )
            + "\n"
        )
