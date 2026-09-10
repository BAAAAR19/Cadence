"""The CI load-test gate: compare one short sweep against a committed baseline.

What this is, and what it is not
--------------------------------
It is **not** a benchmark. A GitHub runner is a shared, oversubscribed VM whose
neighbours are invisible; latency measured on it is a measurement of the
runner's mood. Every number in the README comes from a pinned machine and none
of them come from here.

It **is** a regression gate, which is a different instrument. It runs the whole
stack -- the API, the scheduler, the paged KV cache, the radix cache, the
conformal controller -- against the mock backend, whose per-token cost is a
fixed sleep. That makes the thing being measured the *scheduler's decisions*
rather than the runner's clock speed: at a given offered load the same
arrivals, the same batching decisions and the same admission decisions produce
the same p99 on any machine, to within the noise of the sleeps. So a change in
that p99 is a change in behaviour, and that is exactly what a gate should fire
on.

The gate refuses in three distinct ways, and they are not the same event:

``exit 1`` -- a regression: p99 rose past the tolerance, or goodput fell below
the floor. The PR did something.

``exit 2`` -- an invalid measurement: too few requests, a load generator that
could not keep to its own schedule, or a rate present in the baseline and
missing from the run. Nothing is said about the change, because nothing can be:
a gate that reports "pass" on a run that did not happen is worse than no gate.

``exit 0`` -- neither.

An improvement never fails. A p99 that drops by half is a real result, and the
right response to it is a new baseline in a commit that says why, not a red
tick on the PR that produced it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from analyze import load, summarize  # noqa: E402

# The metrics the gate is allowed to have an opinion about, and the direction
# each one is bad in. Everything else in the summary is reported for context
# and gates nothing -- a gate with fifteen conditions is a gate that gets
# disabled the first week it is flaky.
GATED = ("e2e_p99", "ttft_p99", "goodput_rps")

MIN_REQUESTS = 30
"""Below this a p99 is the second-slowest request, which is not a quantile.
Sized from the gate's own configuration: 45 s at 2 rps is ~90 arrivals, and
half of that is a run that went wrong."""

MAX_SCHED_DELAY_S = 0.25
"""p99 of (actual send time - intended send time), above which the run is not
open loop and its tail is not comparable to anything.

This, and not the KS statistic, is the instrument that detects a starved
runner. The KS test compares the *intended* inter-arrival times against
Exp(rate), and those come out of the load generator's RNG before any I/O
happens: they are identical whether the runner is idle or on fire, so a KS
gate would catch a changed arrival process and never catch a slow machine.
What a slow machine does is delay the *send*, and a client that cannot issue
arrivals on schedule has quietly turned an open-loop benchmark into a
closed-loop one -- the failure mode this whole project is built to avoid. A
quarter of a second is an eighth of the 2 s SLO the gate runs at; past that,
the client's own scheduling is a visible fraction of what is being measured.

The KS p-value is still reported, because a change in it means the arrival
process itself changed, which is a different and equally interesting event.
Its *value* is deliberately not gated: the sweep seed is fixed in advance and
seed 0's realisation sits in the tail of the null distribution at several
rates, which ``bench/validate_loadgen.py`` establishes over 200 seeds and the
README states plainly. Re-picking a seed until its p-value looks better is
selection on the statistic; gating on it would be the same mistake with a
build failure attached."""


def _rows(summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Summary frame -> ``{"2.0": {metric: value}}``, keyed by rate as a string
    so the JSON baseline round-trips without float keys."""
    out: dict[str, dict[str, float]] = {}
    for _, r in summary.iterrows():
        out[f"{r.rate_rps:g}"] = {
            m: (None if pd.isna(r[m]) else float(r[m]))
            for m in (
                "e2e_p50", "e2e_p99", "ttft_p50", "ttft_p99", "itl_p99",
                "goodput_rps", "throughput_rps", "slo_attainment",
                "shed_rate", "n", "n_ok",
            )
            if m in summary.columns
        }
    return out


def measure(paths: list[str], slo_s: float) -> tuple[pd.DataFrame, dict[str, dict], dict]:
    df = load(paths)
    summary = summarize(df, slo_s=slo_s)
    if summary.config.nunique() != 1:
        raise SystemExit(
            f"the gate compares one configuration; got {sorted(summary.config.unique())}"
        )
    def _by_rate(series_name: str, how: str) -> dict[str, float | None]:
        if series_name not in df:
            return {}
        g = df.groupby("rate_rps")[series_name]
        vals = g.min() if how == "min" else g.quantile(0.99)
        return {f"{k:g}": (None if pd.isna(v) else float(v)) for k, v in vals.items()}

    meta = {
        "config": str(summary.config.iloc[0]),
        "slo_s": slo_s,
        "ks_p": _by_rate("ks_p", "min"),
        "sched_delay_p99": _by_rate("sched_delay", "q99"),
    }
    return summary, _rows(summary), meta


def validate(rows: dict[str, dict], meta: dict, baseline: dict) -> list[str]:
    """Reasons this run cannot be compared to anything. Checked before the
    thresholds, so a broken run is never reported as a pass."""
    problems = []
    missing = sorted(set(baseline["rates"]) - set(rows))
    if missing:
        problems.append(f"rates in the baseline and not in the run: {missing}")
    for rate, r in sorted(rows.items()):
        if r.get("n", 0) < MIN_REQUESTS:
            problems.append(
                f"{rate} rps: only {r.get('n', 0):.0f} arrivals; a p99 over fewer "
                f"than {MIN_REQUESTS} is not a quantile"
            )
        if r.get("n_ok", 0) < 1:
            problems.append(f"{rate} rps: nothing completed successfully")
        d = meta.get("sched_delay_p99", {}).get(rate)
        if d is not None and d > MAX_SCHED_DELAY_S:
            problems.append(
                f"{rate} rps: the load generator fell {d * 1e3:.0f} ms behind its "
                f"own schedule at p99 (limit {MAX_SCHED_DELAY_S * 1e3:.0f} ms); "
                f"the runner could not issue arrivals open-loop, so this run's "
                f"tail is not comparable to anything"
            )
    if baseline.get("config") and meta["config"] != baseline["config"]:
        problems.append(
            f"configuration changed: baseline is {baseline['config']!r}, "
            f"run is {meta['config']!r}"
        )
    if baseline.get("slo_s") and abs(baseline["slo_s"] - meta["slo_s"]) > 1e-9:
        problems.append(
            f"SLO changed: baseline {baseline['slo_s']}s, run {meta['slo_s']}s"
        )
    return problems


def compare(
    rows: dict[str, dict],
    baseline: dict,
    max_p99_regression: float,
    min_goodput: float,
) -> tuple[list[str], str]:
    """Returns (failures, report table)."""
    failures: list[str] = []
    lines = [
        "| rate | metric | baseline | run | change | limit | |",
        "|---:|:---|---:|---:|---:|:---|:--|",
    ]
    for rate in sorted(baseline["rates"], key=float):
        base = baseline["rates"][rate]
        run = rows[rate]
        for metric in GATED:
            b, v = base.get(metric), run.get(metric)
            if b is None or v is None:
                continue
            higher_is_better = metric == "goodput_rps"
            if higher_is_better:
                limit = b * min_goodput
                bad = v < limit
                limit_s = f">= {limit:.3f} ({min_goodput:.0%} of baseline)"
            else:
                limit = b * (1.0 + max_p99_regression)
                bad = v > limit
                limit_s = f"<= {limit:.3f} (+{max_p99_regression:.0%})"
            change = (v - b) / b if b else float("nan")
            lines.append(
                f"| {rate} | {metric} | {b:.3f} | {v:.3f} | {change:+.1%} | "
                f"{limit_s} | {'FAIL' if bad else 'ok'} |"
            )
            if bad:
                failures.append(
                    f"{rate} rps {metric}: {v:.3f} vs baseline {b:.3f} "
                    f"({change:+.1%}), limit {limit_s}"
                )
    return failures, "\n".join(lines)


def write_baseline(
    path: Path, rows: dict[str, dict], meta: dict, note: str | None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "_comment": (
                    "Baseline for the CI load gate. Deterministic mock backend, "
                    "not a benchmark: see bench/check_regression.py. Regenerate "
                    "with --update-baseline, and say in the commit message what "
                    "changed and why the new numbers are the right ones."
                ),
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
                "note": note,
                "config": meta["config"],
                "slo_s": meta["slo_s"],
                "ks_p": meta["ks_p"],
                "sched_delay_p99": meta["sched_delay_p99"],
                "rates": rows,
            },
            indent=2,
            sort_keys=False,
        )
        + "\n"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("paths", nargs="+", help="parquet file(s) or directory from the CI sweep")
    p.add_argument("--baseline", default="bench/baselines/ci_baseline.json")
    p.add_argument("--slo", type=float, default=2.0)
    p.add_argument("--max-p99-regression", type=float, default=0.15)
    p.add_argument("--min-goodput", type=float, default=0.90)
    p.add_argument("--update-baseline", action="store_true",
                   help="write the measured numbers to --baseline instead of "
                        "checking against them")
    p.add_argument("--note", default=None, help="recorded in a regenerated baseline")
    p.add_argument("--summary-out", default=None,
                   help="write the report table here as well (GITHUB_STEP_SUMMARY)")
    a = p.parse_args(argv)

    # The baseline is checked before the sweep is even read: a gate with
    # nothing to compare against must not spend a minute of CI discovering
    # that, and must never be able to report a pass for it.
    bpath = Path(a.baseline)
    if not bpath.exists() and not a.update_baseline:
        print(f"no baseline at {bpath}; create one with --update-baseline", file=sys.stderr)
        return 2

    summary, rows, meta = measure(a.paths, a.slo)

    if a.update_baseline:
        write_baseline(bpath, rows, meta, a.note)
        print(f"wrote {bpath}")
        print(json.dumps(rows, indent=2))
        return 0

    baseline = json.loads(bpath.read_text())

    problems = validate(rows, meta, baseline)
    failures, report = compare(rows, baseline, a.max_p99_regression, a.min_goodput)

    out = [f"### CI load gate -- `{meta['config']}`, SLO {meta['slo_s']:g}s", "", report, ""]
    if problems:
        out += ["**Invalid run** -- the measurement cannot be compared:", ""]
        out += [f"- {m}" for m in problems]
    elif failures:
        out += ["**Regression:**", ""]
        out += [f"- {m}" for m in failures]
    else:
        out += [
            f"No regression: every gated metric is within "
            f"+{a.max_p99_regression:.0%} (latency) and "
            f"{a.min_goodput:.0%} (goodput) of the committed baseline."
        ]
    text = "\n".join(out)
    print(text)
    if a.summary_out:
        Path(a.summary_out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.summary_out, "a") as fh:
            fh.write(text + "\n")

    if problems:
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
