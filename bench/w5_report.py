"""Week 5's tables: the ablation ladder, its spread, and what it cost.

Everything here is generated from the committed parquet under
``results/w5_ladder`` and written to ``docs/*.md``, which
``bench/embed_tables.py`` then substitutes into the README between markers.
CI regenerates and diffs, so a number in the writeup cannot drift from the
data that produced it.

Four tables, and the last two are the point.

``ablation.md``    the five-rung table the build guide asks for, one row per
                   rung, five columns, with a spread across three seeds.
``spread.md``      every (rung, rate) cell, mean and min-max over seeds. The
                   headline table hides variance by construction; this is
                   where a reader checks whether the differences survive it.
``tradeoffs.md``   every metric that got *worse* going up the ladder. A ladder
                   in which every rung improves everything is not a ladder, it
                   is a sales deck, and the trade-offs are what make the rest
                   of the numbers believable.
``thermal.md``     the fixed single-stream probe taken before each of the 90
                   runs. This is the evidence that the machine did not change
                   speed underneath the comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from analyze import load, summarize  # noqa: E402

# Rung 5 is the alpha=0.20 arm, and that is a choice with a reason rather
# than a tuning. Alpha is the strength of the promise the controller makes
# about each admitted request, and it is the policy's only real knob. At
# alpha=0.01 -- a 99% guarantee -- the honest bound on this workload fits
# nothing, so the controller refuses 100% of arrivals even at 0.6 rps with an
# idle engine. That is measured here on three seeds and reported as the
# negative result it is (see ARMS below and docs/admission_arms.md), but a
# rung that serves nothing is not a rung. The alpha=0.20 arm is the one that
# makes the ladder's fifth claim, and alpha=0.05 brackets it.
RUNGS = [
    "fifo",
    "static",
    "continuous",
    "continuous+cache",
    "continuous+cache+admission-a20",
]
ARMS = [
    "continuous+cache+admission",
    "continuous+cache+admission-a05",
    "continuous+cache+admission-a20",
]
LABELS = {
    "fifo": "1  FIFO, no batching",
    "static": "2  static batching (8)",
    "continuous": "3  continuous batching",
    "continuous+cache": "4  + paged KV + prefix cache",
    "continuous+cache+admission-a20": "5  + conformal admission (80%)",
    "continuous+cache+admission-a05": "5  + conformal admission (95%)",
    "continuous+cache+admission": "5  + conformal admission (99%)",
}
ISOLATES = {
    "fifo": "the baseline everything is measured against",
    "static": "the cost of head-of-line blocking",
    "continuous": "iteration-level scheduling",
    "continuous+cache": "memory efficiency and prompt reuse",
    "continuous+cache+admission-a20": "tail-latency control under overload",
    "continuous+cache+admission-a05": "the same control, promised harder",
    "continuous+cache+admission": "the same control, promised harder still",
}


def _f(v, nd=2, dash="-") -> str:
    return dash if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def _pct(v, dash="-") -> str:
    return dash if v is None or not np.isfinite(v) else f"{100 * v:.0f}%"


def present(s: pd.DataFrame) -> list[str]:
    """Ladder order, then the remaining rung-5 arms from weakest promise to
    strongest -- so the row that refuses everything is last, where it reads as
    the end of a trend rather than as an anomaly in the middle of the table."""
    seen = list(dict.fromkeys(s.config))
    out = [c for c in RUNGS if c in seen]
    out += [c for c in reversed(ARMS) if c in seen and c not in out]
    return out + [c for c in seen if c not in out]


# --- per-seed aggregation --------------------------------------------------


def per_seed(s: pd.DataFrame, metric: str) -> pd.DataFrame:
    """(config, rate) -> mean, min, max, std of ``metric`` across seeds."""
    g = s.groupby(["config", "rate_rps"])[metric]
    out = pd.DataFrame(
        {
            "mean": g.mean(),
            "min": g.min(),
            "max": g.max(),
            "std": g.std(ddof=0),
            "n_seeds": g.size(),
        }
    ).reset_index()
    return out


def saturation(s: pd.DataFrame, config: str) -> tuple[float, float, float]:
    """(peak goodput, its spread across seeds, the offered load it happens at).

    Saturation is defined as the offered load at which mean goodput peaks --
    measured, never the nominal capacity. Past it, offering more work returns
    less useful work, which is the entire subject of the project.
    """
    g = per_seed(s[s.config == config], "goodput_rps")
    if g.empty:
        return float("nan"), float("nan"), float("nan")
    row = g.loc[g["mean"].idxmax()]
    return float(row["mean"]), float(row["max"] - row["min"]), float(row["rate_rps"])


def _nearest_rate(s: pd.DataFrame, target: float) -> float:
    rates = np.sort(s.rate_rps.unique())
    return float(rates[int(np.argmin(np.abs(rates - target)))])


def at_rate(s: pd.DataFrame, config: str, rate: float, metric: str) -> tuple[float, float]:
    g = per_seed(s[s.config == config], metric)
    row = g[np.isclose(g.rate_rps, rate)]
    if row.empty:
        return float("nan"), float("nan")
    return float(row["mean"].iloc[0]), float((row["max"] - row["min"]).iloc[0])


# --- the tables ------------------------------------------------------------


def ablation_table(s: pd.DataFrame, hits: pd.Series | None, slo: float) -> tuple[str, dict]:
    """One row per rung, five columns, spread across seeds.

    ``+/-`` is the min-max range over the three seeds, not a standard error:
    with n=3 a standard error is a number pretending to be a distribution,
    while the range is exactly what was seen.
    """
    ref_peak, _, ref_sat = saturation(s, "continuous+cache")
    twice = _nearest_rate(s, 2 * ref_sat) if np.isfinite(ref_sat) else float("nan")

    rows, meta = [], {}
    for cfg in present(s):
        peak, peak_spread, at = saturation(s, cfg)
        ttft50, _ = at_rate(s, cfg, at, "ttft_p50")
        ttft99, ttft99_spread = at_rate(s, cfg, at, "ttft_p99")
        e2e99, e2e99_spread = at_rate(s, cfg, twice, "e2e_p99")
        # Rungs 1-3 run without the prefix cache, and the server reports a
        # hit rate of 0.0 for a cache that does not exist. Reporting that as
        # "0%" would invite the reader to compare it with rung 4's 62%, which
        # is a comparison between a measurement and a placeholder.
        hit = float("nan")
        if hits is not None and "cache" in cfg:
            got = [v for (c, _), v in hits.items() if c == cfg]
            hit = float(np.nanmean(got)) if got and np.isfinite(got).any() else float("nan")
        if peak <= 0:
            # An arm that admitted nothing has no peak, no load at which the
            # peak occurred, and no latency quantiles: there are no completed
            # requests to take them over. Printing "0.00 at 0.6 rps" would be
            # inventing three numbers out of one.
            rows.append(
                f"| {LABELS.get(cfg, cfg)} | {ISOLATES.get(cfg, '')} | 0.00 | — "
                f"| — | — | — | — |"
            )
            meta[cfg] = {"peak_goodput_rps": 0.0, "admitted_nothing": True}
            continue
        rows.append(
            "| {label} | {isolates} | {peak} ± {ps} | {at} | {t50} | {t99} ± {ts} "
            "| {e99} ± {es} | {hit} |".format(
                label=LABELS.get(cfg, cfg),
                isolates=ISOLATES.get(cfg, ""),
                peak=_f(peak), ps=_f(peak_spread),
                at=_f(at, 1),
                t50=_f(ttft50), t99=_f(ttft99), ts=_f(ttft99_spread),
                e99=_f(e2e99, 1), es=_f(e2e99_spread, 1),
                hit=_pct(hit, dash="n/a"),
            )
        )
        meta[cfg] = {
            "peak_goodput_rps": peak,
            "peak_goodput_spread": peak_spread,
            "at_offered_rps": at,
            "ttft_p50_s": ttft50,
            "ttft_p99_s": ttft99,
            "e2e_p99_at_2x_s": e2e99,
            "prefix_hit_rate": None if not np.isfinite(hit) else hit,
        }

    header = (
        f"| Config | Isolates | Peak goodput (rps, SLO {slo:g}s) | at offered (rps) "
        f"| p50 TTFT (s) | p99 TTFT (s) | p99 E2E at {twice:g} rps (s) | Prefix hit rate |\n"
        "|:--|:--|--:|--:|--:|--:|--:|--:|"
    )
    note = (
        f"\n\n`±` is the min-max range over three seeds. Peak goodput is the "
        f"maximum over the offered-load grid, and the column beside it is the "
        f"load at which that maximum occurred. The last latency column is read "
        f"at {twice:g} rps -- about twice the {ref_sat:g} rps at which rung 4's "
        f"goodput peaks -- so every rung is compared at the same offered load, "
        f"well past saturation for all five."
    )
    return header + "\n" + "\n".join(rows) + note, {
        "saturation_rps": ref_sat,
        "saturation_goodput_rps": ref_peak,
        "overload_rate_rps": twice,
        "rungs": meta,
    }


def spread_table(s: pd.DataFrame, slo: float) -> str:
    lines = [
        "| Config | offered (rps) | seeds | goodput (rps) | p99 E2E (s) | p99 TTFT (s) | "
        "SLO attainment |",
        "|:--|--:|--:|:--|:--|:--|:--|",
    ]
    for cfg in present(s):
        g = s[s.config == cfg]
        for rate in sorted(g.rate_rps.unique()):
            cell = []
            for metric, nd in (("goodput_rps", 2), ("e2e_p99", 1), ("ttft_p99", 2)):
                r = per_seed(g, metric)
                r = r[np.isclose(r.rate_rps, rate)]
                if r.empty:
                    cell.append("-")
                else:
                    cell.append(
                        f"{r['mean'].iloc[0]:.{nd}f} "
                        f"[{r['min'].iloc[0]:.{nd}f}, {r['max'].iloc[0]:.{nd}f}]"
                    )
            att = per_seed(g, "slo_attainment")
            att = att[np.isclose(att.rate_rps, rate)]
            a = (
                "-"
                if att.empty
                else f"{att['mean'].iloc[0]:.2f} "
                f"[{att['min'].iloc[0]:.2f}, {att['max'].iloc[0]:.2f}]"
            )
            n = int(g[np.isclose(g.rate_rps, rate)].seed.nunique())
            lines.append(
                f"| {LABELS.get(cfg, cfg)} | {rate:g} | {n} | "
                f"{cell[0]} | {cell[1]} | {cell[2]} | {a} |"
            )
    lines.append("")
    lines.append(
        f"Mean over seeds, with `[min, max]` beside it. Goodput and SLO "
        f"attainment are computed at a {slo:g}s target; latency quantiles come "
        f"from the load generator's raw records, never from a Prometheus "
        f"histogram."
    )
    return "\n".join(lines)


def tradeoff_table(s: pd.DataFrame, slo: float) -> tuple[str, list[dict]]:
    """Every metric that a rung made worse, or failed to improve.

    Generated rather than written: the rule is "report at least one result
    that is worse or flat", and a rule enforced by a script cannot be quietly
    dropped from a later revision of the writeup.
    """
    metrics = [
        ("goodput_rps", "goodput", True, 2),
        ("throughput_rps", "completed requests/s", True, 2),
        ("slo_attainment", "SLO attainment", True, 2),
        ("e2e_p99", "p99 end-to-end", False, 2),
        ("ttft_p99", "p99 TTFT", False, 2),
        ("itl_p99", "p99 inter-token", False, 3),
    ]
    order = present(s)
    found: list[dict] = []
    for lower, upper in zip(order, order[1:], strict=False):
        for metric, name, higher_is_better, nd in metrics:
            lo = per_seed(s[s.config == lower], metric).set_index("rate_rps")["mean"]
            up = per_seed(s[s.config == upper], metric).set_index("rate_rps")["mean"]
            common = sorted(set(lo.index) & set(up.index))
            for rate in common:
                a, b = lo.get(rate), up.get(rate)
                if not (np.isfinite(a) and np.isfinite(b)) or a == 0:
                    continue
                delta = (b - a) / abs(a)
                worse = -delta if higher_is_better else delta
                # 5%: below that the three seeds do not separate the two rungs
                # and calling it a regression would be reading noise.
                if worse > 0.05:
                    found.append(
                        {
                            "from": lower, "to": upper, "metric": metric,
                            "name": name, "rate_rps": float(rate),
                            "before": float(a), "after": float(b),
                            "change": float(delta), "nd": nd,
                        }
                    )

    # One row per (transition, metric): the offered load at which the
    # degradation was worst, and how many of the grid's loads showed it. The
    # full grid is in docs/ablation_full.csv -- a table with sixty rows in it
    # does not get read, and the claim being made here is qualitative.
    worst: dict[tuple[str, str, str], dict] = {}
    for f in found:
        key = (f["from"], f["to"], f["metric"])
        f = dict(f, n_rates=1)
        prev = worst.get(key)
        if prev is None:
            worst[key] = f
        else:
            f["n_rates"] = prev["n_rates"] + 1
            worst[key] = f if abs(f["change"]) > abs(prev["change"]) else dict(
                prev, n_rates=f["n_rates"]
            )

    if not found:
        return (
            "No rung made any reported metric more than 5% worse than the rung "
            "below it at any offered load. That is a suspicious result, not a "
            "good one -- see the note in the writeup.\n"
        ), found

    n_grid = s.rate_rps.nunique()
    lines = [
        "| Change | Metric | worst at (rps) | before | after | | loads affected |",
        "|:--|:--|--:|--:|--:|--:|--:|",
    ]
    for f in sorted(worst.values(), key=lambda r: (order.index(r["to"]), -abs(r["change"]))):
        lines.append(
            f"| {LABELS.get(f['from'], f['from'])} → {LABELS.get(f['to'], f['to'])} "
            f"| {f['name']} | {f['rate_rps']:g} | {f['before']:.{f['nd']}f} "
            f"| {f['after']:.{f['nd']}f} | {f['change']:+.0%} | {f['n_rates']}/{n_grid} |"
        )
    lines.append("")
    lines.append(
        f"Every place where adding a component made a metric more than 5% "
        f"worse than the rung below it, at a {slo:g}s SLO. One row per "
        f"(transition, metric), shown at the offered load where the "
        f"degradation was largest, with the number of the grid's {n_grid} loads "
        f"at which it appeared. Generated by `bench/w5_report.py` and not "
        f"selected by hand -- the 5% floor is there because three seeds do not "
        f"separate two rungs below it. The full grid is in "
        f"`docs/ablation_full.csv`."
    )
    return "\n".join(lines), found


def arms_table(s: pd.DataFrame, slo: float) -> str:
    """Rung 5 at three guarantee levels, against rung 4.

    Alpha is not a hyperparameter to be tuned until the numbers look good; it
    is the strength of the promise the controller makes about each request it
    admits, and it is a product decision. Reporting one alpha would present a
    choice as a result. Reporting three shows the trade-off the choice is
    made along -- and includes the level at which the honest answer is "this
    system cannot promise that", which is the most informative row in the
    table.
    """
    base = "continuous+cache"
    order = [base] + [a for a in ARMS if a in set(s.config)]
    rates = sorted(s.rate_rps.unique())
    lines = [
        "| Config | " + " | ".join(f"{r:g} rps" for r in rates) + " |",
        "|:--|" + "--:|" * len(rates),
    ]

    def block(metric: str, fmt, title: str) -> None:
        lines.append(f"| **{title}** |" + " |" * len(rates))
        for cfg in order:
            g = per_seed(s[s.config == cfg], metric).set_index("rate_rps")["mean"]
            cells = [fmt(g.get(r, float("nan"))) for r in rates]
            lines.append(f"| {LABELS.get(cfg, cfg)} | " + " | ".join(cells) + " |")

    block("goodput_rps", lambda v: _f(v, 2), f"Goodput (rps within {slo:g}s)")
    block("e2e_p99", lambda v: _f(v, 1), "p99 end-to-end (s)")
    block("shed_rate", _pct, "Refused (503)")
    block("slo_attainment_admitted",
          lambda v: _pct(v), "SLO attainment among admitted")

    lines.append("")
    lines.append(
        f"Mean over three seeds. The last block is the promise the controller "
        f"actually kept: of the requests it chose to admit, how many finished "
        f"inside {slo:g}s. The row above it is what that cost."
    )
    return "\n".join(lines)


def anchor_table(main: pd.DataFrame, second: pd.DataFrame, slo: float) -> tuple[str, dict]:
    """How much of a difference is "it was measured later"?

    Rungs 1 to 4 and rung 5 were measured in two blocks, hours apart, because
    the first block's rung 5 turned out to be the arm that refuses
    everything. Running the replacement on its own breaks the rate-major
    rotation that protects the ladder from drift, so rung 4 was run again
    alongside it, unchanged, as an anchor. The difference between the two
    copies of rung 4 is the size of the block effect, measured in the same
    metric the ladder is compared in, rather than an assurance that there
    was not one.
    """
    cfg = "continuous+cache"
    rates = sorted(set(main[main.config == cfg].rate_rps) & set(second[second.config == cfg].rate_rps))
    if not rates:
        return "", {}
    lines = [
        "| offered (rps) | metric | block 1 (rungs 1-5, alpha 0.01) | block 2 (rung 5 arms) | |",
        "|--:|:--|--:|--:|--:|",
    ]
    worst_by = {}
    for metric, name, nd in (("e2e_p99", "p99 end-to-end (s)", 1),
                             ("goodput_rps", "goodput (rps)", 2)):
        a = per_seed(main[main.config == cfg], metric).set_index("rate_rps")["mean"]
        b = per_seed(second[second.config == cfg], metric).set_index("rate_rps")["mean"]
        for r in rates:
            av, bv = a.get(r, np.nan), b.get(r, np.nan)
            if not (np.isfinite(av) and np.isfinite(bv)):
                continue
            if av == 0 and bv == 0:
                # Both blocks delivered nothing inside the SLO. That is
                # agreement, not a 0/0 percentage.
                lines.append(
                    f"| {r:g} | {name} | {av:.{nd}f} | {bv:.{nd}f} | both zero |"
                )
                continue
            d = (bv - av) / av if av else float("inf")
            worst_by[metric] = max(worst_by.get(metric, 0.0), abs(d))
            lines.append(
                f"| {r:g} | {name} | {av:.{nd}f} | {bv:.{nd}f} | {d:+.0%} |"
            )
    worst = worst_by.get("e2e_p99", float("nan"))
    worst_goodput = worst_by.get("goodput_rps", float("nan"))
    lines.append("")
    lines.append(
        f"Rung 4, run twice: once interleaved with rungs 1-3, once "
        f"interleaved with the rung-5 arms hours later, with everything else "
        f"identical. The largest disagreement is {100 * worst:.0f}% in p99 "
        f"end-to-end and {100 * worst_goodput:.0f}% in goodput.\n\n"
        f"The goodput figure is the one to take seriously, and it is worse "
        f"than it looks at first: the large disagreements are at the offered "
        f"loads just past the collapse point, where goodput is falling "
        f"steeply and a few percent of extra machine speed moves a lot of "
        f"requests across the SLO line. That is not noise in the "
        f"measurement so much as genuine sensitivity in the thing being "
        f"measured, and it is why the ladder's claims are made about the "
        f"shape of these curves rather than about individual cells.\n\n"
        f"Any cross-block comparison -- which means every comparison "
        f"involving rung 5 -- should be read with this as its floor. The "
        f"rung-5 differences are one to two orders of magnitude, so they "
        f"survive it comfortably; a 10% difference between two rungs "
        f"measured in different blocks would not be a result."
    )
    return "\n".join(lines), {
        "worst_p99_delta": float(worst),
        "worst_goodput_delta": float(worst_goodput),
        "rates": rates,
    }


def thermal_table(canary_paths: list[Path]) -> tuple[str, dict]:
    """What the machine was doing underneath the ladder."""
    if not canary_paths:
        return "No thermal probe was recorded for this sweep.\n", {}
    recs = [
        json.loads(line)
        for path in canary_paths
        for line in path.open()
        if line.strip()
    ]
    ok = [r for r in recs if r.get("ok")]
    failed = [r for r in recs if not r.get("ok")]
    if not ok:
        return "The thermal probe ran but produced no usable readings.\n", {}
    df = pd.DataFrame(ok)
    overall = df.tok_per_s
    by = df.groupby("rung").tok_per_s.agg(["mean", "min", "max", "size"])

    lines = [
        "| Measured before | probes | decode (tok/s), mean | min | max |",
        "|:--|--:|--:|--:|--:|",
    ]
    for rung in [r for r in RUNGS if r in by.index] + [
        r for r in by.index if r not in RUNGS
    ]:
        r = by.loc[rung]
        lines.append(
            f"| {LABELS.get(rung, rung)} | {int(r['size'])} | {r['mean']:.1f} "
            f"| {r['min']:.1f} | {r['max']:.1f} |"
        )
    lines.append(
        f"| **whole sweep** | {len(df)} | **{overall.mean():.1f}** "
        f"| {overall.min():.1f} | {overall.max():.1f} |"
    )

    spread = (overall.max() - overall.min()) / overall.max()
    between = (by["mean"].max() - by["mean"].min()) / by["mean"].max()
    lines.append("")
    lines.append(
        f"A fixed 64-token single-stream generation, greedy, on an idle engine, "
        f"timed immediately before each of the {len(df)} measured runs "
        f"(`bench/run_sweep.py:canary`). It is this project's substitute for "
        f"`powermetrics`, which needs root: the number is a direct reading of "
        f"how fast the machine was at that moment.\n\n"
        f"Across the whole sweep the probe varied by {100 * spread:.1f}%. What "
        f"matters for the ladder is not that spread but whether it fell "
        f"*unevenly* on the rungs, and the per-rung means differ by "
        f"{100 * between:.1f}% -- the rate-major, rung-rotated order is what "
        f"keeps that small, and this table is how the claim is checked rather "
        f"than asserted."
    )
    if failed:
        by_rung = {}
        for r in failed:
            by_rung[r.get("rung", "?")] = by_rung.get(r.get("rung", "?"), 0) + 1
        lines.append("")
        lines.append(
            f"{len(failed)} of the {len(recs)} probes returned nothing, all of "
            f"them on the admission rungs ("
            + ", ".join(f"{LABELS.get(k, k)}: {v}" for k, v in sorted(by_rung.items()))
            + "). The probe goes through the same door as every other "
            "request, so a controller that is shedding sheds it too. That is "
            "a defect in the instrument rather than in the engine -- the "
            "probe should bypass admission, and does not -- and it is left "
            "as it ran: the rungs whose thermal coverage is thinner are named "
            "here rather than quietly averaged in."
        )
    return "\n".join(lines), {
        "n_probes": int(len(df)),
        "tok_per_s_mean": float(overall.mean()),
        "tok_per_s_min": float(overall.min()),
        "tok_per_s_max": float(overall.max()),
        "sweep_spread": float(spread),
        "between_rung_spread": float(between),
        "per_rung_mean": {k: float(v) for k, v in by["mean"].items()},
        "n_failed_probes": len(failed),
    }


ARM_ALPHA = {
    "continuous+cache+admission": 0.01,
    "continuous+cache+admission-a05": 0.05,
    "continuous+cache+admission-a20": 0.20,
}


def online_coverage_table(trace_dirs: list[str]) -> tuple[str, dict]:
    """What the bound achieved live, on the requests it chose to admit.

    This is the number the theory does not cover, and it is the one worth
    reporting. Split conformal guarantees coverage over data exchangeable
    with the calibration set; the calibration set was collected with
    admission *off*, and the controller changes which requests run. The
    guarantee is therefore void the moment the controller is switched on --
    not approximately, but as a matter of what the theorem says.

    So it is measured instead, over three seeds and the whole offered-load
    grid, from the traces the controller itself wrote while deciding.
    """
    from w4_report import online_coverage

    rows, meta = [], {}
    for arm, alpha in ARM_ALPHA.items():
        got = online_coverage([str(d) for d in trace_dirs], alpha, config=arm)
        if not got:
            continue
        rows.append(
            f"| {LABELS.get(arm, arm)} | {1 - alpha:.0%} | "
            f"**{got['empirical']:.1%}** | {got['n']} | {got['n_violations']} | "
            f"{got['mean_bound_s']:.2f} | {got['mean_e2e_s']:.2f} |"
        )
        meta[arm] = got
    if not rows:
        return "", {}
    header = [
        "| Arm | Nominal | Empirical | admitted & completed | violations | "
        "mean bound (s) | mean actual (s) |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    note = (
        "\n\nCoverage over every request the controller admitted and saw "
        "finish, across three seeds and six offered loads, from the traces it "
        "wrote while it was deciding. This is deliberately *not* the held-out "
        "coverage in the Week 4 fit: that one is a check on the method, and "
        "this one is a check on the deployment. The finite-sample guarantee "
        "does not apply here at all -- the calibration set was collected with "
        "admission off, and the controller changes which requests run, so the "
        "exchangeability the theorem needs is gone by construction. The gap "
        "between the mean bound and the mean actual latency is the price of a "
        "bound that is valid rather than sharp."
    )
    return "\n".join(header + rows) + note, meta


def drain_table(path: Path) -> tuple[str, dict]:
    """What the drain actually did, from bench/demo_drain.py --json-out."""
    if not path.exists():
        return "", {}
    d = json.loads(path.read_text())
    inflight = d.get("inflight", [])
    clean = [r for r in inflight if r.get("finish") in {"stop", "length"}]
    after = [r for r in inflight if r.get("t_done", 0.0) > d.get("t_signal", 0.0)]
    lines = [
        "| What SIGTERM did | Measured |",
        "|:--|:--|",
        f"| `/ready` before the signal | {d.get('ready_before_signal')} |",
        f"| `/ready` after the signal | 503 after "
        f"{1e3 * d['unready_after_s']:.0f} ms |"
        if d.get("unready_after_s") is not None
        else "| `/ready` after the signal | never went 503 |",
        f"| a request arriving mid-drain | {d.get('late_status')} with "
        f"`Retry-After: {d.get('late_retry_after')}` |",
        f"| streams in flight when it landed | {len(inflight)} |",
        f"| of those, still running at the signal | {len(after)} |",
        f"| of those, finished with a real `finish_reason` | {len(clean)} |",
        f"| truncated | {len(inflight) - len(clean)} |",
        f"| wall time from signal to exit | {d.get('drain_wall_s', float('nan')):.2f}s |",
    ]
    lines.append("")
    lines.append(
        "One run of `uv run bench/demo_drain.py`, which exits non-zero if any "
        "row above comes out wrong -- including if every stream had already "
        "finished when the signal landed, since that would mean nothing was "
        "drained."
    )
    return "\n".join(lines), {
        "streams": len(inflight),
        "finished_cleanly": len(clean),
        "running_at_signal": len(after),
        "unready_after_s": d.get("unready_after_s"),
    }


def session_check(s: pd.DataFrame, w2_paths: list[str], slo: float) -> str:
    """Does the rung the two sessions share still measure the same thing?

    Week 2's ladder and this one were run days apart on the same machine with
    the same configuration for ``continuous+cache``. Any drift between them is
    an upper bound on what a reader should believe about comparisons made
    across sessions, and it is cheaper to measure it than to promise it.
    """
    try:
        w2 = summarize(load(w2_paths), slo_s=slo)
    except SystemExit:
        return ""
    cfg = "continuous+cache"
    a = per_seed(s[s.config == cfg], "e2e_p99").set_index("rate_rps")["mean"]
    b = per_seed(w2[w2.config == cfg], "e2e_p99").set_index("rate_rps")["mean"]
    common = sorted(set(a.index) & set(b.index))
    if not common:
        return ""
    lines = [
        "| offered (rps) | Week 2 p99 E2E (s) | Week 5 p99 E2E (s) | |",
        "|--:|--:|--:|--:|",
    ]
    for rate in common:
        d = (a[rate] - b[rate]) / b[rate] if b[rate] else float("nan")
        lines.append(f"| {rate:g} | {b[rate]:.2f} | {a[rate]:.2f} | {d:+.0%} |")
    lines.append("")
    lines.append(
        "The one rung both sessions ran, at the offered loads they share. "
        "Week 2 is a single seed and Week 5 is the mean of three, so this is "
        "not a controlled comparison -- it is a bound on how much a "
        "cross-session comparison in this writeup can be trusted."
    )
    return "\n".join(lines)


# --- entry point -----------------------------------------------------------


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="the Week 5 ladder results directory")
    p.add_argument("--r5", default=None,
                   help="the second block: the rung-5 arms plus the rung-4 anchor")
    p.add_argument("--slo", type=float, default=4.0)
    p.add_argument("--outdir", default="docs")
    p.add_argument("--w2", default=None, help="Week 2's ladder, for the session check")
    p.add_argument("--fit", default=None, help="results/w4_fit, for the coverage line")
    p.add_argument("--traces", action="append", default=None,
                   help="admission traces from this sweep; repeatable. Used "
                        "for the online coverage the bound actually achieved.")
    p.add_argument("--drain", default=None, help="bench/demo_drain.py --json-out")
    a = p.parse_args(argv)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load(a.paths)
    s_main = summarize(df, slo_s=a.slo)

    # Two blocks. Rungs 1-4 come from the first, where they were interleaved
    # with each other; rung 5's usable arms come from the second. Rung 4 was
    # run in both, and the copy used everywhere except the anchor table is
    # the first one -- the one measured alongside the rungs it is compared
    # with.
    s, s_r5 = s_main, None
    if a.r5:
        df_r5 = load([a.r5])
        s_r5 = summarize(df_r5, slo_s=a.slo)
        keep = s_r5[s_r5.config.isin(ARMS)]
        s = pd.concat([s_main, keep], ignore_index=True)
        df = pd.concat([df, df_r5[df_r5.config.isin(ARMS)]], ignore_index=True)

    hits = (
        df.groupby(["config", "rate_rps"]).prefix_hit_rate_server.last()
        if "prefix_hit_rate_server" in df
        else None
    )

    table, head = ablation_table(s, hits, a.slo)
    (outdir / "ablation.md").write_text(table + "\n")
    (outdir / "spread.md").write_text(spread_table(s, a.slo) + "\n")
    trade, found = tradeoff_table(s, a.slo)
    (outdir / "tradeoffs.md").write_text(trade + "\n")

    (outdir / "admission_arms.md").write_text(arms_table(s, a.slo) + "\n")

    anchor_meta: dict = {}
    if s_r5 is not None:
        anchor, anchor_meta = anchor_table(s_main, s_r5, a.slo)
        if anchor:
            (outdir / "block_anchor.md").write_text(anchor + "\n")

    canary_paths = [Path(pp) / "canary.jsonl" for pp in a.paths]
    if a.r5:
        canary_paths.append(Path(a.r5) / "canary.jsonl")
    thermal, thermal_meta = thermal_table([p for p in canary_paths if p.exists()])
    (outdir / "thermal.md").write_text(thermal + "\n")

    if a.w2:
        check = session_check(s, [a.w2], a.slo)
        if check:
            (outdir / "session_check.md").write_text(check + "\n")

    s.to_csv(outdir / "ablation_full.csv", index=False)

    head["n_seeds"] = int(s.seed.nunique())
    head["n_runs"] = int(len(s))
    head["slo_s"] = a.slo
    head["thermal"] = thermal_meta
    head["block_anchor"] = anchor_meta
    head["n_tradeoffs"] = len(found)
    if a.traces:
        tdirs = [Path(d) for d in ([a.traces] if isinstance(a.traces, str) else a.traces)]
        cov, cov_meta = online_coverage_table([d for d in tdirs if d.exists()])
        if cov:
            (outdir / "online_coverage.md").write_text(cov + "\n")
            head["online_coverage"] = cov_meta

    if a.drain:
        drain, drain_meta = drain_table(Path(a.drain))
        if drain:
            (outdir / "drain.md").write_text(drain + "\n")
            head["drain"] = drain_meta
    if a.fit and (Path(a.fit) / "fit.json").exists():
        fit = json.loads((Path(a.fit) / "fit.json").read_text())
        head["predictor"] = {
            "alpha": fit.get("alpha"),
            "test_coverage": fit.get("models", {}).get("cqr", {}).get("coverage"),
        }
    (outdir / "w5_headline.json").write_text(json.dumps(head, indent=2, default=str) + "\n")

    print(f"wrote ablation.md, spread.md, tradeoffs.md, thermal.md to {outdir}")
    print(f"  saturation {head['saturation_rps']:g} rps, "
          f"overload column read at {head['overload_rate_rps']:g} rps, "
          f"{head['n_seeds']} seeds, {len(found)} trade-off cells")


if __name__ == "__main__":
    main()
