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

RUNGS = [
    "fifo",
    "static",
    "continuous",
    "continuous+cache",
    "continuous+cache+admission",
]
LABELS = {
    "fifo": "1  FIFO, no batching",
    "static": "2  static batching (8)",
    "continuous": "3  continuous batching",
    "continuous+cache": "4  + paged KV + prefix cache",
    "continuous+cache+admission": "5  + conformal admission",
}
ISOLATES = {
    "fifo": "the baseline everything is measured against",
    "static": "the cost of head-of-line blocking",
    "continuous": "iteration-level scheduling",
    "continuous+cache": "memory efficiency and prompt reuse",
    "continuous+cache+admission": "tail-latency control under overload",
}


def _f(v, nd=2, dash="-") -> str:
    return dash if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def _pct(v, dash="-") -> str:
    return dash if v is None or not np.isfinite(v) else f"{100 * v:.0f}%"


def present(s: pd.DataFrame) -> list[str]:
    seen = list(dict.fromkeys(s.config))
    return [c for c in RUNGS if c in seen] + [c for c in seen if c not in RUNGS]


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


def thermal_table(canary_path: Path) -> tuple[str, dict]:
    """What the machine was doing underneath the ladder."""
    if not canary_path.exists():
        return "No thermal probe was recorded for this sweep.\n", {}
    recs = [json.loads(line) for line in canary_path.open() if line.strip()]
    ok = [r for r in recs if r.get("ok")]
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
    return "\n".join(lines), {
        "n_probes": int(len(df)),
        "tok_per_s_mean": float(overall.mean()),
        "tok_per_s_min": float(overall.min()),
        "tok_per_s_max": float(overall.max()),
        "sweep_spread": float(spread),
        "between_rung_spread": float(between),
        "per_rung_mean": {k: float(v) for k, v in by["mean"].items()},
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
    p.add_argument("--slo", type=float, default=4.0)
    p.add_argument("--outdir", default="docs")
    p.add_argument("--w2", default=None, help="Week 2's ladder, for the session check")
    p.add_argument("--w4", default=None, help="Week 4's sweep (recorded in the headline)")
    p.add_argument("--fit", default=None, help="results/w4_fit, for the coverage line")
    p.add_argument("--traces", default=None, help="admission traces from this sweep")
    p.add_argument("--drain", default=None, help="bench/demo_drain.py --json-out")
    a = p.parse_args(argv)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load(a.paths)
    s = summarize(df, slo_s=a.slo)
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

    canary_path = Path(a.paths[0]) / "canary.jsonl"
    thermal, thermal_meta = thermal_table(canary_path)
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
    head["n_tradeoffs"] = len(found)
    if a.drain and Path(a.drain).exists():
        d = json.loads(Path(a.drain).read_text())
        head["drain"] = {
            "unready_after_s": d.get("unready_after_s"),
            "streams_finished_cleanly": sum(
                1 for r in d.get("inflight", []) if r.get("finish") in {"stop", "length"}
            ),
            "streams": len(d.get("inflight", [])),
            "late_status": d.get("late_status"),
        }
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
