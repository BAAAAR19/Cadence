"""The four figures the writeup opens with.

    1  p99 end-to-end vs offered load, five lines.  The whole thesis.
    2  goodput vs offered load, five lines.         Admission is a gain.
    3  coverage calibration, empirical vs nominal.  The guarantee holds.
    4  TTFT and ITL distributions at one load.      What it cost.

Conventions, applied to all of them and each chosen because the obvious
alternative hides something:

* **x is offered load, never achieved throughput.** Achieved throughput
  saturates, so plotting against it folds the entire overload region -- the
  only region this project is about -- into a single point at the right-hand
  edge.
* **y is log for latency.** The rungs differ by two orders of magnitude past
  saturation; on a linear axis rungs 3 to 5 are one flat line along the
  bottom.
* **a dashed horizontal line at the SLO**, because every latency claim here
  is relative to it.
* **a dashed vertical line at measured saturation**, taken as the offered
  load where rung 4's goodput peaks -- measured, not nominal.
* **a shaded band from min to max across seeds.** Three runs of a latency
  benchmark on a laptop disagree, and a chart that shows only the mean is a
  claim that they did not.

Figure 3 is drawn by ``bench/w4_charts.py`` from the Week 4 fit, and is
rendered here into the same directory so that all four figures the README
opens with are produced by one command from committed data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

from analyze import load, summarize  # noqa: E402
from charts import COLOURS, LABELS, _style, itl_histogram  # noqa: E402
from w5_report import RUNGS, per_seed, saturation  # noqa: E402


def _order(s: pd.DataFrame) -> list[str]:
    seen = list(dict.fromkeys(s.config))
    return [c for c in RUNGS if c in seen] + [c for c in seen if c not in RUNGS]


def _band(ax, s: pd.DataFrame, cfg: str, metric: str, **kw) -> None:
    """Mean line, min-max band. The band is the honest part."""
    g = per_seed(s[s.config == cfg], metric).sort_values("rate_rps")
    if g.empty:
        return
    colour = COLOURS.get(cfg)
    ax.plot(g.rate_rps, g["mean"], marker="o", ms=4.5, lw=1.8, color=colour,
            label=LABELS.get(cfg, cfg), **kw)
    if (g.n_seeds > 1).any():
        ax.fill_between(g.rate_rps, g["min"], g["max"], color=colour, alpha=0.16, lw=0)


def _saturation_marks(ax, sat: float, slo: float | None) -> None:
    if slo is not None:
        ax.axhline(slo, color="#999", ls="--", lw=1.1)
        ax.annotate(f"SLO {slo:g}s", xy=(0.995, slo), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=8, color="#666")
    if np.isfinite(sat):
        ax.axvline(sat, color="#999", ls="--", lw=1.1)
        ax.annotate(
            f"saturation {sat:g} rps",
            xy=(sat, 0.02), xycoords=("data", "axes fraction"),
            rotation=90, ha="right", va="bottom", fontsize=8, color="#666",
        )


# --- figure 1 --------------------------------------------------------------


def _seed_note(s: pd.DataFrame) -> str:
    n = int(s.seed.nunique())
    return "single seed, no band" if n < 2 else f"band is min-max over {n} seeds"


def p99_vs_load(s: pd.DataFrame, out: Path, slo: float, sat: float) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for cfg in _order(s):
        _band(ax, s, cfg, "e2e_p99")
    _saturation_marks(ax, sat, slo)
    ax.set_yscale("log")
    _style(
        ax, "offered load (requests/s)", "p99 end-to-end latency (s, log scale)",
        "p99 end-to-end latency vs offered load",
        "measured from intended arrival, so queueing is included; "
        + _seed_note(s),
    )
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# --- figure 2 --------------------------------------------------------------


def goodput_vs_load(s: pd.DataFrame, out: Path, slo: float, sat: float) -> None:
    """Two panels: goodput, and what it cost in requests refused.

    The right-hand panel exists so the left one cannot be read as a free
    lunch. Rung 5 buys its goodput by refusing work, and the price is on the
    chart next to the benefit.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharex=True)
    for cfg in _order(s):
        _band(axes[0], s, cfg, "goodput_rps")
        _band(axes[1], s, cfg, "shed_rate")

    lim = float(s.rate_rps.max())
    axes[0].plot([0, lim], [0, lim], color="#bbb", ls=":", lw=1.1)
    axes[0].annotate("everything served within the SLO",
                     xy=(lim * 0.52, lim * 0.55), rotation=32, fontsize=7.5,
                     color="#999", ha="center")
    _saturation_marks(axes[0], sat, None)
    _style(
        axes[0], "offered load (requests/s)", f"goodput (requests/s within {slo:g}s)",
        f"Goodput vs offered load (SLO {slo:g}s)",
        "the dotted diagonal is perfect service; distance below it is\n"
        "work that was served late or not at all — " + _seed_note(s),
    )
    axes[0].legend(fontsize=8.5, frameon=False, loc="upper left")

    _saturation_marks(axes[1], sat, None)
    axes[1].set_ylim(-0.02, 1.02)
    _style(
        axes[1], "offered load (requests/s)", "fraction refused (503)",
        "What the goodput cost",
        "only rung 5 refuses anything; the other four accept everything\n"
        "and serve some of it minutes late",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# --- figure 4 --------------------------------------------------------------


def ttft_ccdf(df: pd.DataFrame, out: Path, rate: float, slo: float) -> None:
    """TTFT as a distribution at one offered load, not a quantile.

    A p99 in a table is one number from a shape, and the shape is where the
    scheduling difference is legible: FIFO's head-of-line blocking is a long
    flat shoulder, static batching is a staircase at the wait-to-fill
    timeout, continuous batching is a knee.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    sub = df[np.isclose(df.rate_rps, rate) & df.steady & df.ok.fillna(False)]
    for cfg in _order(sub.rename(columns={"config": "config"})):
        v = sub[sub.config == cfg].ttft.dropna().to_numpy(float)
        v = v[v > 0]
        if not v.size:
            continue
        x = np.sort(v)
        y = 1.0 - np.arange(x.size) / x.size
        axes[0].plot(x, y, lw=1.8, color=COLOURS.get(cfg),
                     label=f"{LABELS.get(cfg, cfg)}  (p99 {np.quantile(v, 0.99):.2f}s)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylim(5e-4, 1.2)
    axes[0].axvline(slo, color="#999", ls="--", lw=1.1)
    for q, lab in ((0.5, "p50"), (0.01, "p99")):
        axes[0].axhline(q, color="#ccc", ls=":", lw=1)
        axes[0].annotate(lab, xy=(0.995, q), xycoords=("axes fraction", "data"),
                         ha="right", va="bottom", fontsize=7.5, color="#888")
    _style(
        axes[0], "time to first token (s, log scale)", "P(TTFT > x)",
        "Time to first token — tail",
        f"every admitted request at {rate:g} rps offered load, all seeds pooled",
    )
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")

    # The other half of the trade-off, on the same page: chunked prefill buys
    # the TTFT on the left by putting prefill work between other requests'
    # decode steps, which is inter-token latency.
    parts = []
    for cfg in _order(sub):
        g = sub[sub.config == cfg]
        if len(g):
            parts.append(g)
    if parts:
        itl_histogram(
            pd.concat(parts, ignore_index=True),
            out.with_name(out.stem + "_itl.png"),
            subtitle=f"all seeds, {rate:g} rps offered load",
        )

    for cfg in _order(sub):
        g = sub[sub.config == cfg]
        chunks = [np.asarray(v, dtype=float) for v in g.itl if len(v)]
        v = np.concatenate(chunks) if chunks else np.array([])
        v = v[v > 0]
        if not v.size:
            continue
        x = np.sort(v) * 1e3
        y = 1.0 - np.arange(x.size) / x.size
        axes[1].plot(x, y, lw=1.8, color=COLOURS.get(cfg),
                     label=f"{LABELS.get(cfg, cfg)}  (p99 {np.quantile(v, 0.99) * 1e3:.0f} ms)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_ylim(5e-5, 1.2)
    for q, lab in ((0.5, "p50"), (0.01, "p99")):
        axes[1].axhline(q, color="#ccc", ls=":", lw=1)
        axes[1].annotate(lab, xy=(0.995, q), xycoords=("axes fraction", "data"),
                         ha="right", va="bottom", fontsize=7.5, color="#888")
    _style(
        axes[1], "inter-token latency (ms, log scale)", "P(ITL > x)",
        "Inter-token latency — tail",
        "the other side of the trade: chunked prefill buys the TTFT on the\n"
        "left by interleaving prefill with other requests' decode steps",
    )
    axes[1].legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# --- entry point -----------------------------------------------------------


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--slo", type=float, default=4.0)
    p.add_argument("--outdir", default="docs/figs/w5")
    p.add_argument("--fit", default=None, help="results/w4_fit, for the coverage figure")
    p.add_argument("--dist-rate", type=float, default=None,
                   help="offered load for figure 4; defaults to measured saturation")
    a = p.parse_args(argv)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = load(a.paths)
    s = summarize(df, slo_s=a.slo)

    _, _, sat = saturation(s, "continuous+cache")
    p99_vs_load(s, out / "p99_vs_load.png", a.slo, sat)
    goodput_vs_load(s, out / "goodput_vs_load.png", a.slo, sat)

    rate = a.dist_rate if a.dist_rate is not None else sat
    ttft_ccdf(df, out / "ttft_itl_distributions.png", float(rate), a.slo)

    if a.fit:
        fitdir = Path(a.fit)
        cov = fitdir / "coverage.csv"
        if cov.exists():
            from w4_charts import coverage_calibration

            online_path = Path("results/w4_admission/online_coverage.json")
            online = json.loads(online_path.read_text()) if online_path.exists() else None
            coverage_calibration(pd.read_csv(cov), out / "coverage.png", online)
        else:
            print(f"no {cov}; skipping the coverage figure", file=sys.stderr)

    print(f"wrote the Week 5 figures to {out}")


if __name__ == "__main__":
    main()
