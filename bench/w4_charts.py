"""The four Week 4 figures, plus the two that keep them honest.

    uv run bench/w4_charts.py results/w4_admission \\
        --fit results/w4_fit --slo 4.0 --outdir docs/figs

1. p99 end-to-end against offered load, with and without admission control.
   The admission line must stay flat past saturation while the other diverges.
   This is the headline.
2. Goodput against offered load. Without admission, goodput collapses past
   saturation as everything misses the SLO; with it, goodput plateaus at
   capacity. (Both of the above come from ``charts.py``, which already draws
   them for every rung; this module adds the ones that only exist once there
   is a controller.)
3. Coverage calibration: empirical against nominal, on the diagonal.
4. Shed rate against offered load, against the theoretical minimum
   ``max(0, 1 - capacity / lambda)``. Distance above that line is work refused
   that could have been served.

And two more, because they are the ones that would expose the failure modes:
the admitted rate over time at a fixed overload, which is where a controller
without hysteresis oscillates, and the bound against the realised latency,
which is where a bound that is uniform rather than adaptive shows itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze import load, summarize  # noqa: E402
from charts import COLOURS, LABELS, ORDER, _style  # noqa: E402

ADM = "continuous+cache+admission"
BASE = "continuous+cache"


def arms(s: pd.DataFrame) -> list[str]:
    present = set(s.config)
    return [c for c in ORDER if c.startswith(ADM) and c in present]


# --- 3. the coverage plot ------------------------------------------------
def coverage_calibration(coverage: pd.DataFrame, out: Path, online: dict | None = None) -> None:
    """Empirical against nominal coverage, with the diagonal.

    The most convincing figure in the project, because it is a verified
    guarantee rather than a benchmark number: every point should sit on the
    diagonal or just above it, and being *above* is not an error -- the
    conformal bound is conservative by construction, and the finite-sample
    correction is what makes it so.

    The online point is the same measurement taken while the controller was
    running, where exchangeability no longer holds because the controller's own
    shedding decides what gets measured. It is drawn in the same axes on
    purpose: the gap between it and the offline point is the honest cost of
    that assumption, and hiding it in a different figure would be a way of not
    saying so.
    """
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot([0.75, 1.005], [0.75, 1.005], color="#bbb", ls=":", lw=1.2,
            label="nominal = empirical")
    lo = coverage.empirical - coverage.ci_lo
    hi = coverage.ci_hi - coverage.empirical
    ax.errorbar(coverage.nominal, coverage.empirical, yerr=np.vstack([lo, hi]),
                fmt="o", ms=6, lw=1.4, capsize=3, color=COLOURS[ADM],
                label="offline, held-out test split")
    for _, r in coverage.iterrows():
        ax.annotate(f"α={r.alpha:g}", xy=(r.nominal, r.empirical), xytext=(6, -10),
                    textcoords="offset points", fontsize=8, color="#555")
    for cfg, o in (online or {}).items():
        # Each arm sits at its own nominal level, because alpha is what the arms
        # differ in: the diamonds are not three readings of one promise.
        ax.plot([o["nominal"]], [o["empirical"]], marker="D", ms=7,
                color=COLOURS.get(cfg, "#B3452F"), ls="none",
                label=f"online: {LABELS.get(cfg, cfg)}  (n={o['n']})")
    ax.set_xlim(0.75, 1.005)
    ax.set_ylim(0.75, 1.005)
    _style(ax, "nominal coverage  (1 − α)", "empirical coverage",
           "Split-conformal coverage, measured",
           "points on or above the diagonal: the bound holds at least as often\n"
           "as claimed. Bars are Wilson 95% intervals.")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# --- 4. shed rate --------------------------------------------------------
def shed_vs_load(s: pd.DataFrame, out: Path, capacity: float) -> None:
    """Shed rate against offered load, with the theoretical minimum.

    A controller that refuses more than ``1 - capacity / lambda`` of the
    offered load is refusing work the machine could have served. A controller
    *below* the line is not virtuous -- it is admitting more than capacity, and
    the excess is showing up as violated deadlines somewhere else.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for cfg in arms(s):
        g = s[s.config == cfg].sort_values("rate_rps")
        ax.plot(g.rate_rps, g.shed_rate, marker="o", ms=5, lw=1.8,
                color=COLOURS.get(cfg), label=LABELS.get(cfg, cfg))
    lam = np.linspace(max(0.05, s.rate_rps.min()), s.rate_rps.max(), 200)
    ax.plot(lam, np.clip(1 - capacity / lam, 0, 1), color="#888", ls="--", lw=1.3,
            label=f"minimum for capacity {capacity:.2f} rps:  max(0, 1 − C/λ)")
    base = s[s.config == BASE].sort_values("rate_rps")
    if len(base):
        ax.plot(base.rate_rps, base.shed_rate, marker="s", ms=4, lw=1.4,
                color=COLOURS[BASE], label="no admission control (sheds nothing)")
    ax.set_ylim(-0.03, 1.0)
    _style(ax, "offered load (requests/s)", "fraction of arrivals shed (503)",
           "Shed rate against the theoretical minimum",
           "above the dashed line is work refused that could have been served;\n"
           "below it, more was admitted than the capacity estimate can serve in time")
    ax.legend(fontsize=8, frameon=False, loc="center left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# --- 5. the oscillation check -------------------------------------------
def admitted_over_time(df: pd.DataFrame, out: Path, rate: float, arm: str,
                       bin_s: float = 5.0) -> None:
    """Admitted and shed arrivals per second, over one overloaded run.

    Shedding is a positive feedback loop -- shed, load falls, predictions
    improve, admit, load rises, shed -- and a controller without damping
    oscillates between the two at a period set by how long a request takes.
    This is the figure that shows whether the hysteresis in
    ``ConformalController`` is doing its job; a sawtooth here would mean it is
    not.
    """
    sub = df[np.isclose(df.rate_rps, rate) & (df.config == arm)].copy()
    if not len(sub):
        return
    sub["bin"] = (sub.t_rel // bin_s) * bin_s
    grp = sub.groupby("bin")
    admitted = grp.apply(lambda g: float((g.status == 200).sum()) / bin_s, include_groups=False)
    shed = grp.apply(lambda g: float((g.status == 503).sum()) / bin_s, include_groups=False)

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.plot(admitted.index, admitted.values, lw=1.8, color=COLOURS.get(arm), label="admitted")
    ax.plot(shed.index, shed.values, lw=1.5, color="#B3452F", ls="--", label="shed (503)")
    ax.axhline(rate, color="#bbb", ls=":", lw=1.2)
    ax.annotate(f"offered {rate:g} rps", xy=(0.99, rate), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=8, color="#888")
    steady = df[df.steady]
    warmup = float(steady.t_rel.min()) if len(steady) else 0.0
    ax.axvspan(0, warmup, color="#eee", zorder=0)
    _style(ax, "time since start of run (s)", f"arrivals/s (mean over {bin_s:g}s bins)",
           f"Admission over one run at {rate:g} rps offered — {LABELS.get(arm, arm)}",
           "a sawtooth would mean the shed/admit loop is oscillating;\n"
           "shaded: warm-up window, excluded from every reported quantile")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# --- 6. is the bound adaptive, or just wide? -----------------------------
def bound_vs_realised(pred: pd.DataFrame, out: Path, slo: float) -> None:
    """The conformal bound against what actually happened, per request.

    Two things to read off it. Every point below the diagonal is a covered
    request, and the 1% that are above it are the violations the level allows.
    And the *spread* of the bound is the whole argument for conditional
    quantile regression: a uniform bound would be a horizontal line, admitting
    the same number of long requests on an idle server as on a saturated one.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    viol = pred.e2e_s > pred.u_s
    sc = axes[0].scatter(pred.u_s, pred.e2e_s, c=pred.queue_depth, cmap="viridis",
                         s=12, alpha=0.75, linewidths=0)
    lim = [0.05, max(pred.u_s.max(), pred.e2e_s.max()) * 1.1]
    axes[0].plot(lim, lim, color="#888", ls=":", lw=1.2)
    axes[0].scatter(pred.u_s[viol], pred.e2e_s[viol], facecolors="none",
                    edgecolors="#B3452F", s=42, lw=1.2,
                    label=f"{int(viol.sum())} of {len(pred)} above the bound")
    axes[0].axvline(slo, color="#999", ls="--", lw=1.1)
    axes[0].annotate("SLO: bound left of this line is admitted", xy=(slo, lim[1]),
                     xytext=(-6, -12), textcoords="offset points", rotation=90,
                     ha="right", va="top", fontsize=8, color="#666")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlim(*lim)
    axes[0].set_ylim(*lim)
    fig.colorbar(sc, ax=axes[0], label="queue depth at admission")
    _style(axes[0], "conformal upper bound U(x)  (s)", "realised end-to-end latency (s)",
           "Bound against outcome, held-out test split",
           "below the dotted line is covered")

    axes[1].hist(pred.u_s, bins=np.logspace(np.log10(max(0.05, pred.u_s.min())),
                                            np.log10(pred.u_s.max()), 50),
                 color=COLOURS[ADM], alpha=0.85)
    axes[1].axvline(slo, color="#999", ls="--", lw=1.1)
    axes[1].set_xscale("log")
    # Matplotlib's default log ticks collide at this range ("3x10^0 4x10^0"),
    # so the decade labels are replaced with plain seconds.
    ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50, 100, 200)
             if pred.u_s.min() * 0.8 <= t <= pred.u_s.max() * 1.2]
    axes[1].set_xticks(ticks)
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    _style(axes[1], "conformal upper bound U(x)  (s)", "requests",
           "The bound is conditional, not constant",
           "a single global quantile would be one vertical line here")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="the Week 4 sweep parquet")
    p.add_argument("--fit", default="results/w4_fit")
    p.add_argument("--slo", type=float, default=4.0)
    p.add_argument("--outdir", default="docs/figs")
    p.add_argument("--timeline-rate", type=float, default=None)
    a = p.parse_args(argv)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    fitdir = Path(a.fit)

    df = load(a.paths)
    s = summarize(df, slo_s=a.slo)

    # Capacity: the best goodput any configuration reached anywhere in the
    # sweep. Measured, not assumed, and deliberately the *most* generous
    # reading of what the machine can do, so the theoretical-minimum shed line
    # is the hardest version of that comparison.
    capacity = float(s.goodput_rps.max())
    shed_vs_load(s, out / "w4_shed_rate.png", capacity)

    rate = a.timeline_rate
    if rate is None:
        rate = float(df.rate_rps.max())
    # The arm that actually admits something: an oscillation plot of a
    # controller that refuses everything is a flat line at zero, and the
    # question the figure exists to answer is about the feedback loop.
    live = [c for c in arms(s) if s[s.config == c].admitted_rps.max() > 0]
    arm = max(live, key=lambda c: s[s.config == c].goodput_rps.max()) if live else ADM
    admitted_over_time(df, out / "w4_admitted_over_time.png", rate, arm)

    cov = fitdir / "coverage.csv"
    if cov.exists():
        online = None
        first = Path(a.paths[0])
        online_path = (first if first.is_dir() else first.parent) / "online_coverage.json"
        if online_path.exists():
            online = json.loads(online_path.read_text())
        coverage_calibration(pd.read_csv(cov), out / "w4_coverage.png", online)
    pred = fitdir / "test_predictions.parquet"
    if pred.exists():
        bound_vs_realised(pd.read_parquet(pred), out / "w4_bound_vs_realised.png", a.slo)

    print(f"wrote Week 4 figures to {out} (capacity estimate {capacity:.2f} rps, "
          f"timeline at {rate:g} rps)")


if __name__ == "__main__":
    main()
