"""Charts that carry the argument.

Each figure exists to defend one sentence, and nothing here smooths, clips or
re-bins in a way that would flatter a configuration. Quantiles come from the
load generator's raw records, never from a Prometheus histogram.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze import load, summarize  # noqa: E402

# One colour per rung, held fixed across every figure so the eye can carry a
# configuration from one chart to the next.
ORDER = ["fifo", "static", "continuous", "continuous+cache", "continuous+cache+admission"]
COLOURS = {
    "fifo": "#B3452F",
    "static": "#D68A2E",
    "continuous": "#3E7CB1",
    "continuous+cache": "#2E7D5B",
    "continuous+cache+admission": "#6B4E9B",
}
# The Week-2 knob experiments share the figure machinery with the ladder.
KNOB_COLOURS = {
    "chunk-128": "#2E7D5B",
    "chunk-256": "#3E7CB1",
    "chunked-prefill-512": "#D68A2E",
    "unchunked-prefill": "#B3452F",
    "prefill-first": "#3E7CB1",
    "decode-first": "#D68A2E",
}
KNOB_LABELS = {
    "chunk-128": "prefill budget 128 tokens",
    "chunk-256": "prefill budget 256 tokens",
    "chunked-prefill-512": "prefill budget 512 tokens",
    "unchunked-prefill": "unchunked (budget 2048 > prompt)",
    "prefill-first": "prefill before decode",
    "decode-first": "decode before prefill",
}

LABELS = {
    "fifo": "1  FIFO, no batching",
    "static": "2  static batching (8)",
    "continuous": "3  continuous batching",
    "continuous+cache": "4  + paged KV + prefix cache",
    "continuous+cache+admission": "5  + conformal admission",
}


COLOURS.update(KNOB_COLOURS)
LABELS.update(KNOB_LABELS)


def _style(ax, xlabel, ylabel, title, subtitle=""):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), loc="left", fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def _configs(s: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(s.config))
    return [c for c in ORDER if c in present] + [c for c in present if c not in ORDER]


def latency_vs_load(s: pd.DataFrame, out: Path, slo: float) -> None:
    """The Week 1 exit test and the project's headline claim: where is the knee,
    and does the p99 line stay flat past it."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, q, name in ((axes[0], "e2e_p50", "median"), (axes[1], "e2e_p99", "p99")):
        for cfg in _configs(s):
            g = s[s.config == cfg].sort_values("rate_rps")
            ax.plot(g.rate_rps, g[q], marker="o", ms=4.5, lw=1.8,
                    color=COLOURS.get(cfg), label=LABELS.get(cfg, cfg))
        ax.axhline(slo, color="#999", ls="--", lw=1.1)
        ax.annotate(f"SLO {slo:g}s", xy=(0.99, slo), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=8, color="#666")
        ax.set_yscale("log")
        _style(ax, "offered load (requests/s)", "end-to-end latency (s)",
               f"End-to-end {name}",
               "measured from intended arrival, so queueing is included")
    axes[1].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def goodput_vs_load(s: pd.DataFrame, out: Path, slo: float) -> None:
    """Throughput and goodput point in opposite directions under overload, and
    goodput is the one that matters.

    Three panels, because the first one is a trap worth showing rather than
    hiding. With no admission control and a client that waits up to 300 s,
    *completed* requests per second simply tracks the offered load: every rung
    eventually finishes everything, some of them minutes late. Read on its own
    it says all four configurations are identical -- and output tokens per
    second says the same thing, for the same reason. What the scheduler decides
    is *when*, not *whether*, so the middle panel asks how much of that
    completed work was still useful and the right panel multiplies the two.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharex=True)
    for cfg in _configs(s):
        g = s[s.config == cfg].sort_values("rate_rps")
        kw = dict(marker="o", ms=4.5, lw=1.8, color=COLOURS.get(cfg),
                  label=LABELS.get(cfg, cfg))
        axes[0].plot(g.rate_rps, g.throughput_rps, **kw)
        axes[1].plot(g.rate_rps, g.slo_attainment, **kw)
        axes[2].plot(g.rate_rps, g.goodput_rps, **kw)

    lim = s.rate_rps.max()
    axes[0].plot([0, lim], [0, lim], color="#bbb", ls=":", lw=1.1)
    axes[2].plot([0, lim], [0, lim], color="#bbb", ls=":", lw=1.1)
    _style(axes[0], "offered load (requests/s)", "requests/s", "Throughput",
           "completed requests -- tracks offered load for every rung,\n"
           "because nothing is shed and the client waits")
    axes[1].set_ylim(0, 1.03)
    _style(axes[1], "offered load (requests/s)", "fraction within SLO",
           f"SLO attainment ({slo:g}s)",
           "of everything that completed, how much was still useful")
    _style(axes[2], "offered load (requests/s)", "requests/s",
           f"Goodput (SLO {slo:g}s)",
           "requests completing within the SLO -- the only number\n"
           "that means anything under overload")
    axes[2].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def ttft_and_itl(s: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for cfg in _configs(s):
        g = s[s.config == cfg].sort_values("rate_rps")
        axes[0].plot(g.rate_rps, g.ttft_p99, marker="o", ms=4.5, lw=1.8,
                     color=COLOURS.get(cfg), label=LABELS.get(cfg, cfg))
        axes[1].plot(g.rate_rps, g.itl_p99, marker="o", ms=4.5, lw=1.8,
                     color=COLOURS.get(cfg), label=LABELS.get(cfg, cfg))
    axes[0].set_yscale("log")
    _style(axes[0], "offered load (requests/s)", "p99 TTFT (s)", "Time to first token, p99",
           "includes queueing: what a user perceives as 'did it start'")
    _style(axes[1], "offered load (requests/s)", "p99 ITL (s)", "Inter-token latency, p99",
           "streaming smoothness; spikes here are batch-step stalls")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def itl_histogram(df: pd.DataFrame, out: Path, group: str = "config",
                  title: str = "Inter-token latency",
                  subtitle: str = "") -> None:
    """The before/after view, in two panels.

    The left panel is the distribution; the right is the *tail*, plotted as a
    complementary CDF on log-log axes. The tail panel is not decoration: a
    batch-step stall is a rare event by construction, so on a density plot it
    is invisible at exactly the point where it matters. A p99 of 300 ms and a
    p99 of 20 ms look identical in a histogram and obviously different here.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    keys = (
        _configs(df.rename(columns={group: "config"}))
        if group == "config"
        else list(dict.fromkeys(df[group]))
    )
    allv = np.concatenate([np.asarray(v) for v in df.itl if len(v)]) if len(df) else np.array([])
    allv = allv[allv > 0] * 1e3 if allv.size else np.array([10.0, 100.0])
    lo, hi = max(1.0, allv.min() * 0.8), allv.max() * 1.3
    bins = np.logspace(np.log10(lo), np.log10(hi), 70)
    for k in keys:
        sub = df[df[group] == k]
        vals = (
            np.concatenate([np.asarray(v) for v in sub.itl if len(v)])
            if len(sub)
            else np.array([])
        )
        vals = vals[vals > 0] if vals.size else vals
        if not vals.size:
            continue
        colour = COLOURS.get(k)
        label = f"{LABELS.get(k, k)}  (p99 {np.quantile(vals, 0.99) * 1e3:.0f} ms)"
        axes[0].hist(vals * 1e3, bins=bins, histtype="step", lw=1.8, density=True,
                     color=colour, label=label)
        # CCDF: P(ITL > x). Plotted from the sorted sample, no binning.
        x = np.sort(vals) * 1e3
        y = 1.0 - np.arange(x.size) / x.size
        axes[1].plot(x, y, lw=1.8, color=colour, label=label)

    axes[0].set_xscale("log")
    axes[0].set_xlim(lo, hi)
    _style(axes[0], "inter-token latency (ms, log scale)", "density",
           title + " — distribution", subtitle)
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(lo, hi)
    axes[1].set_ylim(5e-5, 1.2)
    for q, lab in ((0.5, "p50"), (0.01, "p99"), (0.001, "p999")):
        axes[1].axhline(q, color="#ccc", ls=":", lw=1)
        axes[1].annotate(lab, xy=(0.995, q), xycoords=("axes fraction", "data"),
                         ha="right", va="bottom", fontsize=7.5, color="#888")
    _style(axes[1], "inter-token latency (ms, log scale)", "P(ITL > x)",
           title + " — tail",
           "a stall is a rare event; this is the panel it is visible in")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def latency_over_time(df: pd.DataFrame, out: Path, rate: float) -> None:
    """Does the queue reach steady state, or keep growing? A rising line past
    the warm-up window means the offered load exceeds capacity -- which is the
    open-loop signature, and the thing a closed-loop harness cannot show."""
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sub = df[np.isclose(df.rate_rps, rate)]
    for cfg in _configs(sub.assign(config=sub.config)):
        g = sub[sub.config == cfg].sort_values("t_rel")
        g = g[g.ok.fillna(False)]
        if not len(g):
            continue
        ax.plot(g.t_rel, g.e2e, ".", ms=3, alpha=0.55, color=COLOURS.get(cfg),
                label=LABELS.get(cfg, cfg))
    # Derive the warm-up window from the data rather than restating a constant:
    # the earliest arrival flagged steady is exactly where it ends.
    steady = df[df.steady]
    warmup = float(steady.t_rel.min()) if len(steady) else 0.0
    ax.axvspan(0, warmup, color="#eee", zorder=0)
    _style(ax, "time since start of run (s)", "end-to-end latency (s)",
           f"Latency over one run at {rate:g} rps",
           "shaded: warm-up window, excluded from every reported quantile")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def prefix_hit_rate(s: pd.DataFrame, out: Path) -> None:
    have = s[s.config.str.contains("cache")]
    if not len(have):
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for cfg in _configs(have):
        g = have[have.config == cfg].sort_values("rate_rps")
        if "prefix_hit_rate" not in g:
            continue
        ax.plot(g.rate_rps, g.prefix_hit_rate, marker="o", ms=5, lw=1.8,
                color=COLOURS.get(cfg), label=LABELS.get(cfg, cfg))
    ax.axhline(0.7, color="#999", ls="--", lw=1)
    ax.annotate("70% of requests share a system prompt", xy=(0.99, 0.705),
                xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=8, color="#666")
    ax.set_ylim(0, 1)
    _style(ax, "offered load (requests/s)", "hit rate (prompt tokens)",
           "Radix prefix cache, token-level hit rate",
           "token-level, not request-level: partial hits are the normal case")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--outdir", default="docs/figs")
    p.add_argument("--slo", type=float, default=4.0)
    p.add_argument("--timeline-rate", type=float, default=None)
    a = p.parse_args(argv)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = load(a.paths)
    s = summarize(df, slo_s=a.slo)
    if "prefix_hit_rate_server" in df:
        hits = df.groupby(["config", "rate_rps"]).prefix_hit_rate_server.last()
        s["prefix_hit_rate"] = [
            hits.get((c, r), float("nan")) for c, r in zip(s.config, s.rate_rps, strict=True)
        ]

    latency_vs_load(s, out / "latency_vs_load.png", a.slo)
    goodput_vs_load(s, out / "goodput_vs_load.png", a.slo)
    ttft_and_itl(s, out / "ttft_and_itl.png")
    prefix_hit_rate(s, out / "prefix_hit_rate.png")

    steady = df[df.steady]
    rate = a.timeline_rate
    if rate is None:
        # The most interesting rate is the one just past the best rung's knee.
        rates = sorted(df.rate_rps.unique())
        rate = float(rates[-2] if len(rates) > 1 else rates[-1])
    itl_histogram(
        steady[np.isclose(steady.rate_rps, rate)],
        out / "itl_histogram.png",
        subtitle=f"one run per configuration at {rate:g} rps offered load",
    )
    latency_over_time(df, out / "latency_over_time.png", rate)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
