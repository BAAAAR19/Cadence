"""parquet -> tables and charts.

Every quantile that reaches the README is computed here, from the load
generator's raw records, and never read off a Prometheus histogram: Prometheus
interpolates within a bucket, so its p99 is wrong by exactly the amount that
matters near the SLO.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILES = (0.5, 0.95, 0.99)


def load(paths: list[str | Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        p = Path(p)
        for f in sorted(p.glob("*.parquet")) if p.is_dir() else [p]:
            frames.append(pd.read_parquet(f))
    if not frames:
        raise SystemExit(f"no parquet files under {paths}")
    return pd.concat(frames, ignore_index=True)


def _q(series: pd.Series, q: float) -> float:
    s = series.dropna()
    return float(s.quantile(q)) if len(s) else float("nan")


def summarize(df: pd.DataFrame, steady_only: bool = True, slo_s: float | None = None) -> pd.DataFrame:
    """One row per (config, workload, rate, seed).

    ``throughput`` counts completed requests per second; ``goodput`` counts
    only those that completed *within* the SLO. Under overload the two point in
    opposite directions, and goodput is the one that matters: a gateway serving
    40 rps with a 12 s p99 against a 2 s SLO has a goodput of nearly zero.
    """
    if steady_only and "steady" in df:
        df = df[df.steady]
    if slo_s is not None:
        # Recompute rather than trust the value the run was recorded with, so
        # the SLO can be varied after the fact and the choice audited.
        df = df.copy()
        df["slo_s"] = slo_s
        df["met_slo"] = df.ok.fillna(False) & (df.e2e <= slo_s)
    rows = []
    keys = ["config", "workload", "rate_rps", "seed"]
    for key, g in df.groupby(keys, dropna=False):
        span = float(g.t_rel.max() - g.t_rel.min()) or float("nan")
        ok = g[g.ok.fillna(False)]
        row = dict(zip(keys, key, strict=True))
        row.update(
            {
                "n": len(g),
                "n_ok": len(ok),
                "n_shed": int((g.status == 503).sum()),
                "n_error": int((~g.ok.fillna(False) & (g.status != 503)).sum()),
                "offered_rps": len(g) / span if span else float("nan"),
                "throughput_rps": len(ok) / span if span else float("nan"),
                "goodput_rps": float(g.met_slo.sum()) / span if span else float("nan"),
                "slo_attainment": float(g.met_slo.mean()),
                "shed_rate": float((g.status == 503).mean()),
                "mean_output_tokens": float(ok.n_tokens.mean()) if len(ok) else float("nan"),
                "output_tok_per_s": float(ok.n_tokens.sum()) / span if span else float("nan"),
                "window_s": span,
            }
        )
        for metric in ("ttft", "e2e"):
            for q in QUANTILES:
                row[f"{metric}_p{int(q * 100)}"] = _q(ok[metric], q)
        # Pool every gap rather than averaging per-request quantiles: a
        # 200-token response contributes 199 gaps and a 2-token response
        # contributes 1, and weighting them equally would understate the tail.
        gaps = [np.asarray(v, dtype=float) for v in ok.itl if len(v)] if len(ok) else []
        itl = np.concatenate(gaps) if gaps else np.array([])
        for q in QUANTILES:
            row[f"itl_p{int(q * 100)}"] = float(np.quantile(itl, q)) if itl.size else float("nan")
        if "shared_prompt" in g and g.shared_prompt.notna().any():
            row["shared_fraction"] = float(g.shared_prompt.mean())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["config", "workload", "rate_rps", "seed"])
    return out.reset_index(drop=True)


def across_seeds(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread over seeds, which is what a table in the README should
    report -- a single seed's p99 on a laptop is not a number anyone should
    trust."""
    num = summary.select_dtypes("number").columns.drop("seed", errors="ignore")
    g = summary.groupby(["config", "workload", "rate_rps"])[list(num)]
    mean = g.mean().add_suffix("_mean")
    std = g.std(ddof=0).add_suffix("_std")
    n = g.size().rename("n_seeds")
    return pd.concat([mean, std, n], axis=1).reset_index()


def table(summary: pd.DataFrame) -> str:
    cols = [
        "config", "rate_rps", "n", "throughput_rps", "goodput_rps",
        "slo_attainment", "shed_rate",
        "ttft_p50", "ttft_p99", "itl_p50", "itl_p99", "e2e_p50", "e2e_p99",
    ]
    cols = [c for c in cols if c in summary]
    view = summary[cols].copy()
    for c in view.columns:
        if view[c].dtype.kind == "f":
            view[c] = view[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
    return view.to_markdown(index=False)


def main(argv=None) -> None:  # pragma: no cover - CLI
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", default=None, help="write the per-run summary as CSV")
    p.add_argument("--agg", action="store_true", help="aggregate across seeds")
    p.add_argument("--all", action="store_true", help="include warm-up/cool-down rows")
    p.add_argument("--slo", type=float, default=None,
                   help="recompute goodput and SLO attainment at this target")
    a = p.parse_args(argv)

    df = load(a.paths)
    s = summarize(df, steady_only=not a.all, slo_s=a.slo)
    print(table(s))
    if a.agg:
        print()
        agg = across_seeds(s)
        cols = ["config", "rate_rps", "n_seeds", "goodput_rps_mean", "goodput_rps_std",
                "e2e_p99_mean", "e2e_p99_std"]
        print(agg[[c for c in cols if c in agg]].to_markdown(index=False))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        s.to_csv(a.out, index=False)


if __name__ == "__main__":  # pragma: no cover
    main()
