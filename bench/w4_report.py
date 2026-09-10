"""Week 4's tables, generated from the committed parquet and the fit.

    uv run bench/w4_report.py results/w4_admission --fit results/w4_fit \\
        --traces results/w4_admission/traces --slo 4.0 --outdir docs

Writes ``docs/admission.md``, ``docs/coverage.md`` and ``docs/predictor.md``,
which ``bench/embed_tables.py`` substitutes into the README between markers, so
no number in the writeup can drift from the run that produced it.

The online coverage number deserves a note, because it is the one that is easy
to compute wrongly. It is measured over requests the controller *admitted*, and
only over those that ran to completion: a request the client abandoned is a
censored observation of its own latency, and counting it as a violation would
punish the bound for the client's timeout while counting it as covered would
hide a real one. Both counts are reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analyze import load, summarize  # noqa: E402
from charts import LABELS, ORDER  # noqa: E402
from traces import load_traces  # noqa: E402

BASE = "continuous+cache"
ADM = "continuous+cache+admission"


ALPHA_OF = {
    ADM: 0.01,
    f"{ADM}-a05": 0.05,
    f"{ADM}-a20": 0.20,
}
"""The guarantee each arm was run at. Read from bench/configs.py's env, which
is recorded in every run's meta.json, and repeated here so that the coverage
each arm is judged against is the one it actually promised."""


def arms(s: pd.DataFrame) -> list[str]:
    """The admission arms present, in the ladder's own order.

    Plural because the guarantee level is the policy's knob: the sweep runs the
    same controller at several values of alpha, and a table with one of them in
    it would be a table of a choice rather than of a trade-off.
    """
    present = set(s.config)
    return [c for c in ORDER if c.startswith(ADM) and c in present]


def _f(v, nd=2, dash="-"):
    return dash if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def _pct(v, dash="-"):
    """An arm that admitted nothing has no attainment *among admitted*, and the
    honest rendering of that is an empty cell rather than ``nan%``."""
    return dash if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.0%}"


# --- the A/B ------------------------------------------------------------
def admission_table(s: pd.DataFrame, slo: float) -> str:
    """One row per (rung, offered load). The comparison the week exists for."""
    rows = []
    for cfg in [BASE, *arms(s)]:
        g = s[s.config == cfg].sort_values("rate_rps")
        for _, r in g.iterrows():
            rows.append(
                {
                    "Config": LABELS.get(cfg, cfg),
                    "Offered (rps)": f"{r.rate_rps:g}",
                    "Shed": _pct(r.shed_rate),
                    "Admitted (rps)": _f(r.admitted_rps),
                    "Goodput (rps)": _f(r.goodput_rps),
                    "SLO met, all arrivals": _pct(r.slo_attainment),
                    "SLO met, admitted": _pct(r.slo_attainment_admitted),
                    "p50 E2E (s)": _f(r.e2e_p50),
                    "p99 E2E (s)": _f(r.e2e_p99),
                    "p99 TTFT (s)": _f(r.ttft_p99),
                }
            )
    return pd.DataFrame(rows).to_markdown(index=False)


def headline(s: pd.DataFrame, slo: float) -> dict:
    """The numbers the prose quotes, computed once here rather than by hand."""
    out: dict = {}
    base = s[s.config == BASE].sort_values("rate_rps")
    if not len(base) or not arms(s):
        return out
    sat = float(base.loc[base.goodput_rps.idxmax()].rate_rps)
    out["saturation_rps"] = sat
    out["capacity_rps"] = float(s.goodput_rps.max())
    out["rates"] = sorted(base.rate_rps)
    out["arms"] = {}
    for cfg in arms(s):
        adm = s[s.config == cfg].sort_values("rate_rps")
        rates = sorted(set(base.rate_rps) & set(adm.rate_rps))
        top = max(rates)
        b = base[np.isclose(base.rate_rps, top)].iloc[0]
        a = adm[np.isclose(adm.rate_rps, top)].iloc[0]
        entry = {
            "top_rate": top,
            "top_rate_x_saturation": top / sat if sat else float("nan"),
            "top": {
                "base_p99": float(b.e2e_p99), "adm_p99": float(a.e2e_p99),
                "base_goodput": float(b.goodput_rps), "adm_goodput": float(a.goodput_rps),
                "base_slo": float(b.slo_attainment), "adm_slo": float(a.slo_attainment),
                "adm_slo_admitted": float(a.slo_attainment_admitted),
                "adm_shed": float(a.shed_rate),
            },
            "shed_rate_by_rate": {f"{r:g}": float(adm[np.isclose(adm.rate_rps, r)].iloc[0].shed_rate)
                                  for r in rates},
            "p99_by_rate": {f"{r:g}": float(adm[np.isclose(adm.rate_rps, r)].iloc[0].e2e_p99)
                            for r in rates},
            "goodput_by_rate": {f"{r:g}": float(adm[np.isclose(adm.rate_rps, r)].iloc[0].goodput_rps)
                                for r in rates},
            "max_p99_admitted": float(adm.e2e_p99.max()) if adm.e2e_p99.notna().any() else None,
        }
        over = [r for r in rates if r > sat]
        if over:
            bo = base[base.rate_rps.isin(over)]
            ao = adm[adm.rate_rps.isin(over)]
            entry["past_saturation"] = {
                "rates": over,
                "base_p99_max": float(bo.e2e_p99.max()),
                "adm_p99_max": float(ao.e2e_p99.max()) if ao.e2e_p99.notna().any() else None,
                "base_goodput_mean": float(bo.goodput_rps.mean()),
                "adm_goodput_mean": float(ao.goodput_rps.mean()),
                "goodput_ratio": float(ao.goodput_rps.mean() / bo.goodput_rps.mean())
                if bo.goodput_rps.mean()
                else float("nan"),
            }
        out["arms"][cfg] = entry
    out["base_goodput_by_rate"] = {
        f"{r:g}": float(base[np.isclose(base.rate_rps, r)].iloc[0].goodput_rps)
        for r in sorted(base.rate_rps)
    }
    out["base_p99_by_rate"] = {
        f"{r:g}": float(base[np.isclose(base.rate_rps, r)].iloc[0].e2e_p99)
        for r in sorted(base.rate_rps)
    }
    return out


# --- the predictor ------------------------------------------------------
def _q_label(fit: dict) -> str:
    """The calibration constant is not in seconds under every score.

    Under the deployed ratio score it is a *log*-ratio, so a column headed
    "Q (s)" would be three different units in one table and a reader would have
    no way to know.
    """
    return {
        "ratio": "Q (log-ratio)",
        "scaled": "Q (interval widths)",
        "absolute": "Q (s)",
    }.get(fit.get("score", "absolute"), "Q")


def score_table(fit: dict) -> str:
    """The same model and the same calibration fold under all three scores."""
    names = {
        "absolute": "absolute — `y − q_hi`, the build guide's",
        "ratio": "ratio — `log1p(y) − log1p(q_hi)`  *(deployed)*",
        "scaled": "scaled — `(y − q_hi) / (q_hi − q_lo)`",
    }
    rows = [
        {
            "Nonconformity score": names.get(r["score"], r["score"]),
            "Coverage": f"{r['coverage']:.4f}",
            "Q": f"{r['q_s']:.3f}",
            "Median bound U (s)": _f(r["median_bound_s"]),
            "Would admit": f"{r['admit_frac']:.1%}",
        }
        for r in sorted(fit.get("scores", []), key=lambda r: r["score"])
    ]
    return pd.DataFrame(rows).to_markdown(index=False) if rows else ""


def predictor_table(fit: dict) -> str:
    names = {
        "cqr": "Conformalised quantile regression",
        "throughput": "Throughput arithmetic (fitted)",
        "constant": "Constant quantile (no features)",
    }
    rows = []
    for k in ("cqr", "throughput", "constant"):
        m = fit["models"][k]
        rows.append(
            {
                "Model": names[k],
                "Coverage (target ≥ %.0f%%)" % (100 * (1 - fit["alpha"])): f"{m['coverage']:.3f}",
                _q_label(fit): _f(m["q_s"], 3),
                "Mean bound U (s)": _f(m["mean_bound_s"]),
                "Median bound U (s)": _f(m["median_bound_s"]),
                "Would admit": f"{m['admit_frac']:.1%}",
                "of those, met SLO": f"{m['slo_attainment_if_admitted']:.1%}"
                if np.isfinite(m["slo_attainment_if_admitted"])
                else "-",
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False)


def importance_table(fit: dict, top: int = 8) -> str:
    imp = list(fit["importances"].items())[:top]
    return pd.DataFrame(
        [{"Feature": k, "Importance": f"{v:.3f}"} for k, v in imp]
    ).to_markdown(index=False)


def coverage_table(fit: dict, online: dict[str, dict] | None) -> str:
    rows = []
    for r in fit["coverage"]:
        rows.append(
            {
                "α": f"{r['alpha']:g}",
                "Nominal (1−α)": f"{r['nominal']:.2f}",
                "Empirical, offline": f"{r['empirical']:.4f}",
                "95% CI": f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]",
                _q_label(fit): _f(r["q_s"], 3),
                "Mean U (s)": _f(r["mean_bound_s"]),
                "Would admit": f"{r['admit_frac']:.1%}",
            }
        )
    tbl = pd.DataFrame(rows).to_markdown(index=False)
    if online:
        live = pd.DataFrame(
            [
                {
                    "Arm": LABELS.get(cfg, cfg),
                    "Nominal (1−α)": f"{o['nominal']:.2f}",
                    "Empirical, online": f"{o['empirical']:.4f}",
                    "Admitted and completed": o["n"],
                    "Censored (client gave up)": o["n_censored"],
                    "Mean U (s)": _f(o["mean_bound_s"]),
                    "Mean realised E2E (s)": _f(o["mean_e2e_s"]),
                }
                for cfg, o in online.items()
            ]
        )
        tbl += (
            "\n\nAnd the same quantity measured while the controller was deciding — "
            "where the calibration set no longer describes what runs, because the "
            "controller chose it:\n\n" + live.to_markdown(index=False)
        )
    return tbl


def safety_table(fit: dict) -> str:
    rows = [
        {
            "Safety factor": f"{r['safety']:g}",
            "Would admit": f"{r['admit_frac']:.1%}",
            "of those, met SLO": f"{r['slo_attainment_if_admitted']:.1%}"
            if np.isfinite(r["slo_attainment_if_admitted"])
            else "-",
            "Offline goodput (admitted ∧ in SLO)": f"{r['goodput_frac']:.1%}",
        }
        for r in fit["safety"]
    ]
    return pd.DataFrame(rows).to_markdown(index=False)


def session_check(s: pd.DataFrame, w2_paths: list[str], slo: float) -> dict | None:
    """Rung 4 was measured twice: in Week 2's ladder and again here, as the
    control arm.

    Same workload, same seed, same duration, same SLO, same rung -- different
    day and a differently warm laptop. The two are not part of one interleaved
    run, so the honest thing is not to plot them on one chart but to say by how
    much they disagree at the loads they share, and let that number qualify
    every comparison drawn across the two sessions.
    """
    try:
        w2 = summarize(load(w2_paths), slo_s=slo)
    except SystemExit:
        return None
    a = s[s.config == BASE].set_index("rate_rps")
    b = w2[w2.config == BASE].set_index("rate_rps")
    shared = sorted(set(a.index) & set(b.index))
    if not shared:
        return None
    rows = {
        f"{r:g}": {
            "w2_goodput": float(b.loc[r].goodput_rps),
            "w4_goodput": float(a.loc[r].goodput_rps),
            "w2_e2e_p99": float(b.loc[r].e2e_p99),
            "w4_e2e_p99": float(a.loc[r].e2e_p99),
        }
        for r in shared
    }
    gd = [abs(v["w4_goodput"] - v["w2_goodput"]) / v["w2_goodput"] for v in rows.values()]
    return {
        "rates": shared,
        "per_rate": rows,
        "max_goodput_disagreement": float(max(gd)),
        "mean_goodput_disagreement": float(sum(gd) / len(gd)),
    }


# --- coverage as it actually came out -----------------------------------
def online_coverage(trace_dirs: list[str], alpha: float, config: str | None = None) -> dict | None:
    """Coverage of the bound over the requests the controller admitted.

    This is the number the theory does not cover: the calibration set was
    collected with admission off, and the controller changes what runs, so
    exchangeability is gone by construction. Measuring it anyway is the point.
    """
    try:
        raw = load_traces(trace_dirs, drop_warmup_s=0.0)
    except SystemExit:
        return None
    if config is not None:
        raw = raw[raw.config == config]
    adm = raw[(raw.action == "admit") & raw.u_bound_s.notna()]
    if not len(adm):
        return None
    done = adm[(~adm.censored.fillna(True)) & adm.e2e_s.notna()]
    if not len(done):
        return None
    covered = int((done.e2e_s <= done.u_bound_s).sum())
    return {
        "nominal": 1 - alpha,
        "empirical": covered / len(done),
        "n": int(len(done)),
        "n_censored": int(len(adm) - len(done)),
        "n_violations": int(len(done) - covered),
        "mean_bound_s": float(done.u_bound_s.mean()),
        "mean_e2e_s": float(done.e2e_s.mean()),
    }


def coverage_on_held_out_traces(trace_dirs: list[str], model: str, alpha: float) -> dict | None:
    """The fitted bound, checked against traces it has never seen.

    The rung-4 arm of the sweep runs with admission off and the trace log on,
    which makes it a second test set -- collected on a different day, at
    offered loads the training set did not include, by a process that had no
    part in the fit. Coverage there is the strongest version of the offline
    claim available without collecting a third time, and it is the one place
    the extrapolation to 4 and 6 rps gets checked.
    """
    from cadence.admission.artifact import PredictorArtifact
    from cadence.admission.conformal import SplitConformalUpperBound
    from cadence.admission.features import from_frame

    path = Path(model)
    if not path.exists():
        return None
    art = PredictorArtifact.load(path)
    bound = SplitConformalUpperBound(art.predictor, alpha=alpha).calibrate_scores(
        art.calib_scores
    )
    try:
        raw = load_traces(trace_dirs, drop_warmup_s=25.0)
    except SystemExit:
        return None
    base = raw[(raw.config == BASE) & (raw.action == "admit")]
    done = base[(~base.censored.fillna(True)) & base.e2e_s.notna()]
    if len(done) < 100:
        return None
    u = bound.upper_bound(from_frame(done))
    y = done.e2e_s.to_numpy(float)
    per_rate = {
        f"{r:g}": float(np.mean(y[done.rate_rps.to_numpy() == r] <=
                                u[done.rate_rps.to_numpy() == r]))
        for r in sorted(done.rate_rps.dropna().unique())
    }
    return {
        "nominal": 1 - alpha,
        "empirical": float(np.mean(y <= u)),
        "n": int(len(done)),
        "per_rate": per_rate,
        "rates_beyond_training": [r for r in per_rate if float(r) > 3.4],
    }



def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="the Week 4 sweep parquet")
    p.add_argument("--fit", default="results/w4_fit")
    p.add_argument("--traces", nargs="*", default=["results/w4_admission/traces"])
    p.add_argument("--slo", type=float, default=4.0)
    p.add_argument("--model", default="models/admission.pkl")
    p.add_argument("--w2", nargs="*", default=["results/w2_ladder"],
                   help="the Week 2 ladder, for the shared-rung sanity check")
    p.add_argument("--outdir", default="docs")
    a = p.parse_args(argv)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fit = json.loads((Path(a.fit) / "fit.json").read_text())

    df = load(a.paths)
    s = summarize(df, slo_s=a.slo)
    (outdir / "admission.md").write_text(admission_table(s, a.slo) + "\n")

    online: dict[str, dict] = {}
    if a.traces:
        for cfg in arms(s):
            alpha = ALPHA_OF.get(cfg, fit["alpha"])
            o = online_coverage(a.traces, alpha, config=cfg)
            if o:
                online[cfg] = o
        outp = Path(a.paths[0])
        outp = outp if outp.is_dir() else outp.parent
        (outp / "online_coverage.json").write_text(json.dumps(online, indent=2) + "\n")

    (outdir / "coverage.md").write_text(coverage_table(fit, online or None) + "\n")
    (outdir / "predictor.md").write_text(
        predictor_table(fit)
        + "\n\nThe quantile pair is fitted at "
        + f"{1 - fit.get('base_alpha', fit['alpha']) / 2:.0%} and the bound is "
        + f"calibrated to {1 - fit['alpha']:.0%}; the nonconformity score is "
        + f"`{fit.get('score', 'absolute')}`. Both were chosen on a held-out slice "
        + "of the training fold, and neither can affect validity — only width:\n\n"
        + score_table(fit)
        + "\n\nWhat the upper-quantile model leans on:\n\n"
        + importance_table(fit)
        + "\n\nSafety factor, read off the same held-out split:\n\n"
        + safety_table(fit)
        + "\n"
    )

    head = headline(s, a.slo)
    head["online_coverage"] = online
    head["session_check"] = session_check(s, a.w2, a.slo) if a.w2 else None
    head["held_out_coverage"] = (
        coverage_on_held_out_traces(a.traces, a.model, fit["alpha"]) if a.traces else None
    )
    head["offline_coverage"] = fit["models"]["cqr"]["coverage"]
    head["split_sensitivity"] = fit.get("split_sensitivity")
    (Path(a.fit) / "headline.json").write_text(json.dumps(head, indent=2) + "\n")

    print(json.dumps(head, indent=2))
    print(f"\nwrote {outdir}/admission.md, {outdir}/coverage.md, {outdir}/predictor.md")


if __name__ == "__main__":
    main()
