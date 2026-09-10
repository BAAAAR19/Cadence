"""Fit the latency predictor, calibrate the bound, and check both.

    uv run bench/fit_predictor.py --traces results/w4_traces \\
        --alpha 0.01 --slo 4.0 --out models/admission.pkl

What it writes:

* ``models/admission.pkl``      the artifact the gateway loads
* ``results/w4_fit/fit.json``   every number the writeup quotes
* ``results/w4_fit/test_predictions.parquet``  per-request test-split output
* ``results/w4_fit/coverage.csv``  empirical vs nominal coverage

The part that matters is not the fit; it is the two checks around it.

**The model must beat the baselines.** Conformal calibration gives *any*
predictor the nominal coverage, so coverage cannot tell a good model from a
useless one -- what a good model buys is a tighter bound at the same guarantee,
and therefore more requests admitted under the same SLO. So the constant and
the throughput-arithmetic baselines are fitted, calibrated and evaluated the
same way, and the comparison is on bound width and on how many requests the
bound would let in.

**The guarantee must be validated, not asserted.** Coverage is measured on a
held-out test split at four nominal levels, with the binomial interval that
says whether the deviation from nominal is even distinguishable from sampling
noise at this sample size.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cadence.admission.artifact import PredictorArtifact  # noqa: E402
from cadence.admission.conformal import (  # noqa: E402
    SCORES,
    SplitConformalUpperBound,
    conformal_level,
)
from cadence.admission.features import FEATURES, from_frame  # noqa: E402
from cadence.admission.predictor import (  # noqa: E402
    ConstantQuantile,
    QuantileLatencyPredictor,
    ThroughputArithmetic,
    pinball_loss,
)
from srchash import source_hash  # noqa: E402
from traces import (  # noqa: E402
    load_traces,
    split_by_round,
    split_by_time_within_rate,
    usable,
)

ALPHAS = (0.20, 0.10, 0.05, 0.01)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Used rather than the normal approximation because the whole point is
    coverage near 0.99, where the normal interval runs off the end of [0, 1]
    and stops meaning anything.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (float(centre - half), float(centre + half))


def fit_one(kind: str, alpha: float, Xtr, ytr, Xca, yca, seed: int = 0,
            base_alpha: float | None = None, score: str = "ratio"):
    """Fit a model on train, calibrate it on calib. Returns the bound."""
    if kind == "cqr":
        model = QuantileLatencyPredictor(
            alpha=alpha, base_alpha=base_alpha, random_state=seed
        ).fit(Xtr, ytr)
    elif kind == "constant":
        model = ConstantQuantile(alpha=alpha).fit(Xtr, ytr)
    elif kind == "throughput":
        model = ThroughputArithmetic(alpha=alpha).fit(Xtr, ytr)
    else:  # pragma: no cover - argparse restricts this
        raise ValueError(kind)
    return SplitConformalUpperBound(model, alpha=alpha, score=score).calibrate(Xca, yca)


def select_base_and_score(alpha, Xtr, ytr, tr, slo_s, seed, frac=0.75):
    """Choose the base quantile level and the score function -- on the training
    fold alone.

    Neither choice can affect *validity*: conformal coverage holds for any base
    model and any score. Both decide *efficiency*, which on this workload is the
    difference between a controller that admits a third of the offered load and
    one that admits none of it, so leaving them at the guide's defaults would
    have been a decision too.

    Because they are chosen rather than fixed, they are chosen on a slice of the
    training fold that is held out of the model fit, and the calibration and
    test folds never enter. Selecting on the calibration fold would tune the
    same rows the bound is then calibrated against; selecting on test is
    reporting a number that was optimised for.

    The criterion is the operational one: offered requests that would be both
    admitted and inside the SLO -- an offline stand-in for goodput.
    """
    n = len(ytr)
    cut = int(frac * n)
    Xa, ya = Xtr[:cut], ytr[:cut]
    Xb, yb = Xtr[cut:], ytr[cut:]
    if len(yb) < 100:
        return None, "ratio", []
    rows = []
    # ``scaled`` is evaluated and reported but is not a candidate. Its
    # calibration constant is set by the narrowest interval in the fold -- the
    # easiest request in it -- so it moves by three orders of magnitude between
    # two folds of the same trace (Q = 1.3e3 on calibration here). A quantity
    # that unstable cannot be selected on; the instability is the finding.
    candidates = ("absolute", "ratio")
    for base in (0.5, 0.2, 0.1, 0.05, 0.01):
        model = QuantileLatencyPredictor(
            alpha=alpha, base_alpha=base, random_state=seed
        ).fit(Xa, ya)
        for score in candidates:
            b = SplitConformalUpperBound(model, alpha=alpha, score=score)
            # Calibrate and evaluate on the same held-out slice: this is a
            # selection criterion, not a reported number, and the alternative
            # is a fourth fold carved out of an already small trace.
            try:
                b.calibrate(Xb, yb)
            except ValueError:
                continue
            u = b.upper_bound(Xb)
            adm = u <= slo_s
            rows.append(
                {
                    "base_alpha": base,
                    "score": score,
                    "q": float(b.q_),
                    "median_bound_s": float(np.median(u)),
                    "admit_frac": float(np.mean(adm)),
                    "goodput_frac": float(np.mean(adm & (yb <= slo_s))),
                }
            )
    best = max(rows, key=lambda r: (r["goodput_frac"], -r["median_bound_s"]))
    return best["base_alpha"], best["score"], rows


def evaluate(bound, X, y, slo_s: float) -> dict:
    u = bound.upper_bound(X)
    _, hi = bound.predictor.predict_quantiles(X)
    covered = int(np.sum(y <= u))
    lo_ci, hi_ci = wilson(covered, len(y))
    admit = u <= slo_s
    return {
        "n": int(len(y)),
        "coverage": covered / len(y),
        "coverage_ci_lo": lo_ci,
        "coverage_ci_hi": hi_ci,
        "q_s": float(bound.q_),
        "mean_bound_s": float(np.mean(u)),
        "median_bound_s": float(np.median(u)),
        "pinball_hi": pinball_loss(y, hi, 1 - bound.alpha / 2),
        "mae_hi_s": float(np.mean(np.abs(y - hi))),
        # What the bound would actually do, which is the number that decides
        # whether a tighter model is worth anything: the share of requests it
        # would let in, and the share of *those* that in fact met the SLO.
        "admit_frac": float(np.mean(admit)),
        "slo_attainment_if_admitted": (
            float(np.mean(y[admit] <= slo_s)) if admit.any() else float("nan")
        ),
        "slo_attainment_overall": float(np.mean(y <= slo_s)),
    }


def calibration_domain(paths: list[str]) -> dict:
    """The conditions the calibration set was drawn under.

    Split conformal guarantees coverage on data exchangeable with the
    calibration scores, and nothing else. Change the backend, the machine, the
    context size or the workload and the scores describe a distribution the
    live traffic is not drawn from -- at which point the bound is not
    conservative, it is arbitrary. That is not hypothetical: the artifact
    fitted on llama.cpp with Metal, loaded against the mock backend, refuses
    100% of requests at zero load, because a bound learned from 4-second
    latencies is nonsense about 0.4-second ones.

    So the conditions travel with the model, read from the collection's own
    meta.json rather than from whatever this process happens to be configured
    with, and ``cadence.admission.conformal_controller`` checks them at
    start-up.
    """
    env: dict = {}
    source = None
    for raw_path in paths:
        p = Path(raw_path)
        for meta_path in sorted(p.glob("meta*.json")) if p.is_dir() else []:
            meta = json.loads(meta_path.read_text())
            env = meta.get("env", {}) or {}
            source = str(meta_path)
            break
        if env:
            break
    keys = (
        "CADENCE_BACKEND", "CADENCE_N_CTX", "CADENCE_MAX_BATCH",
        "CADENCE_BLOCK_SIZE", "CADENCE_MAX_TOKENS_CAP", "CADENCE_KV_CORE",
    )
    return {
        "source": source,
        "backend": env.get("CADENCE_BACKEND"),
        "env": {k: env[k] for k in keys if k in env},
        "platform": f"{platform.system()}-{platform.machine()}",
        "python": platform.python_version(),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", nargs="+", default=["results/w4_traces"])
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--slo", type=float, default=4.0)
    p.add_argument("--drop-warmup", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-alpha", type=float, default=None,
                   help="level the quantile pair is fitted at. Default: chosen "
                        "on a held-out slice of the training fold, see "
                        "select_base_and_score.")
    p.add_argument("--score", default=None, choices=sorted(SCORES),
                   help="nonconformity score. Default: chosen the same way.")
    p.add_argument("--split", choices=("round", "time"), default="round",
                   help="round: whole collection rounds to whole folds, the "
                        "episode-level exchangeability this trace actually has. "
                        "time: 60/20/20 by time within each offered load.")
    p.add_argument("--out", default="models/admission.pkl")
    p.add_argument("--outdir", default="results/w4_fit")
    a = p.parse_args(argv)

    domain = calibration_domain(a.traces)
    raw = load_traces(a.traces, drop_warmup_s=a.drop_warmup)
    df = usable(raw)
    split = split_by_round if a.split == "round" else split_by_time_within_rate
    tr, ca, te = split(df)
    Xtr, ytr = from_frame(tr), tr.e2e_s.to_numpy(float)
    Xca, yca = from_frame(ca), ca.e2e_s.to_numpy(float)
    Xte, yte = from_frame(te), te.e2e_s.to_numpy(float)

    censored = int(raw[(raw.action == "admit")].censored.fillna(True).sum())
    print(
        f"{len(raw)} rows -> {len(df)} usable "
        f"({int((raw.action == 'shed').sum())} shed, {censored} censored); "
        f"train {len(tr)} / calib {len(ca)} / test {len(te)}",
        flush=True,
    )
    if len(ca) < 100:
        raise SystemExit(
            f"calibration split has {len(ca)} rows; a {1 - a.alpha:.0%} bound needs "
            f"at least {int(np.ceil((1 - a.alpha) / a.alpha))}. Collect more trace."
        )

    # --- how the bound is built: chosen on the training fold ---------------
    base_alpha, score, selection = select_base_and_score(
        a.alpha, Xtr, ytr, tr, a.slo, a.seed
    )
    if a.base_alpha is not None:
        base_alpha = a.base_alpha
    if a.score is not None:
        score = a.score
    print(f"  base quantile level {1 - (base_alpha or a.alpha) / 2:.3f} "
          f"(base_alpha={base_alpha}), score={score}", flush=True)

    # --- the model, and the two baselines, all calibrated identically -----
    models = {}
    for kind in ("cqr", "constant", "throughput"):
        b = fit_one(kind, a.alpha, Xtr, ytr, Xca, yca, seed=a.seed,
                    base_alpha=base_alpha, score=score)
        models[kind] = b
        m = evaluate(b, Xte, yte, a.slo)
        print(
            f"  {kind:<11} coverage={m['coverage']:.4f} "
            f"mean_U={m['mean_bound_s']:7.2f}s  admit={m['admit_frac']:.3f} "
            f"pinball={m['pinball_hi']:.4f}",
            flush=True,
        )

    cqr = models["cqr"]
    results = {kind: evaluate(b, Xte, yte, a.slo) for kind, b in models.items()}

    # --- coverage against nominal, the figure the guarantee is checked on --
    coverage_rows = []
    for alpha in ALPHAS:
        b = fit_one("cqr", alpha, Xtr, ytr, Xca, yca, seed=a.seed,
                    base_alpha=base_alpha, score=score)
        m = evaluate(b, Xte, yte, a.slo)
        coverage_rows.append(
            {
                "alpha": alpha,
                "nominal": 1 - alpha,
                "empirical": m["coverage"],
                "ci_lo": m["coverage_ci_lo"],
                "ci_hi": m["coverage_ci_hi"],
                "q_s": m["q_s"],
                "mean_bound_s": m["mean_bound_s"],
                "admit_frac": m["admit_frac"],
                "n_calib": int(len(ca)),
                "level": conformal_level(len(ca), alpha),
                "n_test": m["n"],
            }
        )
        print(
            f"  alpha={alpha:<5} nominal={1 - alpha:.2f} "
            f"empirical={m['coverage']:.4f} [{m['coverage_ci_lo']:.4f}, "
            f"{m['coverage_ci_hi']:.4f}]  mean_U={m['mean_bound_s']:.2f}s",
            flush=True,
        )
    coverage = pd.DataFrame(coverage_rows)

    # --- the same model under all three score functions --------------------
    # Reported rather than selected on: the choice was made on the training
    # fold above. What this table is for is the claim that validity is
    # indifferent to the score and efficiency is not, which is only worth
    # making if the numbers are on the record.
    score_rows = []
    for sc in sorted(SCORES):
        b = fit_one("cqr", a.alpha, Xtr, ytr, Xca, yca, seed=a.seed,
                    base_alpha=base_alpha, score=sc)
        m = evaluate(b, Xte, yte, a.slo)
        score_rows.append({"score": sc, **m})
        print(f"  score={sc:<9} coverage={m['coverage']:.4f} Q={m['q_s']:.3f} "
              f"median_U={m['median_bound_s']:7.2f}s admit={m['admit_frac']:.3f}",
              flush=True)

    # --- what a safety factor would do, offline ---------------------------
    # The live sweep varies one thing at a time and costs an hour a point; this
    # is the same trade-off read off the test split for free, and it is what
    # the live points are checked against.
    u_test = cqr.upper_bound(Xte)
    safety_rows = []
    for s in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5):
        adm = u_test * s <= a.slo
        safety_rows.append(
            {
                "safety": s,
                "admit_frac": float(np.mean(adm)),
                "slo_attainment_if_admitted": (
                    float(np.mean(yte[adm] <= a.slo)) if adm.any() else float("nan")
                ),
                # Requests admitted *and* inside the SLO, per request offered:
                # the offline analogue of goodput.
                "goodput_frac": float(np.mean(adm & (yte <= a.slo))),
            }
        )
    safety = pd.DataFrame(safety_rows)

    # --- the same fit under the other split convention ---------------------
    # Not a robustness ritual: the gap between these two numbers is the price
    # of the folds not being samples of a stationary system, and quoting it is
    # the difference between a guarantee that was checked and one that was
    # assumed.
    other = split_by_time_within_rate if a.split == "round" else split_by_round
    tr2, ca2, te2 = other(df)
    b2 = fit_one("cqr", a.alpha, from_frame(tr2), tr2.e2e_s.to_numpy(float),
                 from_frame(ca2), ca2.e2e_s.to_numpy(float), seed=a.seed,
                 base_alpha=base_alpha, score=score)
    alt = evaluate(b2, from_frame(te2), te2.e2e_s.to_numpy(float), a.slo)
    alt["split"] = "time" if a.split == "round" else "round"
    print(
        f"  [{alt['split']} split] coverage={alt['coverage']:.4f} "
        f"mean_U={alt['mean_bound_s']:.2f}s  (primary split: {a.split})",
        flush=True,
    )

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(outdir / "coverage.csv", index=False)
    safety.to_csv(outdir / "safety.csv", index=False)

    lo_te, hi_te = cqr.predictor.predict_quantiles(Xte)
    pd.DataFrame(
        {
            "t_wall": te.t_wall.to_numpy(),
            "rate_rps": te.rate_rps.to_numpy(),
            "e2e_s": yte,
            "q_lo_s": lo_te,
            "q_hi_s": hi_te,
            "u_s": u_test,
            "max_tokens": te.max_tokens.to_numpy(),
            "n_output_tokens": te.n_output_tokens.to_numpy(),
            "queue_depth": te.queue_depth.to_numpy(),
            "n_cached_prefix_tokens": te.n_cached_prefix_tokens.to_numpy(),
        }
    ).to_parquet(outdir / "test_predictions.parquet")

    fit = {
        "alpha": a.alpha,
        "fitted_on": domain,
        "base_alpha": base_alpha,
        "score": score,
        "selection": selection,
        "scores": score_rows,
        "slo_s": a.slo,
        "split": a.split,
        "split_sensitivity": alt,
        "src_hash": source_hash(),
        "traces": [str(t) for t in a.traces],
        "n_rows_raw": int(len(raw)),
        "n_shed_rows": int((raw.action == "shed").sum()),
        "n_censored": censored,
        "n_train": int(len(tr)),
        "n_calib": int(len(ca)),
        "n_test": int(len(te)),
        "conformal_level": conformal_level(len(ca), a.alpha),
        "rates": sorted(df.rate_rps.dropna().unique().tolist()),
        "models": results,
        "importances": models["cqr"].predictor.importances(),
        "y_quantiles": {
            str(q): float(np.quantile(yte, q)) for q in (0.5, 0.9, 0.99, 1.0)
        },
        "coverage": coverage_rows,
        "safety": safety_rows,
    }
    (outdir / "fit.json").write_text(json.dumps(fit, indent=2) + "\n")

    art = PredictorArtifact(
        predictor=cqr.predictor,
        # In the order the calibration fold was collected in, because the
        # rolling mode seeds its window from the most recent of them.
        calib_scores=cqr.scores_in_order_,
        alpha=a.alpha,
        score=score,
        feature_names=tuple(FEATURES),
        meta={
            "src_hash": fit["src_hash"],
            # The domain the bound is valid on. A conformal guarantee is a
            # statement about exchangeability with the calibration set, so an
            # artifact carried to a machine or a backend whose latency
            # distribution is different is not a conservative bound -- it is
            # an arbitrary one. Recorded here so that the controller can say
            # so at start-up instead of silently shedding every request, which
            # is what a Metal-fitted model does on a CPU container.
            "fitted_on": domain,
            "base_alpha": base_alpha,
            "traces": fit["traces"],
            "n_train": fit["n_train"],
            "n_calib": fit["n_calib"],
            "slo_s": a.slo,
            "test_coverage": results["cqr"]["coverage"],
            "test_mean_bound_s": results["cqr"]["mean_bound_s"],
        },
    )
    art.save(a.out)
    print(f"\nwrote {a.out} and {outdir}/fit.json", flush=True)


if __name__ == "__main__":
    main()
