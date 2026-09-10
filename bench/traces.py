"""Reading the admission trace, and splitting it the way conformal needs.

Shared by ``fit_predictor.py`` and ``w4_report.py`` so that the split the model
was fitted on and the split the report describes cannot drift apart.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Both producers encode the offered load in the file name -- collect_traces.py
# as ``r0_rate1.2.jsonl`` and run_ladder.py as
# ``continuous_cache_admission_rate1.2_seed0.jsonl`` -- because it is the one
# thing the server cannot know: the gateway is told a configuration, not a rate.
_RATE = re.compile(r"rate(?P<rate>[0-9]+(?:\.[0-9]+)?)")
_ROUND = re.compile(r"^r(?P<round>\d+)_")
_SEED = re.compile(r"seed(?P<seed>\d+)")


def load_traces(paths: list[str | Path], drop_warmup_s: float = 20.0) -> pd.DataFrame:
    """Every trace row under ``paths``, with the block it came from attached.

    ``drop_warmup_s`` discards the opening seconds of each block. They are not
    representative of anything: the prefix cache is cold, the batch is empty
    and the arrival-rate estimator has not converged, so the rows describe a
    transient the controller will only ever see once per process start. The
    load generator drops the same window from the client side for the same
    reason.
    """
    rows: list[pd.DataFrame] = []
    for p in paths:
        p = Path(p)
        # Anything under a directory called ``contaminated`` is excluded by
        # name. Two blocks of this trace were re-collected because a test
        # suite was started on the same machine while they were running; the
        # originals are kept, because a discarded measurement that nobody can
        # see is indistinguishable from one that was never taken, but they are
        # not part of the training set.
        files = (
            sorted(f for f in p.rglob("*.jsonl") if "contaminated" not in f.parts)
            if p.is_dir()
            else [p]
        )
        for f in files:
            recs = [json.loads(line) for line in f.open() if line.strip()]
            if not recs:
                continue
            df = pd.DataFrame(recs)
            rate = _RATE.search(f.name)
            rnd = _ROUND.search(f.name)
            seed = _SEED.search(f.name)
            df["block"] = f.stem
            df["rate_rps"] = float(rate.group("rate")) if rate else np.nan
            df["round"] = int(rnd.group("round")) if rnd else 0
            df["seed"] = int(seed.group("seed")) if seed else 0
            df["t_rel"] = df["t_wall"] - df["t_wall"].min()
            rows.append(df[df.t_rel >= drop_warmup_s])
    if not rows:
        raise SystemExit(f"no trace rows under {paths}")
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values("t_wall").reset_index(drop=True)


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Admitted requests that ran to completion.

    Three exclusions, each of which would bias the fit in a knowable
    direction if it were left in:

    * shed rows have no latency at all;
    * censored rows -- the client hung up, or the request failed -- are lower
      bounds on their own latency, and treating a 300 s timeout as a 300 s
      completion teaches the model that the worst case is exactly the client's
      timeout;
    * rows with a null latency, which is the same thing without the timeout.
    """
    ok = df[(df.action == "admit") & (~df.censored.fillna(True)) & df.e2e_s.notna()]
    return ok.reset_index(drop=True)


def split_by_round(
    df: pd.DataFrame, n_train: int = 2, n_calib: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Whole collection rounds to whole folds -- the primary split.

    The build guide's rule is to split by time and never at random, because
    adjacent requests share queue conditions and a random split leaks system
    state across the fold boundary. That rule is right, and following it
    literally on this trace is still wrong, in two ways that pull in opposite
    directions.

    A *global* time split puts whole offered loads into whole folds, because
    the collection visits several of them; the folds then sample three
    different distributions and the exchangeability the conformal guarantee is
    made of is broken by the split rather than by the system. A split *within*
    each load fixes that but introduces a subtler failure, and it is worth
    stating because it is the one that actually bit: inside an overloaded
    block the queue grows monotonically, so the last 20% of the block is the
    part with the deepest queues, and a tree model asked to predict it is
    extrapolating past every split point it was fitted with. Coverage collapses
    -- not because conformal prediction failed, but because the calibration and
    test folds were not samples of the same thing.

    So the unit of exchangeability here is the *run*, not the request: each
    (round, load) block is an independent realisation of the same experiment,
    with its own arrival seed and its own freshly-started gateway, and whole
    blocks are assigned to folds. Requests inside a fold remain correlated with
    each other -- which widens the confidence interval on measured coverage,
    and is why that interval is reported -- but a calibration request and a
    test request never shared a queue, and both folds span the whole range of
    load and the whole life of a block.

    One honest caveat: rounds are visited in time order, so the test fold is
    the last-collected and therefore the warmest laptop. The rate order is
    rotated between rounds to spread that within a round, and the direction of
    the remaining bias is the conservative one -- coverage is checked against a
    fold that is, if anything, slower than the one it was calibrated on.
    """
    rounds = sorted(df["round"].unique())
    if len(rounds) < n_train + n_calib + 1:
        raise SystemExit(
            f"need at least {n_train + n_calib + 1} collection rounds to split by round; "
            f"the trace has {len(rounds)}. Re-run collect_traces.py with --rounds, or "
            f"pass --split time."
        )
    folds = (
        rounds[:n_train],
        rounds[n_train:n_train + n_calib],
        rounds[n_train + n_calib:],
    )
    return tuple(  # type: ignore[return-value]
        df[df["round"].isin(f)].sort_values("t_wall").reset_index(drop=True) for f in folds
    )


def split_by_time_within_rate(
    df: pd.DataFrame, train: float = 0.6, calib: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal 60/20/20 within each offered load -- the sensitivity check.

    Kept, and reported alongside the primary split, because it is the split the
    build guide's instruction most directly describes and because the
    difference between the two coverage numbers *is* a result: it measures how
    much of the guarantee depends on the folds being drawn from a stationary
    system. See :func:`split_by_round` for why this one is not the default.
    """
    parts: list[list[pd.DataFrame]] = [[], [], []]
    for _, g in df.groupby("rate_rps", dropna=False):
        g = g.sort_values("t_wall")
        n = len(g)
        a, b = int(train * n), int(calib * n)
        for i, chunk in enumerate((g.iloc[:a], g.iloc[a:b], g.iloc[b:])):
            parts[i].append(chunk)
    return tuple(  # type: ignore[return-value]
        pd.concat(p, ignore_index=True).sort_values("t_wall").reset_index(drop=True)
        for p in parts
    )
