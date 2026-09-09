"""The analysis code produces every number in the writeup, so it is tested on
frames whose answers are known by construction.

The definitions that matter and are easy to get quietly wrong: throughput
counts completions, goodput counts only completions *within* the SLO, and both
are attributed to the steady-state arrival window rather than to whenever the
response happened to land.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analyze import across_seeds, summarize


def _frame(rows: list[dict], **common) -> pd.DataFrame:
    base = dict(
        config="c", workload="mixed", rate_rps=1.0, seed=0, slo_s=4.0,
        duration_s=100.0, warmup_s=0.0, cooldown_s=0.0, steady=True,
        status=200, ok=True, itl=[], n_tokens=10, shared_prompt=True,
    )
    out = []
    for i, r in enumerate(rows):
        d = {**base, **common, **r}
        d.setdefault("t_rel", float(i))
        d.setdefault("t_intended", float(i))
        d.setdefault("met_slo", bool(d["ok"]) and d["e2e"] <= d["slo_s"])
        out.append(d)
    return pd.DataFrame(out)


def test_goodput_counts_only_requests_inside_the_slo():
    # 10 requests over a 9 s span; 6 inside a 4 s SLO, 4 outside.
    df = _frame([{"e2e": 1.0, "ttft": 0.1} for _ in range(6)]
                + [{"e2e": 9.0, "ttft": 0.1} for _ in range(4)])
    s = summarize(df).iloc[0]
    assert s.n == 10
    assert s.throughput_rps == pytest.approx(10 / 9)
    assert s.goodput_rps == pytest.approx(6 / 9)
    assert s.slo_attainment == pytest.approx(0.6)


def test_slo_can_be_recomputed_after_the_fact():
    """The SLO is a choice; the raw records are the evidence. Changing the
    target must change the verdict without re-running anything."""
    df = _frame([{"e2e": e, "ttft": 0.1} for e in (1.0, 3.0, 5.0, 7.0)])
    assert summarize(df, slo_s=4.0).iloc[0].slo_attainment == pytest.approx(0.5)
    assert summarize(df, slo_s=8.0).iloc[0].slo_attainment == pytest.approx(1.0)
    assert summarize(df, slo_s=0.5).iloc[0].slo_attainment == pytest.approx(0.0)


def test_shed_and_error_requests_are_separated_and_never_counted_as_goodput():
    df = pd.concat([
        _frame([{"e2e": 1.0, "ttft": 0.1} for _ in range(5)]),
        _frame([{"e2e": 0.2, "ttft": None, "status": 503, "ok": False,
                 "met_slo": False} for _ in range(3)]),
        _frame([{"e2e": 30.0, "ttft": None, "status": "ReadTimeout", "ok": False,
                 "met_slo": False} for _ in range(2)]),
    ], ignore_index=True)
    s = summarize(df).iloc[0]
    assert s.n == 10 and s.n_ok == 5 and s.n_shed == 3 and s.n_error == 2
    assert s.shed_rate == pytest.approx(0.3)
    assert s.slo_attainment == pytest.approx(0.5)


def test_warmup_rows_are_excluded_but_still_present_in_the_raw_frame():
    df = _frame([{"e2e": 20.0, "ttft": 0.1, "steady": False} for _ in range(5)]
                + [{"e2e": 1.0, "ttft": 0.1} for _ in range(5)])
    assert summarize(df).iloc[0].n == 5, "warm-up rows leaked into the summary"
    assert summarize(df, steady_only=False).iloc[0].n == 10
    assert len(df) == 10, "the raw frame must keep everything"


def test_quantiles_come_from_successful_requests_only():
    df = _frame([{"e2e": 1.0, "ttft": 0.5} for _ in range(99)]
                + [{"e2e": 99.0, "ttft": None, "status": 503, "ok": False,
                    "met_slo": False}])
    s = summarize(df).iloc[0]
    assert s.e2e_p99 == pytest.approx(1.0), "a shed request inflated the latency tail"
    assert s.ttft_p50 == pytest.approx(0.5)


def test_itl_quantiles_pool_every_gap_not_every_request():
    """A long response contributes hundreds of gaps and a short one contributes
    a handful. Taking each request's own quantile and averaging those would
    weight the two requests equally -- here that would report a median of about
    0.5 s for a stream whose gaps are almost all 10 ms."""
    df = _frame(
        [{"e2e": 1.0, "ttft": 0.1, "itl": np.full(990, 0.01)}]
        + [{"e2e": 1.0, "ttft": 0.1, "itl": np.full(10, 1.0)}]
    )
    s = summarize(df).iloc[0]
    assert s.itl_p50 == pytest.approx(0.01), "quantile weighted by request, not by gap"
    # 1% of 1000 gaps sits exactly at the boundary, so the interpolated p99
    # lands between the bulk and the stall -- but strictly above the bulk.
    assert s.itl_p99 > s.itl_p50, "the slow gaps vanished from the tail entirely"

    naive = float(np.mean([np.median(np.full(990, 0.01)), np.median(np.full(10, 1.0))]))
    assert naive == pytest.approx(0.505)
    assert s.itl_p50 != pytest.approx(naive)


def test_across_seeds_reports_a_spread():
    df = pd.concat(
        [_frame([{"e2e": e, "ttft": 0.1} for _ in range(10)], seed=i)
         for i, e in enumerate((1.0, 2.0, 3.0))],
        ignore_index=True,
    )
    agg = across_seeds(summarize(df))
    assert len(agg) == 1
    assert agg.iloc[0].n_seeds == 3
    assert agg.iloc[0].e2e_p50_mean == pytest.approx(2.0)
    assert agg.iloc[0].e2e_p50_std > 0
