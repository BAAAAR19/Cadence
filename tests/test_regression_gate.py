"""The CI load gate's own logic.

The gate is a piece of test infrastructure, which is exactly the kind of code
that is never tested and quietly stops working. The failure is silent in the
worst direction: a gate that always passes looks identical, on every PR, to a
gate that is working and a codebase with no regressions.

So the three things it must do -- fail a regression, pass an improvement,
refuse to judge a broken run -- are asserted here against synthetic
summaries, with no server and no sweep.
"""

from __future__ import annotations

import json

import pytest

from check_regression import compare, main, validate, write_baseline

BASELINE = {
    "config": "continuous+cache+admission",
    "slo_s": 2.0,
    "rates": {
        "4": {"e2e_p99": 1.00, "ttft_p99": 0.20, "goodput_rps": 4.0, "n": 180, "n_ok": 175},
        "8": {"e2e_p99": 1.60, "ttft_p99": 0.40, "goodput_rps": 6.0, "n": 360, "n_ok": 300},
    },
}


def _run(over: dict | None = None):
    """A run identical to the baseline, with the named metrics overridden.

    Keyed by ``(rate, metric)`` -- e.g. ``_run({("8", "goodput_rps"): 2.0})``.
    """
    rows = {rate: dict(vals) for rate, vals in BASELINE["rates"].items()}
    for (rate, metric), value in (over or {}).items():
        rows[rate][metric] = value
    return rows


META = {
    "config": "continuous+cache+admission",
    "slo_s": 2.0,
    "ks_p": {"4": 0.4, "8": 0.3},
    "sched_delay_p99": {"4": 0.004, "8": 0.010},
}


# --- what must pass -------------------------------------------------------


def test_an_identical_run_passes():
    failures, _ = compare(_run(), BASELINE, 0.15, 0.90)
    assert failures == []


def test_an_improvement_never_fails():
    """A p99 that halves is a result, not a build break. The response to it is
    a new baseline in a commit that explains itself."""
    rows = _run({("4", "e2e_p99"): 0.5, ("8", "e2e_p99"): 0.8,
                 ("4", "goodput_rps"): 6.0, ("8", "goodput_rps"): 9.0})
    failures, _ = compare(rows, BASELINE, 0.15, 0.90)
    assert failures == []


def test_noise_inside_the_tolerance_passes():
    """The runner is shared and the sleeps are not exact. A gate that fires on
    5% is a gate that gets switched off in a fortnight."""
    rows = _run({("4", "e2e_p99"): 1.10, ("8", "goodput_rps"): 5.7})
    failures, _ = compare(rows, BASELINE, 0.15, 0.90)
    assert failures == []


# --- what must fail -------------------------------------------------------


@pytest.mark.parametrize(
    "metric,value",
    [("e2e_p99", 1.30), ("ttft_p99", 0.30)],
)
def test_a_latency_regression_past_the_tolerance_fails(metric, value):
    rows = _run({("4", metric): value})
    failures, report = compare(rows, BASELINE, 0.15, 0.90)
    assert len(failures) == 1
    assert metric in failures[0]
    assert "FAIL" in report


def test_a_goodput_collapse_fails_even_with_a_flat_p99():
    """The interesting regression: a controller that starts shedding
    everything holds a beautiful p99 over the handful of requests it still
    admits. Latency alone would call that an improvement."""
    rows = _run({("8", "goodput_rps"): 2.0, ("8", "e2e_p99"): 0.9})
    failures, _ = compare(rows, BASELINE, 0.15, 0.90)
    assert len(failures) == 1
    assert "goodput_rps" in failures[0]


# --- what must not be judged at all ---------------------------------------


def test_a_run_with_too_few_arrivals_is_invalid_not_passing():
    rows = _run({("4", "n"): 5})
    problems = validate(rows, META, BASELINE)
    assert problems and "not a quantile" in problems[0]


def test_a_run_where_nothing_completed_is_invalid():
    rows = _run({("8", "n_ok"): 0})
    problems = validate(rows, META, BASELINE)
    assert any("nothing completed" in m for m in problems)


def test_a_starved_load_generator_is_invalid():
    """The runner, not the server, fell behind: the arrivals were issued late,
    so the run is closed-loop and its tail is optimistic. That is not a pass
    and it is not a regression -- it is not a measurement."""
    meta = META | {"sched_delay_p99": {"4": 0.004, "8": 0.9}}
    problems = validate(_run(), meta, BASELINE)
    assert any("open-loop" in m for m in problems)


def test_a_missing_rate_is_invalid():
    rows = {k: v for k, v in _run().items() if k != "8"}
    problems = validate(rows, META, BASELINE)
    assert any("not in the run" in m for m in problems)


def test_changing_the_configuration_invalidates_the_comparison():
    problems = validate(_run(), META | {"config": "continuous"}, BASELINE)
    assert any("configuration changed" in m for m in problems)


def test_changing_the_slo_invalidates_the_comparison():
    problems = validate(_run(), META | {"slo_s": 4.0}, BASELINE)
    assert any("SLO changed" in m for m in problems)


# --- the artefact ---------------------------------------------------------


def test_a_written_baseline_round_trips(tmp_path):
    path = tmp_path / "b.json"
    write_baseline(path, _run(), META, note="from a test")
    got = json.loads(path.read_text())
    assert got["config"] == META["config"]
    assert got["note"] == "from a test"
    assert set(got["rates"]) == {"4", "8"}
    # And the file it wrote is one the gate can read back.
    assert validate(_run(), META, got) == []
    assert compare(_run(), got, 0.15, 0.90)[0] == []


def test_a_missing_baseline_is_an_invalid_run_not_a_pass(tmp_path, capsys):
    """Exit 2, not 0. A gate that passes because it had nothing to compare
    against is the failure mode this whole file exists to prevent."""
    assert main([str(tmp_path / "nothing.parquet"),
                 "--baseline", str(tmp_path / "absent.json")]) == 2
