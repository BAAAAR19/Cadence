"""Week 3's three tables: the profile, the microbenchmark, the end-to-end A/B.

Same contract as ``make_report.py``: committed inputs, generated tables, no
number typed into the README by hand. ``bench/embed_tables.py --check``
enforces it in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import flamegraph as fg  # noqa: E402
from analyze import load, summarize  # noqa: E402

ARMS = {"kv-core-python": "Python reference", "kv-core-cpp": "C++17 core"}


def _pct(x: float | None) -> str:
    return f"{100 * x:.3f}%" if x else "n/a"


# --- the profile ----------------------------------------------------------
def profile_table(meta_paths: dict[str, Path]) -> str:
    """What the scheduler thread spends its time on, per KV core.

    Reported inclusive (how much of a step involves this component at all) and
    self (how much is spent executing it); for a leaf data structure the two
    are nearly equal, and a large gap between them means the buckets overlap.
    """
    rows: list[dict] = []
    for arm, path in meta_paths.items():
        if not path.exists():
            continue
        meta = json.loads(path.read_text())
        run = meta["runs"][0]
        for name, v in sorted(run["buckets"].items(), key=lambda kv: -kv[1]["inclusive_pct"]):
            rows.append(
                {
                    "KV core": arm,
                    "Component": name,
                    "Inclusive": f"{v['inclusive_pct']:.2f}%",
                    "Self": f"{v['self_pct']:.2f}%",
                }
            )
    return pd.DataFrame(rows).to_markdown(index=False)


def profile_notes(meta_paths: dict[str, Path]) -> str:
    out = []
    for arm, path in meta_paths.items():
        if not path.exists():
            continue
        meta = json.loads(path.read_text())
        run = meta["runs"][0]
        s, st = run["sampler"], run.get("steps") or {}
        mean_ms = f"{1000 * st['mean_step_s']:.1f} ms" if st.get("mean_step_s") else "n/a"
        out.append(
            f"* **{arm}**: {s['n_samples']:,} samples at {s['hz_achieved']:.0f} Hz over "
            f"{run['sampled_s']:.0f} s, of which the scheduler thread was busy "
            f"{100 * run['busy_frac']:.0f}%. {st.get('n_steps', 0):,} engine steps, "
            f"mean {mean_ms}. {run['completed']} completed requests at "
            f"{run['rate_rps']:g} rps."
        )
    return "\n".join(out)


# --- the microbenchmark ---------------------------------------------------
def bench_tables(path: Path) -> tuple[str, str, str]:
    b = json.loads(path.read_text())
    match = pd.DataFrame(
        [
            {
                "Prompt tokens": r["prompt_tokens"],
                "Python (us)": f"{r['python']:.2f}",
                "C++ (us)": f"{r['cpp']:.2f}",
                "of which pybind11 marshalling": f"{r['cpp_marshalling']:.2f}",
                "Speedup": f"{r['speedup']:.1f}x",
            }
            for r in b["match_us_by_prompt_tokens"]
        ]
    ).to_markdown(index=False)

    a = b["allocator"]
    alloc = pd.DataFrame(
        [
            {
                "Operation": label,
                "Python (us)": f"{a['python'][k]:.3f}",
                "C++ (us)": f"{a['cpp'][k]:.3f}",
                "Speedup": f"{a['speedup'][k]:.1f}x",
            }
            for k, label in (
                ("can_append_us", "can_append (once per sequence per token)"),
                ("alloc_release_34_us", "alloc + release of a 34-block table"),
            )
        ]
    ).to_markdown(index=False)

    g = b["gil"]
    gil = pd.DataFrame(
        [
            {
                "Measurement": "a bound no-op, GIL held throughout",
                "ns per call": f"{g['call_ns']:.0f}",
            },
            {
                "Measurement": "the same no-op, GIL released and re-acquired",
                "ns per call": f"{g['call_with_gil_release_ns']:.0f}",
            },
            {
                "Measurement": "cost of the release/re-acquire pair",
                "ns per call": f"{g['release_cost_ns']:.0f}",
            },
        ]
    ).to_markdown(index=False)

    s = b["step_share"]
    step = pd.DataFrame(
        [
            {
                "Running batch": row["batch"],
                "Python": f"{row['python']['per_step_us']:.1f} us",
                "C++": f"{row['cpp']['per_step_us']:.1f} us",
                "Python, share of a step": _pct(row["python"]["share_of_step"]),
                "C++, share of a step": _pct(row["cpp"]["share_of_step"]),
            }
            for row in s["by_batch"]
        ]
    ).to_markdown(index=False)
    return match, alloc, step, gil


# --- the end-to-end A/B ---------------------------------------------------
def ab_table(paths: list[str], slo: float) -> str:
    df = load(paths)
    s = summarize(df, slo_s=slo)
    cols = {
        "config": "KV core", "rate_rps": "Offered (rps)",
        "throughput_rps": "Throughput", "goodput_rps": "Goodput",
        "slo_attainment": "SLO met", "ttft_p50": "TTFT p50",
        "itl_p99": "ITL p99", "e2e_p50": "E2E p50", "e2e_p99": "E2E p99",
    }
    v = s[s.config.isin(ARMS)].copy()
    v = v.sort_values(["rate_rps", "config"])[list(cols)].rename(columns=cols)
    v["KV core"] = v["KV core"].map(ARMS)
    for c in v.columns:
        if v[c].dtype.kind == "f":
            v[c] = v[c].map(lambda x: "-" if pd.isna(x) else f"{x:.3f}")
    return v.to_markdown(index=False)


def ab_deltas(paths: list[str], slo: float) -> str:
    """The line that matters: cpp relative to python, per rate.

    Reported next to the run-to-run spread the Week 2 replicates measured
    (1.7%), because a delta smaller than the noise is not a delta.
    """
    s = summarize(load(paths), slo_s=slo)
    rows = []
    for rate, g in s[s.config.isin(ARMS)].groupby("rate_rps"):
        py = g[g.config == "kv-core-python"]
        cpp = g[g.config == "kv-core-cpp"]
        if py.empty or cpp.empty:
            continue
        py, cpp = py.iloc[0], cpp.iloc[0]
        row = {"Offered (rps)": f"{rate:g}"}
        for label, col in (
            ("Goodput", "goodput_rps"), ("Throughput", "throughput_rps"),
            ("TTFT p50", "ttft_p50"), ("ITL p99", "itl_p99"), ("E2E p99", "e2e_p99"),
        ):
            a, b = float(py[col]), float(cpp[col])
            row[label] = "-" if not a else f"{100 * (b - a) / a:+.1f}%"
        rows.append(row)
    return pd.DataFrame(rows).to_markdown(index=False)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ab", default="results/w3_ab", help="directory of A/B parquet")
    p.add_argument("--profile", default="results/w3_profile")
    p.add_argument("--bench", default="results/w3_bench/bench.json")
    p.add_argument("--slo", type=float, default=4.0)
    p.add_argument("--outdir", default="docs")
    p.add_argument("--figdir", default="docs/figs")
    a = p.parse_args(argv)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    prof = Path(a.profile)

    metas = {
        "Python reference": prof / "python" / "meta.json",
        "C++17 core": prof / "cpp" / "meta.json",
    }

    # The flame graph is rendered from the committed folded stacks rather than
    # copied from wherever the profiling run left it, so `git diff --exit-code
    # docs` in CI is a real check on the figure too.
    figs = Path(a.figdir)
    figs.mkdir(parents=True, exist_ok=True)
    for arm, sub in (("python", "python"), ("cpp", "cpp")):
        folded = prof / sub / "llamacpp.folded"
        if not folded.exists():
            continue
        (figs / f"w3_flamegraph_{arm}.svg").write_text(
            fg.render_svg(
                fg.build_tree(fg.read_folded(folded)),
                f"cadence scheduler thread - {arm} KV core, 1.9 rps",
            )
        )
    (out / "core_profile.md").write_text(
        profile_table(metas) + "\n\n" + profile_notes(metas) + "\n"
    )

    match, alloc, step, gil = bench_tables(Path(a.bench))
    ms = json.loads(Path(a.bench).read_text())["step_share"]["mean_step_s"]
    mean_step = f"{1000 * ms:.1f} ms, measured" if ms else "not measured"
    (out / "core_bench.md").write_text(
        "**Longest-prefix match, by prompt length**\n\n" + match
        + "\n\n**Allocator operations**\n\n" + alloc
        + f"\n\n**What that is as a share of one engine step** (mean step {mean_step})"
        + "\n\n" + step
        + "\n\n**The price of releasing the GIL, which is why the port does not**"
          "\n\n" + gil + "\n"
    )

    ab_paths = sorted(str(p) for p in Path(a.ab).glob("*.parquet"))
    (out / "core_ab.md").write_text(
        ab_table(ab_paths, a.slo) + "\n\n**C++ relative to Python**\n\n"
        + ab_deltas(ab_paths, a.slo) + "\n"
    )
    print(f"wrote {out}/core_profile.md, {out}/core_bench.md, {out}/core_ab.md")


if __name__ == "__main__":
    main()
