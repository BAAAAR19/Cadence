"""The two Week-2 knob experiments, reported as a table and an ITL figure.

Both are trade-offs rather than wins, which is the point: finding the knob and
quantifying what it costs on the other axis is worth more than picking a
default and asserting it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from analyze import load, summarize  # noqa: E402
from charts import KNOB_LABELS, itl_histogram  # noqa: E402

PAIRS = [
    ("Chunked prefill", ["chunk-128", "chunk-256", "chunked-prefill-512",
                         "unchunked-prefill"]),
    ("Prefill/decode order", ["prefill-first", "decode-first"]),
    ("Replicates of the unmodified configuration",
     ["replicate-1", "replicate-2", "replicate-3"]),
]


def table(s: pd.DataFrame, arms: list[str]) -> str:
    cols = {
        "config": "Arm", "goodput_rps": "Goodput", "slo_attainment": "SLO met",
        "ttft_p50": "TTFT p50", "ttft_p99": "TTFT p99",
        "itl_p50": "ITL p50", "itl_p99": "ITL p99",
        "e2e_p50": "E2E p50", "e2e_p99": "E2E p99",
    }
    v = s[s.config.isin(arms)].copy()
    v["_k"] = v.config.map({a: i for i, a in enumerate(arms)})
    v = v.sort_values("_k")[list(cols)].rename(columns=cols)
    v["Arm"] = v["Arm"].map(lambda a: KNOB_LABELS.get(a, a))
    for c in v.columns:
        if v[c].dtype.kind == "f":
            v[c] = v[c].map(lambda x: "-" if pd.isna(x) else f"{x:.3f}")
    return v.to_markdown(index=False)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--slo", type=float, default=4.0)
    p.add_argument("--outdir", default="docs")
    p.add_argument("--figdir", default="docs/figs")
    a = p.parse_args(argv)

    out, figs = Path(a.outdir), Path(a.figdir)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    df = load(a.paths)
    s = summarize(df, slo_s=a.slo)
    steady = df[df.steady]
    rate = float(df.rate_rps.iloc[0])

    blocks = []
    for title, arms in PAIRS:
        present = [x for x in arms if x in set(s.config)]
        if len(present) < 2:
            continue
        if title.startswith("Replicates"):
            blocks.append(
                f"**{title}** — identical settings, {len(present)} runs, "
                f"{rate:g} rps offered load. The spread here is the floor any "
                f"knob difference has to clear.\n\n" + table(s, present)
            )
            continue
        blocks.append(f"**{title}** — one run each at {rate:g} rps offered load.\n\n"
                      + table(s, present))
    (out / "knobs.md").write_text("\n\n".join(blocks) + "\n")

    chunk_arms = [x for x in PAIRS[0][1] if x in set(steady.config)]
    if len(chunk_arms) >= 2:
        itl_histogram(
            steady[steady.config.isin(chunk_arms)],
            figs / "itl_chunked_prefill.png",
            title="Inter-token latency: chunked vs unchunked prefill",
            subtitle=f"continuous+cache at {rate:g} rps offered load",
        )

    print("\n\n".join(blocks))
    for _, arms in PAIRS:
        for arm in arms:
            sub = steady[steady.config == arm]
            gaps = [np.asarray(v, dtype=float) for v in sub.itl if len(v)]
            if gaps:
                g = np.concatenate(gaps)
                print(f"{arm:24s} n_gaps={g.size:7d} "
                      f"p999={np.quantile(g, 0.999) * 1e3:7.1f}ms "
                      f"max={g.max() * 1e3:7.1f}ms")


if __name__ == "__main__":
    main()
