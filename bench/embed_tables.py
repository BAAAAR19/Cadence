"""Substitute the generated tables into the README.

The README's numbers live between paired markers and are filled from
``docs/*.md``, which ``bench/make_report.py`` writes from the committed
parquet. A number in the writeup therefore cannot drift from the data that
produced it: regenerate, re-embed, and any change shows up in the diff.

``--check`` makes that enforceable in CI: it fails if re-embedding would change
anything, which means someone edited a number by hand or forgot to regenerate.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

TABLES = {
    "LADDER": "ladder.md",
    "SWEEP": "sweep.md",
    "SLO": "slo_sensitivity.md",
    "METHOD": "methodology.md",
    "KNOBS": "knobs.md",
    "CORE_PROFILE": "core_profile.md",
    "CORE_BENCH": "core_bench.md",
    "CORE_AB": "core_ab.md",
    "ADMISSION": "admission.md",
    "COVERAGE": "coverage.md",
    "PREDICTOR": "predictor.md",
    # Week 5.
    "ABLATION": "ablation.md",
    "ARMS": "admission_arms.md",
    "SPREAD": "spread.md",
    "TRADEOFFS": "tradeoffs.md",
    "THERMAL": "thermal.md",
    "ANCHOR": "block_anchor.md",
    "SESSION": "session_check.md",
    "DRAIN": "drain.md",
}


def embed(text: str, docs: Path) -> str:
    for name, filename in TABLES.items():
        body = (docs / filename).read_text().strip()
        pattern = re.compile(
            rf"(<!-- {name} -->)(.*?)(<!-- /{name} -->)", re.DOTALL
        )
        if not pattern.search(text):
            raise SystemExit(f"README is missing the <!-- {name} --> ... <!-- /{name} --> pair")
        text = pattern.sub(
            lambda m, body=body: f"{m.group(1)}\n\n{body}\n\n{m.group(3)}", text
        )
    return text


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--readme", default="README.md")
    p.add_argument("--docs", default="docs")
    p.add_argument("--check", action="store_true",
                   help="fail if the README is not already up to date")
    a = p.parse_args(argv)

    readme = Path(a.readme)
    text = readme.read_text()
    out = embed(text, Path(a.docs))

    if a.check:
        if out != text:
            raise SystemExit(
                f"{readme} is stale: regenerate with "
                f"`uv run bench/make_report.py results/w2_ladder --slo 4.0 --outdir docs`, "
                f"`uv run bench/core_report.py`, "
                f"`uv run bench/w4_report.py results/w4_admission`, "
                f"`bash bench/report_week5.sh`, "
                f"then `uv run bench/embed_tables.py`"
            )
        print(f"{readme} is up to date")
        return
    readme.write_text(out)
    print(f"embedded {len(TABLES)} tables into {readme}")


if __name__ == "__main__":
    main()
