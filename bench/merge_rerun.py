"""Merge a repeated run back into a sweep, and record that it was repeated.

A results file that silently contains work from two sittings is not a results
file. This writes the merge into the sweep's ``meta.json`` so the provenance
travels with the data.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--into", required=True, help="the sweep directory to merge into")
    p.add_argument("--from-dir", required=True, help="directory holding the repeated run")
    p.add_argument("--rung", required=True)
    p.add_argument("--reason", required=True)
    a = p.parse_args(argv)

    stem = a.rung.replace("+", "_")
    target = Path(a.into) / f"{stem}.parquet"
    source = Path(a.from_dir) / f"{stem}.parquet"
    base, extra = pd.read_parquet(target), pd.read_parquet(source)

    dup = set(map(tuple, base[["rate_rps", "seed"]].drop_duplicates().to_numpy())) & set(
        map(tuple, extra[["rate_rps", "seed"]].drop_duplicates().to_numpy())
    )
    if dup:
        raise SystemExit(f"refusing to merge: {sorted(dup)} already present in {target}")

    merged = pd.concat([base, extra], ignore_index=True)
    merged["status"] = pd.to_numeric(merged["status"], errors="coerce").astype("Int64")
    if "error" in merged:
        merged["error"] = merged["error"].astype("string")
    merged.to_parquet(target)

    meta_path = Path(a.into) / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.setdefault("reruns", []).append(
        {
            "rung": a.rung,
            "rates": sorted(float(r) for r in extra.rate_rps.unique()),
            "seeds": sorted(int(s) for s in extra.seed.unique()),
            "reason": a.reason,
            "merged_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "source": str(source),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"merged {len(extra)} rows into {target} "
        f"({len(base)} -> {len(merged)}); recorded in {meta_path}"
    )


if __name__ == "__main__":
    main()
