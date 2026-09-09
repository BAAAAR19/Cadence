"""A reproducible fingerprint of the engine source a measurement ran on.

The README already claims that every number comes from one recorded engine
revision. Until now the recorded value in ``results/src.hash`` was produced by
hand, which makes it a note rather than a check. This makes it a function:
same source, same twelve characters, on any machine.

What is hashed is the engine and the C++ core -- the code that can change a
measurement -- and not the benchmark harness, which changes what is measured
rather than how the server behaves.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACKED = (
    ("src/cadence", (".py", ".pyi")),
    ("src/cpp", (".cpp", ".hpp", ".txt")),
)


def source_files() -> list[Path]:
    out: list[Path] = []
    for rel, suffixes in TRACKED:
        base = ROOT / rel
        if not base.exists():
            continue
        out += [
            p
            for p in base.rglob("*")
            if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts
        ]
    return sorted(out)


def source_hash() -> str:
    h = hashlib.sha256()
    for p in source_files():
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:12]


def main() -> None:
    print(source_hash())


if __name__ == "__main__":
    main()
