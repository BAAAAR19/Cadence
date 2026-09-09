"""Folded stacks -> flame graph SVG, and folded stacks -> attribution table.

Brendan Gregg's ``flamegraph.pl`` is the usual renderer; it is Perl, and the
input format is three lines of parsing, so it is vendored here as Python
instead of added as a build dependency for one picture.

The input is what ``cadence.obs.profiler`` writes: one line per distinct stack,
``root;...;leaf <microseconds>``, where the weight is elapsed wall time on the
sampled thread rather than a sample count.
"""

from __future__ import annotations

import argparse
import colorsys
import html
import zlib
from dataclasses import dataclass, field
from pathlib import Path

# Where a frame belongs, for the attribution table. Order matters: the first
# match wins, so the KV structures are claimed before the scheduler that calls
# them.
BUCKETS: list[tuple[str, tuple[str, ...]]] = [
    ("kv: block manager", ("block_manager.py",)),
    ("kv: radix cache", ("radix_cache.py",)),
    ("kv: C++ core", ("_core", "block_allocator.cpp", "radix_cache.cpp")),
    ("model runner", ("llamacpp.py", "llamacpp_http.py", "mock.py")),
    # llama.cpp's own tokenizer/detokenizer, and numpy under greedy sampling.
    # Both are reached only from the runner, so they land inside "model
    # runner" on an inclusive basis and only separate out as self time.
    ("tokenizer (llama.cpp)", ("_internals.py", "llama_types.py")),
    ("sampling (numpy)", ("fromnumeric.py", "numeric.py", "_methods.py")),
    ("scheduler", ("continuous.py", "fifo.py", "static_batch.py", "base.py")),
    ("request lifecycle", ("request.py",)),
    ("metrics", ("metrics.py", "tracing.py")),
]

IDLE = ("base.py:_idle_wait",)
"""Frames that mean the scheduler had nothing to do. Excluded from the
denominator: "x% of step time" must not be diluted by how idle the run was."""


@dataclass
class Frame:
    name: str
    value: int = 0
    """Total microseconds with this frame on the stack (inclusive)."""
    self_value: int = 0
    children: dict[str, Frame] = field(default_factory=dict)

    def child(self, name: str) -> Frame:
        f = self.children.get(name)
        if f is None:
            f = self.children[name] = Frame(name)
        return f


def read_folded(path: Path) -> list[tuple[list[str], int]]:
    out: list[tuple[list[str], int]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        stack, _, weight = line.rpartition(" ")
        out.append((stack.split(";"), int(weight)))
    return out


def is_idle(stack: list[str]) -> bool:
    return any(f.endswith(i) or f == i for f in stack for i in IDLE)


def bucket_of(frame: str) -> str | None:
    for name, needles in BUCKETS:
        if any(n in frame for n in needles):
            return name
    return None


def attribute(rows: list[tuple[list[str], int]]) -> dict[str, dict[str, int]]:
    """Inclusive and self time per bucket, over the busy samples only.

    *Inclusive* answers "how much of a step involves this component at all";
    *self* answers "how much of a step is spent executing it". For a data
    structure that calls nothing else the two are nearly equal, which is a
    useful check that the buckets are not double-counting.
    """
    totals: dict[str, dict[str, int]] = {}
    for stack, w in rows:
        if is_idle(stack):
            continue
        seen: set[str] = set()
        for frame in stack:
            b = bucket_of(frame)
            if b is not None and b not in seen:
                seen.add(b)
                totals.setdefault(b, {"inclusive": 0, "self": 0})["inclusive"] += w
        leaf = bucket_of(stack[-1]) if stack else None
        if leaf is not None:
            totals.setdefault(leaf, {"inclusive": 0, "self": 0})["self"] += w
    return totals


def busy_micros(rows: list[tuple[list[str], int]]) -> int:
    return sum(w for stack, w in rows if not is_idle(stack))


def build_tree(rows: list[tuple[list[str], int]]) -> Frame:
    root = Frame("all")
    for stack, w in rows:
        root.value += w
        node = root
        for name in stack:
            node = node.child(name)
            node.value += w
        node.self_value += w
    return root


def _colour(name: str) -> str:
    """Hot palette, hashed by frame name so a function keeps its colour between
    renders -- two flame graphs of the same server are then comparable at a
    glance."""
    h = zlib.crc32(name.encode()) & 0xFFFFFFFF
    if "block_manager" in name or "radix_cache" in name or "_core" in name:
        hue = 0.58 + (h % 100) / 2000.0  # blue: the structures under test
        sat, val = 0.55, 0.85
    elif "llamacpp" in name or "mock.py" in name:
        hue = 0.33 + (h % 100) / 2500.0  # green: the model
        sat, val = 0.40, 0.72
    else:
        hue = 0.05 + (h % 100) / 1400.0  # amber: everything else
        sat, val = 0.62, 0.95
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def render_svg(root: Frame, title: str, width: int = 1200, row_h: int = 17) -> str:
    depth = _depth(root)
    height = depth * row_h + 60
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Menlo,DejaVu Sans Mono,monospace" '
        f'font-size="11">',
        f'<rect width="{width}" height="{height}" fill="#f8f8f8"/>',
        f'<text x="{width // 2}" y="22" text-anchor="middle" font-size="14" '
        f'fill="#111">{html.escape(title)}</text>',
    ]

    def emit(node: Frame, x: float, y: float, w: float) -> None:
        if w < 0.35:  # narrower than a pixel: drawing it would be noise
            return
        label = node.name if w > 60 else ""
        pct = 100.0 * node.value / root.value if root.value else 0.0
        parts.append(
            f'<g><title>{html.escape(node.name)} — {node.value / 1000:.1f} ms '
            f'({pct:.2f}%)</title>'
            f'<rect x="{x:.2f}" y="{y:.1f}" width="{w:.2f}" height="{row_h - 1}" '
            f'fill="{_colour(node.name)}" stroke="#fff" stroke-width="0.4"/>'
        )
        if label:
            parts.append(
                f'<text x="{x + 3:.2f}" y="{y + row_h - 5:.1f}" fill="#111">'
                f'{html.escape(_clip(label, w))}</text>'
            )
        parts.append("</g>")
        cx = x
        for ch in sorted(node.children.values(), key=lambda c: c.name):
            cw = w * ch.value / node.value if node.value else 0.0
            emit(ch, cx, y - row_h, cw)
            cx += cw

    emit(root, 0.0, height - row_h - 4, float(width))
    parts.append("</svg>")
    return "\n".join(parts)


def _clip(text: str, w: float) -> str:
    n = max(0, int(w / 6.2) - 1)
    return text if len(text) <= n else text[: max(0, n - 1)] + "…"


def _depth(node: Frame) -> int:
    return 1 + max((_depth(c) for c in node.children.values()), default=0)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("folded", type=Path)
    p.add_argument("--svg", type=Path, default=None)
    p.add_argument("--title", default="cadence scheduler thread")
    p.add_argument("--md", type=Path, default=None, help="write the table as markdown")
    a = p.parse_args(argv)

    rows = read_folded(a.folded)
    if a.svg:
        a.svg.parent.mkdir(parents=True, exist_ok=True)
        a.svg.write_text(render_svg(build_tree(rows), a.title))
        print(f"wrote {a.svg}")
    busy = busy_micros(rows)
    total = sum(w for _, w in rows)
    print(f"sampled {total / 1e6:.1f} s, busy {busy / 1e6:.1f} s "
          f"({100.0 * busy / total:.0f}% of it)")
    table = sorted(attribute(rows).items(), key=lambda kv: -kv[1]["inclusive"])
    for name, v in table:
        print(f"  {name:22s} inclusive {100.0 * v['inclusive'] / busy:6.2f}%   "
              f"self {100.0 * v['self'] / busy:6.2f}%")
    if a.md:
        a.md.parent.mkdir(parents=True, exist_ok=True)
        out = ["| Component | Inclusive | Self |", "|---|---:|---:|"]
        out += [
            f"| {name} | {100.0 * v['inclusive'] / busy:.2f}% | {100.0 * v['self'] / busy:.2f}% |"
            for name, v in table
        ]
        a.md.write_text("\n".join(out) + "\n")
        print(f"wrote {a.md}")


if __name__ == "__main__":
    main()
