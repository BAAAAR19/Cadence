"""Generate the Grafana dashboard JSON.

Written as a script rather than hand-edited JSON for one reason: every query it
emits is checked against the collector names actually registered in
``cadence.obs.metrics``, so a renamed metric breaks the build instead of
silently producing an empty panel three weeks later.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cadence.obs.metrics import REGISTRY  # noqa: E402

DS = {"type": "prometheus", "uid": "cadence-prom"}
KNOWN = {
    name
    for collector in list(REGISTRY._collector_to_names)
    for name in REGISTRY._collector_to_names[collector]
}


def q(expr: str, legend: str) -> dict:
    cleaned = expr
    for ch in "()[],":
        cleaned = cleaned.replace(ch, " ")
    for token in cleaned.split():
        base = token.split("{")[0]
        if base.startswith("cadence_"):
            stem = base.removesuffix("_bucket").removesuffix("_sum").removesuffix("_count")
            if stem not in KNOWN and base not in KNOWN:
                raise SystemExit(f"panel references unknown metric {base!r}")
    return {"expr": expr, "legendFormat": legend, "datasource": DS, "refId": "A"}


def panel(title, targets, x, y, w=12, h=7, unit="s", desc="", stack=False) -> dict:
    for i, t in enumerate(targets):
        t["refId"] = chr(ord("A") + i)
    return {
        "type": "timeseries",
        "title": title,
        "description": desc,
        "datasource": DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": targets,
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "custom": {
                    "lineWidth": 2,
                    "fillOpacity": 8 if not stack else 30,
                    "showPoints": "never",
                    "stacking": {"mode": "normal" if stack else "none"},
                },
            },
            "overrides": [],
        },
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}},
    }


def hist(metric: str, quantiles=(0.5, 0.95, 0.99)) -> list[dict]:
    return [
        q(
            f'histogram_quantile({p}, sum by (le, config) (rate({metric}_bucket[30s])))',
            f"p{int(p * 100)} {{{{config}}}}",
        )
        for p in quantiles
    ]


def build() -> dict:
    y = 0
    panels = [
        panel(
            "End-to-end latency (from arrival)",
            hist("cadence_e2e_seconds"),
            0, y,
            desc="The SLO variable. Interpolated within Prometheus buckets -- the "
                 "numbers in the README come from the load generator's raw parquet.",
        ),
        panel("Time to first token", hist("cadence_ttft_seconds"), 12, y),
    ]
    y += 7
    panels += [
        panel(
            "Inter-token latency",
            hist("cadence_itl_seconds"),
            0, y,
            desc="p99 ITL is where unchunked prefill shows up: a long prompt "
                 "prefilled in one shot stalls every decoding sequence.",
        ),
        panel("Engine step duration", hist("cadence_step_seconds", (0.5, 0.99)), 12, y),
    ]
    y += 7
    panels += [
        panel(
            "Queue depth and running batch size",
            [
                q("cadence_queue_depth", "waiting {{config}}"),
                q("cadence_batch_size", "running {{config}}"),
            ],
            0, y, unit="short",
            desc="Under continuous batching the running batch should be "
                 "continuously changing size, not stepping between waves.",
        ),
        panel(
            "Throughput and goodput",
            [
                q("sum by (config) (rate(cadence_requests_total{state=\"done\"}[30s]))",
                  "throughput {{config}}"),
                q("sum by (config) (rate(cadence_slo_met_total[30s]))", "goodput {{config}}"),
                q("sum by (config) (rate(cadence_shed_total[30s]))", "shed {{config}}"),
            ],
            12, y, unit="reqps",
            desc="Under overload these diverge, and goodput is the one that matters.",
        ),
    ]
    y += 7
    panels += [
        panel(
            "KV blocks",
            [
                q("cadence_kv_blocks_free", "free {{config}}"),
                q("cadence_kv_blocks_total", "total {{config}}"),
            ],
            0, y, unit="short",
            desc="The memory ceiling on batch size.",
        ),
        panel(
            "KV fragmentation ratio",
            [q("cadence_kv_fragmentation_ratio", "{{config}}")],
            12, y, unit="percentunit",
            desc="allocated tokens / (allocated blocks x block size). "
                 "Paged allocation should hold this above ~0.9.",
        ),
    ]
    y += 7
    panels += [
        panel(
            "Prefix cache token-level hit rate",
            [
                q(
                    "sum by (config) (rate(cadence_prefix_cache_hit_tokens_total[1m]))"
                    " / sum by (config) (rate(cadence_prefix_cache_query_tokens_total[1m]))",
                    "{{config}}",
                )
            ],
            0, y, unit="percentunit",
            desc="Token-level, not request-level: partial hits are the normal case.",
        ),
        panel(
            "Preemptions and copy-on-write",
            [
                q("sum by (config) (rate(cadence_preemptions_total[1m]))", "preemptions {{config}}"),
                q("sum by (config) (rate(cadence_kv_copy_on_write_total[1m]))", "CoW {{config}}"),
            ],
            12, y, unit="short",
        ),
    ]
    y += 7
    panels += [
        panel(
            "Token throughput",
            [q("sum by (config, kind) (rate(cadence_tokens_total[30s]))", "{{kind}} {{config}}")],
            0, y, w=24, unit="short", stack=True,
        )
    ]

    return {
        "uid": "cadence-overview",
        "title": "Cadence -- SLO-aware inference gateway",
        "tags": ["cadence"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "1s",
        "time": {"from": "now-15m", "to": "now"},
        "templating": {"list": []},
        "panels": panels,
    }


if __name__ == "__main__":
    out = Path(__file__).parent / "dashboards" / "cadence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"wrote {out} ({len(build()['panels'])} panels)")
