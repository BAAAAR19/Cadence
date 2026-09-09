"""SSE framing helpers.

Three things break SSE benchmarks and all three are invisible until you measure
TTFT:

1. Any buffering proxy in front of the app inflates TTFT. ``X-Accel-Buffering:
   no`` and ``Cache-Control: no-cache`` are set on every stream, and the
   benchmark is run without a proxy.
2. ``StreamingResponse`` will not flush if the generator yields nothing for the
   first frame, so an opening ``role`` delta is always emitted immediately.
3. If a client disconnects mid-stream and the ``finally`` block does not run,
   KV blocks leak and every subsequent measurement drifts.
"""

from __future__ import annotations

import json
import time

DONE = "data: [DONE]\n\n"

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def chunk(cid: str, model: str, delta: dict, finish: str | None = None) -> str:
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
