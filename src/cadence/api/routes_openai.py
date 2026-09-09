from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from cadence.api.schemas import ChatCompletionRequest, ModelCard, ModelList
from cadence.api.sse import DONE, SSE_HEADERS, chunk

router = APIRouter()


@router.get("/v1/models")
async def list_models(request: Request) -> ModelList:
    cfg = request.app.state.cfg
    return ModelList(data=[ModelCard(id=cfg.served_model_name)])


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    engine = request.app.state.engine
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]

    decision = await engine.admit(body)
    if decision.action == "shed":
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": f"{decision.retry_after_s:.3f}"},
            content={
                "error": {
                    "message": "overloaded: predicted SLO violation",
                    "type": "server_overloaded",
                    "code": "slo_shed",
                }
            },
        )

    rq = await engine.submit(body, decision)

    if not body.stream:
        try:
            await rq.result()
        finally:
            engine.release(rq)
        return JSONResponse(content=rq.to_openai(cid, body.model))

    async def gen():
        # Flush an opening frame immediately: without it StreamingResponse can
        # sit on the response until the first token, which shows up as inflated
        # TTFT that has nothing to do with the scheduler.
        yield chunk(cid, body.model, {"role": "assistant", "content": ""})
        try:
            async for tok in rq.stream():
                if await request.is_disconnected():
                    rq.cancel()
                    break
                yield chunk(cid, body.model, {"content": tok})
            yield chunk(cid, body.model, {}, finish=rq.finish_reason or "stop")
        except Exception:
            yield chunk(cid, body.model, {}, finish="error")
        finally:
            yield DONE
            engine.release(rq)  # always free KV blocks

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)
