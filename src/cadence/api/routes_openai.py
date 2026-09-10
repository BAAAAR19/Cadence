from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from cadence.api.lifecycle import Lifecycle
from cadence.api.schemas import ChatCompletionRequest, ModelCard, ModelList
from cadence.api.sse import DONE, SSE_HEADERS, chunk

router = APIRouter()


@router.get("/v1/models")
async def list_models(request: Request) -> ModelList:
    cfg = request.app.state.cfg
    return ModelList(data=[ModelCard(id=cfg.served_model_name)])


def _refusal(message: str, code: str, retry_after_s: float) -> JSONResponse:
    """A 503 in the shape the OpenAI clients already understand, always with a
    ``Retry-After``. A refusal without one is just a failure; with one it is a
    scheduling instruction, and the load generator records it as such."""
    return JSONResponse(
        status_code=503,
        headers={"Retry-After": f"{retry_after_s:.3f}"},
        content={
            "error": {
                "message": message,
                "type": "server_overloaded",
                "code": code,
            }
        },
    )


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    engine = request.app.state.engine
    cfg = request.app.state.cfg
    life: Lifecycle = request.app.state.lifecycle
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]

    # Two refusals that come before the controller, because neither is a
    # statement about latency. Draining is "not here, try the next instance";
    # the cap is "this process is out of room". Both are counted separately
    # from the conformal sheds, so a coverage number is never quietly diluted
    # by an operational one.
    if life.draining:
        life.refuse_draining()
        engine.metrics.shed(reason="draining")
        return _refusal("server is draining", "draining", cfg.retry_after_s)
    if not life.acquire():
        engine.metrics.shed(reason="capacity")
        return _refusal(
            "server at its concurrency cap", "capacity", cfg.retry_after_s
        )

    try:
        decision = await engine.admit(body)
    except BaseException:
        life.release()
        raise
    if decision.action == "shed":
        life.release()
        return _refusal(
            "overloaded: predicted SLO violation", "slo_shed", decision.retry_after_s
        )

    try:
        rq = await engine.submit(body, decision)
    except BaseException:
        life.release()
        raise

    if not body.stream:
        try:
            await rq.result()
        finally:
            engine.release(rq)
            life.release()
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
            life.release()

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)
