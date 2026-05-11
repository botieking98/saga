from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
import uvicorn

from saga.serving.args import ServerArgs
from saga.serving.engine_client import SagaEngineClient
from saga.serving.protocol import (
    normalize_chat_messages,
    normalize_prompts,
    parse_sampling_params,
)


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int = 64
    temperature: float = 1.0
    stream: bool = False
    ignore_eos: bool = False


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: Any
    max_tokens: int = 64
    temperature: float = 1.0
    stream: bool = False
    ignore_eos: bool = False
    n: int = 1


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "saga"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


def create_app(args: ServerArgs) -> FastAPI:
    engine = SagaEngineClient(
        model=args.model_path,
        llm_kwargs=args.llm_kwargs,
        startup_timeout=args.startup_timeout,
        request_timeout=args.request_timeout,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine
        app.state.model_name = args.effective_model_name
        app.state.engine.start()
        try:
            yield
        finally:
            app.state.engine.close()

    app = FastAPI(title="saga OpenAI-Compatible API", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": app.state.model_name}

    @app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
    def v1_root():
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models():
        return ModelList(data=[ModelCard(id=app.state.model_name)]).model_dump()

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        payload = req.model_dump()
        try:
            sampling_params = parse_sampling_params(payload)
            messages = normalize_chat_messages(req.messages)
            result = await run_in_threadpool(
                app.state.engine.generate,
                prompt_kind="chat",
                sampling_params=sampling_params,
                messages=messages,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=repr(exc)) from exc

        created = int(time.time())
        completion_text = result["text"]
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": created,
            "model": app.state.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": completion_text},
                    "finish_reason": result["finish_reason"],
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            },
        }

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        if req.n != 1:
            raise HTTPException(status_code=400, detail="only n=1 is supported")

        payload = req.model_dump()
        try:
            sampling_params = parse_sampling_params(payload)
            prompts = normalize_prompts(req.prompt)

            tasks = [
                run_in_threadpool(
                    app.state.engine.generate,
                    prompt_kind="text",
                    sampling_params=sampling_params,
                    prompt=prompt,
                )
                for prompt in prompts
            ]
            results = await asyncio.gather(*tasks)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=repr(exc)) from exc

        created = int(time.time())
        choices = []
        prompt_tokens = 0
        completion_tokens = 0
        for i, result in enumerate(results):
            prompt_tokens += result["prompt_tokens"]
            completion_tokens += result["completion_tokens"]
            choices.append(
                {
                    "index": i,
                    "text": result["text"],
                    "logprobs": None,
                    "finish_reason": result["finish_reason"],
                }
            )

        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": created,
            "model": app.state.model_name,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def run_api_server(args: ServerArgs) -> None:
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
