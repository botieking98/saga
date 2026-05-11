from __future__ import annotations

import queue
import threading
import uuid
from multiprocessing import get_context
from typing import Any

from saga.serving.protocol import build_chat_prompt


def _worker_main(
    model: str,
    llm_kwargs: dict[str, Any],
    request_queue,
    response_queue,
):
    # Delay heavy CUDA/triton imports until worker boot so API process can start/help cleanly.
    from saga.engine.llm_engine import LLMEngine
    from saga.sampling_params import SamplingParams

    try:
        llm = LLMEngine(model, **llm_kwargs)
    except Exception as exc:  # pragma: no cover
        response_queue.put({"type": "ready", "ok": False, "error": repr(exc)})
        return

    response_queue.put({"type": "ready", "ok": True})
    active: dict[int, dict[str, Any]] = {}
    shutting_down = False

    def fail_request(request_id: str, exc: Exception):
        response_queue.put(
            {
                "type": "result",
                "request_id": request_id,
                "ok": False,
                "error": repr(exc),
            }
        )

    def process_request(req: dict[str, Any]) -> bool:
        req_type = req.get("type")
        if req_type == "shutdown":
            return True
        if req_type != "generate":
            return False

        request_id = req.get("request_id")
        if not isinstance(request_id, str):
            return False

        try:
            sampling = SamplingParams(**req["sampling_params"])
            prompt_kind = req.get("prompt_kind", "text")

            if prompt_kind == "chat":
                prompt_text = build_chat_prompt(llm.tokenizer, req["messages"])
                prompt_token_ids = llm.tokenizer.encode(prompt_text)
            else:
                prompt = req["prompt"]
                if isinstance(prompt, str):
                    prompt_token_ids = llm.tokenizer.encode(prompt)
                elif isinstance(prompt, list) and all(isinstance(token, int) for token in prompt):
                    prompt_token_ids = prompt
                else:
                    raise ValueError("prompt must be str or list[int]")

            seq_id = llm.add_request(prompt_token_ids, sampling)
            active[seq_id] = {
                "request_id": request_id,
                "prompt_tokens": len(prompt_token_ids),
                "max_tokens": sampling.max_tokens,
            }
        except Exception as exc:
            fail_request(request_id, exc)
        return False

    while True:
        while True:
            try:
                req = request_queue.get_nowait()
            except queue.Empty:
                break
            shutting_down = process_request(req) or shutting_down

        if active:
            try:
                outputs, _ = llm.step()
            except Exception as exc:
                for state in active.values():
                    fail_request(state["request_id"], exc)
                active.clear()
                continue

            for seq_id, token_ids in outputs:
                state = active.pop(seq_id, None)
                if state is None:
                    continue
                completion_tokens = len(token_ids)
                finish_reason = "length" if completion_tokens >= state["max_tokens"] else "stop"
                response_queue.put(
                    {
                        "type": "result",
                        "request_id": state["request_id"],
                        "ok": True,
                        "text": llm.tokenizer.decode(token_ids),
                        "prompt_tokens": state["prompt_tokens"],
                        "completion_tokens": completion_tokens,
                        "finish_reason": finish_reason,
                    }
                )
            continue

        if shutting_down:
            break

        try:
            req = request_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        shutting_down = process_request(req) or shutting_down


class SagaEngineClient:

    def __init__(
        self,
        model: str,
        llm_kwargs: dict[str, Any],
        startup_timeout: float,
        request_timeout: float,
    ):
        ctx = get_context("spawn")
        self._request_queue = ctx.Queue()
        self._response_queue = ctx.Queue()
        self._process = ctx.Process(
            target=_worker_main,
            args=(model, llm_kwargs, self._request_queue, self._response_queue),
        )
        self._pending: dict[str, queue.Queue] = {}
        self._pending_lock = threading.Lock()
        self._closed = threading.Event()
        self._dispatcher = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._startup_timeout = startup_timeout
        self._request_timeout = request_timeout

    def start(self):
        self._process.start()
        try:
            ready = self._response_queue.get(timeout=self._startup_timeout)
        except queue.Empty as exc:
            raise RuntimeError("engine worker startup timed out") from exc

        if ready.get("type") != "ready":
            raise RuntimeError(f"unexpected startup response: {ready}")
        if not ready.get("ok", False):
            raise RuntimeError(f"engine worker failed to start: {ready.get('error')}")

        self._dispatcher.start()

    def _dispatch_loop(self):
        while not self._closed.is_set():
            try:
                message = self._response_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if message.get("type") != "result":
                continue
            request_id = message.get("request_id")
            if not isinstance(request_id, str):
                continue

            with self._pending_lock:
                mailbox = self._pending.get(request_id)
            if mailbox is None:
                continue

            try:
                mailbox.put_nowait(message)
            except queue.Full:
                continue

    def generate(
        self,
        *,
        prompt_kind: str,
        sampling_params: dict[str, Any],
        prompt: str | list[int] | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        if not self._process.is_alive():
            raise RuntimeError("engine worker is not alive")

        request_id = uuid.uuid4().hex
        mailbox: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = mailbox

        payload = {
            "type": "generate",
            "request_id": request_id,
            "prompt_kind": prompt_kind,
            "sampling_params": sampling_params,
        }
        if prompt_kind == "chat":
            payload["messages"] = messages or []
        else:
            payload["prompt"] = prompt

        self._request_queue.put(payload)

        try:
            result = mailbox.get(timeout=self._request_timeout)
        except queue.Empty as exc:
            raise TimeoutError("request timeout while waiting for model output") from exc
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

        if not result.get("ok", False):
            raise RuntimeError(result.get("error", "unknown worker error"))
        return result

    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()

        try:
            self._request_queue.put({"type": "shutdown"}, timeout=0.5)
        except Exception:
            pass

        if self._process.is_alive():
            self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)

        if self._dispatcher.is_alive():
            self._dispatcher.join(timeout=1)
