from __future__ import annotations

from typing import Any


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("chat content list entries must be objects")
            part_type = part.get("type")
            if part_type not in (None, "text"):
                raise ValueError(f"unsupported content type: {part_type}")
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError("text content must be a string")
            text_parts.append(text)
        return "".join(text_parts)
    raise ValueError("chat message content must be a string or a list of text parts")


def normalize_chat_messages(raw_messages: Any) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty list")

    normalized: list[dict[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("message.role must be a non-empty string")
        content = extract_text_content(message.get("content"))
        normalized.append({"role": role, "content": content})
    return normalized


def normalize_prompts(raw_prompt: Any) -> list[str | list[int]]:
    if isinstance(raw_prompt, str):
        return [raw_prompt]
    if isinstance(raw_prompt, list):
        if not raw_prompt:
            return [""]
        if all(isinstance(token, int) for token in raw_prompt):
            return [raw_prompt]
        if all(isinstance(item, str) for item in raw_prompt):
            return raw_prompt
        if all(
            isinstance(item, list) and all(isinstance(token, int) for token in item)
            for item in raw_prompt
        ):
            return raw_prompt
    raise ValueError(
        "prompt must be string, token id list, list of strings, or list of token id lists"
    )


def parse_sampling_params(payload: dict[str, Any]) -> dict[str, Any]:
    stream = payload.get("stream", False)
    if stream:
        raise ValueError("stream=true is not supported")

    max_tokens = int(payload.get("max_tokens", 64))
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")

    temperature = float(payload.get("temperature", 1.0))
    if temperature <= 1e-10:
        raise ValueError("temperature must be > 1e-10; greedy sampling is unsupported")

    ignore_eos = bool(payload.get("ignore_eos", False))
    return {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "ignore_eos": ignore_eos,
    }


def build_chat_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False)

    lines = [f"{m['role']}: {m['content']}" for m in messages]
    lines.append("assistant:")
    return "\n".join(lines)
