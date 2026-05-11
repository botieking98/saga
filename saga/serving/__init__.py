from __future__ import annotations

__all__ = ["ServerArgs", "create_app", "parse_args", "run_api_server"]


def __getattr__(name: str):
    if name in ("ServerArgs", "parse_args"):
        from saga.serving.args import ServerArgs, parse_args

        return {"ServerArgs": ServerArgs, "parse_args": parse_args}[name]

    if name in ("create_app", "run_api_server"):
        from saga.serving.api_server import create_app, run_api_server

        return {"create_app": create_app, "run_api_server": run_api_server}[name]

    raise AttributeError(f"module 'saga.serving' has no attribute {name!r}")
