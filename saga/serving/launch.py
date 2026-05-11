from __future__ import annotations

from saga.serving.args import parse_args


def main() -> None:
    args = parse_args()
    from saga.serving.api_server import run_api_server

    run_api_server(args)


if __name__ == "__main__":
    main()
