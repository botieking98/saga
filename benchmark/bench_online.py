from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from transformers import AutoTokenizer


@dataclass(frozen=True)
class BenchResult:
    ok: bool
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ttft_s: float | None = None
    tpot_s: float | None = None
    error: str = ""


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = min(len(arr) - 1, max(0, int((len(arr) - 1) * p)))
    return arr[idx]


def http_get_json(url: str, timeout_s: float) -> Any:
    req = urlrequest.Request(url, method="GET")
    with urlrequest.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post_json(url: str, payload: dict[str, Any], timeout_s: float) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urlrequest.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_model_name(base_url: str, timeout_s: float) -> str:
    models = http_get_json(f"{base_url}/models", timeout_s)
    data = models.get("data", [])
    if not data:
        raise RuntimeError("/v1/models returned empty model list")
    model = data[0].get("id")
    if not isinstance(model, str) or not model:
        raise RuntimeError("invalid model id from /v1/models")
    return model


def generate_prompt(tokenizer: Any, n_tokens: int, rng: random.Random) -> str:
    vocab_size = max(1000, tokenizer.vocab_size // 2)
    token_ids = [rng.randint(0, vocab_size) for _ in range(n_tokens)]

    for _ in range(48):
        prompt = tokenizer.decode(token_ids)
        new_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(new_ids) == n_tokens:
            return prompt
        if len(new_ids) < n_tokens:
            new_ids.extend(rng.randint(0, vocab_size) for _ in range(n_tokens - len(new_ids)))
        else:
            new_ids = new_ids[:n_tokens]
        token_ids = new_ids

    raise RuntimeError("failed to generate prompt with exact token length")


def build_prompts(
    tokenizer: Any,
    *,
    n_requests: int,
    input_len: int,
    rng: random.Random,
    prompt_mode: str,
    shared_prefix_len: int,
    num_shared_prefixes: int,
) -> tuple[list[str], dict[str, Any]]:
    if prompt_mode == "random":
        prompts = [generate_prompt(tokenizer, input_len, rng) for _ in range(n_requests)]
        return prompts, {"prompt_mode": "random"}

    if prompt_mode != "shared_prefix":
        raise ValueError(f"unsupported prompt mode: {prompt_mode}")

    if not (0 <= shared_prefix_len <= input_len):
        raise ValueError("--shared-prefix-len must be within [0, --input-len]")
    if num_shared_prefixes <= 0:
        raise ValueError("--num-shared-prefixes must be > 0")

    suffix_len = input_len - shared_prefix_len
    prefixes = [
        generate_prompt(tokenizer, shared_prefix_len, rng) if shared_prefix_len > 0 else ""
        for _ in range(num_shared_prefixes)
    ]
    prompts: list[str] = []
    for i in range(n_requests):
        suffix = generate_prompt(tokenizer, suffix_len, rng) if suffix_len > 0 else ""
        prompts.append(prefixes[i % num_shared_prefixes] + suffix)

    return prompts, {
        "prompt_mode": "shared_prefix",
        "shared_prefix_len": shared_prefix_len,
        "suffix_len": suffix_len,
        "num_shared_prefixes": num_shared_prefixes,
    }


def one_request_non_stream(
    base_url: str,
    model: str,
    prompt: str,
    prompt_tokens: int,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> BenchResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    start = time.perf_counter()
    try:
        result = http_post_json(f"{base_url}/chat/completions", payload, timeout_s)
        latency = time.perf_counter() - start

        usage = result.get("usage", {})
        prompt_tokens_used = int(usage.get("prompt_tokens", prompt_tokens))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", prompt_tokens_used + completion_tokens))

        tpot_est = (latency / completion_tokens) if completion_tokens > 0 else None
        return BenchResult(
            ok=True,
            latency_s=latency,
            prompt_tokens=prompt_tokens_used,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            ttft_s=None,
            tpot_s=tpot_est,
        )
    except (urlerror.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        latency = time.perf_counter() - start
        return BenchResult(
            ok=False,
            latency_s=latency,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error=repr(exc),
        )


def one_request_stream(
    base_url: str,
    model: str,
    prompt: str,
    prompt_tokens: int,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> BenchResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        f"{base_url}/chat/completions",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    inter_token_gaps: list[float] = []
    completion_tokens = 0

    try:
        with urlrequest.urlopen(req, timeout=timeout_s) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choice = ((chunk.get("choices") or [{}])[0])
                delta = choice.get("delta") or {}
                text = delta.get("content", "")

                if text:
                    now = time.perf_counter()
                    completion_tokens += 1
                    if first_token_at is None:
                        first_token_at = now
                    if last_token_at is not None:
                        inter_token_gaps.append(now - last_token_at)
                    last_token_at = now

                finish_reason = choice.get("finish_reason")
                if finish_reason is not None:
                    break

        latency = time.perf_counter() - start
        ttft = (first_token_at - start) if first_token_at is not None else None
        tpot = (sum(inter_token_gaps) / len(inter_token_gaps)) if inter_token_gaps else None

        return BenchResult(
            ok=True,
            latency_s=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            ttft_s=ttft,
            tpot_s=tpot,
        )
    except (urlerror.URLError, TimeoutError, ValueError) as exc:
        latency = time.perf_counter() - start
        return BenchResult(
            ok=False,
            latency_s=latency,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error=repr(exc),
        )


def one_request(
    base_url: str,
    model: str,
    prompt: str,
    prompt_tokens: int,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    stream: bool,
) -> BenchResult:
    if stream:
        return one_request_stream(
            base_url,
            model,
            prompt,
            prompt_tokens,
            max_tokens,
            temperature,
            timeout_s,
        )
    return one_request_non_stream(
        base_url,
        model,
        prompt,
        prompt_tokens,
        max_tokens,
        temperature,
        timeout_s,
    )


async def run_benchmark(args: argparse.Namespace) -> list[BenchResult]:
    base_url = args.base_url.rstrip("/") + "/v1"
    model = args.model or get_model_name(base_url, args.timeout)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model_path)
    rng = random.Random(args.seed)

    prompts, prompt_meta = build_prompts(
        tokenizer,
        n_requests=args.num_requests,
        input_len=args.input_len,
        rng=rng,
        prompt_mode=args.prompt_mode,
        shared_prefix_len=args.shared_prefix_len,
        num_shared_prefixes=args.num_shared_prefixes,
    )
    prompt_token_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    print(f"Prompt Mode: {prompt_meta['prompt_mode']}")
    if prompt_meta["prompt_mode"] == "shared_prefix":
        print(
            "Shared Prefix Config: "
            f"prefix_len={prompt_meta['shared_prefix_len']}, "
            f"suffix_len={prompt_meta['suffix_len']}, "
            f"num_prefixes={prompt_meta['num_shared_prefixes']}"
        )
    print(
        "Prompt Token Lens: "
        f"target={args.input_len}, "
        f"avg={sum(prompt_token_lens)/len(prompt_token_lens):.2f}, "
        f"min={min(prompt_token_lens)}, "
        f"max={max(prompt_token_lens)}"
    )

    # Warm-up request.
    _ = one_request(
        base_url,
        model,
        prompts[0],
        prompt_token_lens[0],
        min(args.output_len, 8),
        args.temperature,
        args.timeout,
        args.stream,
    )

    sem = asyncio.Semaphore(args.concurrency)
    loop = asyncio.get_running_loop()

    def submit_one(prompt: str, prompt_tokens: int) -> asyncio.Future:
        return loop.run_in_executor(
            executor,
            one_request,
            base_url,
            model,
            prompt,
            prompt_tokens,
            args.output_len,
            args.temperature,
            args.timeout,
            args.stream,
        )

    async def wrapped(prompt: str, prompt_tokens: int) -> BenchResult:
        async with sem:
            return await submit_one(prompt, prompt_tokens)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        tasks = [wrapped(prompt, prompt_tokens) for prompt, prompt_tokens in zip(prompts, prompt_token_lens)]
        results = await asyncio.gather(*tasks)
    ended = time.perf_counter()

    summarize(results, ended - started, stream=args.stream)

    if args.output_json:
        payload = {
            "config": {
                "base_url": base_url,
                "model": model,
                "num_requests": args.num_requests,
                "concurrency": args.concurrency,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "temperature": args.temperature,
                "seed": args.seed,
                "stream": args.stream,
                "prompt_mode": args.prompt_mode,
                "shared_prefix_len": args.shared_prefix_len,
                "num_shared_prefixes": args.num_shared_prefixes,
            },
            "results": [asdict(r) for r in results],
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved detailed results to {args.output_json}")

    return results


def summarize(results: list[BenchResult], duration_s: float, stream: bool) -> None:
    total = len(results)
    ok_results = [r for r in results if r.ok]
    err_results = [r for r in results if not r.ok]

    latencies = [r.latency_s for r in ok_results]
    ttfts = [r.ttft_s for r in ok_results if r.ttft_s is not None]
    tpots = [r.tpot_s for r in ok_results if r.tpot_s is not None]

    prompt_tokens = sum(r.prompt_tokens for r in ok_results)
    completion_tokens = sum(r.completion_tokens for r in ok_results)
    total_tokens = sum(r.total_tokens for r in ok_results)

    req_per_s = len(ok_results) / duration_s if duration_s > 0 else 0.0
    tok_per_s = total_tokens / duration_s if duration_s > 0 else 0.0
    gen_tok_per_s = completion_tokens / duration_s if duration_s > 0 else 0.0

    print("=" * 72)
    print("Online Serving Benchmark (OpenAI-compatible)")
    print("=" * 72)
    print(f"Mode: {'stream' if stream else 'non-stream'}")
    print(f"Requests: total={total}, success={len(ok_results)}, failed={len(err_results)}")
    print(f"Duration: {duration_s:.3f}s")
    print(f"Throughput: {req_per_s:.2f} req/s")
    print(f"Token Throughput: total={tok_per_s:.2f} tok/s, completion={gen_tok_per_s:.2f} tok/s")
    print(f"Token Count: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")

    if latencies:
        print(
            "E2E Latency(s): "
            f"avg={sum(latencies)/len(latencies):.4f}, "
            f"p50={percentile(latencies, 0.50):.4f}, "
            f"p95={percentile(latencies, 0.95):.4f}, "
            f"p99={percentile(latencies, 0.99):.4f}, "
            f"max={max(latencies):.4f}"
        )

    if ttfts:
        print(
            "TTFT(s): "
            f"avg={sum(ttfts)/len(ttfts):.4f}, "
            f"p50={percentile(ttfts, 0.50):.4f}, "
            f"p95={percentile(ttfts, 0.95):.4f}, "
            f"p99={percentile(ttfts, 0.99):.4f}, "
            f"max={max(ttfts):.4f}"
        )
    else:
        print("TTFT: N/A (requires stream mode)")

    if tpots:
        tag = "TPOT(s)" if stream else "TPOT(estimated, s)"
        print(
            f"{tag}: "
            f"avg={sum(tpots)/len(tpots):.4f}, "
            f"p50={percentile(tpots, 0.50):.4f}, "
            f"p95={percentile(tpots, 0.95):.4f}, "
            f"p99={percentile(tpots, 0.99):.4f}, "
            f"max={max(tpots):.4f}"
        )

    if err_results:
        print("Sample errors:")
        for err in err_results[:5]:
            print(f"  - {err.error}")
    print("=" * 72)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online serving benchmark for saga OpenAI API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="server URL without /v1")
    parser.add_argument("--model", default="", help="model id; empty means auto detect via /v1/models")
    parser.add_argument(
        "--model-path",
        default="~/huggingface/Qwen3-0.6B",
        help="local model path for tokenizer",
    )
    parser.add_argument("--tokenizer", default="", help="optional tokenizer path/name override")

    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument(
        "--prompt-mode",
        choices=["random", "shared_prefix"],
        default="random",
        help="prompt generation mode",
    )
    parser.add_argument(
        "--shared-prefix-len",
        type=int,
        default=-1,
        help="prefix length used when --prompt-mode=shared_prefix; -1 means auto (75%% of input-len)",
    )
    parser.add_argument(
        "--num-shared-prefixes",
        type=int,
        default=1,
        help="number of distinct shared prefixes when --prompt-mode=shared_prefix",
    )

    parser.add_argument("--stream", action="store_true", help="use stream=true to collect TTFT/TPOT")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="", help="optional output json path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_requests <= 0:
        raise ValueError("--num-requests must be > 0")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")
    if args.input_len <= 0:
        raise ValueError("--input-len must be > 0")
    if args.output_len <= 0:
        raise ValueError("--output-len must be > 0")

    if args.temperature <= 1e-10:
        raise ValueError("--temperature must be > 1e-10")

    if not args.model_path:
        raise ValueError("--model-path is required for tokenizer loading")
    args.model_path = os.path.expanduser(args.model_path)

    if args.prompt_mode == "shared_prefix":
        if args.shared_prefix_len == -1:
            args.shared_prefix_len = max(1, int(args.input_len * 0.75))
        if not (0 <= args.shared_prefix_len <= args.input_len):
            raise ValueError("--shared-prefix-len must be within [0, --input-len]")
        if args.num_shared_prefixes <= 0:
            raise ValueError("--num-shared-prefixes must be > 0")
    else:
        if args.shared_prefix_len == -1:
            args.shared_prefix_len = 0
        if args.num_shared_prefixes <= 0:
            raise ValueError("--num-shared-prefixes must be > 0")

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
