import atexit
import socket
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from saga.config import Config
from saga.sampling_params import SamplingParams
from saga.engine.sequence import Sequence
from saga.engine.scheduler import Scheduler, Batch
from saga.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        if config.dist_init_port == 0:
            config.dist_init_port = self._find_free_tcp_port()
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.daemon = True
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self._inflight: tuple[int, Batch] | None = None
        atexit.register(self.exit)

    @staticmethod
    def _find_free_tcp_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> int:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        outputs, num_tokens, _ = self.step_with_updates()
        return outputs, num_tokens

    def _launch_one_batch(self, batch: Batch) -> int:
        ticket = self.model_runner.new_ticket()
        self.model_runner.call("launch", batch, ticket)
        return ticket

    def step_with_updates(self):
        if self._inflight is None:
            batch = self.scheduler.schedule_next_batch()
            if batch is None:
                raise RuntimeError("scheduler has neither prefill nor decode work")
            ticket = self._launch_one_batch(batch)
            self._inflight = (ticket, batch)

        next_inflight: tuple[int, Batch] | None = None
        next_batch = self.scheduler.schedule_next_batch()
        if next_batch is not None:
            next_ticket = self._launch_one_batch(next_batch)
            next_inflight = (next_ticket, next_batch)

        ticket, batch = self._inflight
        num_tokens = sum(seq.num_scheduled_tokens for seq in batch.seqs) if batch.is_prefill else -len(batch.seqs)
        prev_completion_tokens = {seq.seq_id: seq.num_completion_tokens for seq in batch.seqs}
        token_ids = self.model_runner.call("collect", ticket)
        self.scheduler.postprocess(batch, token_ids)
        self._inflight = next_inflight
        step_updates = [
            (seq.seq_id, token_id)
            for seq, token_id in zip(batch.seqs, token_ids)
            if seq.num_completion_tokens > prev_completion_tokens[seq.seq_id]
        ]
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in batch.seqs if seq.is_finished]
        return outputs, num_tokens, step_updates

    def is_finished(self):
        return self.scheduler.is_finished() and self._inflight is None

    def get_prefix_cache_stats(self, reset: bool = False) -> dict[str, int | float]:
        return self.scheduler.get_prefix_cache_stats(reset=reset)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
