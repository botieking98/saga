from collections import deque

from saga.config import Config
from saga.engine.sequence import Sequence, SequenceStatus
from saga.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.enable_continuous_batching = config.enable_continuous_batching
        self.decode_steps_per_prefill = config.decode_steps_per_prefill
        self.max_prefill_tokens_per_step = config.max_prefill_tokens_per_step
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self._decode_steps_since_prefill = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def get_prefix_cache_stats(self, reset: bool = False) -> dict[str, int | float]:
        return self.block_manager.get_prefix_cache_stats(reset=reset)

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _schedule_prefill(self, token_budget: int) -> list[Sequence]:
        scheduled_seqs = []
        num_batched_tokens = 0

        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = token_budget - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)
        return scheduled_seqs

    def _schedule_decode(self) -> list[Sequence]:
        scheduled_seqs = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if not scheduled_seqs:
            return []
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.enable_continuous_batching and self.running and self.waiting:
            if self._decode_steps_since_prefill < self.decode_steps_per_prefill:
                decode_seqs = self._schedule_decode()
                if decode_seqs:
                    self._decode_steps_since_prefill += 1
                    return decode_seqs, False

                prefill_seqs = self._schedule_prefill(self.max_prefill_tokens_per_step)
                if prefill_seqs:
                    self._decode_steps_since_prefill = 0
                    return prefill_seqs, True
            else:
                prefill_seqs = self._schedule_prefill(self.max_prefill_tokens_per_step)
                if prefill_seqs:
                    self._decode_steps_since_prefill = 0
                    return prefill_seqs, True

                decode_seqs = self._schedule_decode()
                if decode_seqs:
                    self._decode_steps_since_prefill += 1
                    return decode_seqs, False

        prefill_seqs = self._schedule_prefill(self.max_num_batched_tokens)
        if prefill_seqs:
            self._decode_steps_since_prefill = 0
            return prefill_seqs, True

        decode_seqs = self._schedule_decode()
        if decode_seqs:
            self._decode_steps_since_prefill += 1
            return decode_seqs, False
        raise RuntimeError("scheduler has neither prefill nor decode work")

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
