from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from saga.config import Config
from saga.engine.sequence import Sequence, SequenceStatus
from saga.engine.block_manager import BlockManager


def _seq_in_flight(seq: Sequence) -> bool:
    return bool(getattr(seq, "in_flight", False))


@dataclass
class Batch:
    seqs: list[Sequence]
    phase: Literal["prefill", "decode"]

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"


@dataclass
class DecodeManager:
    block_size: int
    running: deque[Sequence] = field(default_factory=deque)

    def add(self, seq: Sequence):
        if seq.status == SequenceStatus.FINISHED:
            return
        if _seq_in_flight(seq):
            return
        if seq in self.running:
            return
        self.running.append(seq)

    def remove(self, seq: Sequence):
        try:
            self.running.remove(seq)
        except ValueError:
            return

    @property
    def runnable(self) -> bool:
        return bool(self.running)

    @property
    def inflight_tokens(self) -> int:
        tokens_reserved = (self.block_size - 1) * len(self.running)
        remain_tokens = sum(
            max(0, seq.max_tokens - seq.num_completion_tokens)
            for seq in self.running
            if seq.status != SequenceStatus.FINISHED
        )
        return tokens_reserved + remain_tokens


@dataclass
class PrefillManager:
    block_size: int
    pending: list[Sequence] = field(default_factory=list)

    def add(self, seq: Sequence):
        if _seq_in_flight(seq):
            return
        self.pending.append(seq)

    def prepend(self, seq: Sequence):
        self.pending.insert(0, seq)

    @property
    def runnable(self) -> bool:
        return bool(self.pending)

    def schedule_next_batch(
        self,
        block_manager: BlockManager,
        token_budget: int,
        reserved_tokens: int,
        max_num_seqs: int,
    ) -> Batch | None:
        if token_budget <= 0 or not self.pending:
            return None

        available_tokens = len(block_manager.free_block_ids) * self.block_size
        scheduled: list[Sequence] = []
        chunked: list[Sequence] = []
        remaining: list[Sequence] = []
        blocked = False

        for seq in self.pending:
            if _seq_in_flight(seq):
                remaining.append(seq)
                continue
            if blocked or token_budget <= 0 or len(scheduled) >= max_num_seqs:
                remaining.append(seq)
                continue

            # Continuation of a previously admitted chunked prefill request.
            if seq.block_table:
                remain_len = seq.num_tokens - seq.num_cached_tokens
                if remain_len <= 0:
                    raise RuntimeError(
                        f"invalid chunked prefill state: seq_id={seq.seq_id}, "
                        f"num_tokens={seq.num_tokens}, num_cached_tokens={seq.num_cached_tokens}"
                    )
                chunk_size = min(token_budget, remain_len)
                seq.num_scheduled_tokens = chunk_size
                seq.is_prefill = True
                token_budget -= chunk_size
                reserved_tokens += remain_len + max(0, seq.max_tokens - seq.num_completion_tokens)
                seq.in_flight = True
                if chunk_size < remain_len:
                    chunked.append(seq)
                scheduled.append(seq)
                continue

            num_cached_blocks = block_manager.can_allocate(seq)
            if num_cached_blocks == -1:
                blocked = True
                remaining.append(seq)
                continue

            cached_tokens = num_cached_blocks * self.block_size
            if cached_tokens == seq.num_tokens and seq.num_tokens > 0:
                cached_tokens -= 1
            remain_len = seq.num_tokens - cached_tokens
            estimated_len = remain_len + max(0, seq.max_tokens - seq.num_completion_tokens)
            if estimated_len + reserved_tokens > available_tokens:
                blocked = True
                remaining.append(seq)
                continue

            block_manager.allocate(seq, num_cached_blocks)
            remain_len = seq.num_tokens - seq.num_cached_tokens
            chunk_size = min(token_budget, remain_len)
            seq.num_scheduled_tokens = chunk_size
            seq.is_prefill = True
            token_budget -= chunk_size
            reserved_tokens += estimated_len
            seq.in_flight = True
            if chunk_size < remain_len:
                chunked.append(seq)
            scheduled.append(seq)

        if not scheduled:
            return None

        self.pending = chunked + remaining
        return Batch(seqs=scheduled, phase="prefill")


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.enable_continuous_batching = config.enable_continuous_batching
        self.decode_steps_per_prefill = config.decode_steps_per_prefill
        self.max_prefill_tokens_per_step = config.max_prefill_tokens_per_step
        self.prefill_budget = (
            config.max_prefill_tokens_per_step
            if config.enable_continuous_batching
            else config.max_num_batched_tokens
        )
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.prefill_manager = PrefillManager(self.block_size)
        self.decode_manager = DecodeManager(self.block_size)
        self._num_inflight_seqs = 0

    def is_finished(self):
        return (
            not self.prefill_manager.runnable
            and not self.decode_manager.runnable
            and self._num_inflight_seqs == 0
        )

    def get_prefix_cache_stats(self, reset: bool = False) -> dict[str, int | float]:
        return self.block_manager.get_prefix_cache_stats(reset=reset)

    def add(self, seq: Sequence):
        seq.in_flight = False
        self.prefill_manager.add(seq)

    def _schedule_prefill(self) -> Batch | None:
        return self.prefill_manager.schedule_next_batch(
            block_manager=self.block_manager,
            token_budget=self.prefill_budget,
            reserved_tokens=self.decode_manager.inflight_tokens,
            max_num_seqs=self.max_num_seqs,
        )

    def _schedule_decode(self) -> Batch | None:
        if not self.decode_manager.running:
            return None
        scheduled: list[Sequence] = []
        attempts = len(self.decode_manager.running)
        while attempts > 0 and len(scheduled) < self.max_num_seqs and self.decode_manager.running:
            attempts -= 1
            seq = self.decode_manager.running.popleft()
            if _seq_in_flight(seq):
                self.decode_manager.running.append(seq)
                continue
            while not self.block_manager.can_append(seq):
                if self.decode_manager.running:
                    self.preempt(self.decode_manager.running.pop())
                else:
                    self.preempt(seq)
                    seq = None
                    break
            if seq is None:
                continue
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)
            seq.in_flight = True
            scheduled.append(seq)
            self.decode_manager.running.append(seq)
        if not scheduled:
            return None
        return Batch(seqs=scheduled, phase="decode")

    def schedule_next_batch(self) -> Batch | None:
        # Ported from mini-sglang's scheduling order: prefill first, decode second.
        batch = self._schedule_prefill() or self._schedule_decode()
        if batch is not None:
            self._num_inflight_seqs += len(batch.seqs)
        return batch

    def schedule(self) -> tuple[list[Sequence], bool]:
        batch = self.schedule_next_batch()
        if batch is None:
            raise RuntimeError("scheduler has neither prefill nor decode work")
        return batch.seqs, batch.is_prefill

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.in_flight = False
        self.block_manager.deallocate(seq)
        self.decode_manager.remove(seq)
        self.prefill_manager.prepend(seq)

    def postprocess(
        self,
        batch: Batch | list[Sequence],
        token_ids: list[int],
        is_prefill: bool | None = None,
    ):
        if isinstance(batch, list):
            if is_prefill is None:
                raise ValueError("is_prefill is required when postprocess() receives seq list")
            batch = Batch(seqs=batch, phase="prefill" if is_prefill else "decode")
        assert len(token_ids) == len(batch.seqs)
        self._num_inflight_seqs -= len(batch.seqs)
        if self._num_inflight_seqs < 0:
            self._num_inflight_seqs = 0
        for seq, token_id in zip(batch.seqs, token_ids):
            seq.in_flight = False
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if batch.is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue

            seq.append_token(token_id)
            finished = (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens
            if finished:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.decode_manager.remove(seq)
            elif batch.is_prefill:
                seq.status = SequenceStatus.RUNNING
                self.decode_manager.add(seq)
