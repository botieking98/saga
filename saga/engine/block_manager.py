from collections import deque
import xxhash
import numpy as np

from saga.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_ids: dict[int, set[int]] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self._reset_prefix_cache_stats()

    def _reset_prefix_cache_stats(self):
        self.prefix_cache_queries = 0
        self.prefix_cache_query_tokens = 0
        self.prefix_cache_hit_queries = 0
        self.prefix_cache_hit_blocks = 0
        self.prefix_cache_hit_tokens = 0
        self.prefix_cache_full_hit_queries = 0
        self.prefix_cache_max_hit_blocks = 0
        self.prefix_cache_max_hit_tokens = 0

    def get_prefix_cache_stats(self, reset: bool = False) -> dict[str, int | float]:
        queries = self.prefix_cache_queries
        query_tokens = self.prefix_cache_query_tokens
        hit_queries = self.prefix_cache_hit_queries
        hit_blocks = self.prefix_cache_hit_blocks
        hit_tokens = self.prefix_cache_hit_tokens
        stats: dict[str, int | float] = {
            "queries": queries,
            "query_tokens": query_tokens,
            "hit_queries": hit_queries,
            "hit_blocks": hit_blocks,
            "hit_tokens": hit_tokens,
            "full_hit_queries": self.prefix_cache_full_hit_queries,
            "max_hit_blocks": self.prefix_cache_max_hit_blocks,
            "max_hit_tokens": self.prefix_cache_max_hit_tokens,
            "query_hit_rate": (hit_queries / queries) if queries > 0 else 0.0,
            "token_hit_rate": (hit_tokens / query_tokens) if query_tokens > 0 else 0.0,
        }
        if reset:
            self._reset_prefix_cache_stats()
        return stats

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1:
            self._remove_block_hash(block.hash, block_id)
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _add_block_hash(self, hash_value: int, block_id: int):
        block_ids = self.hash_to_block_ids.get(hash_value)
        if block_ids is None:
            self.hash_to_block_ids[hash_value] = {block_id}
            return
        block_ids.add(block_id)

    def _remove_block_hash(self, hash_value: int, block_id: int):
        block_ids = self.hash_to_block_ids.get(hash_value)
        if block_ids is None:
            return
        block_ids.discard(block_id)
        if not block_ids:
            del self.hash_to_block_ids[hash_value]

    def _touch_free_block(self, block_id: int):
        if block_id in self.used_block_ids:
            return
        try:
            self.free_block_ids.remove(block_id)
        except ValueError:
            return
        self.free_block_ids.append(block_id)

    def _find_cached_block(self, hash_value: int, token_ids: list[int]) -> int:
        block_ids = self.hash_to_block_ids.get(hash_value)
        if not block_ids:
            return -1
        free_candidate = -1
        for block_id in block_ids:
            if self.blocks[block_id].token_ids != token_ids:
                continue
            # Prefer already-used blocks to reduce pressure on the free queue.
            if block_id in self.used_block_ids:
                return block_id
            if free_candidate == -1:
                free_candidate = block_id
        return free_candidate

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        # Prefix cache only contains full blocks.
        num_full_blocks = seq.num_tokens // self.block_size
        for i in range(num_full_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self._find_cached_block(h, token_ids)
            if block_id == -1:
                break
            num_cached_blocks += 1
            self._touch_free_block(block_id)
            if block_id in self.used_block_ids:
                num_new_blocks -= 1

        hit_tokens = min(seq.num_tokens, num_cached_blocks * self.block_size)
        self.prefix_cache_queries += 1
        self.prefix_cache_query_tokens += seq.num_tokens
        self.prefix_cache_hit_blocks += num_cached_blocks
        self.prefix_cache_hit_tokens += hit_tokens
        self.prefix_cache_max_hit_blocks = max(self.prefix_cache_max_hit_blocks, num_cached_blocks)
        self.prefix_cache_max_hit_tokens = max(self.prefix_cache_max_hit_tokens, hit_tokens)
        if num_cached_blocks > 0:
            self.prefix_cache_hit_queries += 1
        if seq.num_tokens > 0 and hit_tokens == seq.num_tokens:
            self.prefix_cache_full_hit_queries += 1

        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self._find_cached_block(h, token_ids)
            assert block_id != -1
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size
        # If full prompt is cache hit, we still recompute one tail token
        # to produce the first-step logits with the current request context.
        if seq.num_cached_tokens == seq.num_tokens and seq.num_tokens > 0:
            seq.num_cached_tokens -= 1

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            if block.hash != -1 and block.hash != h:
                self._remove_block_hash(block.hash, block.block_id)
            block.update(h, token_ids)
            self._add_block_hash(h, block.block_id)
