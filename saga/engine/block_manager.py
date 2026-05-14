from collections import deque
import time

from saga.engine.sequence import Sequence


class RadixTreeNode:

    def __init__(self, key: tuple[int, ...] | None, parent: "RadixTreeNode | None"):
        self.key = key
        self.parent = parent
        self.children: dict[tuple[int, ...], RadixTreeNode] = {}
        self.block_ids: set[int] = set()
        self.timestamp_ns = time.monotonic_ns()

    def touch(self):
        self.timestamp_ns = time.monotonic_ns()

    @property
    def is_root(self) -> bool:
        return self.parent is None


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.token_ids: tuple[int, ...] = ()
        self.cache_node: RadixTreeNode | None = None

    def update(self, token_ids: tuple[int, ...], cache_node: RadixTreeNode):
        self.token_ids = token_ids
        self.cache_node = cache_node

    def reset(self):
        self.ref_count = 1
        self.token_ids = ()
        self.cache_node = None


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.radix_root = RadixTreeNode(None, None)
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

    def _cleanup_node(self, node: RadixTreeNode):
        while not node.is_root and not node.children and not node.block_ids:
            parent = node.parent
            assert parent is not None and node.key is not None
            del parent.children[node.key]
            node = parent

    def _detach_block_from_cache(self, block_id: int):
        block = self.blocks[block_id]
        node = block.cache_node
        if node is None:
            return
        node.block_ids.discard(block_id)
        block.cache_node = None
        block.token_ids = ()
        self._cleanup_node(node)

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self._detach_block_from_cache(block_id)
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _touch_free_block(self, block_id: int):
        if block_id in self.used_block_ids:
            return
        try:
            self.free_block_ids.remove(block_id)
        except ValueError:
            return
        self.free_block_ids.append(block_id)

    def _pick_cached_block_id(self, node: RadixTreeNode, token_ids: tuple[int, ...]) -> int:
        stale_block_ids: list[int] = []
        used_candidate = -1
        free_candidate = -1
        for block_id in tuple(node.block_ids):
            block = self.blocks[block_id]
            if block.cache_node is not node or block.token_ids != token_ids:
                stale_block_ids.append(block_id)
                continue
            if block_id in self.used_block_ids:
                used_candidate = block_id
                break
            if free_candidate == -1:
                free_candidate = block_id
        for block_id in stale_block_ids:
            node.block_ids.discard(block_id)
        if used_candidate != -1:
            return used_candidate
        return free_candidate

    def _match_cached_prefix(self, seq: Sequence) -> list[int]:
        node = self.radix_root
        matched: list[int] = []
        num_full_blocks = seq.num_tokens // self.block_size
        for i in range(num_full_blocks):
            token_ids = tuple(seq.block(i))
            child = node.children.get(token_ids)
            if child is None:
                break
            block_id = self._pick_cached_block_id(child, token_ids)
            if block_id == -1:
                self._cleanup_node(child)
                break
            if block_id not in self.used_block_ids:
                self._touch_free_block(block_id)
            child.touch()
            matched.append(block_id)
            node = child
        return matched

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        matched_block_ids = self._match_cached_prefix(seq)
        num_cached_blocks = len(matched_block_ids)
        num_new_blocks = seq.num_blocks
        for block_id in matched_block_ids:
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
        node = self.radix_root
        for i in range(num_cached_blocks):
            token_ids = tuple(seq.block(i))
            child = node.children.get(token_ids)
            if child is None:
                raise RuntimeError("radix cache inconsistent: missing cached child node")
            block_id = self._pick_cached_block_id(child, token_ids)
            if block_id == -1:
                raise RuntimeError("radix cache inconsistent: missing cached block id")
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            child.touch()
            seq.block_table.append(block_id)
            node = child
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
        if start == end:
            return

        node = self.radix_root
        for i in range(start):
            token_ids = tuple(seq.block(i))
            child = node.children.get(token_ids)
            if child is None:
                child = RadixTreeNode(token_ids, node)
                node.children[token_ids] = child
            node = child
            node.touch()

        for i in range(start, end):
            token_ids = tuple(seq.block(i))
            child = node.children.get(token_ids)
            if child is None:
                child = RadixTreeNode(token_ids, node)
                node.children[token_ids] = child
            block_id = seq.block_table[i]
            block = self.blocks[seq.block_table[i]]
            if block.cache_node is not child:
                self._detach_block_from_cache(block_id)
                child.block_ids.add(block_id)
                block.update(token_ids, child)
            else:
                child.block_ids.add(block_id)
                block.token_ids = token_ids
            child.touch()
            node = child
