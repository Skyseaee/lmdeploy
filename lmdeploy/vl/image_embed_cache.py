################################################################################
# @Copyright: 2019-2025 Shopee. All Rights Reserved.
# @Author   : wenlong.cao@shopee.com
# @Date     : 2025-04-10 14:36:54
# @Details  : This module provides functionality for managing image embeddings 
#             with a Least Recently Used (LRU) cache.
################################################################################
import time
import math
import os
import sys
from typing import Any, Optional
from collections import deque
from threading import Lock, RLock

import numpy as np
import torch

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

IMAGE_EMBEDDING_CACHE_MAX_MEMORY_SIZE = int(os.environ.get("ONELLM_VL_CACHE_MEMORY_SIZE ", 10*(1024**3))) # 10GB
IMAGE_EMBEDDING_CACHE_FLASH_TIME = int(os.environ.get("ONELLM_VL_CACHE_FLASH_TIME", 600))  # seconds
IMAGE_EMBEDDING_CACHE_STAT_WINDOW_SIZE = int(os.environ.get("ONELLM_VL_CACHE_STAT_WINDOW_SIZE", 1000))

def generate_get_total_size_fn(sample):
    code_lines = []
    var_counter = [0]
    seen = set()

    def trace(obj, var_name):
        obj_id = id(obj)
        if obj_id in seen:
            code_lines.append(f"# skip already seen {var_name}")
            code_lines.append(f"size += 0")
            return
        seen.add(obj_id)

        if isinstance(obj, torch.Tensor):
            # 改这里！动态调用 element_size() 和 numel()
            code_lines.append(f"size += {var_name}.element_size() * {var_name}.numel()")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                # key
                key_var = f"{var_name}_k{var_counter[0]}"
                var_counter[0] += 1
                code_lines.append(f"{key_var} = {repr(k)}")
                code_lines.append(f"size += sys.getsizeof({key_var})")
                # value
                value_var = f"{var_name}_v{var_counter[0]}"
                var_counter[0] += 1
                code_lines.append(f"{value_var} = {var_name}[{repr(k)}]")
                trace(v, value_var)
            code_lines.append(f"size += sys.getsizeof({var_name})")
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for idx, item in enumerate(obj):
                item_var = f"{var_name}_{var_counter[0]}"
                var_counter[0] += 1
                code_lines.append(f"{item_var} = {var_name}[{idx}]")
                trace(item, item_var)
            code_lines.append(f"size += sys.getsizeof({var_name})")
        else:
            code_lines.append(f"size += sys.getsizeof({var_name})")

    trace(sample, 'obj')

    body_code = "\n    ".join(code_lines)
    src = f"""
def specialized_get_total_size(obj):
    import sys
    size = 0
    {body_code}
    return size
""".strip()

    local_env = {}
    exec(src, {'sys': sys, 'torch': torch}, local_env)
    specialized_fn = local_env['specialized_get_total_size']

    print("[VLCache🍬] Generated specialized get_total_size function:\n" + "=="*50)
    print(src)
    print("=="*50)
    return specialized_fn


def generate_move_to_cpu_fn(sample):
    """
    def move_to_cpu(result):
        if isinstance(result, torch.Tensor):
            return result.cpu()
        elif isinstance(result, list):
            return [move_to_cpu(x) for x in result]
        elif isinstance(result, dict):
            return {k: move_to_cpu(v) for k, v in result.items()}
        else:
            return result
    """
    code_lines = []
    var_counter = [0]
    def trace(obj, var_name):
        if isinstance(obj, torch.Tensor):
            code_lines.append(f"{var_name} = {var_name}.cpu()")
        elif isinstance(obj, list):
            items = []
            code_lines.append(f"if {var_name} != None:")
            for idx, elem in enumerate(obj):
                child_var = f"{var_name}_{var_counter[0]}"
                var_counter[0] += 1
                code_lines.append(f"    {child_var} = {var_name}[{idx}]")
                trace(elem, child_var)
                items.append(child_var)
            code_lines.append(f"    {var_name} = [{', '.join(items)}]")
        elif isinstance(obj, dict):
            items = []
            for key, val in obj.items():
                child_var = f"{var_name}_{var_counter[0]}"
                var_counter[0] += 1
                code_lines.append(f"{child_var} = {var_name}[{repr(key)}]")
                trace(val, child_var)
                items.append(f"{repr(key)}: {child_var}")
            code_lines.append(f"{var_name} = {{{', '.join(items)}}}")
        else:
            pass  # 其他类型不需要处理

    trace(sample, 'result')  # 注意这里第一个参数是名字

    full_code = "\n    ".join(code_lines)  # 缩进 4 格，适配函数体
    src = f"def specialized_move_to_cpu(result):\n    {full_code}\n    return result"

    # 创建函数
    local_env = {}
    exec(src, {}, local_env)
    specialized_fn = local_env['specialized_move_to_cpu']
    
    print("[VLCache🍬] Generated specialized move_to_cpu function:\n" + "=="*50)
    print(src)
    print("=="*50)
    return specialized_fn


class LazyRunTracer:
    compile_ops = {
        "to_cpu": generate_move_to_cpu_fn,
        "get_total_size" : generate_get_total_size_fn,
    }
    def __init__(self, key="to_cpu"):
        if key not in LazyRunTracer.compile_ops:
            raise ValueError(f"LazyRunTracer Invalid key: {key}")
        self.key = key
        self._compiled_fn = None
        self._lock = Lock()

    def __call__(self, obj):
        if self._compiled_fn is None:
            with self._lock:
                self._compiled_fn = LazyRunTracer.compile_ops[self.key](obj)
        return self._compiled_fn(obj)

def get_image_embedding_max_capacity(
        max_memory: int=1024**3, 
        embedding_size: int = 5120*256*6) -> int:
    """5120*256*2*patch_size = 2.5MB*patch_size
    1: 2.5MB 
    6: 15MB
    12: 30MB
    """
    dtype_size = 2  # float16
    return int(max_memory / (embedding_size * dtype_size))

get_total_size = LazyRunTracer("get_total_size")

class LRUCacheNode:
    def __init__(self, key: str, value):
        self.key = key
        self.value = value  # Embedding 向量
        self.prev: Optional[LRUCacheNode] = None
        self.next: Optional[LRUCacheNode] = None

    @property
    def memory_bytes(self) -> int:
        return get_total_size(self.value)

class EmbeddingCacheInterface(object):
    """
    An interface for embedding cache implementations.
    
    This interface defines the basic methods that any embedding cache should implement.
    It is used to ensure that different cache implementations can be used interchangeably.
    """
    def get(self, key: str):
        """Retrieve the embedding for the given key."""
        raise NotImplementedError

    def put(self, key: str, value) -> None:
        """Insert or update the embedding for the given key."""
        raise NotImplementedError

    @property
    def size(self):
        """Get the current size of the cache."""
        raise NotImplementedError

    @property
    def max_capacity(self):
        """Get the maximum capacity of the cache."""
        raise NotImplementedError

    @property
    def hit_rate(self) -> float:
        """Get the hit rate of the cache."""
        raise NotImplementedError

class EmbeddingLRUCache(EmbeddingCacheInterface):
    """
    A Least Recently Used (LRU) cache implementation for storing embeddings.

    This class maintains a fixed-capacity cache of embeddings, where the least recently
    used items are evicted when the cache exceeds its capacity.

    It also tracks the hit rate of cache requests over a sliding window.
    """
    def __init__(self, capacity: int,
                 max_memory_size: int=IMAGE_EMBEDDING_CACHE_MAX_MEMORY_SIZE,
                 stats_window_size: int=IMAGE_EMBEDDING_CACHE_STAT_WINDOW_SIZE):
        self.capacity = capacity
        self.cache = {}
        self.head = LRUCacheNode("head", np.array([]))
        self.tail = LRUCacheNode("tail", np.array([]))
        self.head.next = self.tail
        self.tail.prev = self.head

        self._request_history = deque(maxlen=stats_window_size)
        self.lock = Lock()
        self.max_memory_size = max_memory_size
        self.total_memory_size = 0
        self.capacity_timestamp = time.time()
        logger.info(f"[VLCache🍬] Initialized LRU Cache with capacity: {capacity}, "
                    f"max_memory_size: {max_memory_size/(1024*1024):.2f}MB, "
                    f"stats_windows_size:{stats_window_size}")

    def get(self, key: str):
        """获取缓存中的 Embedding，若存在则移至链表头部"""
        with self.lock:
            if key not in self.cache:
                self._request_history.append(0)
                return None
            node = self.cache[key]
            self._move_to_head(node)
            self._request_history.append(1)
            return node.value

    @property
    def size(self):
        return len(self.cache)

    @property
    def max_capacity(self):
        if time.time() - self.capacity_timestamp < IMAGE_EMBEDDING_CACHE_FLASH_TIME:
            return self.capacity

        def calculate_total_bytes_and_size():
            total_bytes = sum(c.memory_bytes for c in self.cache.values())
            avg_size = max(total_bytes / max(self.size, 1),  1024)
            return total_bytes, avg_size

        total_bytes, avg_size = calculate_total_bytes_and_size()
        tmp_capacity = math.floor(self.max_memory_size / avg_size)
        if total_bytes > self.max_memory_size or tmp_capacity < self.size:
            excess = self.size - tmp_capacity

            with self.lock:
                for _ in range(excess):
                    self._remove_tail()

            total_bytes, avg_size = calculate_total_bytes_and_size()
            self.total_memory_size = total_bytes
            self.capacity = tmp_capacity
            logger.info(f"[VLCache🍬] Capacity Update, Average size: {avg_size/(1024*1024):.2f}MB, Total memory: {total_bytes/(1024*1024):.2}MB")
        self.capacity_timestamp = time.time()
        return self.capacity

    def reset_cache(self):
        with self.lock:
            self.cache.clear()
            self.head.next = self.tail
            self.tail.prev = self.head
            self.total_memory_size = 0
            self._request_history.clear()
            self.capacity_timestamp = time.time()
            logger.info("[VLCache🍬] Cache has been reset.")
        
    def put(self, key: str, value) -> None:
        """插入或更新 Embedding"""
        with self.lock:
            if key in self.cache:
                node = self.cache[key]
                node.value = value
                self._move_to_head(node)
            else:
                if len(self.cache) >= self.capacity:
                    self._remove_tail()
                new_node = LRUCacheNode(key, value)
                self.cache[key] = new_node
                self._add_to_head(new_node)

    def _move_to_head(self, node: LRUCacheNode) -> None:
        """将节点移动到链表头部"""
        self._remove_node(node)
        self._add_to_head(node)

    def _add_to_head(self, node: LRUCacheNode) -> None:
        """在头部插入节点"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: LRUCacheNode) -> None:
        """移除指定节点"""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _remove_tail(self) -> None:
        """移除尾部节点（最近最少使用）"""
        tail_node = self.tail.prev
        if tail_node != self.head:
            del self.cache[tail_node.key]
            self._remove_node(tail_node)

    @property
    def hit_rate(self) -> float:
        """滑动窗口命中率"""
        if len(self._request_history) == 0:
            return 0.0
        hits = sum(self._request_history)
        return (hits / len(self._request_history)) * 100

class EmbeddingLRUCacheWithLock:
    """
    A Least Recently Used (LRU) cache implementation for storing embeddings.

    This class maintains a fixed-capacity cache of embeddings, where the least recently
    used items are evicted when the cache exceeds its capacity.

    It also tracks the hit rate of cache requests over a sliding window.
    
    TODO: 
    - Add a method to clear the cache.
    """
    def __init__(self, capacity: int, max_memory_size:int=IMAGE_EMBEDDING_CACHE_MAX_MEMORY_SIZE):
        self.capacity = capacity
        self.cache = {}
        self.head = LRUCacheNode("head", np.array([]))
        self.tail = LRUCacheNode("tail", np.array([]))
        self.head.next = self.tail
        self.tail.prev = self.head

        self.lock = Lock()
        self.max_memory_size = max_memory_size
        self.total_memory_size = 0
        self.capacity_timestamp = time.time()

    def get(self, key: str):
        """获取缓存中的 Embedding，若存在则移至链表头部"""
        with self.lock:
            if key not in self.cache:
                return None
            node = self.cache[key]
            self._move_to_head(node)
            return node.value

    @property
    def size(self):
        return len(self.cache)

    @property
    def max_capacity(self):
        if time.time() - self.capacity_timestamp < IMAGE_EMBEDDING_CACHE_FLASH_TIME:
            return self.capacity

        def calculate_total_bytes_and_size():
            total_bytes = sum(c.memory_bytes for c in self.cache.values())
            avg_size = max(total_bytes / max(self.size, 1),  1024)
            return total_bytes, avg_size

        total_bytes, avg_size = calculate_total_bytes_and_size()
        tmp_capacity = math.floor(self.max_memory_size / avg_size)
        if total_bytes > self.max_memory_size or tmp_capacity < self.size:
            excess = self.size - tmp_capacity

            with self.lock:
                for _ in range(excess):
                    self._remove_tail()

            total_bytes, avg_size = calculate_total_bytes_and_size()
            self.total_memory_size = total_bytes
            self.capacity = tmp_capacity
            logger.info(f"[VLCache🍬] Capacity Update, Average size: {avg_size/(1024*1024):.2f}MB, Total memory: {total_bytes/(1024*1024):.2}MB")
        self.capacity_timestamp = time.time()
        return self.capacity

    def reset_cache(self):
        with self.lock:
            self.cache.clear()
            self.head.next = self.tail
            self.tail.prev = self.head
            self.total_memory_size = 0
            # self._request_history.clear()
            self.capacity_timestamp = time.time()
            logger.info("[VLCache🍬] Cache has been reset.")

    def put(self, key: str, value) -> None:
        """插入或更新 Embedding"""
        with self.lock:
            if key in self.cache:
                node = self.cache[key]
                node.value = value
                self._move_to_head(node)
            else:
                if len(self.cache) >= self.capacity:
                    self._remove_tail()
                new_node = LRUCacheNode(key, value)
                self.cache[key] = new_node
                self._add_to_head(new_node)

    def _move_to_head(self, node: LRUCacheNode) -> None:
        """将节点移动到链表头部"""
        self._remove_node(node)
        self._add_to_head(node)

    def _add_to_head(self, node: LRUCacheNode) -> None:
        """在头部插入节点"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: LRUCacheNode) -> None:
        """移除指定节点"""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _remove_tail(self) -> None:
        """移除尾部节点（最近最少使用）"""
        tail_node = self.tail.prev
        if tail_node != self.head:
            del self.cache[tail_node.key]
            self._remove_node(tail_node)
            
class ShardsEmbeddingLRUCache(EmbeddingCacheInterface):
    def __init__(self, capacity: int, 
                 stats_window_size=IMAGE_EMBEDDING_CACHE_STAT_WINDOW_SIZE, 
                 shards=16):
        self.stats_window_size = stats_window_size
        self.shards = [{"lock": RLock(), "cache": EmbeddingLRUCacheWithLock(capacity//shards, max_memory_size=IMAGE_EMBEDDING_CACHE_MAX_MEMORY_SIZE//shards)} for _ in range(shards)]
        self._request_history = deque(maxlen=stats_window_size)
        self.capacity = capacity
        self.capacity_timestamp = time.time()
    
    def _get_shard(self, key):
        return self.shards[hash(key) % len(self.shards)]
    
    def get(self, key):
        shard = self._get_shard(key)
        with shard["lock"]:
            value = shard["cache"].get(key)
            self._request_history.append(1 if value != None else 0)
            return value
    
    def put(self, key, value):
        shard = self._get_shard(key)
        with shard["lock"]:
            shard["cache"].put(key, value)

    @property
    def size(self):
        total_size = sum(shard["cache"].size for shard in self.shards)
        return total_size

    @property
    def max_capacity(self):
        if time.time() - self.capacity_timestamp < IMAGE_EMBEDDING_CACHE_FLASH_TIME:
            return self.capacity
        self.capacity = sum(shard["cache"].max_capacity for shard in self.shards)
        logger.info(f"[VLCache🍬] Cache size: {self.size}/{self.capacity}")
        self.capacity_timestamp = time.time()
        return self.capacity

    @property
    def hit_rate(self) -> float:
        """滑动窗口命中率"""
        if len(self._request_history) == 0:
            return 0.0
        hits = sum(self._request_history)
        return (hits / len(self._request_history)) * 100

def check_kvcache_deps_install():
    try:
        import ais_kvcache  # noqa: F401
    except ImportError:
        logger.warning("[VLCache🍬] try to prepare environment for ais_kvcache, about 10min")
        os.system(f"{sys.executable} -m pip install shopee-ais-kvcache-sdk -i https://pypi.shopee.io/")
        os.execv(sys.executable, [sys.executable] + sys.argv)
        try:
            import ais_kvcache  # noqa: F401
        except ImportError:
            raise ImportError(
                'please install ais_kvcache by pip install ais_kvcache'  # noqa: E501
            )

class EmbeddingMooncake(EmbeddingCacheInterface):
    def __init__(self, model_name, 
                    stats_window_size=IMAGE_EMBEDDING_CACHE_STAT_WINDOW_SIZE, 
                    *args, **kwargs): 
        import ais_kvcache
        self.store = ais_kvcache.Store(
            config=ais_kvcache.StoreConfig(
                storage_backend_name=os.getenv("BACKEND", "MOONCAKE")
            ),
            metadata=ais_kvcache.StoreMetadata(
                model_name=model_name,
                world_size=2,
                worker_id=0,
                layer_id=0,
                kv_shape=(1, 1, 1, 256, 1024),
            ),
        )

        self._request_history = deque(maxlen=stats_window_size)
        self._fake_tokens = torch.tensor([1])
        self.count = 0
        self.lock = Lock()
        import atexit
        atexit.register(self.cleanup)
        logger.info(f"[VLCache🍬] Initialized Cache with model_name: {model_name}, "
                    f"stats_windows_size:{stats_window_size}")
    def get(self, key: str):
        """Get the value from the cache."""
        image_embedding, _ = self.store.get(tokens=self._fake_tokens, key=key)
        with self.lock:
            if image_embedding is None:
                self._request_history.append(0)
                return None
            else:
                self._request_history.append(1)
                return image_embedding[0]
    
    def put(self, key: str, value: torch.Tensor):
        """Put the value into the cache."""
        self.store.put(tokens=self._fake_tokens, kv_caches=[value], key=key)

    def cleanup(self):
        """Clear the cache."""
        self.store.clear()
        self.store.close()
        del self.store
        
    @property
    def hit_rate(self) -> float:
        """slide window for hit rate"""
        if len(self._request_history) == 0:
            return 0.0
        hits = sum(self._request_history)
        return (hits / len(self._request_history)) * 100
    
    @property
    def size(self):
        return "inf"

    @property
    def max_capacity(self):
        return "inf"
    