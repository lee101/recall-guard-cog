"""Cog entrypoint for the RecallGuard HNSW certifier."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from time import perf_counter

import numpy as np
from cog import BasePredictor, Input

from recall_guard import build_recall_index, search, synthetic_vectors


class Predictor(BasePredictor):
    def setup(self) -> None:
        self._cache: OrderedDict[str, object] = OrderedDict()
        self._cache_lock = threading.Lock()

    def _cache_key(self, vectors: np.ndarray, config: tuple[object, ...]) -> str:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(memoryview(vectors).cast("B"))
        digest.update(repr(config).encode())
        return digest.hexdigest()

    def run(
        self,
        query_json: str = Input(
            description="Optional JSON query vector. Omit to generate a reproducible hard query.",
            default="",
        ),
        vectors_json: str = Input(
            description="Optional JSON matrix of vectors. Omit to use a clustered synthetic benchmark.",
            default="",
        ),
        dataset_size: int = Input(default=5000, ge=100, le=100000),
        dimensions: int = Input(default=64, ge=4, le=512),
        neighbors: int = Input(default=10, ge=1, le=100),
        ef_search: int = Input(default=32, ge=2, le=2000),
        calibration_queries: int = Input(default=200, ge=20, le=2000),
        target_recall: float = Input(default=0.95, ge=0.1, le=1.0),
        risk_alpha: float = Input(default=0.05, ge=0.001, le=0.5),
        seed: int = Input(default=7),
        audit: bool = Input(
            description="Compute exact neighbors even on accepted fast-path queries and report observed recall.",
            default=False,
        ),
    ) -> dict:
        parse_started = perf_counter()
        if vectors_json.strip():
            vectors = np.asarray(json.loads(vectors_json), dtype=np.float32)
        else:
            vectors = synthetic_vectors(dataset_size, dimensions, seed)
        if neighbors >= len(vectors):
            raise ValueError("neighbors must be smaller than the dataset")
        if ef_search < neighbors + 1:
            ef_search = neighbors + 1

        if query_json.strip():
            query = np.asarray(json.loads(query_json), dtype=np.float32)
        else:
            rng = np.random.default_rng(seed ^ 0xA11CE)
            anchor = int(rng.integers(0, len(vectors)))
            query = vectors[anchor] + rng.normal(0, 0.03, vectors.shape[1]).astype(np.float32)

        config = (neighbors, ef_search, calibration_queries, target_recall, risk_alpha, seed)
        key = self._cache_key(np.ascontiguousarray(vectors), config)
        with self._cache_lock:
            index = self._cache.get(key)
            if index is not None:
                self._cache.move_to_end(key)
        cache_hit = index is not None
        if index is None:
            index = build_recall_index(
                vectors,
                k=neighbors,
                ef_search=ef_search,
                calibration_queries=calibration_queries,
                target_recall=target_recall,
                risk_alpha=risk_alpha,
                seed=seed,
            )
            with self._cache_lock:
                self._cache[key] = index
                self._cache.move_to_end(key)
                while len(self._cache) > 3:
                    self._cache.popitem(last=False)

        result = search(index, query, audit=audit)
        result["dataset"] = {
            "vectors": int(len(vectors)),
            "dimensions": int(vectors.shape[1]),
            "source": "json" if vectors_json.strip() else "synthetic_clustered",
            "index_cache_hit": cache_hit,
            "index_build_ms": 0.0 if cache_hit else index.build_ms,
            "parse_and_lookup_ms": (perf_counter() - parse_started) * 1_000,
        }
        return result
