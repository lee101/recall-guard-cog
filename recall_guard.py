"""Production-friendly wrapper around the paper's HNSW CRC certifier.

The fast path is the paper's post-hoc conformal-risk-control (CRC) gate.  When
the gate declines a result, the default rectifier is an exact, vectorized L2
scan.  That fallback is intentionally conservative: it makes the service useful
without pretending that a dataset-specific spanner stretch bound is universal.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import hnswlib
import numpy as np

from crc_core import CRCCertifier, crc_decision, fit_crc_certifier


MAX_VECTOR_VALUES = 5_000_000


@dataclass
class RecallIndex:
    vectors: np.ndarray
    squared_norms: np.ndarray
    index: Any
    certifier: CRCCertifier
    calibration_queries: int
    calibration_acceptance: float
    calibration_mean_recall: float
    build_ms: float


def as_vectors(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 4 or array.shape[1] < 2:
        raise ValueError(f"{name} must be a 2D array with at least 4 rows and 2 columns")
    if array.size > MAX_VECTOR_VALUES:
        raise ValueError(f"{name} exceeds the {MAX_VECTOR_VALUES:,}-value safety limit")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return np.ascontiguousarray(array)


def synthetic_vectors(size: int, dimensions: int, seed: int) -> np.ndarray:
    """Create a clustered benchmark where low-ef ANN has meaningful hard cases."""

    if not 100 <= size <= 100_000:
        raise ValueError("dataset_size must be between 100 and 100000")
    if not 4 <= dimensions <= 512:
        raise ValueError("dimensions must be between 4 and 512")
    if size * dimensions > MAX_VECTOR_VALUES:
        raise ValueError(f"requested dataset exceeds the {MAX_VECTOR_VALUES:,}-value safety limit")
    rng = np.random.default_rng(seed)
    cluster_count = min(64, max(4, size // 100))
    centers = rng.normal(size=(cluster_count, dimensions)).astype(np.float32)
    assignments = rng.integers(0, cluster_count, size=size)
    noise = rng.normal(scale=0.18, size=(size, dimensions)).astype(np.float32)
    return np.ascontiguousarray(centers[assignments] + noise)


def calibration_set(vectors: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count < 20:
        raise ValueError("calibration_queries must be at least 20")
    count = min(count, 2_000)
    rng = np.random.default_rng(seed ^ 0x5EEDC0DE)
    source = rng.integers(0, len(vectors), size=count)
    scale = np.std(vectors, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, np.float32(1e-3))
    noise = rng.normal(size=(count, vectors.shape[1])).astype(np.float32)
    return np.ascontiguousarray(vectors[source] + noise * scale * np.float32(0.025))


def exact_topk(
    vectors: np.ndarray,
    squared_norms: np.ndarray,
    query: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact squared-L2 top-k without allocating an N x D difference matrix."""

    q = np.asarray(query, dtype=np.float32).reshape(-1)
    if q.size != vectors.shape[1]:
        raise ValueError(f"query has {q.size} dimensions; expected {vectors.shape[1]}")
    distances = squared_norms - np.float32(2.0) * (vectors @ q) + np.dot(q, q)
    np.maximum(distances, np.float32(0), out=distances)
    candidate_ids = np.argpartition(distances, k - 1)[:k]
    order = np.argsort(distances[candidate_ids], kind="stable")
    labels = candidate_ids[order].astype(np.int64, copy=False)
    return labels, distances[labels].astype(np.float64, copy=False)


def _exact_ground_truth(
    vectors: np.ndarray,
    squared_norms: np.ndarray,
    queries: np.ndarray,
    k: int,
) -> np.ndarray:
    return np.vstack([exact_topk(vectors, squared_norms, q, k)[0] for q in queries])


def build_recall_index(
    vectors: Any,
    *,
    k: int = 10,
    ef_search: int = 32,
    ef_construction: int = 160,
    graph_degree: int = 16,
    target_recall: float = 0.95,
    risk_alpha: float = 0.05,
    calibration_queries: int = 200,
    seed: int = 7,
) -> RecallIndex:
    started = perf_counter()
    x = as_vectors(vectors, name="vectors")
    if not 1 <= k < len(x):
        raise ValueError("k must be positive and smaller than the dataset")
    if ef_search < k + 1:
        raise ValueError("ef_search must be at least k + 1")
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    if not 0 < risk_alpha < 1:
        raise ValueError("risk_alpha must be in (0, 1)")

    index = hnswlib.Index(space="l2", dim=x.shape[1])
    index.init_index(
        max_elements=len(x),
        ef_construction=max(ef_construction, ef_search, k + 1),
        M=max(4, graph_degree),
        random_seed=seed,
    )
    index.add_items(x, np.arange(len(x), dtype=np.int64), num_threads=-1)
    index.set_ef(ef_search)
    index.set_num_threads(-1)

    norms = np.einsum("ij,ij->i", x, x, dtype=np.float32)
    cal = calibration_set(x, calibration_queries, seed)
    gt = _exact_ground_truth(x, norms, cal, k)
    certifier, rows = fit_crc_certifier(
        index,
        cal,
        gt,
        k=k,
        ef_search=ef_search,
        tau=target_recall,
        alpha=risk_alpha,
        use_augmented=False,
    )
    accepted = [row for row in rows if float(row["score"]) >= certifier.theta_hat]
    mean_recall = float(np.mean([row["recall"] for row in accepted])) if accepted else 1.0
    return RecallIndex(
        vectors=x,
        squared_norms=norms,
        index=index,
        certifier=certifier,
        calibration_queries=len(rows),
        calibration_acceptance=len(accepted) / max(len(rows), 1),
        calibration_mean_recall=mean_recall,
        build_ms=(perf_counter() - started) * 1_000,
    )


def search(index: RecallIndex, query: Any, *, audit: bool = False) -> dict[str, Any]:
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    if q.size != index.vectors.shape[1] or not np.isfinite(q).all():
        raise ValueError(f"query must contain {index.vectors.shape[1]} finite values")

    started = perf_counter()
    decision = crc_decision(index.index, q, index.certifier)
    ann_ms = (perf_counter() - started) * 1_000
    accepted = bool(decision["accepted"])
    labels = np.asarray(decision["labels"], dtype=np.int64)
    distances = np.asarray(decision["dists"], dtype=np.float64)
    exact_labels: np.ndarray | None = None
    exact_distances: np.ndarray | None = None
    rectify_ms = 0.0

    if not accepted or audit:
        rectify_started = perf_counter()
        exact_labels, exact_distances = exact_topk(
            index.vectors, index.squared_norms, q, index.certifier.k
        )
        rectify_ms = (perf_counter() - rectify_started) * 1_000
    if not accepted:
        labels, distances = exact_labels, exact_distances

    result: dict[str, Any] = {
        "path": "crc_fast_path" if accepted else "exact_rectifier",
        "accepted_by_crc": accepted,
        "rectified": not accepted,
        "labels": labels.tolist(),
        "distances_squared_l2": distances.tolist(),
        "certificate": {
            "score": float(decision["score"]),
            "threshold": float(index.certifier.theta_hat),
            "target_recall": float(index.certifier.tau),
            "risk_alpha": float(index.certifier.alpha),
            "risk_bound": float(index.certifier.target_bound),
            "calibration_queries": index.calibration_queries,
            "calibration_acceptance": index.calibration_acceptance,
            "calibration_mean_recall_accepted": index.calibration_mean_recall,
        },
        "timing_ms": {
            "ann_and_certificate": ann_ms,
            "rectifier": rectify_ms,
        },
    }
    if audit and exact_labels is not None:
        overlap = len(set(labels.tolist()).intersection(exact_labels.tolist()))
        result["audit"] = {
            "recall": overlap / index.certifier.k,
            "exact_labels": exact_labels.tolist(),
        }
    return result
