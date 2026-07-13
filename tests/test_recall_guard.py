import numpy as np

from recall_guard import build_recall_index, exact_topk, search, synthetic_vectors


def test_exact_topk_matches_bruteforce():
    vectors = np.asarray([[0, 0], [1, 0], [0, 2], [4, 4]], dtype=np.float32)
    norms = np.einsum("ij,ij->i", vectors, vectors, dtype=np.float32)
    labels, distances = exact_topk(vectors, norms, np.asarray([0.8, 0.1]), 2)
    assert labels.tolist() == [1, 0]
    np.testing.assert_allclose(distances, [0.05, 0.65], atol=1e-6)


def test_certify_then_rectify_returns_valid_neighbors():
    vectors = synthetic_vectors(600, 16, seed=11)
    built = build_recall_index(
        vectors,
        k=5,
        ef_search=8,
        calibration_queries=40,
        target_recall=0.95,
        risk_alpha=0.05,
        seed=11,
    )
    result = search(built, vectors[17] + np.float32(0.01), audit=True)
    assert result["path"] in {"crc_fast_path", "exact_rectifier"}
    assert len(result["labels"]) == 5
    assert 0 <= result["audit"]["recall"] <= 1
    if result["rectified"]:
        assert result["audit"]["recall"] == 1.0


def test_rejects_bad_query_dimension():
    vectors = synthetic_vectors(200, 8, seed=3)
    built = build_recall_index(vectors, k=3, ef_search=5, calibration_queries=20)
    try:
        search(built, [1, 2, 3])
    except ValueError as exc:
        assert "8" in str(exc)
    else:
        raise AssertionError("expected query validation error")
