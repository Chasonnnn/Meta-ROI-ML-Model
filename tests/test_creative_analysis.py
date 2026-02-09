import numpy as np
import pytest

from meta_elv.creatives.analysis import cosine_topk_neighbors, kmeans_clusters


def test_cosine_topk_neighbors_excludes_self_and_has_expected_rows() -> None:
    ids = ["a", "b", "c"]
    emb = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    df = cosine_topk_neighbors(ids, emb, top_k=1)
    assert set(df.columns) == {"ad_id", "neighbor_ad_id", "similarity"}
    assert len(df) == 3  # n * k
    assert not (df["ad_id"] == df["neighbor_ad_id"]).any()

    row_a = df[df["ad_id"] == "a"].iloc[0]
    assert row_a["neighbor_ad_id"] == "c"
    assert row_a["similarity"] == pytest.approx(1.0, abs=1e-6)

    row_c = df[df["ad_id"] == "c"].iloc[0]
    assert row_c["neighbor_ad_id"] == "a"
    assert row_c["similarity"] == pytest.approx(1.0, abs=1e-6)

    row_b = df[df["ad_id"] == "b"].iloc[0]
    assert row_b["neighbor_ad_id"] in {"a", "c"}
    assert row_b["similarity"] == pytest.approx(0.0, abs=1e-6)

    df0 = cosine_topk_neighbors(ids, emb, top_k=0)
    assert len(df0) == 0


def test_kmeans_clusters_handles_edge_cases_and_shapes() -> None:
    empty = np.empty((0, 4), dtype=np.float32)
    res0 = kmeans_clusters(empty, n_clusters=5)
    assert res0.n_clusters == 0
    assert res0.labels.shape == (0,)

    emb = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    res1 = kmeans_clusters(emb, n_clusters=1)
    assert res1.n_clusters == 1
    assert res1.labels.shape == (3,)
    assert (res1.labels == 0).all()

    res2 = kmeans_clusters(emb, n_clusters=10, random_seed=123)
    assert res2.n_clusters == 3  # clipped to n
    assert res2.labels.shape == (3,)
    assert res2.labels.dtype.kind in {"i", "u"}
    assert set(res2.labels.tolist()).issubset({0, 1, 2})

    emb_nan = emb.copy()
    emb_nan[0, 0] = np.nan
    res3 = kmeans_clusters(emb_nan, n_clusters=2, random_seed=123)
    assert res3.n_clusters == 2
    assert res3.labels.shape == (3,)

