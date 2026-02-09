from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def cosine_topk_neighbors(
    ids: list[str],
    embeddings: np.ndarray,
    *,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Return a long-form neighbor table: (id, neighbor_id, cosine_similarity).
    """
    if len(ids) != embeddings.shape[0]:
        raise ValueError("ids length must match embeddings rows")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D array")
    n = embeddings.shape[0]
    if n == 0:
        return pd.DataFrame(columns=["ad_id", "neighbor_ad_id", "similarity"])
    k = int(top_k)
    if k <= 0:
        return pd.DataFrame(columns=["ad_id", "neighbor_ad_id", "similarity"])
    k = min(k, max(0, n - 1))

    x = _l2_normalize(embeddings.astype(np.float32, copy=False))
    sim = x @ x.T
    # Exclude self
    np.fill_diagonal(sim, -np.inf)

    rows: list[dict[str, Any]] = []
    for i in range(n):
        # argsort is fine for typical creative counts (tens/hundreds)
        nn = np.argsort(-sim[i])[:k]
        for j in nn:
            rows.append(
                {
                    "ad_id": ids[i],
                    "neighbor_ad_id": ids[int(j)],
                    "similarity": float(sim[i, int(j)]),
                }
            )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    n_clusters: int


def kmeans_clusters(
    embeddings: np.ndarray,
    *,
    n_clusters: int = 10,
    random_seed: int = 7,
) -> ClusterResult:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D array")
    n = embeddings.shape[0]
    if n == 0:
        return ClusterResult(labels=np.array([], dtype=int), n_clusters=0)
    k = int(n_clusters)
    if k <= 1:
        return ClusterResult(labels=np.zeros(n, dtype=int), n_clusters=1)
    k = min(k, n)

    x = embeddings.astype(np.float32, copy=False)
    # KMeans expects finite values
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    km = KMeans(n_clusters=k, n_init=10, random_state=int(random_seed))
    labels = km.fit_predict(x)
    return ClusterResult(labels=labels.astype(int), n_clusters=k)

