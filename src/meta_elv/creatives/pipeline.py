from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import cosine_topk_neighbors, kmeans_clusters
from .embeddings import CreativeMediaMap, embed_with_clip, load_creative_media_map, write_embeddings_npz


@dataclass(frozen=True)
class CreativeAnalysisOutputs:
    assets_csv: Path
    embeddings_npz: Path
    neighbors_csv: Path
    clusters_csv: Path
    cluster_summary_csv: Path | None
    warnings: list[str]


def _ad_perf_from_predictions(pred_path: Path) -> pd.DataFrame | None:
    try:
        d = pd.read_csv(pred_path)
    except Exception:
        return None
    if "ad_id" not in d.columns:
        return None
    # Minimal aggregate
    agg: dict[str, tuple[str, str]] = {"lead_count": ("lead_id", "size") if "lead_id" in d.columns else ("ad_id", "size")}
    if "p_qualified_14d" in d.columns:
        agg["avg_p_qualified_14d"] = ("p_qualified_14d", "mean")
    if "elv" in d.columns:
        agg["avg_elv"] = ("elv", "mean")
        agg["predicted_elv"] = ("elv", "sum")
    out = d.dropna(subset=["ad_id"]).groupby("ad_id", as_index=False).agg(**agg)
    out["lead_count"] = out["lead_count"].fillna(0).astype(int)
    return out


def run_creative_analysis(
    *,
    creative_map_path: Path,
    media_dir: Path | None,
    out_dir: Path,
    run_dir: Path | None = None,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: str = "cpu",
    batch_size: int = 16,
    num_video_frames: int = 4,
    neighbors: int = 10,
    clusters: int = 10,
    random_seed: int = 7,
) -> CreativeAnalysisOutputs:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    media_map: CreativeMediaMap = load_creative_media_map(Path(creative_map_path), media_dir=media_dir)
    emb = embed_with_clip(
        media_map,
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        batch_size=batch_size,
        num_video_frames=num_video_frames,
    )

    # Clusters + neighbors
    cluster = kmeans_clusters(emb.embeddings, n_clusters=clusters, random_seed=random_seed)
    clusters_df = pd.DataFrame({"ad_id": emb.ids, "cluster_id": cluster.labels.astype(int)})
    neighbors_df = cosine_topk_neighbors(emb.ids, emb.embeddings, top_k=neighbors)

    # Assets table (no embeddings)
    assets = media_map.df.copy()
    keep = [c for c in ["ad_id", "media_path", "media_type"] if c in assets.columns]
    assets = assets[keep].copy()
    assets = assets.merge(clusters_df, on="ad_id", how="left")

    # Optional: summarize cluster performance if a run_dir is provided and has predictions.csv
    cluster_summary_csv = None
    if run_dir is not None:
        pred_path = Path(run_dir) / "predictions.csv"
        perf = _ad_perf_from_predictions(pred_path)
        if perf is not None and len(perf):
            merged = assets.merge(perf, on="ad_id", how="left")
            # Weighted average by lead_count where possible
            def wavg(x: pd.Series, w: pd.Series) -> float:
                if w.sum() <= 0:
                    return float(x.mean()) if len(x) else 0.0
                return float((x.fillna(0.0) * w).sum() / w.sum())

            g = merged.groupby("cluster_id", as_index=False)
            summary_rows: list[dict[str, Any]] = []
            for cid, sub in g:
                w = sub.get("lead_count", pd.Series([0] * len(sub)))
                row: dict[str, Any] = {
                    "cluster_id": int(cid) if cid is not None and not (isinstance(cid, float) and np.isnan(cid)) else -1,
                    "n_ads": int(len(sub)),
                    "lead_count": int(sub.get("lead_count", pd.Series([0] * len(sub))).fillna(0).sum()),
                }
                if "avg_p_qualified_14d" in sub.columns:
                    row["avg_p_qualified_14d"] = wavg(sub["avg_p_qualified_14d"], w)
                if "avg_elv" in sub.columns:
                    row["avg_elv"] = wavg(sub["avg_elv"], w)
                if "predicted_elv" in sub.columns:
                    row["predicted_elv"] = float(sub["predicted_elv"].fillna(0.0).sum())
                summary_rows.append(row)
            summary = pd.DataFrame(summary_rows).sort_values(["predicted_elv", "lead_count"], ascending=False)
            cluster_summary_csv = out_dir / "creative_cluster_summary.csv"
            summary.to_csv(cluster_summary_csv, index=False)

    # Write artifacts
    assets_csv = out_dir / "creative_assets.csv"
    assets.to_csv(assets_csv, index=False)
    neighbors_csv = out_dir / "creative_neighbors.csv"
    neighbors_df.to_csv(neighbors_csv, index=False)
    clusters_csv = out_dir / "creative_clusters.csv"
    clusters_df.to_csv(clusters_csv, index=False)
    embeddings_npz = write_embeddings_npz(out_dir / "creative_embeddings.npz", ids=emb.ids, embeddings=emb.embeddings)

    return CreativeAnalysisOutputs(
        assets_csv=assets_csv,
        embeddings_npz=embeddings_npz,
        neighbors_csv=neighbors_csv,
        clusters_csv=clusters_csv,
        cluster_summary_csv=cluster_summary_csv,
        warnings=list(emb.warnings),
    )

