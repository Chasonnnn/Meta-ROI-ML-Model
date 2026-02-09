# Creative Embeddings (Similarity + Clustering)

This repo can optionally compute **creative media embeddings** (image/video) to support:
- “find similar creatives”
- creative clustering and cluster-level performance summaries

This is separate from the ELV predictive model (it does not affect `elv train` / `elv score`).

## Install (Optional Extra)

This feature is heavier (PyTorch + OpenCLIP + OpenCV). Install it explicitly:

```bash
uv sync --extra embeddings
```

## Inputs

Create a CSV mapping `ad_id` to a media file.

`creative_media.csv`:
- required: `ad_id`, `media_path`
- optional: `creative_type` (image/video)

Example:

```csv
ad_id,media_path,creative_type
12345,creatives/ad_12345.mp4,video
67890,creatives/ad_67890.png,image
```

If `media_path` is relative, pass `--media-dir` so files can be resolved.

## Run

```bash
elv creative-analyze \
  --creative-map data/byo/creative_media.csv \
  --media-dir data/byo \
  --neighbors 10 \
  --clusters 12 \
  --num-video-frames 4
```

To also compute cluster performance summaries, point at an existing ELV run dir:

```bash
elv creative-analyze \
  --creative-map data/byo/creative_media.csv \
  --media-dir data/byo \
  --run-dir runs/<run_id>
```

## Outputs

Written to `--out-dir` (or `runs/<new_run_id>/` by default):
- `creative_embeddings.npz` (embeddings matrix)
- `creative_neighbors.csv` (top-K cosine neighbors per creative)
- `creative_clusters.csv` (cluster assignment per creative)
- `creative_assets.csv` (ad_id + media mapping + cluster_id)
- `creative_cluster_summary.csv` (only when `--run-dir` is provided and has `predictions.csv`)

## Video Frame Sampling

For video creatives, we extract a small number of frames (default `--num-video-frames 4`),
embed each frame, and **mean-pool** into one embedding vector.

