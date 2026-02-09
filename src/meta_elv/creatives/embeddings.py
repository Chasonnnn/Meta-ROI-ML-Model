from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class EmbeddingDependencyError(RuntimeError):
    pass


class CreativeMediaError(ValueError):
    pass


@dataclass(frozen=True)
class CreativeMediaMap:
    df: pd.DataFrame
    warnings: list[str]


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def load_creative_media_map(path: Path, *, media_dir: Path | None = None) -> CreativeMediaMap:
    """
    Required CSV columns:
    - ad_id
    - media_path (filename or relative path; resolved against media_dir if provided)

    Optional:
    - creative_type (image/video)
    """
    path = Path(path)
    df = pd.read_csv(path).reset_index(drop=True)
    warnings: list[str] = []

    for c in ["ad_id", "media_path"]:
        if c not in df.columns:
            raise CreativeMediaError(f"creative media map missing required column: {c}")

    df["ad_id"] = df["ad_id"].astype("object").astype(str)
    df["media_path"] = df["media_path"].astype("object").astype(str)

    if df["ad_id"].duplicated().any():
        dupes = df.loc[df["ad_id"].duplicated(), "ad_id"].head(5).tolist()
        raise CreativeMediaError(
            f"creative media map has duplicate ad_id values (example: {dupes}). "
            "Provide exactly one media_path per ad_id."
        )

    # Infer media type
    if "creative_type" in df.columns:
        ct = df["creative_type"].astype("object").fillna("").astype(str).str.lower().str.strip()
        df["media_type"] = np.where(ct.str.contains("video"), "video", np.where(ct.str.contains("image"), "image", ""))
    else:
        df["media_type"] = ""

    def _infer_from_ext(p: str) -> str:
        ext = Path(p).suffix.lower()
        if ext in _IMAGE_EXTS:
            return "image"
        if ext in _VIDEO_EXTS:
            return "video"
        return ""

    df["media_type"] = df["media_type"].where(df["media_type"].ne(""), df["media_path"].map(_infer_from_ext))
    unknown = int((df["media_type"] == "").sum())
    if unknown:
        warnings.append(f"{unknown} rows have unknown media_type; will attempt to treat as image.")
        df["media_type"] = df["media_type"].replace({"": "image"})

    if media_dir is not None:
        base = Path(media_dir)
        df["media_abs_path"] = df["media_path"].map(lambda p: str((base / p).resolve()))
    else:
        df["media_abs_path"] = df["media_path"].map(lambda p: str(Path(p).resolve()))

    # Existence checks
    missing = df["media_abs_path"].map(lambda p: not Path(p).exists())
    if bool(missing.any()):
        ex = df.loc[missing, ["ad_id", "media_path"]].head(5).to_dict(orient="records")
        raise CreativeMediaError(f"Missing media files for some rows (example: {ex}).")

    return CreativeMediaMap(df=df, warnings=warnings)


def _load_clip_model(model_name: str, pretrained: str, device: str):
    try:
        import torch
        import open_clip
    except Exception as e:  # pragma: no cover
        raise EmbeddingDependencyError(
            "Creative embeddings require the optional embeddings extra. Install with:\n"
            "  uv sync --extra embeddings\n"
            "or:\n"
            "  pip install -e '.[embeddings]'\n"
        ) from e

    if device == "cuda" and not torch.cuda.is_available():
        raise CreativeMediaError("device=cuda requested but CUDA is not available.")

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    return model, preprocess, torch


def _load_pil_image(path: Path):
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover
        raise EmbeddingDependencyError(
            "Pillow is required for creative embeddings. Install with: uv sync --extra embeddings"
        ) from e
    img = Image.open(path).convert("RGB")
    return img


def _extract_video_frames_cv2(path: Path, num_frames: int) -> list[Any]:
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise EmbeddingDependencyError(
            "Video frame extraction requires opencv-python-headless. Install with: uv sync --extra embeddings"
        ) from e

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise CreativeMediaError(f"Failed to open video: {path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        # Fallback: attempt to read sequentially to get some frames.
        frames = []
        while len(frames) < num_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise CreativeMediaError(f"Failed to decode video frames: {path}")
        # Uniformly sample from what we got
        idxs = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = [frames[int(i)] for i in idxs if 0 <= int(i) < len(frames)]
    else:
        idxs = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames.append(frame)
        cap.release()
        if not frames:
            raise CreativeMediaError(f"Failed to decode video frames: {path}")

    # Convert to PIL Images
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover
        raise EmbeddingDependencyError(
            "Pillow is required for creative embeddings. Install with: uv sync --extra embeddings"
        ) from e

    pil_frames = []
    for frame in frames:
        # OpenCV frames are BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(frame_rgb))
    return pil_frames


@dataclass(frozen=True)
class EmbedResult:
    ids: list[str]
    embeddings: np.ndarray
    warnings: list[str]


def embed_with_clip(
    media_map: CreativeMediaMap,
    *,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: str = "cpu",
    batch_size: int = 16,
    num_video_frames: int = 4,
) -> EmbedResult:
    df = media_map.df.copy()
    warnings = list(media_map.warnings)

    model, preprocess, torch = _load_clip_model(model_name, pretrained, device)

    ids = df["ad_id"].astype(str).tolist()
    out = np.zeros((len(df), 1), dtype=np.float32)
    out_dim = None

    # Embed images in batches; embed videos item-by-item (frames are small count).
    img_idx = df.index[df["media_type"] == "image"].tolist()
    vid_idx = df.index[df["media_type"] == "video"].tolist()

    def _embed_pil_images(pil_images: list[Any]) -> np.ndarray:
        with torch.no_grad():
            xs = torch.stack([preprocess(im) for im in pil_images]).to(device)
            feats = model.encode_image(xs)
            feats = feats.float().cpu().numpy()
        return feats

    # Images
    if img_idx:
        bs = max(1, int(batch_size))
        for start in range(0, len(img_idx), bs):
            chunk = img_idx[start : start + bs]
            pil_images = [_load_pil_image(Path(df.loc[i, "media_abs_path"])) for i in chunk]
            feats = _embed_pil_images(pil_images)
            if out_dim is None:
                out_dim = feats.shape[1]
                out = np.zeros((len(df), out_dim), dtype=np.float32)
            out[chunk, :] = feats

    # Videos
    for i in vid_idx:
        p = Path(df.loc[i, "media_abs_path"])
        n = max(1, int(num_video_frames))
        frames = _extract_video_frames_cv2(p, n)
        feats = _embed_pil_images(frames)
        if out_dim is None:
            out_dim = feats.shape[1]
            out = np.zeros((len(df), out_dim), dtype=np.float32)
        out[i, :] = feats.mean(axis=0)

    if out_dim is None:
        # Should not happen because we coerce unknown types to image, but keep safe.
        raise CreativeMediaError("No media rows could be embedded.")

    return EmbedResult(ids=ids, embeddings=out.astype(np.float32, copy=False), warnings=warnings)


def write_embeddings_npz(path: Path, *, ids: list[str], embeddings: np.ndarray) -> Path:
    path = Path(path)
    arr_ids = np.array([str(i) for i in ids], dtype=object)
    np.savez_compressed(path, ad_id=arr_ids, embedding=embeddings.astype(np.float32, copy=False))
    return path
