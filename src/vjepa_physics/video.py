"""Decode an mp4 into RGB uint8 frames.

The decoder is the one swappable part of the pipeline. Every backend must honour
the same contract, enforced by `_validate`:

    (T, C, H, W)  uint8  RGB  with T = 16, H = W = 256

OpenCV returns BGR and torchcodec returns RGB, so channel order is the classic
place for a silent bug. tests/test_video.py and scripts/00_sanity.py check that
the (blue) disk actually comes out blue.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DECODERS = ("torchcodec", "opencv")


def torchcodec_available() -> bool:
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
    except Exception:  # ImportError, or FFmpeg libraries missing at load time
        return False
    return True


def resolve_decoder(name: str) -> str:
    if name == "auto":
        return "torchcodec" if torchcodec_available() else "opencv"
    if name not in DECODERS:
        raise ValueError(f"unknown decoder {name!r}; expected auto or one of {DECODERS}")
    return name


def _decode_opencv(path: Path) -> torch.Tensor:
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame[:, :, ::-1])  # BGR -> RGB
    cap.release()
    if not frames:
        raise IOError(f"OpenCV decoded no frames from {path}")
    array = np.ascontiguousarray(np.stack(frames))  # (T, H, W, C)
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()


def _decode_torchcodec(path: Path) -> torch.Tensor:
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(path))
    return decoder.get_frames_at(indices=list(range(len(decoder)))).data  # (T, C, H, W), RGB


def _validate(frames: torch.Tensor, path: Path, num_frames: int, frame_size: int) -> None:
    expected = (num_frames, 3, frame_size, frame_size)
    if frames.dtype != torch.uint8 or tuple(frames.shape) != expected:
        raise ValueError(
            f"{path}: decoded {tuple(frames.shape)} {frames.dtype}, expected {expected} uint8. "
            "Refusing to pad, crop or resample silently."
        )


def load_clip(path: str | Path, decoder: str = "opencv", num_frames: int = 16,
              frame_size: int = 256) -> torch.Tensor:
    """Decode one clip to a (T, C, H, W) uint8 RGB tensor."""
    path = Path(path)
    backend = resolve_decoder(decoder)
    frames = _decode_torchcodec(path) if backend == "torchcodec" else _decode_opencv(path)
    _validate(frames, path, num_frames, frame_size)
    return frames


def pixel_features(frames: torch.Tensor, size: int = 16) -> torch.Tensor:
    """[OURS] Raw-pixel baseline: each frame to `size`x`size` grayscale, flattened.

    For 16 frames at size 16 this is 16 * 256 = 4096 values in [0, 1]. Area
    averaging keeps the small disk visible as a faint blob rather than aliasing
    it away.
    """
    rgb = frames.float() / 255.0
    luminance = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    gray = (rgb * luminance).sum(dim=1, keepdim=True)                      # (T, 1, H, W)
    small = F.interpolate(gray, size=(size, size), mode="area")             # (T, 1, s, s)
    return small.flatten()
