import pytest
import torch

from conftest import SPEED_DIR, needs_data
from vjepa_physics.video import _validate, load_clip, pixel_features

CLIP = SPEED_DIR / "videos" / "scene_0100" / "video.mp4"


@needs_data
def test_decode_contract():
    frames = load_clip(CLIP, decoder="opencv")
    assert frames.shape == (16, 3, 256, 256)
    assert frames.dtype == torch.uint8


def disk_rgb(frames: torch.Tensor) -> list[float]:
    background = frames.flatten(2).median(dim=2).values[..., None, None]
    disk = (frames.int() - background.int()).abs().sum(dim=1) > 60
    assert disk.sum() > 100
    return frames.permute(1, 0, 2, 3)[:, disk].float().mean(dim=1).tolist()


@needs_data
def test_disk_has_the_files_true_colour():
    """Catches a BGR/RGB swap -- the single most likely preprocessing bug.

    DATA.md calls the disk blue, but the files actually contain RGB (210, 105, 41),
    an orange. Verified bit-identical against an independent PyAV/FFmpeg decode to
    rgb24. The reversed triple (41, 105, 210) is blue, so the dataset generator most
    likely swapped channels when writing. We decode the file faithfully; a swap in
    *our* decoder would produce the blue triple and fail here.
    """
    r, g, b = disk_rgb(load_clip(CLIP, decoder="opencv"))
    assert abs(r - 210) < 15 and abs(g - 105) < 15 and abs(b - 41) < 15, \
        f"disk colour R={r:.0f} G={g:.0f} B={b:.0f}; expected ~(210, 105, 41)"


@needs_data
def test_torchcodec_matches_opencv():
    """If torchcodec is installed (e.g. on the GPU box), it must produce the same
    frames as OpenCV. Both sit on FFmpeg; locally they agree bit-for-bit with PyAV."""
    pytest.importorskip("torchcodec.decoders")
    a = load_clip(CLIP, decoder="opencv")
    b = load_clip(CLIP, decoder="torchcodec")
    assert (a.int() - b.int()).abs().max() <= 1


def test_validate_refuses_wrong_shapes():
    with pytest.raises(ValueError):
        _validate(torch.zeros(15, 3, 256, 256, dtype=torch.uint8), CLIP, 16, 256)
    with pytest.raises(ValueError):
        _validate(torch.zeros(16, 3, 256, 256, dtype=torch.float32), CLIP, 16, 256)


def test_pixel_features_shape_and_range():
    frames = torch.randint(0, 256, (16, 3, 256, 256), dtype=torch.uint8)
    feats = pixel_features(frames, size=16)
    assert feats.shape == (16 * 16 * 16,)
    assert 0.0 <= feats.min() and feats.max() <= 1.0


@needs_data
def test_processor_keeps_full_frame():
    """Regression guard for D1: with geometry disabled the processor must be pure
    (x/255 - mean)/std. If resize/crop ever comes back, this fails."""
    from transformers import AutoVideoProcessor

    processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    frames = load_clip(CLIP, decoder="opencv")
    out = processor([frames], do_resize=False, do_center_crop=False,
                    return_tensors="pt")["pixel_values_videos"]
    mean = torch.tensor(processor.image_mean).view(1, 1, 3, 1, 1)
    std = torch.tensor(processor.image_std).view(1, 1, 3, 1, 1)
    assert torch.allclose(out, (frames[None].float() / 255 - mean) / std, atol=1e-5)
