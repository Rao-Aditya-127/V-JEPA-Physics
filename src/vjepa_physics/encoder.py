"""The frozen V-JEPA 2 encoder: preprocess, forward, mean-pool.

    clips (list of (T, C, H, W) uint8)
      -> VJEPA2VideoProcessor(do_resize=False, do_center_crop=False)   (B, T, C, H, W) float
      -> model(..., skip_predictor=True, output_hidden_states=True)    25 x (B, 2048, 1024)
      -> mean over the 2048 tokens                                     (B, 25, 1024)

Index 0 of the 25 is the patch embedding before any block; index l+1 is the output
of block l. So the paper's "layer l" is hidden_states[l + 1] (see features.py).

We deliberately avoid `model.get_vision_features()`: it returns last_hidden_state
only -- one layer, and with the final LayerNorm applied, unlike hidden_states[24].
"""

from __future__ import annotations

import contextlib

import torch
from transformers import AutoModel, AutoVideoProcessor

AUTOCAST_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


class FrozenVJEPA2:
    def __init__(self, model_id: str, device: str = "cpu", autocast_dtype: str | None = None,
                 do_resize: bool = False, do_center_crop: bool = False):
        self.model_id = model_id
        self.device = torch.device(device)
        self.do_resize = do_resize
        self.do_center_crop = do_center_crop
        self.autocast_dtype = AUTOCAST_DTYPES[autocast_dtype] if autocast_dtype else None

        self.processor = AutoVideoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size

    def _autocast(self):
        if self.device.type == "cuda" and self.autocast_dtype is not None:
            return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
        return contextlib.nullcontext()

    def preprocess(self, clips: list[torch.Tensor]) -> torch.Tensor:
        """Rescale + normalise with the checkpoint's own mean/std; geometry off (D1)."""
        batch = self.processor(clips, do_resize=self.do_resize,
                               do_center_crop=self.do_center_crop, return_tensors="pt")
        return batch["pixel_values_videos"]

    @torch.inference_mode()
    def hidden_states(self, clips: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """All num_layers + 1 residual-stream snapshots, each (B, tokens, hidden)."""
        pixels = self.preprocess(clips).to(self.device, non_blocking=True)
        with self._autocast():
            out = self.model(pixel_values_videos=pixels, skip_predictor=True,
                             output_hidden_states=True)
        states = out.hidden_states
        if states is None or len(states) != self.num_layers + 1:
            got = None if states is None else len(states)
            raise RuntimeError(
                f"expected {self.num_layers + 1} hidden states, got {got}. "
                "output_hidden_states behaviour may differ in this transformers version."
            )
        return states

    @torch.inference_mode()
    def pooled(self, clips: list[torch.Tensor]) -> torch.Tensor:
        """Mean-pool every hidden state over its tokens -> (B, num_layers + 1, hidden), float32.

        The mean is taken in float32: averaging 2048 values in fp16 loses precision.
        """
        states = self.hidden_states(clips)
        return torch.stack([h.float().mean(dim=1) for h in states], dim=1).cpu()
