"""Learned image enhancement in front of a detector, in the style of IA-YOLO (Liu et al., 2022).

A small CNN looks at a 256x256 thumbnail of each input image and predicts the parameters of a
chain of differentiable Kornia ops: white balance, gamma, contrast, fine sharpness and a coarse
unsharp mask. The chain is applied to the full-size image before the detector backbone and is
trained end to end by the detection loss, so the enhancement is whatever helps detection.

The last layer starts at zero, so at initialisation every op is the identity and the pretrained
detector sees unchanged images. The ops run in float32 outside autocast (gamma's gradient involves
log(x)).
"""

import math

import kornia
import torch
from torch import nn

PARAMS = ("wb_r", "wb_g", "wb_b", "gamma", "contrast", "sharpness", "unsharp")


class KorniaDIP(nn.Module):
    def __init__(self, width: int = 32, thumb: int = 256) -> None:
        super().__init__()
        self.thumb = thumb
        layers, ch = [], 3
        for out in (16, width, width, width, width):
            layers += [nn.Conv2d(ch, out, 3, stride=2, padding=1), nn.LeakyReLU(0.1)]
            ch = out
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(ch, len(PARAMS))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def params(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-image op parameters, shape (B,), identity values at a zero head output."""
        small = nn.functional.interpolate(x, size=(self.thumb, self.thumb), mode="bilinear")
        a = torch.tanh(self.head(self.features(small)))
        p = dict(zip(PARAMS, a.unbind(1), strict=True))
        out = {k: torch.exp(p[k] * 0.2) for k in ("wb_r", "wb_g", "wb_b")}  # gains 0.82..1.22
        out["gamma"] = torch.exp(p["gamma"] * math.log(2))  # 0.5..2
        out["contrast"] = torch.exp(p["contrast"] * math.log(2))  # 0.5..2
        out["sharpness"] = torch.exp(p["sharpness"] * math.log(3))  # 1/3 (blur)..3, 1 = identity
        out["unsharp"] = p["unsharp"]  # -1 (blur)..1 (sharpen), 0 = identity
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            p = self.params(x)
            gains = torch.stack([p["wb_r"], p["wb_g"], p["wb_b"]], 1)[:, :, None, None]
            x = (x * gains).clamp(1e-4, 1)
            x = kornia.enhance.adjust_gamma(x, p["gamma"])
            x = kornia.enhance.adjust_contrast(x, p["contrast"])
            x = kornia.enhance.sharpness(x, p["sharpness"])
            blur = kornia.filters.gaussian_blur2d(x, (9, 9), (2.0, 2.0))
            x = (x + p["unsharp"][:, None, None, None] * (x - blur)).clamp(0, 1)
        return x
