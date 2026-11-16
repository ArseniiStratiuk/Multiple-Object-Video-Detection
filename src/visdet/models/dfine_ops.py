"""Transforms and models added to D-FINE's registry; import only with D-FINE on sys.path.

`Albu` applies photometric Albumentations transforms (boxes unchanged) to the PIL image; put it
before ConvertPILImage. Config entry, same format as visdet.models.ultralytics.albumentations:
    {type: Albu, transforms: [{Downscale: {scale_range: [0.35, 1.0], p: 0.5}},
                              {MotionBlur: {blur_limit: [3, 17], p: 0.3}}]}

`RandomSlice` is slicing-aided fine-tuning (SAHI, Akyon et al. 2022): with probability p the
sample becomes one random square slice of the original frame, which the later Resize upscales.
The slice size in original pixels matches the SAHI slices used at inference.

`MosaicTiles` is D-FINE's (RT-DETR's) Mosaic with two fixes: the sampled image is one of the four
tiles (upstream draws three random images and leaves the fourth tile empty), and per-image fields
come from the sampled image instead of being concatenated. Put it first in the ops list.

`DFINEEnhanced` is D-FINE with a learned Kornia enhancement (visdet.models.enhance.KorniaDIP)
in front of the backbone. Its parameters live under `enhance.`, outside the backbone's small
learning-rate group, and the pretrained D-FINE keys are unchanged.
"""

import random

import numpy as np
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from PIL import Image
from src.core import register
from src.data._misc import convert_to_tv_tensor
from src.zoo.dfine.dfine import DFINE

from visdet.models.enhance import KorniaDIP
from visdet.models.ultralytics import albumentations


@register()
class Albu(T.Transform):
    _transformed_types = (Image.Image,)

    def __init__(self, transforms: list[dict]) -> None:
        super().__init__()
        import albumentations as A

        self.aug = A.Compose(albumentations(transforms))

    def transform(self, inpt, params):
        return Image.fromarray(self.aug(image=np.asarray(inpt))["image"])


@register()
class RandomSlice(T.RandomCrop):
    def __init__(self, size: int = 640, p: float = 0.5) -> None:
        super().__init__(size=size)
        self.p = p

    def forward(self, *inputs):
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        return super().forward(*inputs)


@register()
class MosaicTiles(T.Transform):
    def __init__(self, size: int = 1280, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self.resize = T.Resize(size=[size, size])
        self.affine = T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.5, 1.5), fill=0)
        self.crop = T.RandomCrop(size=size)

    def forward(self, *inputs):
        image, target, dataset = inputs if len(inputs) > 1 else inputs[0]
        if torch.rand(1) >= self.p:
            return image, target, dataset
        others = random.choices(range(len(dataset)), k=3)
        items = [(image, target)] + [dataset.load_item(i) for i in others]
        items = [self.resize(im, tg) for im, tg in items]
        w, h = F.get_size(items[0][0])[::-1]
        canvas = Image.new(mode=items[0][0].mode, size=(2 * w, 2 * h), color=0)
        offsets = [(0, 0), (w, 0), (0, h), (w, h)]
        for (im, _), off in zip(items, offsets, strict=True):
            canvas.paste(im, off)
        shift = torch.tensor([[ox, oy, ox, oy] for ox, oy in offsets], dtype=torch.float32)
        mosaic = dict(target)
        mosaic["boxes"] = convert_to_tv_tensor(
            torch.cat([tg["boxes"] + shift[i] for i, (_, tg) in enumerate(items)]),
            "boxes", box_format="xyxy", spatial_size=[2 * h, 2 * w],
        )  # fmt: skip
        for key in ("labels", "area", "iscrowd"):
            if key in target:
                mosaic[key] = torch.cat([tg[key] for _, tg in items])
        canvas, mosaic = self.affine(canvas, mosaic)
        canvas, mosaic = self.crop(canvas, mosaic)
        return canvas, mosaic, dataset


@register()
class DFINEEnhanced(DFINE):
    def __init__(self, backbone, encoder, decoder) -> None:
        super().__init__(backbone, encoder, decoder)
        self.enhance = KorniaDIP()

    def forward(self, x, targets=None):
        return super().forward(self.enhance(x), targets)
