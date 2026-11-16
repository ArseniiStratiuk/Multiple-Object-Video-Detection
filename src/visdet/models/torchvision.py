"""Faster R-CNN (two-stage) and RetinaNet (one-stage, anchor-based) from torchvision.

Both start from COCO weights with a new 8-class head. Images are resized inside the model so the
long side equals `imgsz`, the same input size the Ultralytics models use. Class ids are shifted
by one because index 0 is background in torchvision heads.
"""

import math
import os
import time
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.io import ImageReadMode, decode_jpeg, read_file
from torchvision.models import detection
from torchvision.models.detection import roi_heads
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.ops import generalized_box_iou_loss

from visdet import CLASSES, track

NUM_CLASSES = len(CLASSES) + 1
MODEL_KEYS = {
    "fasterrcnn_resnet50_fpn_v2": (
        "rpn_pre_nms_top_n_train", "rpn_post_nms_top_n_train", "rpn_pre_nms_top_n_test",
        "rpn_post_nms_top_n_test", "rpn_batch_size_per_image", "rpn_positive_fraction",
        "box_batch_size_per_image", "box_positive_fraction", "box_fg_iou_thresh",
        "box_bg_iou_thresh",
    ),
    "retinanet_resnet50_fpn_v2": ("fg_iou_thresh", "bg_iou_thresh", "topk_candidates"),
}  # fmt: skip


class CachedFrames(Dataset):
    """Cached JPEGs and their YOLO label files, boxes returned in pixels (x1, y1, x2, y2)."""

    def __init__(self, paths: list[str], train: bool):
        self.paths, self.train = paths, train

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        path = self.paths[i]
        image = decode_jpeg(read_file(path), mode=ImageReadMode.RGB)
        _, height, width = image.shape
        label = Path(path.replace("/images/", "/labels/")).with_suffix(".txt")
        text = label.read_text() if label.exists() else ""  # test frames have no labels
        rows = [list(map(float, r.split())) for r in text.splitlines() if r]
        t = torch.tensor(rows, dtype=torch.float32).reshape(-1, 5)
        cx, cy, w, h = t[:, 1] * width, t[:, 2] * height, t[:, 3] * width, t[:, 4] * height
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
        if self.train and torch.rand(()) < 0.5:
            image = image.flip(-1)
            boxes = torch.stack(
                [width - boxes[:, 2], boxes[:, 1], width - boxes[:, 0], boxes[:, 3]], 1
            )
        keep = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
        target = {"boxes": boxes[keep], "labels": t[keep, 0].long() + 1}
        return image, target


def build_model(cfg: dict) -> torch.nn.Module:
    name = cfg["model"]
    extra = {k: cfg[k] for k in MODEL_KEYS[name] if k in cfg}
    size = {"min_size": cfg["imgsz"], "max_size": cfg["imgsz"]}
    if name == "fasterrcnn_resnet50_fpn_v2":
        model = detection.fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT", box_detections_per_img=cfg["max_det"],
            box_score_thresh=cfg["predict_conf"], **size, **extra,
        )  # fmt: skip
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
        if "anchor_sizes" in cfg:
            # One size per FPN level and three ratios keep 3 anchors per location, so the
            # pretrained RPN head still fits.
            ratios = (tuple(cfg.get("anchor_ratios", (0.5, 1.0, 2.0))),) * len(cfg["anchor_sizes"])
            model.rpn.anchor_generator = AnchorGenerator(
                tuple((x,) for x in cfg["anchor_sizes"]), ratios
            )
        if cfg.get("box_loss") == "giou":
            roi_heads.fastrcnn_loss = _fastrcnn_giou_loss
    elif name == "retinanet_resnet50_fpn_v2":
        model = detection.retinanet_resnet50_fpn_v2(
            weights="DEFAULT", detections_per_img=cfg["max_det"],
            score_thresh=cfg["predict_conf"], **size, **extra,
        )  # fmt: skip
        anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            256, anchors, NUM_CLASSES, norm_layer=partial(torch.nn.GroupNorm, 32)
        )
        if "anchor_sizes" in cfg:
            # Three scales per level and the given ratios keep 9 anchors per location, so the
            # pretrained box head still fits.
            sizes = tuple(
                (x, round(x * 2 ** (1 / 3)), round(x * 2 ** (2 / 3))) for x in cfg["anchor_sizes"]
            )
            ratios = (tuple(cfg.get("anchor_ratios", (0.5, 1.0, 2.0))),) * len(sizes)
            model.anchor_generator = AnchorGenerator(sizes, ratios)
        if "focal_alpha" in cfg or "focal_gamma" in cfg:
            _set_focal(model, cfg.get("focal_alpha", 0.25), cfg.get("focal_gamma", 2.0))
    else:
        raise ValueError(f"unknown torchvision model {name}")
    return model


def _fastrcnn_giou_loss(class_logits, box_regression, labels, regression_targets):
    """Fast R-CNN loss with GIoU instead of smooth L1 on the box deltas.

    IoU and GIoU are unchanged by scaling each axis, so both delta sets are decoded against one
    unit reference box instead of their proposals. Weight 10 follows the MMDetection setting.
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)
    positive = torch.where(labels > 0)[0]
    deltas = box_regression.reshape(len(labels), -1, 4)[positive, labels[positive]]

    def decode(d):
        d = d.float() / torch.tensor([10.0, 10.0, 5.0, 5.0], device=d.device)
        w, h = d[:, 2].clamp(max=4.0).exp(), d[:, 3].clamp(max=4.0).exp()
        return torch.stack([d[:, 0] - w / 2, d[:, 1] - h / 2, d[:, 0] + w / 2, d[:, 1] + h / 2], 1)

    loss = generalized_box_iou_loss(decode(deltas), decode(regression_targets[positive]), "sum")
    return classification_loss, 10.0 * loss / labels.numel()


def _set_focal(model, alpha: float, gamma: float) -> None:
    from torchvision.ops import sigmoid_focal_loss

    head = model.head.classification_head

    def compute_loss(targets, head_outputs, matched_idxs):
        losses = []
        for target, logits, matched in zip(
            targets, head_outputs["cls_logits"], matched_idxs, strict=True
        ):
            foreground = matched >= 0
            gt = torch.zeros_like(logits)
            gt[foreground, target["labels"][matched[foreground]]] = 1.0
            valid = matched != head.BETWEEN_THRESHOLDS
            loss = sigmoid_focal_loss(logits[valid], gt[valid], alpha, gamma, reduction="sum")
            losses.append(loss / max(1, foreground.sum()))
        return sum(losses) / len(targets)

    head.compute_loss = compute_loss


def _device(device: str) -> str:
    return f"cuda:{device}" if device.isdigit() else device


def _collate(batch):
    return tuple(zip(*batch, strict=True))


def _distributed() -> tuple[int, int]:
    """Rank and world size; torchrun sets the environment for multi-GPU runs."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return (dist.get_rank() if world > 1 else 0), world


def train(cfg: dict, data_yaml: Path, out_dir: Path, device: str, on_epoch_end=None) -> Path | None:
    """Train on one GPU, or on all torchrun processes with DDP (linear learning-rate scaling).

    Only rank 0 logs, saves and returns the weights path; other ranks return None.
    `on_epoch_end(predict_fn, epoch)` runs on rank 0 after each epoch, where
    `predict_fn(images)` predicts with the current weights (used for validation curves).
    """
    rank, world = _distributed()
    device = f"cuda:{os.environ['LOCAL_RANK']}" if world > 1 else _device(device)
    torch.manual_seed(cfg["seed"] + rank)
    spec = yaml.safe_load(Path(data_yaml).read_text())
    paths = (Path(spec["path"]) / spec["train"]).read_text().split()
    dataset = CachedFrames(paths, train=True)
    sampler = DistributedSampler(dataset, seed=cfg["seed"]) if world > 1 else None
    loader = DataLoader(
        dataset, batch_size=cfg["batch"], shuffle=sampler is None, sampler=sampler,
        num_workers=cfg["workers"], collate_fn=_collate, pin_memory=True,
        persistent_workers=True, drop_last=True,
    )  # fmt: skip
    model = build_model(cfg).to(device)
    if world > 1:
        model = DDP(model, device_ids=[torch.device(device).index])
    params = [p for p in model.parameters() if p.requires_grad]
    lr = cfg["lr"] * world
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    total = cfg["epochs"] * len(loader)
    warmup = min(500, total // 10)

    def factor(step: int) -> float:
        if step < warmup:
            return 0.001 + (1 - 0.001) * step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "train_log.csv"
    if rank == 0:
        log.write_text("epoch,step,loss,lr,seconds\n")
    start = time.time()
    step = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        for images, targets in loader:
            images = [im.to(device, non_blocking=True).float().div_(255) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {losses}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            optimizer.step()
            scheduler.step()
            if rank == 0 and step % 50 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                with log.open("a") as fh:
                    fh.write(f"{epoch},{step},{loss.item():.4f},{lr_now:.6f},{elapsed:.0f}\n")
                parts = {f"train/{k}": v.item() for k, v in losses.items()}
                track.log({"train/loss": loss.item(), "lr": lr_now, "epoch": epoch} | parts)
            step += 1
        if rank == 0:
            bare = model.module if world > 1 else model
            torch.save(bare.state_dict(), out_dir / "last.pt")
            if on_epoch_end is not None:
                on_epoch_end(lambda images, m=bare: _predict_with(m, cfg, images, device), epoch)
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()
    return out_dir / "last.pt" if rank == 0 else None


def predict(cfg: dict, weights: Path, images: pd.DataFrame, device: str) -> pd.DataFrame:
    device = _device(device)
    model = build_model(cfg)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return _predict_with(model.to(device), cfg, images, device)


@torch.inference_mode()
def _predict_with(model, cfg: dict, images: pd.DataFrame, device: str) -> pd.DataFrame:
    """Predict in eval mode; cfg["imgsz"] sets the inference size, cfg["flip"] mirrors inputs."""
    was_training = model.training
    model.eval()
    model.transform.min_size, model.transform.max_size = (cfg["imgsz"],), cfg["imgsz"]
    flip = cfg.get("flip", False)
    loader = DataLoader(
        CachedFrames(images["path"].tolist(), train=False), batch_size=8,
        num_workers=cfg["workers"], collate_fn=_collate,
    )  # fmt: skip
    ids = iter(images["image_id"].tolist())
    parts = []
    for batch, _ in loader:
        batch = [im.to(device).float().div_(255) for im in batch]
        if flip:
            batch = [im.flip(-1) for im in batch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(batch)
        for im, out in zip(batch, outputs, strict=True):
            b = out["boxes"].float().cpu()
            if flip:
                width = im.shape[-1]
                b = torch.stack([width - b[:, 2], b[:, 1], width - b[:, 0], b[:, 3]], 1)
            parts.append(
                pd.DataFrame(
                    {
                        "image_id": next(ids),
                        "cls": out["labels"].cpu().numpy() - 1,
                        "score": out["scores"].float().cpu().numpy(),
                        "x": b[:, 0].numpy(),
                        "y": b[:, 1].numpy(),
                        "w": (b[:, 2] - b[:, 0]).numpy(),
                        "h": (b[:, 3] - b[:, 1]).numpy(),
                    }
                )
            )
    model.train(was_training)
    return pd.concat(parts, ignore_index=True)


@torch.inference_mode()
def proposal_recall(cfg: dict, weights: Path, images: pd.DataFrame, device: str) -> pd.DataFrame:
    """Faster R-CNN region proposals per image in cached-image pixels (class-agnostic).

    Returns the same columns as `predict` with cls = -1 and the objectness rank as score, so the
    proposal stage can be scored for recall separately from the second stage.
    """
    device = _device(device)
    model = build_model(cfg)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.to(device).eval()
    loader = DataLoader(
        CachedFrames(images["path"].tolist(), train=False), batch_size=4,
        num_workers=cfg["workers"], collate_fn=_collate,
    )  # fmt: skip
    ids = iter(images["image_id"].tolist())
    parts = []
    for batch, _ in loader:
        batch = [im.to(device).float().div_(255) for im in batch]
        sizes = [im.shape[-2:] for im in batch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            image_list, _ = model.transform(batch)
            features = model.backbone(image_list.tensors)
            proposals, _ = model.rpn(image_list, features)
        for boxes, resized, original in zip(proposals, image_list.image_sizes, sizes, strict=True):
            sy, sx = original[0] / resized[0], original[1] / resized[1]
            b = boxes.float().cpu() * torch.tensor([sx, sy, sx, sy])
            parts.append(
                pd.DataFrame(
                    {
                        "image_id": next(ids), "cls": -1,
                        "score": 1.0 - torch.arange(len(b)).numpy() / max(1, len(b)),
                        "x": b[:, 0].numpy(), "y": b[:, 1].numpy(),
                        "w": (b[:, 2] - b[:, 0]).numpy(), "h": (b[:, 3] - b[:, 1]).numpy(),
                    }
                )
            )  # fmt: skip
    return pd.concat(parts, ignore_index=True)
