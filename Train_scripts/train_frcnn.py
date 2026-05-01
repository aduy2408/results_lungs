"""
train_frcnn.py
==============
Fine-tune a torchvision Faster R-CNN on the VinBigData Chest X-ray dataset
(COCO-format annotations, 22 pathology classes).

Architecture choices
--------------------
  backbone=resnet50_fpn_v2  → FasterRCNN_ResNet50_FPN_V2  (default, strongest)
  backbone=resnet50_fpn     → FasterRCNN_ResNet50_FPN     (lighter)
  backbone=mobilenet_v3     → FasterRCNN_MobileNet_V3_Large_FPN (fastest)

LR schedule
-----------
  1. Linear warmup for --warmup_epochs epochs
  2. Multi-step decay (default) at --milestones epochs  OR  cosine annealing

Usage
-----
    python train_frcnn.py                             # all defaults
    python train_frcnn.py \\
        --data_root  /home/Ubuntu/DATA \\
        --train_json /home/Ubuntu/DATA/annotations/instances_train.json \\
        --val_json   /home/Ubuntu/DATA/annotations/instances_val.json \\
        --backbone   resnet50_fpn_v2 \\
        --epochs 20 --batch 4 --imgsz 1024 --amp

Pre-requisites
--------------
    pip install -r requirements.txt
"""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,  FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn,     FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset_frcnn import build_dataloaders, NUM_CLASSES, CLASSES, CLASS_MODE


# ──────────────────────────────────────────────────────────────────────────────
# Defaults  (mirror custom_train_dino.py / train_yolo26.py where sensible)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_DATA_ROOT   = "/home/Ubuntu/DATA"
DEFAULT_TRAIN_JSON  = "/home/Ubuntu/DATA/annotations/instances_train.json"
DEFAULT_VAL_JSON    = "/home/Ubuntu/DATA/annotations/instances_val.json"
DEFAULT_BACKBONE    = "resnet50_fpn_v2"
DEFAULT_LR          = 1e-4         # Target peak LR
DEFAULT_LRF         = 0.1          # Reduction factor for Plateau
DEFAULT_BATCH       = 16
DEFAULT_IMGSZ       = 512
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_WARMUP_ITERS = 1000        # Warmup for 1000 iterations
DEFAULT_WARMUP_EPOCHS = 3.0        # Warmup for 3 epochs (overrides warmup_iters if > 0)
DEFAULT_EPOCHS       = 50
DEFAULT_EARLY_STOPPING = 10
DEFAULT_WORKERS     = 8
DEFAULT_PROJECT     = "runs/frcnn"
DEFAULT_NAME        = "frcnn_lung"


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Faster R-CNN on VinBigData Chest X-ray"
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--data_root",  default=DEFAULT_DATA_ROOT,
                   help="Root dir containing images/ subfolder")
    p.add_argument("--train_json", default=DEFAULT_TRAIN_JSON,
                   help="COCO train annotation JSON")
    p.add_argument("--val_json",   default=DEFAULT_VAL_JSON,
                   help="COCO val annotation JSON")
    p.add_argument("--test_json",  default=None,
                   help="COCO test annotation JSON (for final unseen evaluation)")

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--backbone", default=DEFAULT_BACKBONE,
                   choices=["resnet50_fpn_v2", "resnet50_fpn", "mobilenet_v3"],
                   help="Feature extractor backbone (default: resnet50_fpn_v2)")
    p.add_argument("--pretrained", action="store_true", default=True,
                   help="Load ImageNet/COCO pretrained backbone (default: True)")
    p.add_argument("--freeze_backbone", action="store_true", default=False,
                   help="Freeze backbone — only the detection head is trained")

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--epochs",         type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch",          type=int,   default=DEFAULT_BATCH)
    p.add_argument("--imgsz",          type=int,   default=DEFAULT_IMGSZ)
    p.add_argument("--lr",             type=float, default=DEFAULT_LR,
                   help="Peak learning rate (default: 2e-4)")
    p.add_argument("--lrf",            type=float, default=DEFAULT_LRF,
                   help="Final lr ratio for cosine schedule (default: 0.01)")
    p.add_argument("--weight_decay",   type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--warmup_epochs",  type=float, default=DEFAULT_WARMUP_EPOCHS,
                   help="Linear warmup length in epochs (default: 3)")
    p.add_argument("--clip_grad",      type=float, default=0.0,
                   help="Max gradient norm; 0 to disable (default: 0.0)")
    p.add_argument("--optimizer",      default="AdamW",
                   choices=["SGD", "Adam", "AdamW"])
    p.add_argument("--scheduler",      default="plateau",
                   choices=["multistep", "cosine", "plateau"],
                   help="LR decay policy after warmup (default: plateau)")
    p.add_argument("--warmup_iters",   type=int,   default=DEFAULT_WARMUP_ITERS,
                   help="Linear warmup length in iterations (default: 1000)")
    p.add_argument("--patience",       type=int,   default=2,
                   help="Patience for ReduceLROnPlateau (default: 2)")
    p.add_argument("--early_stop",     type=int,   default=DEFAULT_EARLY_STOPPING,
                   help="Patience for early stopping (default: 10)")
    p.add_argument("--milestones",     type=int, nargs="+", default=[70, 90],
                   help="Epochs to apply gamma decay (multistep; default: 70 90)")
    p.add_argument("--gamma",          type=float, default=0.1,
                   help="Multiplicative LR decay factor (default: 0.1)")
    p.add_argument("--num_classes",    type=int,   default=None,
                   help="Override number of classes (14 or 22). If None, uses YOLO_CLASS_MODE env.")

    # ── System ────────────────────────────────────────────────────────────────
    p.add_argument("--workers", type=int,   default=DEFAULT_WORKERS)
    p.add_argument("--device",  type=str,   default="",
                   help='"" = auto, "cpu", "0", "0,1"')
    p.add_argument("--amp",     action="store_true", default=True,
                   help="Mixed-precision training (default: True)")

    # ── Validation ────────────────────────────────────────────────────────────
    p.add_argument("--val_interval", type=int, default=1,
                   help="Evaluate COCO mAP every N epochs (default: 1)")
    p.add_argument("--val_batch",    type=int, default=1,
                   help="Batch size for validation (default: 1)")

    # ── Checkpointing ─────────────────────────────────────────────────────────
    p.add_argument("--save_period", type=int, default=1,
                   help="Save last.pt every N epochs (default: 1)")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--project",   default=DEFAULT_PROJECT)
    p.add_argument("--name",      default=DEFAULT_NAME)
    p.add_argument("--exist_ok",  action="store_true", default=False,
                   help="Allow overwriting an existing run directory")

    # ── Resume ────────────────────────────────────────────────────────────────
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint .pt to resume from "
                        "(e.g. runs/frcnn/frcnn_lung/weights/last.pt)")

    p.add_argument("--aspect_ratios", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0],
                   help="Custom aspect ratios for anchors (default: 0.5 1.0 2.0 3.0)")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────────────────────
def build_model(backbone: str, num_classes: int, pretrained: bool, imgsz: int = 512, aspect_ratios: list = [0.5, 1.0, 2.0, 3.0]):
    """
    Load a pretrained Faster R-CNN and swap the box predictor head.
    Sets min_size/max_size to imgsz to avoid redundant internal upscaling.
    """
    if backbone == "resnet50_fpn_v2":
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model   = fasterrcnn_resnet50_fpn_v2(weights=weights, min_size=imgsz, max_size=imgsz)
    elif backbone == "resnet50_fpn":
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model   = fasterrcnn_resnet50_fpn(weights=weights, min_size=imgsz, max_size=imgsz)
    elif backbone == "mobilenet_v3":
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
        model   = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, min_size=imgsz, max_size=imgsz)
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}")

    # 1. Custom Anchor Generator
    # VinBigData has many small/horizontal boxes, adding 3.0 helps.
    new_anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=(tuple(aspect_ratios),) * 5
    )
    # Swap anchor generator
    model.rpn.anchor_generator = new_anchor_generator

    # 2. Re-initialize RPN head 
    # Because we changed number of anchors per location (k), we MUST resize RPN outputs
    out_channels = model.backbone.out_channels
    num_anchors = new_anchor_generator.num_anchors_per_location()[0]
    
    from torchvision.models.detection.rpn import RPNHead
    model.rpn.head = RPNHead(out_channels, num_anchors)

    # 3. Replace the classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Optimizer
# ──────────────────────────────────────────────────────────────────────────────
def build_optimizer(model: torch.nn.Module, args: argparse.Namespace):
    """
    Separate backbone and detection-head parameters.
    Backbone uses 10× lower LR (same ratio as DINO config paramwise_cfg).
    """
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": args.lr * 0.1, "name": "backbone"},
        {"params": head_params,     "lr": args.lr,       "name": "head"},
    ]

    if args.optimizer == "AdamW":
        return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        return torch.optim.Adam(param_groups)
    else:  # SGD
        return torch.optim.SGD(
            param_groups, momentum=0.9, weight_decay=args.weight_decay
        )


# ──────────────────────────────────────────────────────────────────────────────
# LR scheduler
# ──────────────────────────────────────────────────────────────────────────────
def build_scheduler(optimizer, args: argparse.Namespace):
    """
    Returns the chosen scheduler. 
    Note: ReduceLROnPlateau requires a metric during step().
    """
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.gamma, patience=args.patience
        )
    
    total  = args.epochs
    # Note: Warmup is now handled PER ITERATION in the train loop,
    # so these LambdaLRs now only handle the post-warmup decay.
    if args.scheduler == "cosine":
        def lr_lambda(epoch: int) -> float:
            progress = epoch / max(1, total)
            return args.lrf + 0.5 * (1.0 - args.lrf) * (1.0 + math.cos(math.pi * progress))
    else:  # multistep
        milestones = set(args.milestones)
        def lr_lambda(epoch: int) -> float:
            factor = 1.0
            for m in milestones:
                if epoch >= m:
                    factor *= args.gamma
            return factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop (one epoch)
# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model,
    optimizer,
    loader,
    device,
    scaler:    GradScaler,
    epoch:     int,
    args:      argparse.Namespace,
    global_step: int = 0,
) -> tuple[dict, int]:
    """
    Returns a dict of averaged loss components for the epoch.
    """
    model.train()
    totals = defaultdict(float)
    n_batches = len(loader)

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch:03d} [train]",
        leave=False,
        dynamic_ncols=True,
    )

    for images, targets in pbar:
        global_step += 1
        
        # ── Iteration-based Warmup ───────────────────────────────────────────
        if global_step <= args.warmup_iters:
            # Scale LR linearly from 0 to peak (args.lr)
            warmup_factor = global_step / float(args.warmup_iters)
            for param_group in optimizer.param_groups:
                # Use name to distinguish backbone if needed
                base_lr = args.lr * 0.1 if param_group.get("name") == "backbone" else args.lr
                param_group["lr"] = base_lr * warmup_factor
        
        images  = [img.to(device, non_blocking=True) for img in images]
        targets = [
            {k: v.to(device, non_blocking=True) for k, v in t.items()}
            for t in targets
        ]

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=args.amp):
            loss_dict = model(images, targets)
            # torchvision returns:
            #   loss_classifier, loss_box_reg,
            #   loss_objectness, loss_rpn_box_reg
            total_loss = sum(loss_dict.values())

        if args.amp:
            scaler.scale(total_loss).backward()
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad
                )
            optimizer.step()

        # Accumulate losses for logging
        for k, v in loss_dict.items():
            totals[k] += v.item()
        totals["loss_total"] += total_loss.item()

        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "lr":   f"{optimizer.param_groups[-1]['lr']:.2e}",
        })

    metrics = {k: v / n_batches for k, v in totals.items()}
    return metrics, global_step


# ──────────────────────────────────────────────────────────────────────────────
# COCO evaluation
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model,
    loader,
    val_json_path: str,
    device,
    use_amp: bool,
) -> dict:
    """
    Run inference on the val set and compute COCO mAP / mAP@50 / mAP@75
    using pycocotools.

    Returns
    -------
    dict with keys: mAP, mAP_50, mAP_75
    """
    model.eval()

    coco_gt = COCO(val_json_path)

    dataset = loader.dataset
    label_to_cat_id = {v: k for k, v in dataset.cat_id_to_label.items()}

    results = []

    for images, targets in tqdm(
        loader, desc="  Validating", leave=False, dynamic_ncols=True
    ):
        images = [img.to(device, non_blocking=True) for img in images]

        with autocast(enabled=use_amp):
            preds = model(images)   # list[{boxes, labels, scores}]

        for pred, tgt in zip(preds, targets):
            image_id = tgt["image_id"].item()
            boxes  = pred["boxes"].cpu().numpy()   # [N, 4] xyxy
            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if int(label) == 0:
                    continue   # skip background

                x1, y1, x2, y2 = box.tolist()
                cat_id = label_to_cat_id.get(int(label), int(label))

                results.append({
                    "image_id":    image_id,
                    "category_id": cat_id,
                    "bbox":        [x1, y1, x2 - x1, y2 - y1],  # COCO xywh
                    "score":       float(score),
                })

    if not results:
        print("[WARN] No predictions produced — mAP set to 0.0")
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}

    coco_dt   = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.catIds = sorted(dataset.cat_id_to_label.keys())
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats   # [mAP, mAP@50, mAP@75, ...]
    return {
        "mAP":    float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────
def save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  [ckpt] Saved → {path}")


def load_checkpoint(path: str, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"] + 1
    best_map    = ckpt.get("best_map", 0.0)
    history     = ckpt.get("history", [])
    print(
        f"[INFO] Resumed from epoch {ckpt['epoch']} "
        f"(best mAP={best_map:.4f})"
    )
    return start_epoch, best_map, history


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Fix all relevant RNGs for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (small perf cost, but ensures reproducibility)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark     = False
    print(f"[INFO] Global seed set to {seed}")


def main() -> None:
    args = parse_args()

    # ── Seed (fixed at 42 for reproducibility) ────────────────────────────────
    set_seed(42)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        if args.device.isdigit():
            device = torch.device(f"cuda:{args.device}")
        else:
            device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device  : {device}")

    # ── Output directory (auto-increment if exists) ────────────────────────────
    save_dir = Path(args.project) / args.name
    if save_dir.exists() and not args.exist_ok and not args.resume:
        idx = 1
        while (save_dir.parent / f"{args.name}{idx}").exists():
            idx += 1
        save_dir = save_dir.parent / f"{args.name}{idx}"
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    print(f"[INFO] Save dir: {save_dir}")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    print("[INFO] Building dataloaders …")
    train_loader, val_loader = build_dataloaders(
        data_root   = args.data_root,
        train_json  = args.train_json,
        val_json    = args.val_json,
        img_size    = args.imgsz,
        batch_size  = args.batch,
        val_batch   = args.val_batch,
        seed        = 42,
    )
    # ── Warmup logic: calculate iterations from epochs if specified ───────────
    if args.warmup_epochs > 0:
        args.warmup_iters = int(args.warmup_epochs * len(train_loader))
        
    print(
        f"[INFO] Train: {len(train_loader.dataset):,} images  "
        f"({len(train_loader)} batches) | "
        f"Val: {len(val_loader.dataset):,} images"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    from dataset_frcnn import NUM_CLASSES as DEFAULT_NUM_CLASSES, CLASSES as DEFAULT_CLASSES
    
    current_num_classes = args.num_classes if args.num_classes is not None else DEFAULT_NUM_CLASSES
    
    print(f"[INFO] CLASS_MODE: {CLASS_MODE}")
    print(
        f"[INFO] Model   : Faster R-CNN "
        f"(backbone={args.backbone}, num_classes={current_num_classes})"
    )
    model = build_model(args.backbone, current_num_classes, args.pretrained, args.imgsz, args.aspect_ratios)

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad_(False)
        n_frozen = sum(
            1 for n, p in model.named_parameters()
            if "backbone" in n and not p.requires_grad
        )
        print(f"[INFO] Frozen  : {n_frozen} backbone parameter tensors")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable params: {n_trainable:,}")
    model.to(device)

    # ── Optimizer + Scheduler + AMP ───────────────────────────────────────────
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    scaler    = GradScaler(enabled=args.amp)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_map    = 0.0
    history     = []
    global_step = 0
    epochs_without_improvement = 0

    if args.resume:
        start_epoch, best_map, history = load_checkpoint(
            args.resume, model, optimizer, scaler, device
        )
        # Fast-forward the scheduler to current epoch
        for _ in range(start_epoch):
            scheduler.step()

    # ── Print config ──────────────────────────────────────────────────────────
    print("\n[INFO] Training configuration:")
    for k, v in vars(args).items():
        print(f"  {k:<20} = {v}")
    print()

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        epoch_num = epoch + 1
        t0 = time.time()

        # ── Train one epoch ───────────────────────────────────────────────────
        train_metrics, global_step = train_one_epoch(
            model, optimizer, train_loader, device, scaler, epoch_num, args, global_step
        )

        # ── Step LR (unless plateau, which steps after val) ──────────────────
        if args.scheduler != "plateau":
            scheduler.step()

        elapsed = time.time() - t0

        # ── Build log line ────────────────────────────────────────────────────
        current_lr = optimizer.param_groups[-1]['lr']
        log = (
            f"Epoch [{epoch_num:03d}/{args.epochs}] "
            f"loss={train_metrics['loss_total']:.4f}  "
            f"lr={current_lr:.2e}  "
            f"time={elapsed:.0f}s"
        )

        # ── Validation ────────────────────────────────────────────────────────
        val_metrics = {}
        if epoch_num % args.val_interval == 0:
            val_metrics = evaluate(
                model, val_loader, args.val_json, device, args.amp
            )
            mAP = val_metrics["mAP"]
            log += (
                f"  │  mAP={mAP:.4f}  "
                f"mAP@50={val_metrics['mAP_50']:.4f}  "
                f"mAP@75={val_metrics['mAP_75']:.4f}"
            )

            if mAP > best_map:
                best_map = mAP
                epochs_without_improvement = 0
                log += "  ★ best"
                save_checkpoint(
                    {
                        "epoch":       epoch,
                        "model":       model.state_dict(),
                        "optimizer":   optimizer.state_dict(),
                        "scaler":      scaler.state_dict(),
                        "best_map":    best_map,
                        "history":     history,
                        "args":        vars(args),
                    },
                    weights_dir / "best.pt",
                )
            else:
                epochs_without_improvement += 1

            if args.scheduler == "plateau":
                scheduler.step(mAP)

        print(log)

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if epoch_num % args.save_period == 0:
            save_checkpoint(
                {
                    "epoch":     epoch,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler":    scaler.state_dict(),
                    "best_map":  best_map,
                    "history":   history,
                    "args":      vars(args),
                },
                weights_dir / "last.pt",
            )

        # ── Early Stopping ────────────────────────────────────────────────────
        if epochs_without_improvement >= args.early_stop:
            print(f"\n[INFO] Early stopping triggered after {args.early_stop} epochs without improvement.")
            break

        # ── History ───────────────────────────────────────────────────────────
        record = {"epoch": epoch_num, **train_metrics, **val_metrics}
        history.append(record)

    # ── Persist history ───────────────────────────────────────────────────────
    history_path = save_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n[✓] Training complete.")
    print(f"    Best mAP     : {best_map:.4f}")
    print(f"    Best weights : {weights_dir / 'best.pt'}")
    print(f"    Last weights : {weights_dir / 'last.pt'}")
    print(f"    History      : {history_path}")
    print(f"\n    Classes ({len(CLASSES)}):")
    for i, cls in enumerate(CLASSES, start=1):
        print(f"      {i:2d}. {cls}")

    # ── Final Test Evaluation ─────────────────────────────────────────────────
    if args.test_json and Path(args.test_json).exists():
        print("\n\n" + "="*80)
        print("[🌟] BẮT ĐẦU THI TỐT NGHIỆP TRÊN TẬP TEST (20% DỮ LIỆU CÁCH LY)")
        print("="*80)
        best_weight_path = weights_dir / "best.pt"
        if best_weight_path.exists():
            print(f"[INFO] Nạp trọng số tốt nhất từ: {best_weight_path}")
            ckpt = torch.load(best_weight_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            
            from dataset_frcnn import VinBigFRCNNDataset, get_val_transforms, collate_fn
            from torch.utils.data import DataLoader
            
            test_ds = VinBigFRCNNDataset(
                args.data_root, args.test_json, args.imgsz, transforms=get_val_transforms()
            )
            test_loader = DataLoader(
                test_ds, batch_size=args.val_batch, shuffle=False,
                num_workers=args.workers, pin_memory=True, collate_fn=collate_fn
            )
            print(f"[INFO] Khởi tạo Test Dataset: {len(test_ds)} ảnh.")
            test_metrics = evaluate(model, test_loader, args.test_json, device, args.amp)
            
            print("\n" + "★"*55)
            print("🏆 KẾT QUẢ FINAL TEST ĐỘC LẬP TỪ BEST MODEL")
            print("★"*55)
            print(f"   mAP    (0.50:0.95) : {test_metrics['mAP']:.4f}")
            print(f"   mAP@50 (IoU 0.50)  : {test_metrics['mAP_50']:.4f}")
            print(f"   mAP@75 (IoU 0.75)  : {test_metrics['mAP_75']:.4f}")
            print("★"*55 + "\n")
        else:
            print("⚠️ Cảnh báo: Không tìm thấy file best.pt để thực hiện bài test cuối.")


if __name__ == "__main__":
    main()
