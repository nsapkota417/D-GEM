import collections
import csv
import gc
import json
import os
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.cuda import amp

from utils import banner, build_optimizer

class Trainer:
    def __init__(
        self,
        cfg,
        model: torch.nn.Module,
        device: torch.device,
        device_idx: int,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler = None,
    ):
        self.cfg = cfg
        self.device = device
        self.device_idx = device_idx
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.num_classes = int(self.cfg.data.num_class)
        self.ignore_index = int(
            getattr(self.cfg.data, "ignore_index",
            getattr(self.cfg.train, "ignore_index", 255))
        )

        self.val_ce = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index
        ).to(self.device)

    def get_gpu_memory_map(self):
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
            ],
            encoding="utf-8",
        )
        gpu_memory = [int(x) for x in result.strip().split("\n")]
        return dict(zip(range(len(gpu_memory)), gpu_memory))

    def mem(self, gpu_indices):
        if not torch.cuda.is_available():
            return -1.0
        mem_map = self.get_gpu_memory_map()
        prim_card_num = (
            gpu_indices[0]
            if isinstance(gpu_indices, collections.abc.Sequence)
            else int(gpu_indices)
        )
        return mem_map[prim_card_num] / 1000.0

    def save_model(self, name):
        ts = datetime.now().strftime("%m%d%H%M")
        save_path = self.cfg.train.model_save_path
        os.makedirs(save_path, exist_ok=True)
        model_full_path = os.path.join(save_path, f"{name}_{ts}.pth")
        torch.save(self.model.state_dict(), model_full_path)
        return model_full_path

    def _as_hw_labels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize mask shapes to (B,H,W) long.
        Accepts:
          (B,H,W), (B,1,H,W)
        """
        if x.ndim == 4 and x.shape[1] == 1:
            x = x[:, 0]
        return x.long()

    def _forward_logits(self, image):
        """
        Accept either:
          model(image) -> logits
        or
          model(image) -> (logits, ...)
        """
        out = self.model(image)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def train(self, train_loader, epoch, log_run=True):
        epoch_start = time.time()
        model = self.model.train()

        # optional encoder unfreeze
        if getattr(self.cfg.train, "ft_encoder", False) and epoch == getattr(self.cfg.train, "unfreeze_epoch", -1):
            if hasattr(model, "encoder"):
                for p in model.encoder.parameters():
                    p.requires_grad = True
                self.optimizer = build_optimizer(self.cfg, model)

                print("✅ Warmup completed — unfreezing encoder and enabling fine-tuning\n")

                if self.lr_scheduler is not None and getattr(self.cfg.train, "use_scheduler", False):
                    self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer,
                        T_max=int(self.cfg.train.epochs),
                        eta_min=float(getattr(self.cfg.train, "min_lr", 1e-6)),
                    )

        train_losses = []
        train_mious = []

        use_amp = (
            self.cfg.train.amp
            and torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).major >= 7
        )
        scaler = amp.GradScaler(enabled=use_amp)

        if epoch == 0:
            with banner(top=False):
                print(f"AMP Training {'Enabled' if use_amp else 'Not Enabled'}...")

        for idx, batch_data in enumerate(train_loader):
            image = batch_data["image"].to(
                self.device, dtype=torch.float, non_blocking=True
            )
            mask = batch_data["mask"].to(
                self.device, non_blocking=True
            )
            mask = self._as_hw_labels(mask)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = self._forward_logits(image)   # (B,C,H,W)
                loss = self.criterion(logits, mask)

            loss_item = float(loss.item())

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            train_losses.append(loss_item)

            with torch.no_grad():
                pred = logits.detach().argmax(dim=1).cpu()
                tgt = mask.detach().cpu()
                miou = self._miou(
                    pred=pred,
                    target=tgt,
                    num_classes=self.num_classes,
                    ignore_index=self.ignore_index,
                )
                train_mious.append(miou)

            del logits, loss

        lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        curr_lr = " & ".join([f"{lr:.6f}" for lr in lrs])

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        epoch_time = time.time() - epoch_start

        if log_run:
            print(
                f"  EP: {(epoch):03d}/{(self.cfg.train.epochs):03d} "
                f"[{epoch_time:.1f} s] | "
                f"lr : {curr_lr} | "
                f"loss: {np.mean(train_losses):.4f} | "
                f"mIoU: {np.nanmean(train_mious)*100:.2f} | "
                f"usage: {self.mem(self.device_idx):.1f} GB -- "
                f"[AMP:{use_amp}]."
            )

        return float(np.mean(train_losses)), float(np.nanmean(train_mious) * 100)

    @torch.no_grad()
    def validate(self, val_loader, epoch=0, log_run=True):
        val_start = time.time()

        self.model.eval()

        val_losses = []

        # video_src -> list of frame-level mIoUs
        video_frame_mious = collections.defaultdict(list)

        use_autocast = (
            torch.cuda.is_available()
            and str(self.device).startswith("cuda")
        )

        for bi, batch_data in enumerate(val_loader):
            image = batch_data["image"].to(
                self.device,
                dtype=torch.float,
                non_blocking=True,
            )
            mask = batch_data["mask"].to(
                self.device,
                non_blocking=True,
            )
            mask = self._as_hw_labels(mask)

            # Dataset must return video_src for every frame.
            if "video_src" not in batch_data:
                raise KeyError(
                    "Validation dataset must return 'video_src' in each sample."
                )

            video_sources = batch_data["video_src"]

            # Handles batch size 1 or unusual collate behavior.
            if isinstance(video_sources, str):
                video_sources = [video_sources]

            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_autocast,
            ):
                logits = self._forward_logits(image)

            loss = self.val_ce(logits.float(), mask)
            val_losses.append(float(loss.item()))

            preds = logits.argmax(dim=1).cpu()
            targets = mask.cpu()

            # Calculate mIoU independently for every frame.
            for i in range(preds.shape[0]):
                frame_miou = self._miou(
                    pred=preds[i:i + 1],
                    target=targets[i:i + 1],
                    num_classes=self.num_classes,
                    ignore_index=self.ignore_index,
                )

                if not np.isnan(frame_miou):
                    video_id = str(video_sources[i])
                    video_frame_mious[video_id].append(frame_miou)

            del logits, loss, preds, targets

        ep_val_loss = (
            float(np.mean(val_losses))
            if val_losses
            else float("nan")
        )

        # Average frame mIoUs separately for each video.
        per_video_miou = {
            video_id: float(np.mean(frame_mious))
            for video_id, frame_mious in video_frame_mious.items()
            if frame_mious
        }

        # Macro average: every video receives equal weight.
        ep_val_miou = (
            float(np.mean(list(per_video_miou.values())))
            if per_video_miou
            else float("nan")
        )

        if log_run:
            print("\nValidation mIoU by video")
            print("-" * 100)

            for video_id in sorted(per_video_miou):
                num_frames = len(video_frame_mious[video_id])
                video_miou = per_video_miou[video_id] * 100.0

                print(
                    f"{video_id:<35} "
                    f"{video_miou:>7.2f} "
                    f"({num_frames:,} frames)"
                )

            print("-" * 100)
            print(
                f"🔍 Val [{time.time() - val_start:.1f}s] | "
                f"loss: {ep_val_loss:.4f} | "
                f"Video-Avg mIoU: {ep_val_miou * 100:.2f} | "
                f"videos: {len(per_video_miou)} | "
                f"usage: {self.mem(self.device_idx):.1f} GB"
            )
            print("-" * 100)


        # Preserve the existing return signature.
        ep_val_miou_p1 = ep_val_miou
        ep_val_miou_p2 = ep_val_miou
        ep_val_miou_p3 = ep_val_miou

        return (
            ep_val_loss,
            ep_val_miou * 100.0,
            ep_val_miou_p1 * 100.0,
            ep_val_miou_p2 * 100.0,
            ep_val_miou_p3 * 100.0,
        )

    def _miou(self, pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> float:
        """
        pred/target: (B,H,W) long
        mIoU averaged over classes that appear in union.
        """
        mask = (target != ignore_index)
        pred = pred[mask]
        target = target[mask]

        if pred.numel() == 0:
            return float("nan")

        ious = []
        for c in range(num_classes):
            p = (pred == c)
            t = (target == c)
            inter = (p & t).sum().item()
            union = (p | t).sum().item()
            if union == 0:
                continue
            ious.append(inter / union)

        return float(np.mean(ious)) if len(ious) else float("nan")

@torch.no_grad()
def inference_with_miou(
    cfg,
    inf_df,
    trainer,
    model,
    data_loader,
    save_dir,
    device,
    save_outputs=True,
):
    def _safe_write_csv(df, out_path):
        tmp_path = str(out_path) + ".tmp"
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, out_path)

    num_classes = int(cfg.data.num_class)
    ignore_index = int(cfg.data.ignore_index)

    model.eval()

    # -------------------------------------------------
    # Output preparation
    # -------------------------------------------------
    if save_outputs:
        save_dir = Path(save_dir)
        preds_dir = save_dir / "preds"
        metrics_dir = save_dir / "metrics"

        preds_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        if "img" not in inf_df.columns:
            raise ValueError("inf_df must contain an 'img' column")

        inf_df = inf_df.copy()
        inf_df["img"] = inf_df["img"].astype(str)

        if inf_df["img"].duplicated().any():
            raise ValueError("Duplicate values found in inf_df['img']")

        df_lookup = (
            inf_df
            .set_index("img", drop=False)
            .to_dict(orient="index")
        )
        df_cols = list(inf_df.columns)

    else:
        preds_dir = None
        metrics_dir = None
        df_lookup = None
        df_cols = []

    processed = 0
    print_every = int(cfg.val.eval_print_every)
    chunk_size = int(cfg.val.eval_chunk_size)
    total = len(data_loader.dataset)

    running_miou = deque(maxlen=chunk_size)

    chunk_rows = []
    chunk_id = 0

    # video_id -> list of frame-level mIoUs, stored in [0, 1]
    video_frame_mious = collections.defaultdict(list)

    # video_id -> class_id -> list of frame-level class IoUs
    video_class_ious = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )

    use_autocast = (
        torch.cuda.is_available()
        and str(device).startswith("cuda")
    )

    start_time = time.time()

    for bi, batch_data in enumerate(data_loader):
        image = batch_data["image"].to(
            device,
            dtype=torch.float,
            non_blocking=True,
        )
        mask = batch_data["mask"].to(
            device,
            non_blocking=True,
        )
        mask = trainer._as_hw_labels(mask)

        batch_size = image.shape[0]

        # video_src is always needed for video-level evaluation.
        if "video_src" not in batch_data:
            raise KeyError(
                "batch_data must contain 'video_src' for every frame"
            )

        video_sources = batch_data["video_src"]

        if isinstance(video_sources, str):
            video_sources = [video_sources]

        if len(video_sources) != batch_size:
            raise ValueError(
                f"Expected {batch_size} video_src values, "
                f"received {len(video_sources)}"
            )

        # img_path is needed only when saving predictions/metrics.
        if save_outputs:
            img_paths = batch_data.get("img_path")

            if img_paths is None:
                raise ValueError(
                    "batch_data must contain 'img_path' when "
                    "save_outputs=True"
                )

        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_autocast,
        ):
            logits = trainer._forward_logits(image)

        preds = logits.argmax(dim=1).cpu()
        targets = mask.cpu()

        for i in range(batch_size):
            pred = preds[i]
            tgt = targets[i]

            video_id = str(video_sources[i]).strip()

            if video_id in {"", "-", "nan", "None"}:
                raise ValueError(
                    f"Invalid video_src for sample {processed}: "
                    f"{video_sources[i]!r}"
                )

            # ------------------------------------------
            # Prepare row and save prediction
            # ------------------------------------------
            if save_outputs:
                img_path = str(img_paths[i])

                if img_path not in df_lookup:
                    raise KeyError(
                        f"{img_path} not found in inf_df['img']"
                    )

                row = dict(df_lookup[img_path])

                # Prevent duplicate filenames from different videos.
                safe_video_id = (
                    video_id
                    .replace("/", "_")
                    .replace("\\", "_")
                    .replace(" ", "_")
                )

                video_pred_dir = preds_dir / safe_video_id
                video_pred_dir.mkdir(parents=True, exist_ok=True)

                name = Path(img_path).stem
                pred_path = video_pred_dir / f"{name}.png"

                Image.fromarray(
                    pred.numpy().astype(np.uint8)
                ).save(pred_path)

                row["pred_path"] = str(pred_path)
                row["miou"] = None

                for c in range(num_classes):
                    row[f"class_{c}_miou"] = None

            # ------------------------------------------
            # Frame-level IoU calculation
            # Matches Trainer._miou()
            # ------------------------------------------
            valid = tgt != ignore_index

            if valid.any():
                pred_v = pred[valid]
                tgt_v = tgt[valid]

                frame_ious = []

                for c in range(num_classes):
                    pred_c = pred_v == c
                    target_c = tgt_v == c

                    intersection = (pred_c & target_c).sum().item()
                    union = (pred_c | target_c).sum().item()

                    # Match Function 1: exclude absent classes.
                    if union == 0:
                        continue

                    class_iou = intersection / union

                    frame_ious.append(class_iou)
                    video_class_ious[video_id][c].append(class_iou)

                    if save_outputs:
                        row[f"class_{c}_miou"] = class_iou * 100.0

                if frame_ious:
                    # Mean over classes present in this frame.
                    frame_miou = float(np.mean(frame_ious))

                    video_frame_mious[video_id].append(frame_miou)
                    running_miou.append(frame_miou * 100.0)

                    if save_outputs:
                        row["miou"] = frame_miou * 100.0

            if save_outputs:
                chunk_rows.append(row)

            processed += 1

            # ------------------------------------------
            # Running progress
            # ------------------------------------------
            if processed % print_every == 0:
                elapsed = time.time() - start_time

                avg_frame_time_ms = (
                    elapsed / processed * 1000.0
                    if processed > 0
                    else float("nan")
                )

                fps = (
                    processed / elapsed
                    if elapsed > 0
                    else float("nan")
                )

                recent_miou = (
                    float(np.mean(running_miou))
                    if running_miou
                    else float("nan")
                )

                print(
                    f"[{processed:,}/{total:,}] | "
                    f"Recent mIoU: {recent_miou:.2f} | "
                    f"{avg_frame_time_ms:.2f} ms/frame | "
                    f"{fps:.2f} frames/s"
                )

            # ------------------------------------------
            # Save metric chunk
            # ------------------------------------------
            if save_outputs and len(chunk_rows) >= chunk_size:
                df = pd.DataFrame(chunk_rows)

                ordered_cols = (
                    df_cols
                    + ["pred_path", "miou"]
                    + [
                        f"class_{c}_miou"
                        for c in range(num_classes)
                    ]
                )

                df = df[
                    [c for c in ordered_cols if c in df.columns]
                ]

                path = metrics_dir / f"metrics_{chunk_id:04d}.csv"
                _safe_write_csv(df, path)

                chunk_rows.clear()
                chunk_id += 1

                del df
                gc.collect()

        del logits, preds, targets
        gc.collect()

    # -------------------------------------------------
    # Save remaining rows
    # -------------------------------------------------
    if save_outputs and chunk_rows:
        df = pd.DataFrame(chunk_rows)

        ordered_cols = (
            df_cols
            + ["pred_path", "miou"]
            + [
                f"class_{c}_miou"
                for c in range(num_classes)
            ]
        )

        df = df[
            [c for c in ordered_cols if c in df.columns]
        ]

        path = metrics_dir / f"metrics_{chunk_id:04d}.csv"
        _safe_write_csv(df, path)

        chunk_rows.clear()

        del df
        gc.collect()

    # -------------------------------------------------
    # Per-video mIoU
    #
    # Each video:
    # mean of its valid frame-level mIoUs.
    # -------------------------------------------------
    per_video_miou = {
        video_id: float(np.mean(frame_mious) * 100.0)
        for video_id, frame_mious in video_frame_mious.items()
        if frame_mious
    }

    # -------------------------------------------------
    # Global mIoU
    #
    # Exact Function 1 behavior:
    # equal-weight average across video mIoUs.
    # -------------------------------------------------
    global_miou = (
        float(np.mean(list(per_video_miou.values())))
        if per_video_miou
        else None
    )

    # -------------------------------------------------
    # Per-class mIoU
    #
    # First average each class across frames in a video,
    # then average that class equally across eligible videos.
    # -------------------------------------------------
    per_video_class_miou = {}

    for video_id, class_values in video_class_ious.items():
        per_video_class_miou[video_id] = {
            c: float(np.mean(values) * 100.0)
            for c, values in class_values.items()
            if values
        }

    per_class_miou = {}

    for c in range(num_classes):
        class_video_scores = [
            video_scores[c]
            for video_scores in per_video_class_miou.values()
            if c in video_scores
        ]

        per_class_miou[c] = (
            float(np.mean(class_video_scores))
            if class_video_scores
            else None
        )

    # -------------------------------------------------
    # Timing
    # -------------------------------------------------
    total_time = time.time() - start_time

    avg_time_per_frame = (
        total_time / processed
        if processed > 0
        else float("nan")
    )

    fps = (
        processed / total_time
        if total_time > 0
        else float("nan")
    )

    # -------------------------------------------------
    # Final printing
    # -------------------------------------------------
    print("\nInference mIoU by video")
    print("-" * 100)

    for video_id in sorted(per_video_miou):
        num_frames = len(video_frame_mious[video_id])

        print(
            f"{video_id:<45} "
            f"{per_video_miou[video_id]:>7.2f} "
            f"({num_frames:,} valid frames)"
        )

    print("-" * 100)
    print("Per-class mIoU")
    print("-" * 100)

    for c in range(num_classes):
        class_miou = per_class_miou[c]

        if class_miou is None:
            score_text = "N/A"
            num_videos = 0
        else:
            score_text = f"{class_miou:.2f}"
            num_videos = sum(
                c in video_scores
                for video_scores in per_video_class_miou.values()
            )

        print(
            f"Class {c:<4} "
            f"{score_text:>7} "
            f"({num_videos:,} videos)"
        )

    print("-" * 100)
    print(
        f"Global video-averaged mIoU : "
        f"{global_miou:.2f}"
        if global_miou is not None
        else "Global video-averaged mIoU : N/A"
    )
    print(f"Total time                  : {total_time:.2f} s")
    print(
        f"Average time per frame      : "
        f"{avg_time_per_frame * 1000.0:.2f} ms"
    )
    print(f"Average throughput          : {fps:.2f} frames/s")
    print(f"Processed frames            : {processed:,}")
    print(f"Evaluated videos            : {len(per_video_miou):,}")
    print(
        f"Output saving               : "
        f"{'enabled' if save_outputs else 'disabled'}"
    )
    print("-" * 100)

    return {
        "per_video_miou": per_video_miou,
        "per_video_class_miou": per_video_class_miou,
        "per_class_miou": per_class_miou,
        "global_miou": global_miou,
        "total_time_seconds": total_time,
        "avg_time_per_frame_seconds": avg_time_per_frame,
        "frames_per_second": fps,
        "processed_frames": processed,
    }