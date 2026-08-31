# Modifications for SVSS (Semi-supervised Video Semantic Segmentation)
# Assumes each batch_data from loader has:
#   support_img  : (B,3,H,W)
#   support_mask : (B,H,W) or (B,1,H,W)   labels 0..C-1 (and ignore_index)
#   query_imgs   : (B,T,3,H,W)
#   query_masks  : (B,T,H,W) or (B,T,1,H,W)
#
# And model forward is:
#   out = model(support_img, support_mask, query_imgs)
# where out is:
#   (B,T,C,H,W)
#
# Criterion should accept (N,C,H,W) logits and (N,H,W) targets (long),
# with ignore_index handled inside (e.g. CrossEntropyLoss(ignore_index=...)).

import os, time, collections, subprocess
from datetime import datetime
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.cuda import amp
from utils import banner, build_optimizer


class Trainer:
    def __init__(self,
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

        # SVSS params
        self.num_classes = int(self.cfg.data.num_class)          # e.g. 13
        self.ignore_index = int(getattr(self.cfg.train, "ignore_index", 255))
        self.class_map = {i: str(i) for i in range(self.num_classes)}

        self.val_ce = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index
        ).to(self.device)

    def get_gpu_memory_map(self):
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        return dict(zip(range(len(gpu_memory)), gpu_memory))

    def mem(self, gpu_indices):
        """Get primary GPU card memory usage (GB)."""
        if not torch.cuda.is_available():
            return -1.
        mem_map = self.get_gpu_memory_map()
        prim_card_num = gpu_indices[0] if isinstance(gpu_indices, collections.abc.Sequence) else int(gpu_indices)
        return mem_map[prim_card_num] / 1000

    def save_model(self, name):
        ts = datetime.now().strftime("%m%d%H%M")
        save_path = os.path.join(
            self.cfg.train.model_path,
            self.cfg.meta.name,
            self.cfg.train.model
        )
        os.makedirs(save_path, exist_ok=True)
        model_full_path = os.path.join(save_path, f'{name}_{ts}.pth')
        torch.save(self.model.state_dict(), model_full_path)
        return model_full_path

    def _as_hw_labels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize mask shapes to (N,H,W) long.
        Accepts:
          (N,H,W), (N,1,H,W)
        """
        if x.ndim == 4 and x.shape[1] == 1:
            x = x[:, 0]
        return x.long()

    def train(self, train_loader, epoch, log_run=True):
        epoch_start = time.time()
        
        # ---- IMPORTANT: vary allowed-frame sampling per epoch
        ds = getattr(train_loader, "dataset", None)
        if ds is not None:
            base = getattr(ds, "dataset", ds)  # unwrap Subset if present
            if hasattr(base, "epoch_salt"):
                base.epoch_salt = int(epoch)
            model = self.model.train()

        # --- unfreeze at chosen epoch
        if self.cfg.train.ft_encoder and epoch == self.cfg.train.unfreeze_epoch:
            for p in model.frame_model.encoder.parameters():
                p.requires_grad = True
            self.optimizer = build_optimizer(self.cfg, model)

            print("✅ Warmup completed — unfreezing encoder and enabling fine-tuning \n")

            # rebuild scheduler too (otherwise scheduler still controls old optimizer)
            if self.lr_scheduler is not None and getattr(self.cfg.train, "use_scheduler", False):
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=int(self.cfg.train.epochs),
                    eta_min=float(getattr(self.cfg.train, "min_lr", 1e-6)),
                )

        train_losses = []
        train_mious = []

        use_amp = self.cfg.train.amp and torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 7
        scaler = amp.GradScaler(enabled=use_amp)

        if epoch == 0:
            with banner(top=False):
                print(f"AMP Training {'Enabled' if use_amp else 'Not Enabled'}...")

        for idx, batch_data in enumerate(train_loader):
            # ---- SVSS batch
            support_img  = batch_data["support_img"]    # (B,S,3,H,W)   OR (B,3,H,W) if old
            support_mask = batch_data["support_mask"]   # (B,S,H,W)     OR (B,H,W)

            query_imgs   = batch_data["query_imgs"]    # (B,T,3,H,W)
            query_masks  = batch_data["query_masks"]   # (B,T,H,W) or (B,T,1,H,W)

            if self.cfg.train.debug:
                print("Batched support_img:", batch_data["support_img"].shape)   # should be (B,S,3,H,W)
                print("Batched support_mask:", batch_data["support_mask"].shape) # should be (B,S,H,W)
                print("Batched support_indices:", batch_data["support_indices"].shape, batch_data["support_indices"])


            support_img  = support_img.to(self.device, dtype=torch.float, non_blocking=True)
            support_mask = support_mask.to(self.device, non_blocking=True)
            query_imgs   = query_imgs.to(self.device, dtype=torch.float, non_blocking=True)
            query_masks  = query_masks.to(self.device, non_blocking=True)

            # print(f'before changing support mask {support_mask.shape}')
            # shape normalize
            # shape normalize support_mask
            # supports:
            #   (B,H,W)          -> keep
            #   (B,1,H,W)        -> squeeze
            #   (B,S,H,W)        -> keep
            #   (B,S,1,H,W)      -> squeeze channel
            if support_mask.ndim == 4 and support_mask.shape[1] == 1:
                support_mask = support_mask[:, 0]                   # (B,H,W)
            elif support_mask.ndim == 5 and support_mask.shape[2] == 1:
                support_mask = support_mask[:, :, 0]                # (B,S,H,W)
            support_mask = support_mask.long()

            # print(f'after changing support mask {support_mask.shape}')

            if self.cfg.train.debug:
                print("support_img shape:", support_img.shape)
                print("support_mask shape:", support_mask.shape)

            if query_masks.ndim == 5 and query_masks.shape[2] == 1:
                # (B,T,1,H,W) -> (B,T,H,W)
                query_masks = query_masks[:, :, 0]
            query_masks = query_masks.long()

            B, T, _, H, W = query_imgs.shape

            # ---- zero grad
            self.optimizer.zero_grad(set_to_none=True)

            # ---- forward + loss
            # with amp.autocast(enabled=use_amp):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                # Expect (B,T,C,H,W)
                support_indices = batch_data.get("support_indices", None)  # (B,S) or None
                query_indices   = batch_data.get("query_indices", None)    # (B,T) or None

                if support_indices is not None:
                    support_indices = support_indices.to(self.device, non_blocking=True)
                if query_indices is not None:
                    query_indices = query_indices.to(self.device, non_blocking=True)

                if self.cfg.train.debug:
                    print("loop support_mask uniq:", torch.unique(batch_data["support_mask"]).cpu().tolist()[:30])
                    print("loop query_masks uniq:", torch.unique(batch_data["query_masks"]).cpu().tolist()[:30])

                # print(support_img.shape, support_mask.shape, query_imgs.shape)
                out = model(
                    support_img,
                    support_mask,
                    query_imgs,
                    support_indices=support_indices,
                    query_indices=query_indices,
                )

                if out.ndim != 5:
                    raise ValueError(f"Expected model output (B,T,C,H,W), got {tuple(out.shape)}")

                if out.shape[:2] != (B, T) or out.shape[3:] != (H, W):
                    raise ValueError(
                        f"Output shape mismatch. out={tuple(out.shape)} "
                        f"expected (B={B},T={T},C, H={H},W={W})"
                    )

                # Flatten time into batch for criterion: (B*T,C,H,W) vs (B*T,H,W)
                out_bt = out.reshape(B * T, out.shape[2], H, W)
                tgt_bt = query_masks.reshape(B * T, H, W)

                # Your criterion should already handle ignore_index if needed.
                # e.g., nn.CrossEntropyLoss(ignore_index=255)
                # loss = self.criterion(out_bt.float(), tgt_bt)
                loss = self.criterion(out_bt, tgt_bt)

                del out

            loss_item = float(loss.item())

            # ---- backward + step
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # ---- Debug learnable memory weights (occasionally)
            # if idx % 50 == 0:   # print every 50 iters (adjust if noisy)
            #     if hasattr(model, "alpha_am_logit"):
            #         with torch.no_grad():
            #             am = torch.sigmoid(model.alpha_am_logit).item()
            #             tm = torch.sigmoid(model.alpha_tm_logit).item()
            #             print(f"[AlphaVal] it={idx} am={am:.4f} tm={tm:.4f}")

            train_losses.append(loss_item)

            with torch.no_grad():
                # pred_bt = out_bt.argmax(dim=1)   # (B*T,H,W)
                # pred_bt = out_bt.detach().float().cpu().argmax(dim=1)
                pred_bt = out_bt.detach().cpu().argmax(dim=1)                
                tgt_cpu = tgt_bt.detach().cpu()
                miou = self._miou(
                    pred=pred_bt,
                    target=tgt_cpu,
                    num_classes=self.num_classes,
                    ignore_index=self.ignore_index,
                )

                train_mious.append(miou)
            
            del out_bt, tgt_bt, loss


        lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        curr_lr = " & ".join([f"{lr:.6f}" for lr in lrs])
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        epoch_time = time.time() - epoch_start

        if log_run:
            print(
                f'  EP: {(epoch):03d}/{(self.cfg.train.epochs):03d} '
                f'[{epoch_time:.1f} s] | '
                f'lr : {curr_lr} | '
                f'loss: {np.mean(train_losses):.4f} | '
                f'mIoU: {np.nanmean(train_mious)*100:.2f} | '
                f'usage: {self.mem(self.device_idx):.1f} GB -- '
                f'[AMP:{use_amp}].'
            )

        return float(np.mean(train_losses)), float(np.nanmean(train_mious)*100)

    @torch.no_grad()
    def validate(self, val_loader, epoch=0, log_run=True):
        val_start = time.time()
        self.model.eval()

        val_losses = []
        miou_list = []
        miou_p1_list, miou_p2_list, miou_p3_list = [], [], []

        num_classes = self.num_classes
        ignore_index = self.ignore_index

        use_wandb = bool(getattr(self.cfg.val, "wandb_vis", False))
        vis_every = int(getattr(self.cfg.val, "wandb_vis_every", 50))
        vis_max_batches = int(getattr(self.cfg.train, "wandb_vis_max", 2))
        vis_frames = int(getattr(self.cfg.train, "wandb_vis_frames", 3))
        vis_logged = 0

        for bi, batch_data in enumerate(val_loader):
            support_img  = batch_data["support_img"].to(self.device, dtype=torch.float, non_blocking=True)
            support_mask = batch_data["support_mask"].to(self.device, non_blocking=True)
            query_imgs   = batch_data["query_imgs"].to(self.device, dtype=torch.float, non_blocking=True)
            query_masks  = batch_data["query_masks"].to(self.device, non_blocking=True)

            # support_mask can be (B,H,W), (B,1,H,W), (B,S,H,W), (B,S,1,H,W)
            if support_mask.ndim == 4 and support_mask.shape[1] == 1:
                support_mask = support_mask[:, 0]            # (B,H,W)
            elif support_mask.ndim == 5 and support_mask.shape[2] == 1:
                support_mask = support_mask[:, :, 0]         # (B,S,H,W)
            support_mask = support_mask.long()

            if query_masks.ndim == 5 and query_masks.shape[2] == 1:
                query_masks = query_masks[:, :, 0]
            query_masks = query_masks.long()

            B, T, _, H, W = query_imgs.shape
            # with amp.autocast(enabled=False):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                support_indices = batch_data.get("support_indices", None)  # (B,S) or None
                query_indices   = batch_data.get("query_indices", None)    # (B,T) or None
                if support_indices is not None:
                    support_indices = support_indices.to(self.device, non_blocking=True)
                if query_indices is not None:
                    query_indices = query_indices.to(self.device, non_blocking=True)
                
                # --- FORCE single-support for validation
                if support_img.ndim == 5:          # (B,S,3,H,W)
                    support_img = support_img[:, 0]  # (B,3,H,W)

                if support_mask.ndim == 4:         # (B,S,H,W)
                    support_mask = support_mask[:, 0]  # (B,H,W)

                if support_indices is not None:
                    if torch.is_tensor(support_indices) and support_indices.ndim == 2:  # (B,S)
                        support_indices = support_indices[:, 0]  # (B,)

                # import IPython; IPython.embed()
                out = self.model(
                    support_img, support_mask, query_imgs,
                    support_indices=support_indices,
                    query_indices=query_indices,
                )  # (B,T,C,H,W)

            out_bt = out.reshape(B * T, out.shape[2], H, W)
            del out
            tgt_bt = query_masks.reshape(B * T, H, W)

            loss = self.val_ce(out_bt.float(), tgt_bt)
            val_losses.append(float(loss.item()))

            # ---- mIoU overall + phase-wise
            pred_bt = torch.argmax(out_bt, dim=1).to(torch.int32)          # (B*T,H,W)
            pred_bthw = pred_bt.view(B, T, H, W)           # (B,T,H,W)
            tgt_bthw  = query_masks.view(B, T, H, W)       # (B,T,H,W)

            q_idx = query_indices  # use the already-to(device) version

            for b in range(B):
                v_val_start = time.time()
                qi_b = None
                if q_idx is not None:
                    qi_b = q_idx[b]  # (T,)

                # ---- get original video length for phase assignment
                video_len_b = None
                if "video_len" in batch_data:
                    vl = batch_data["video_len"]
                    video_len_b = int(vl[b].item()) if torch.is_tensor(vl) else int(vl[b])

                mi_all, mi_p1, mi_p2, mi_p3, _ = self.miou_per_frame_fast(
                    pred_bthw[b], tgt_bthw[b],
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    include_bg=True,
                    query_indices=qi_b,
                    video_len=video_len_b,   # <-- HERE
                )

                # ---- per-video print (works for B=1 or B>1)
                clip = batch_data.get("video_clip", f"bi{bi}_b{b}")
                if isinstance(clip, (list, tuple)):
                    clip_b = clip[b]
                elif torch.is_tensor(clip):
                    clip_b = clip[b].item() if clip.numel() > b else f"bi{bi}_b{b}"
                else:
                    clip_b = clip

                if epoch == self.cfg.train.epochs - 1 or getattr(self.cfg.train, "eval_only", False):
                    print(
                        f"[{(time.time()-v_val_start):.1f} s] VAL Clip {clip_b} -- {vl.item()} Frames |  "
                        f"mIoU: {mi_all*100:.2f} | "
                        f"Phases mIoU: {mi_p1*100:.2f} / {mi_p2*100:.2f} / {mi_p3*100:.2f}"
                    )

                if not np.isnan(mi_all): miou_list.append(mi_all)
                if not np.isnan(mi_p1):  miou_p1_list.append(mi_p1)
                if not np.isnan(mi_p2):  miou_p2_list.append(mi_p2)
                if not np.isnan(mi_p3):  miou_p3_list.append(mi_p3)

            del pred_bthw, tgt_bthw, pred_bt, out_bt
            del support_img, support_mask, query_imgs, query_masks
            torch.cuda.empty_cache()
            # # ---- W&B visualization (unchanged, but uses pred_bt now)
            # if use_wandb and (vis_logged < vis_max_batches) and ((bi % vis_every) == 0):
            #     b0 = 0
            #     f = min(vis_frames, T)
            #     frame_ids = np.linspace(0, T - 1, num=f, dtype=int).tolist()
            #     step = int(epoch) * int(len(val_loader)) + int(bi)

            #     for ti in frame_ids:
            #         img_chw = query_imgs[b0, ti]
            #         gt_hw   = query_masks[b0, ti]
            #         pd_hw   = pred_bthw[b0, ti]  # (H,W)

            #         self._wandb_log_triplet(
            #             img_chw=img_chw,
            #             gt_hw=gt_hw,
            #             pred_hw=pd_hw,
            #             step=step,
            #             tag=f"val/b{bi:04d}_t{ti:03d}",
            #         )

            #     vis_logged += 1

        ep_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        ep_val_miou = float(np.mean(miou_list)) if miou_list else float("nan")
        ep_val_miou_p1 = float(np.mean(miou_p1_list)) if miou_p1_list else float("nan")
        ep_val_miou_p2 = float(np.mean(miou_p2_list)) if miou_p2_list else float("nan")
        ep_val_miou_p3 = float(np.mean(miou_p3_list)) if miou_p3_list else float("nan")

        if log_run:
            print("\n")
            print(
                f"     🔍 Val {epoch:03d}/{self.cfg.train.epochs:03d} [{(time.time()-val_start):.1f} s] | "
                f"loss: {ep_val_loss:.4f} | "
                f"Avg mIoU: {ep_val_miou*100:.2f} | "
                f"Phases mIoU: {ep_val_miou_p1*100:.2f} / "
                f"{ep_val_miou_p2*100:.2f} / "
                f"{ep_val_miou_p3*100:.2f}"
            )
            print("\n")

        return (
            ep_val_loss,
            ep_val_miou * 100.0,
            ep_val_miou_p1 * 100.0, 
            ep_val_miou_p2 * 100.0, 
            ep_val_miou_p3 * 100.0
        )       


    def _miou(self, pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> float:
        """
        pred/target: (N,H,W) long
        mIoU averaged over classes that appear in union (ignoring ignore_index).
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

    def miou_per_frame_fast(
        self,
        pred_thw: torch.Tensor,
        tgt_thw: torch.Tensor,
        num_classes: int,
        ignore_index: int = 255,
        include_bg: bool = True,
        query_indices=None,
        video_len: int | None = None,
        p1_frac: float = 0.2,   # early fraction
        p3_frac: float = 0.2,   # late fraction
    ):

        """
        Returns:
            miou_all (float),
            miou_p1 (float), miou_p2 (float), miou_p3 (float),
            miou_t  (list[float])  # per-sampled-frame mIoU

        - mIoU is computed ONLY on sampled query frames
        - Phase buckets (early/mid/late) are assigned based on the original
        video timeline using query_indices / video_len
        """

        assert pred_thw.ndim == 3 and tgt_thw.ndim == 3
        T = pred_thw.shape[0]
        if T == 0:
            nan = float("nan")
            return nan, nan, nan, nan, []

        device = pred_thw.device
        C = int(num_classes)
        eps = 1e-12

        # ---------------- per-frame mIoU ----------------
        if include_bg:
            cls = torch.arange(C, device=device)
        else:
            cls = torch.arange(1, C, device=device)

        miou_t = []

        for t in range(T):
            p = pred_thw[t].reshape(-1)
            y = tgt_thw[t].reshape(-1)

            valid = (y != ignore_index)
            if not valid.any():
                miou_t.append(float("nan"))
                continue

            p = p[valid]
            y = y[valid]

            # fast confusion via bincount
            k = (y * C + p).to(torch.int64)
            conf = torch.bincount(k, minlength=C * C).reshape(C, C).float()

            tp = conf.diag()
            fp = conf.sum(0) - tp
            fn = conf.sum(1) - tp
            denom = tp + fp + fn

            iou = (tp + eps) / (denom + eps)

            valid_cls = denom[cls] > 0
            if valid_cls.any():
                miou_t.append(float(iou[cls][valid_cls].mean().item()))
            else:
                miou_t.append(float("nan"))

        # ---------------- tensorize ----------------
        x = torch.tensor(miou_t, device=device, dtype=torch.float32)
        good = torch.isfinite(x)

        miou_all = float(x[good].mean().item()) if good.any() else float("nan")

        # ---------------- phase assignment ----------------
        # rel \in [0,1] is the position in the ORIGINAL video timeline
        if query_indices is not None:
            if not torch.is_tensor(query_indices):
                q = torch.tensor(query_indices, device=device, dtype=torch.float32).view(-1)
            else:
                q = query_indices.to(device=device, dtype=torch.float32).view(-1)

            if q.numel() != T:
                # fallback: sampled order only
                rel = (torch.arange(T, device=device, dtype=torch.float32) + 0.5) / T
            else:
                if video_len is not None and int(video_len) > 1:
                    rel = (q / float(int(video_len) - 1)).clamp(0.0, 1.0)
                else:
                    # fallback: normalize by max observed query index
                    qmax = float(q.max().item())
                    rel = (q / max(qmax, 1.0)).clamp(0.0, 1.0)
        else:
            # fallback: sampled order only
            rel = (torch.arange(T, device=device, dtype=torch.float32) + 0.5) / T

        # ---------------- 3 phase buckets (custom) ----------------
        # early = [0, p1_frac)
        # mid   = [p1_frac, 1 - p3_frac)
        # late  = [1 - p3_frac, 1]

        # clamp + sanity
        p1 = float(max(0.0, min(0.49, p1_frac)))
        p3 = float(max(0.0, min(0.49, p3_frac)))
        mid_end = 1.0 - p3
        if mid_end <= p1:
            # fallback to thirds if invalid
            p1, mid_end = 1/3, 2/3

        p1_idx = (rel < p1)
        p2_idx = (rel >= p1) & (rel < mid_end)
        p3_idx = (rel >= mid_end)        

        def masked_nanmean(mask):
            m = mask & good
            return float(x[m].mean().item()) if m.any() else float("nan")

        miou_p1 = masked_nanmean(p1_idx)
        miou_p2 = masked_nanmean(p2_idx)
        miou_p3 = masked_nanmean(p3_idx)

        return miou_all, miou_p1, miou_p2, miou_p3, miou_t