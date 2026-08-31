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
from PIL import Image

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

        self.ignore_index = int(getattr(self.cfg.data, "ignore_index", getattr(self.cfg.train, "ignore_index", 255)))

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

    def detach_state(self, state):
        if torch.is_tensor(state):
            return state.detach()

        if isinstance(state, (list, tuple)):
            return type(state)(self.detach_state(x) for x in state)

        if isinstance(state, dict):
            return {k: self.detach_state(v) for k, v in state.items()}

        # custom object: try common pattern
        if hasattr(state, "__dict__"):
            for k, v in vars(state).items():
                setattr(state, k, self.detach_state(v))
            return state

        return state

    def train(self, train_loader, epoch, log_run=True):
        epoch_start = time.time()
        
        # ---- IMPORTANT: vary allowed-frame sampling per epoch
        model = self.model.train()
        ds = getattr(train_loader, "dataset", None)
        if ds is not None:
            base = getattr(ds, "dataset", ds)  # unwrap Subset if present
            if hasattr(base, "epoch_salt"):
                base.epoch_salt = int(epoch)

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

            # ---- SVSS batch (supports always present)
            support_img  = batch_data["support_img"]      # (B,S,3,H,W)
            support_mask = batch_data["support_mask"]     # (B,S,H,W) (or variants)

            # ---- move supports to GPU
            support_img  = support_img.to(self.device, dtype=torch.float, non_blocking=True)
            support_mask = support_mask.to(self.device, non_blocking=True)

            # ---- normalize support_mask shapes
            if support_mask.ndim == 4 and support_mask.shape[1] == 1:
                support_mask = support_mask[:, 0]                   # (B,H,W)
            elif support_mask.ndim == 5 and support_mask.shape[2] == 1:
                support_mask = support_mask[:, :, 0]                # (B,S,H,W)
            support_mask = support_mask.long()

            # indices
            support_indices = batch_data.get("support_indices", None)
            query_indices   = batch_data.get("query_indices", None)
            if support_indices is not None and torch.is_tensor(support_indices):
                support_indices = support_indices.to(self.device, non_blocking=True)
            if query_indices is not None and torch.is_tensor(query_indices):
                query_indices = query_indices.to(self.device, non_blocking=True)

            # ---- zero grad
            self.optimizer.zero_grad(set_to_none=True)

            use_stream = ("query_img_paths" in batch_data) and ("query_mask_paths" in batch_data)

            if self.cfg.train.debug[0]:
                print(f"\n[TRAIN] idx={idx} use_stream={use_stream}")
                print("support_img:", tuple(support_img.shape))
                print("support_mask:", tuple(support_mask.shape))
                if support_indices is not None:
                    print("support_indices:", tuple(support_indices.shape), support_indices[0, :min(10, support_indices.shape[1])].tolist())

            # ============================================================
            # (A) NON-STREAMING (your current behavior)
            # ============================================================
            if not use_stream:
                query_imgs  = batch_data["query_imgs"].to(self.device, dtype=torch.float, non_blocking=True)   # (B,T,3,H,W)
                query_masks = batch_data["query_masks"].to(self.device, non_blocking=True)                     # (B,T,H,W) or (B,T,1,H,W)

                if query_masks.ndim == 5 and query_masks.shape[2] == 1:
                    query_masks = query_masks[:, :, 0]
                query_masks = query_masks.long()
                B, T, _, H, W = query_imgs.shape
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):

                    if self.cfg.train.use_raw_logits:
                        out, out_raw = model(
                            support_img,
                            support_mask,
                            query_imgs,
                            support_indices=support_indices,
                            query_indices=query_indices,
                            return_raw=self.cfg.train.use_raw_logits,
                        )
                    else:
                        out = model(
                            support_img,
                            support_mask,
                            query_imgs,
                            support_indices=support_indices,
                            query_indices=query_indices,
                            return_raw=False
                        )
                        out_raw = None

                    out_bt = out.reshape(B * T, out.shape[2], H, W)

                    if self.cfg.train.use_raw_logits:
                        out_raw_bt = out_raw.reshape(B * T, out_raw.shape[2], H, W)

                    tgt_bt = query_masks.reshape(B * T, H, W)
                    aux_w = float(getattr(self.cfg.train, "aux_raw_w", 0.2))
                    loss_fused = self.criterion(out_bt, tgt_bt)
                    if self.cfg.train.use_raw_logits:
                        loss_raw = self.criterion(out_raw_bt, tgt_bt)
                        loss = loss_fused + aux_w * loss_raw
                    else:
                        loss = loss_fused

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
                    pred_bt = out_bt.detach().cpu().argmax(dim=1)
                    tgt_cpu = tgt_bt.detach().cpu()
                    miou = self._miou(pred=pred_bt, target=tgt_cpu,
                                    num_classes=self.num_classes,
                                    ignore_index=self.ignore_index)
                    train_mious.append(miou)              

                del out, out_bt, tgt_bt, loss
                continue

            # ============================================================
            # (B) STREAMING / ROLLOUT mode (paths, no big query tensor)
            # ============================================================

            q_roll = batch_data.get("query_roll_indices", None)
            if q_roll is None:
                raise RuntimeError("streaming requires query_roll_indices")

            q_sup = batch_data.get("query_indices", None)
            if q_sup is None:
                raise RuntimeError("streaming requires query_indices")

            # ---- normalize q_roll/q_sup to tensors on GPU
            if not torch.is_tensor(q_roll):
                q_roll = torch.as_tensor(q_roll, dtype=torch.long)
            if not torch.is_tensor(q_sup):
                q_sup = torch.as_tensor(q_sup, dtype=torch.long)

            q_roll = q_roll.to(self.device, non_blocking=True)
            q_sup  = q_sup.to(self.device, non_blocking=True)

            # ---- ensure shapes are (B,T_rollout) and (B,t_query)
            if q_roll.dim() == 1:
                q_roll = q_roll.unsqueeze(0)
            if q_sup.dim() == 1:
                q_sup = q_sup.unsqueeze(0)

            # ---- get paths (collate_stream returns list-of-lists)
            q_img_paths  = batch_data["query_img_paths"]
            q_mask_paths = batch_data["query_mask_paths"]

            # normalize paths to "per-sample lists"
            # - if batch_size=1 and someone returned a flat list, wrap it
            if len(q_img_paths) > 0 and isinstance(q_img_paths[0], str):
                q_img_paths = [q_img_paths]
            if len(q_mask_paths) > 0 and isinstance(q_mask_paths[0], str):
                q_mask_paths = [q_mask_paths]

            # pick sample-0 (you said bs=1 recommended)
            q_img_paths  = q_img_paths[0]
            q_mask_paths = q_mask_paths[0]

            T_paths = len(q_img_paths)
            T_roll  = int(q_roll.shape[-1])
            assert T_roll == T_paths, f"q_roll T={T_roll} != paths T={T_paths}"

            # membership mask: supervise ONLY when streamed abs idx ∈ query_indices
            # (B,T_rollout) bool
            sup_mask = (q_roll[:, None, :] == q_sup[:, :, None]).any(dim=1)

            if self.cfg.train.debug[0]:
                hit = torch.where(sup_mask[0])[0]
                # compare hit positions -> absolute ids -> should equal q_sup
                hit_abs = q_roll[0, hit].detach().cpu().tolist()
                print("q_sup abs   =", q_sup[0].detach().cpu().tolist())
                print("hit_abs     =", hit_abs)

            ds = getattr(train_loader, "dataset", None)
            base = getattr(ds, "dataset", ds)

            if not (hasattr(model, "init_state") and hasattr(model, "step")):
                raise RuntimeError("Streaming batch detected but model lacks init_state()/step().")

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                state = model.init_state(
                    support_img=support_img,
                    support_mask=support_mask,
                    support_indices=support_indices,
                )

            losses = []
            miou_accum = []

            for ti, ip in enumerate(q_img_paths):

                # ---- load image
                img_np = base._read_rgb(ip)
                img_t = torch.from_numpy(img_np).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(
                    self.device, non_blocking=True
                )  # (1,3,H,W)

                qi_t = q_roll[0, ti:ti+1]  # (1,) absolute index
                is_sup = bool(sup_mask[0, ti].item())


                if not is_sup:
                    # ---- rollout-only frame: NO GRAD
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        logits_t, logits_raw_t, state = model.step(query_img=img_t, state=state, query_index=qi_t)

                    state = self.detach_state(state)   # truncate + keep memory as cache only
                    del img_t, logits_t
                    continue

                # ---- supervised frame: allow GRAD
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits_t, logits_raw_t, state = model.step(query_img=img_t, state=state, query_index=qi_t)



                state = self.detach_state(state)       # truncate BPTT each step

                # ---- load GT mask only for supervised frames
                mp = q_mask_paths[ti]
                msk_np = base._read_and_map_mask(mp)
                msk_t = torch.from_numpy(msk_np.astype(np.int64)).unsqueeze(0).to(self.device, non_blocking=True)  # (1,H,W)


                # print(f'query_roll: {q_roll.shape}, q_img_paths: {len(q_img_paths)}, train: img: {img_t.shape}, msk: {msk_t.shape}')
                # ---- loss (same as before)
                aux_w = float(getattr(self.cfg.train, "aux_raw_w", 0.2))
                loss_fused = self.criterion(logits_t, msk_t)

                if self.cfg.train.use_raw_logits:
                    loss_raw = self.criterion(logits_raw_t, msk_t)
                    losses.append(loss_fused + aux_w * loss_raw)
                else:
                    losses.append(loss_fused)

                # ---- miou (optional; and do NOT append train_mious inside loop)
                with torch.no_grad():
                    pred = logits_t.detach().cpu().argmax(dim=1)
                    tgt  = msk_t.detach().cpu()
                    miou_accum.append(self._miou(pred=pred, target=tgt,
                                                num_classes=self.num_classes,
                                                ignore_index=self.ignore_index))

                del img_t, logits_t, msk_t

            if len(losses) == 0:
                loss = torch.zeros([], device=self.device, dtype=torch.float32)
            else:
                loss = torch.stack(losses).mean()

            loss_item = float(loss.item())

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            train_losses.append(loss_item)
            train_mious.append(float(np.nanmean(miou_accum)) if len(miou_accum) else float("nan"))

            del loss, losses, miou_accum, state

            if self.cfg.train.debug[0] and idx == 0:
                print("len(q_img_paths) =", len(q_img_paths))
                print("sup_mask sum     =", int(sup_mask[0].sum().item()))
                if q_roll is not None:
                    print("q_roll[:10]      =", q_roll[0, :10].detach().cpu().tolist())
                print("sup_mask[0,:20]  =", sup_mask[0, :20].detach().cpu().int().tolist())

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

        model = self.model
        model.eval()

        can_use_stream = hasattr(model, "init_state") and hasattr(model, "step")
        use_stream = bool(getattr(self.cfg.val, "use_stream", False)) and can_use_stream

        val_losses = []
        miou_list = []
        miou_p1_list, miou_p2_list, miou_p3_list = [], [], []

        num_classes = self.num_classes
        ignore_index = self.ignore_index

        for bi, batch_data in enumerate(val_loader):

            batch_is_stream = ("query_img_paths" in batch_data) and ("query_roll_indices" in batch_data)
            can_use_stream = hasattr(model, "init_state") and hasattr(model, "step")
            use_stream = batch_is_stream and can_use_stream


            # ============================================================
            # (A) STREAM VAL: uses rollout indices + reads frames by path
            # ============================================================
            if use_stream:
                # must come from svss_collate_stream
                need = ["support_img", "support_mask", "query_img_paths", "query_mask_paths", "query_roll_indices"]
                miss = [k for k in need if k not in batch_data]
                if miss:
                    raise KeyError(f"use_stream=True but batch missing {miss}. Use svss_collate_stream for val_loader.")

                support_img  = batch_data["support_img"].to(self.device, dtype=torch.float, non_blocking=True)  # (B,S,3,H,W)
                support_mask = batch_data["support_mask"].to(self.device, non_blocking=True)                   # (B,S,H,W)
                q_roll       = batch_data["query_roll_indices"].to(self.device, non_blocking=True)            # (B,Troll)
                q_idx        = batch_data.get("query_indices", None)
                if q_idx is not None and torch.is_tensor(q_idx):
                    q_idx = q_idx.to(self.device, non_blocking=True)

                # print(f'support_img: {support_img.shape}, support_mask: {support_mask.shape}, q_roll: {q_roll.shape}, q_idx: {q_idx.shape}')
                sup_idx = batch_data.get("support_indices", None)
                if sup_idx is not None and torch.is_tensor(sup_idx):
                    sup_idx = sup_idx.to(self.device, non_blocking=True)
                    if sup_idx.ndim == 2:   # (B,S)
                        sup_idx = sup_idx[:, 0]
                else:
                    sup_idx = None

                # bs=1 safest (your pipeline assumes this)
                B = support_img.shape[0]
                if B != 1:
                    raise RuntimeError("streamed validation currently expects batch_size=1")

                # # force single support
                # if support_img.ndim == 5:
                #     support_img = support_img[:, 0]  # (B,3,H,W)
                # if support_mask.ndim == 4:
                #     support_mask = support_mask[:, 0]  # (B,H,W)
                # support_mask = support_mask.long()

                # paths (collate_stream returns list-of-lists)
                q_img_paths  = batch_data["query_img_paths"][0]
                q_mask_paths = batch_data["query_mask_paths"][0]

                T = int(q_roll.shape[1])
                assert len(q_img_paths) == T and len(q_mask_paths) == T, "paths length must match query_roll_indices"

                # init state (unique id avoids mixing)
                clip = batch_data.get("video_clip", [f"bi{bi}"])
                clip_b = clip[0] if isinstance(clip, (list, tuple)) else str(clip)
                video_id = f"val_{clip_b}_ep{epoch}"

                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    state = model.init_state(
                        support_img=support_img,
                        support_mask=support_mask,
                        support_indices=sup_idx,
                        video_id=video_id,
                    )

                    logits_seq = []
                    tgt_seq = []

                    # use dataset readers (consistent mapping)
                    ds = getattr(val_loader, "dataset", None)
                    base = getattr(ds, "dataset", ds)

                    for t in range(T):
                        ip = q_img_paths[t]
                        mp = q_mask_paths[t]

                        img_np = base._read_rgb(ip)
                        msk_np = base._read_and_map_mask(mp)

                        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(
                            self.device, non_blocking=True
                        )
                        msk_t = torch.from_numpy(msk_np.astype(np.int64)).unsqueeze(0).to(
                            self.device, non_blocking=True
                        )  # (1,H,W)

                        qi_t = q_roll[0, t:t+1]  # absolute index (1,)
                        # logits_t, state = model.step(query_img=img_t, state=state, query_index=qi_t)
                        logits_t, logits_raw_t, state = model.step(query_img=img_t, state=state, query_index=qi_t)


                        logits_seq.append(logits_t.squeeze(0))  # (C,H,W)
                        tgt_seq.append(msk_t.squeeze(0))        # (H,W)

                    out = torch.stack(logits_seq, dim=0).unsqueeze(0)     # (1,T,C,H,W)
                    query_masks = torch.stack(tgt_seq, dim=0).unsqueeze(0)  # (1,T,H,W)

                    if hasattr(model, "clear_video"):
                        model.clear_video(video_id)
                    elif hasattr(model, "_streams"):
                        model._streams.pop(str(video_id), None)

                B, T, C, H, W = out.shape
                out_bt = out.reshape(B * T, C, H, W)
                tgt_bt = query_masks.reshape(B * T, H, W)

            # ============================================================
            # (B) TENSOR VAL: your original behavior
            # ============================================================
            else:
                support_img  = batch_data["support_img"].to(self.device, dtype=torch.float, non_blocking=True)
                support_mask = batch_data["support_mask"].to(self.device, non_blocking=True)
                query_imgs   = batch_data["query_imgs"].to(self.device, dtype=torch.float, non_blocking=True)
                query_masks  = batch_data["query_masks"].to(self.device, non_blocking=True)

                if support_mask.ndim == 4 and support_mask.shape[1] == 1:
                    support_mask = support_mask[:, 0]
                elif support_mask.ndim == 5 and support_mask.shape[2] == 1:
                    support_mask = support_mask[:, :, 0]
                support_mask = support_mask.long()

                if query_masks.ndim == 5 and query_masks.shape[2] == 1:
                    query_masks = query_masks[:, :, 0]
                query_masks = query_masks.long()

                B, T, _, H, W = query_imgs.shape

                support_indices = batch_data.get("support_indices", None)
                query_indices   = batch_data.get("query_indices", None)
                if support_indices is not None:
                    support_indices = support_indices.to(self.device, non_blocking=True)
                if query_indices is not None:
                    query_indices = query_indices.to(self.device, non_blocking=True)

                # if support_img.ndim == 5:
                #     support_img = support_img[:, 0]
                # if support_mask.ndim == 4:
                #     support_mask = support_mask[:, 0]
                # if support_indices is not None and torch.is_tensor(support_indices) and support_indices.ndim == 2:
                #     support_indices = support_indices[:, 0]

                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(
                        support_img, support_mask, query_imgs,
                        support_indices=support_indices,
                        query_indices=query_indices,
                    )  # (B,T,C,H,W)

                B, T, C, H, W = out.shape
                out_bt = out.reshape(B * T, C, H, W)
                tgt_bt = query_masks.reshape(B * T, H, W)
                q_idx = query_indices

            # ----- loss
            loss = self.val_ce(out_bt.float(), tgt_bt)
            val_losses.append(float(loss.item()))

            # ----- metrics
            pred_bt = torch.argmax(out_bt, dim=1).to(torch.int32)  # (B*T,H,W)
            pred_bthw = pred_bt.view(B, T, H, W)
            tgt_bthw = tgt_bt.view(B, T, H, W)

            # ---------------------------------------------------------------

            # ------------------------------------------------------------
            # Optional: save predictions + GT masks every N epochs
            # Saves label maps as PNG (uint8/uint16 depending on num_classes)
            # ------------------------------------------------------------
            save_every = int(getattr(self.cfg.val, "save_preds_every", 0))
            do_save = (save_every > 0) and ((int(epoch) % save_every) == 0)

            if do_save:
                results_root = str(getattr(self.cfg.val, "results_root", "results"))
                out_dir = os.path.join(results_root, f"ep{int(epoch):03d}")
                pred_dir = os.path.join(out_dir, "pred")
                gt_dir   = os.path.join(out_dir, "gt")

                os.makedirs(pred_dir, exist_ok=True)
                os.makedirs(gt_dir, exist_ok=True)

                max_batches = int(getattr(self.cfg.val, "save_preds_max_batches", 2))
                max_frames  = int(getattr(self.cfg.val, "save_preds_max_frames", 50))

                if bi < max_batches:
                    # use a stable clip id if available
                    clip = batch_data.get("video_clip", [f"bi{bi}"])
                    clip_b = clip[0] if isinstance(clip, (list, tuple)) else str(clip)

                    # indices for naming (absolute if provided)
                    if use_stream:
                        q_abs = q_roll[0].detach().cpu().tolist()  # (T,)
                    else:
                        q_abs = None
                        if q_idx is not None:
                            q_abs = q_idx[0].detach().cpu().tolist()

                    # save per-frame (limit IO)
                    t_lim = min(T, max_frames)
                    for t in range(t_lim):
                        pred_hw = pred_bthw[0, t].detach().cpu().to(torch.int32).numpy()
                        gt_hw   = tgt_bthw[0, t].detach().cpu().to(torch.int32).numpy()

                        # choose uint16 if needed
                        use_u16 = (self.num_classes > 255) or (pred_hw.max() > 255) or (gt_hw.max() > 255)
                        pred_img = Image.fromarray(pred_hw.astype(np.uint16 if use_u16 else np.uint8))
                        gt_img   = Image.fromarray(gt_hw.astype(np.uint16 if use_u16 else np.uint8))

                        frame_id = t
                        if q_abs is not None and t < len(q_abs):
                            frame_id = int(q_abs[t])

                        pred_path = os.path.join(pred_dir, f"{clip_b}_{frame_id:06d}.png")
                        gt_path   = os.path.join(gt_dir,   f"{clip_b}_{frame_id:06d}.png")

                        pred_img.save(pred_path)
                        gt_img.save(gt_path)

                    if self.cfg.train.debug[0]:
                        print(f"🖼️ Saved preds/GT to: {out_dir} (clip={clip_b}, frames={t_lim})")



            # ---------------------------------------------------------------



            # query indices for phase assignment (ABSOLUTE timeline)
            if use_stream:
                q_idx_use = q_roll          # (B,T) absolute frame indices
            else:
                q_idx_use = q_idx           # (B,T) absolute frame indices

            for b in range(B):
                video_len_b = None
                if "video_len" in batch_data:
                    vl = batch_data["video_len"]
                    video_len_b = int(vl[b].item()) if torch.is_tensor(vl) else int(vl[b])

                qi_b = q_idx_use[b] if q_idx_use is not None else None

                mi_all, mi_p1, mi_p2, mi_p3, _ = self.miou_per_frame_fast(
                    pred_bthw[b], tgt_bthw[b],
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    include_bg=True,
                    query_indices=qi_b,
                    video_len=video_len_b,
                )

                if not np.isnan(mi_all): miou_list.append(mi_all)
                if not np.isnan(mi_p1):  miou_p1_list.append(mi_p1)
                if not np.isnan(mi_p2):  miou_p2_list.append(mi_p2)
                if not np.isnan(mi_p3):  miou_p3_list.append(mi_p3)

            del out_bt, tgt_bt, pred_bt, pred_bthw, tgt_bthw

        ep_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        ep_val_miou = float(np.mean(miou_list)) if miou_list else float("nan")
        ep_val_miou_p1 = float(np.mean(miou_p1_list)) if miou_p1_list else float("nan")
        ep_val_miou_p2 = float(np.mean(miou_p2_list)) if miou_p2_list else float("nan")
        ep_val_miou_p3 = float(np.mean(miou_p3_list)) if miou_p3_list else float("nan")

        if log_run:
            print(
                f"🔍 Val "
                f"[{(time.time()-val_start):.1f}s] | use_stream={use_stream} | "
                f"loss: {ep_val_loss:.4f} | Avg mIoU: {ep_val_miou*100:.2f} | "
                f"Phases mIoU: {ep_val_miou_p1*100:.2f}/{ep_val_miou_p2*100:.2f}/{ep_val_miou_p3*100:.2f} | "
                f'usage: {self.mem(self.device_idx):.1f} GB'
            )

        return (
            ep_val_loss,
            ep_val_miou * 100.0,
            ep_val_miou_p1 * 100.0,
            ep_val_miou_p2 * 100.0,
            ep_val_miou_p3 * 100.0,
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