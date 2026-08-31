# =========================
# COPY-PASTE PATCH (STM wrapper)
# - Fix class union across multi-support
# - Make memory append/cap robust (concat+cap along LAST dim)
# - Add return_raw support in STMCalibWrapper (trainer compatibility)
# =========================

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.STM.model import STM


class _EncoderProxy(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder


class STMBaselineWrapper(nn.Module):
    """
    STM baseline wrapper for semantic segmentation videos (INFERENCE-ONLY core).
    Produces logits (B,T,C,H,W). Safe to use under torch.no_grad().

    NOTE: This class intentionally keeps the STM core frozen.
          If you want training, use STMCalibWrapper below.
    """
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        bg_index: int = 0,
        device: str = "cuda",
        memorize_all_support: bool = False,
        online_update: bool = True,
        write_stride: int = 2,
        keep_last: int = 20,
        conf_thr: float = 0.55,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.bg_index = int(bg_index)
        self.device = torch.device(device)

        self.memorize_all_support = bool(memorize_all_support)
        self.online_update = bool(online_update)
        self.write_stride = int(write_stride)
        self.keep_last = int(keep_last)
        self.conf_thr = float(conf_thr)

        self.stm = STM().to(self.device).eval()
        self.frame_model = _EncoderProxy(self.stm)

    def load_pretrained(self, ckpt_path: str, strict: bool = True):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if isinstance(ckpt, dict):
            ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        self.stm.load_state_dict(ckpt, strict=strict)
        self.stm.eval()
        return self

    @staticmethod
    def _cap_lastdim(x: torch.Tensor, keep_last: int) -> torch.Tensor:
        if keep_last and keep_last > 0 and x.size(-1) > keep_last:
            x = x[..., -keep_last:]
        return x

    @staticmethod
    def _cap_time(keys: torch.Tensor, values: torch.Tensor, keep_last: int):
        """
        STM memory tensors:
          keys/values are (1, K, C, T, H, W)
        We must cap along time dimension T = dim=3.
        """
        if keep_last and keep_last > 0 and keys.size(3) > keep_last:
            keys = keys[:, :, :, -keep_last:]     # cap T
            values = values[:, :, :, -keep_last:]
        return keys, values


    @torch.no_grad()
    def forward(self, support_img, support_mask, query_imgs, **kwargs):
        if support_img.ndim == 4:
            support_img = support_img.unsqueeze(1)   # (B,1,3,H,W)
            support_mask = support_mask.unsqueeze(1) # (B,1,H,W)

        B, S, _, H, W = support_img.shape
        _, T, _, _, _ = query_imgs.shape

        if B != 1:
            raise ValueError("STMBaselineWrapper expects batch_size=1 (STM baseline).")

        support_img = support_img.to(self.device, dtype=torch.float32)
        support_mask = support_mask.to(self.device, dtype=torch.long)
        query_imgs = query_imgs.to(self.device, dtype=torch.float32)


        # pick a valid support index (mask not all ignore)
        valid_s = None
        for s in range(S):
            m = support_mask[0, s]
            # valid if any pixel is not ignore
            if (m != self.ignore_index).any():
                valid_s = s
                break

        if valid_s is None:
            # no labeled support => predict bg for all queries
            out = torch.zeros((1, T, self.num_classes, H, W), device=self.device)
            out[:, :, self.bg_index] = 1.0
            return out

        # rotate supports so valid_s becomes first
        if valid_s != 0:
            support_img = torch.cat([support_img[:, valid_s:valid_s+1], support_img[:, :valid_s], support_img[:, valid_s+1:]], dim=1)
            support_mask = torch.cat([support_mask[:, valid_s:valid_s+1], support_mask[:, :valid_s], support_mask[:, valid_s+1:]], dim=1)


        # Decide which supports we will actually memorize
        s_range = range(S) if self.memorize_all_support else range(1)

        # Build obj_classes ONLY from memorized supports
        ys = support_mask[0, list(s_range)]  # (S_mem,H,W)
        present = torch.unique(ys)

        present = present[present != self.ignore_index]
        present = present[(present != self.bg_index) & (present >= 0) & (present < self.num_classes)]
        obj_classes = sorted([int(x) for x in present.detach().cpu().tolist()])

        # print("S=", S, "S_mem=", len(list(s_range)), "K=", 1+len(obj_classes), "obj_classes=", obj_classes[:10])
        # print("support0 uniq:", torch.unique(support_mask[0,0]).detach().cpu().tolist()[:20])


        if len(obj_classes) == 0:
            out = torch.zeros((1, T, self.num_classes, H, W), device=self.device)
            out[:, :, self.bg_index] = 1.0
            return out

        K = 1 + len(obj_classes)  # bg + objects
        num_objects = torch.tensor([len(obj_classes)], device=self.device, dtype=torch.long)

        def semantic_to_stm_masks(y_sem_hw: torch.Tensor):
            masks = torch.zeros((1, K, H, W), device=self.device, dtype=torch.float32)
            bg = torch.ones((H, W), device=self.device, dtype=torch.bool)

            # ignore -> bg
            if self.ignore_index is not None:
                y_sem_hw = y_sem_hw.clone()
                y_sem_hw[y_sem_hw == self.ignore_index] = self.bg_index

            for oi, cls_id in enumerate(obj_classes, start=1):
                m = (y_sem_hw == cls_id)
                masks[0, oi] = m.float()
                bg &= ~m

            masks[0, 0] = bg.float()
            return masks

        keys, values = None, None

        def mem_append(k_new, v_new):
            nonlocal keys, values
            # k_new/v_new: (1, K, C, T, H, W)
            if keys is None:
                keys, values = k_new, v_new
            else:
                # ✅ append along time dimension (dim=3), NOT last dim
                keys = torch.cat([keys, k_new], dim=3)
                values = torch.cat([values, v_new], dim=3)

            keys, values = self._cap_time(keys, values, self.keep_last)
            keys = keys.contiguous()
            values = values.contiguous()

        s_range = range(S) if self.memorize_all_support else range(1)
        for s in s_range:
            masks_s = semantic_to_stm_masks(support_mask[0, s])
            k_s, v_s = self.stm(support_img[0, s].unsqueeze(0), masks_s, num_objects)  # memorize
            mem_append(k_s, v_s)

        logits_sem = []

        for t in range(T):
            logit_k = self.stm(query_imgs[0, t].unsqueeze(0), keys, values, num_objects)  # segment (1,K,H,W)

            logit_c = torch.full((1, self.num_classes, H, W), -50.0, device=self.device)
            logit_c[:, self.bg_index] = logit_k[:, 0]
            for oi, cls_id in enumerate(obj_classes, start=1):
                logit_c[:, cls_id] = logit_k[:, oi]
            logits_sem.append(logit_c)

            if self.online_update and (t % self.write_stride == 0):
                prob_k = F.softmax(logit_k, dim=1)            # (1,K,H,W)
                pred_obj = torch.argmax(prob_k, dim=1)        # (1,H,W)
                maxp = torch.max(prob_k, dim=1).values[0]     # (H,W)

                pred_sem = torch.full((H, W), self.bg_index, device=self.device, dtype=torch.long)
                for oi, cls_id in enumerate(obj_classes, start=1):
                    pred_sem[pred_obj[0] == oi] = cls_id

                fg = (pred_obj[0] != 0)
                conf = float(maxp[fg].mean().item()) if fg.any() else 0.0

                if conf >= self.conf_thr:
                    masks_pred = semantic_to_stm_masks(pred_sem)
                    k_new, v_new = self.stm(query_imgs[0, t].unsqueeze(0), masks_pred, num_objects)  # memorize
                    mem_append(k_new, v_new)

        out = torch.stack(logits_sem, dim=1)  # (1,T,C,H,W)
        return out


class STMCalibWrapper(nn.Module):
    """
    TRAINABLE wrapper:
      - STM baseline (frozen) produces logits (B,T,C,H,W)
      - A small head refines logits (trainable) so loss.backward() works
    """
    def __init__(
        self,
        stm: STMBaselineWrapper,
        num_classes: int,
        head: str = "1x1",
        hidden: int = 64,
        dropout: float = 0.0,
        train_stm: bool = False,
    ):
        super().__init__()
        self.stm = stm
        self.num_classes = int(num_classes)

        self.frame_model = _EncoderProxy(self.stm.stm)
        self.tau = 5.0
        if not train_stm:
            for p in self.stm.parameters():
                p.requires_grad = False
            self.stm.eval()

        if head == "1x1":
            self.head = nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=True)
        elif head == "2layer":
            self.head = nn.Sequential(
                nn.Conv2d(num_classes, hidden, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=float(dropout)) if dropout > 0 else nn.Identity(),
                nn.Conv2d(hidden, num_classes, kernel_size=1, bias=True),
            )
        else:
            raise ValueError(f"Unknown head='{head}'")

    def forward(self, support_img, support_mask, query_imgs, return_raw: bool = False, **kwargs):
        # run STM frozen unless train_stm=True
        if any(p.requires_grad for p in self.stm.parameters()):
            stm_logits = self.stm(support_img, support_mask, query_imgs, **kwargs)  # (B,T,C,H,W)
        else:
            with torch.no_grad():
                stm_logits = self.stm(support_img, support_mask, query_imgs, **kwargs)

        stm_logits = stm_logits / 5.0
        B, T, C, H, W = stm_logits.shape
        if C != self.num_classes:
            raise ValueError(f"Expected C={self.num_classes}, got {C}")

        # --- temperature to tame STM logit scale
        tau = float(getattr(self, "tau", 5.0))  # default 5.0
        stm_logits = stm_logits / tau

        x = stm_logits.reshape(B * T, C, H, W)
        delta = self.head(x)

        # --- residual: refine, not replace
        out = (x + delta).reshape(B, T, C, H, W)

        if return_raw:
            return out, stm_logits
        return out