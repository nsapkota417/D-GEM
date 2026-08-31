import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class _EncoderProxy(nn.Module):
    """Trainer compat: model.frame_model.encoder.parameters()"""
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder


def _prob_to_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p)

class DotDict(dict):
    """dict with attribute access + normal dict membership."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _one_hot_semantic(mask_hw: torch.Tensor, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    """
    mask_hw: (H,W) long
    returns: (C,H,W) float in {0,1}
    ignore_index treated as background (0)
    """
    if ignore_index is not None:
        mask_hw = mask_hw.clone()
        mask_hw[mask_hw == ignore_index] = 0
    mask_hw = torch.clamp(mask_hw, 0, num_classes - 1)
    return F.one_hot(mask_hw, num_classes=num_classes).permute(2, 0, 1).float()  # (C,H,W)


def _warp_with_flow(x_bchw: torch.Tensor, flow_b2hw: torch.Tensor) -> torch.Tensor:
    """
    Warp tensor x using forward flow (dx,dy) in pixels.
    x:    (B,C,H,W)
    flow: (B,2,H,W) where flow[:,0]=dx, flow[:,1]=dy (pixel units)
    returns warped x: (B,C,H,W)
    """
    B, C, H, W = x_bchw.shape
    device = x_bchw.device
    dtype = x_bchw.dtype

    # base grid in pixel coords
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # (B,2,H,W)

    coords = base - flow_b2hw

    # normalize to [-1,1]
    x_norm = 2.0 * (coords[:, 0] / max(W - 1, 1)) - 1.0
    y_norm = 2.0 * (coords[:, 1] / max(H - 1, 1)) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)  # (B,H,W,2)

    return F.grid_sample(
        x_bchw,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


class RAFTFlowPropWrapper(nn.Module):
    """
    RAFT-flow propagation baseline for semantic video segmentation.

    Protocol:
      - initialize prob from support_mask (one-hot)
      - for each query frame:
          flow(prev_img -> cur_img) via RAFT
          prob = warp(prob, flow)
          renormalize
      - output logits = log(prob)

    Inputs:
      support_img  : (B,3,H,W) or (B,S,3,H,W)
      support_mask : (B,H,W)   or (B,S,H,W)  (semantic labels)
      query_imgs   : (B,T,3,H,W)

    Output:
      logits       : (B,T,C,H,W)
    """

    def __init__(
        self,
        raft_ckpt: str,
        num_classes: int,
        device: str = "cuda",
        ignore_index: int = 255,

        # where RAFT repo lives; default assumes:
        #   VOS/src/baselines/RAFT
        raft_root: str | None = None,

        # RAFT options (match demo defaults)
        raft_small: bool = False,
        mixed_precision: bool = False,
        alternate_corr: bool = False,

        # input handling
        imagenet_norm: bool = False,  # if your pipeline applies ImageNet norm, set True to undo
        expects_01: bool = True,       # if True: assumes images in [0,1]; will convert to 0..255 for RAFT
        flow_iters: int = 20,          # RAFT refinement iterations (demo uses 20)
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.device = torch.device(device)

        self.imagenet_norm = bool(imagenet_norm)
        self.expects_01 = bool(expects_01)
        self.flow_iters = int(flow_iters)

        # ----- RAFT import (robust for original RAFT absolute imports inside core/)
        from pathlib import Path
        import sys

        if raft_root is None:
            raft_root = Path(__file__).resolve().parents[1] / "baselines" / "RAFT"
        else:
            raft_root = Path(raft_root)

        raft_root = raft_root.resolve()
        core_dir = raft_root / "core"
        if not (core_dir / "raft.py").is_file():
            raise FileNotFoundError(f"Expected RAFT core at: {core_dir} (missing raft.py)")

        # IMPORTANT: insert core_dir FIRST so "from update import ..." works
        for p in (str(core_dir), str(raft_root)):
            if p in sys.path:
                sys.path.remove(p)
        for p in (str(core_dir), str(raft_root)):
            sys.path.insert(0, p)

        # Now import using core_dir directly (NOT core.raft)
        from raft import RAFT
        from core.utils.utils import InputPadder
        self._InputPadder = InputPadder
        args = DotDict(
            small=bool(raft_small),
            mixed_precision=bool(mixed_precision),
            alternate_corr=bool(alternate_corr),
            dropout=0.0,
        )
        self.raft = RAFT(args).to(self.device).eval()

        ckpt = torch.load(raft_ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if isinstance(ckpt, dict):
            ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        self.raft.load_state_dict(ckpt, strict=True)

        # trainer compat
        self.frame_model = _EncoderProxy(self.raft)

    def _undo_imagenet_norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: (3,H,W) or (B,3,H,W)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(3, 1, 1)
        if x.dim() == 4:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return x * std + mean

    def _prep_img_3hw(self, img_3hw: torch.Tensor) -> torch.Tensor:
        """
        Convert your pipeline tensor -> RAFT expected scale.
        We’ll pass images in 0..255 float like RAFT demo.
        """
        if img_3hw.dim() != 3:
            raise ValueError(f"_prep_img_3hw expects (3,H,W), got {tuple(img_3hw.shape)}")

        x = img_3hw.float()

        if self.imagenet_norm:
            x = self._undo_imagenet_norm(x)

        # auto-handle [0,1] vs [0,255]
        mx = float(x.max().item())
        if self.expects_01 or mx <= 1.5:
            x = x * 255.0
        else:
            # assume already 0..255-ish
            pass

        return x

    @torch.no_grad()
    def _flow_12(self, img1_3hw: torch.Tensor, img2_3hw: torch.Tensor) -> torch.Tensor:
        """
        Compute flow img1 -> img2 using RAFT.
        Inputs must be (3,H,W) in 0..255 float.
        Returns flow: (1,2,H,W) in pixels.
        """
        img1 = img1_3hw.unsqueeze(0).to(self.device)
        img2 = img2_3hw.unsqueeze(0).to(self.device)

        padder = self._InputPadder(img1.shape)
        img1p, img2p = padder.pad(img1, img2)

        flow_low, flow_up = self.raft(img1p, img2p, iters=self.flow_iters, test_mode=True)
        flow_up = padder.unpad(flow_up)
        return flow_up  # (1,2,H,W)

    @torch.no_grad()
    def forward(self, support_img, support_mask, query_imgs, **kwargs):
        # shapes
        if support_img.ndim == 5:
            # (B,S,3,H,W) -> use first support frame for baseline
            img0 = support_img[:, 0]
            m0 = support_mask[:, 0]
        else:
            img0 = support_img
            m0 = support_mask

        B, T, _, H, W = query_imgs.shape
        C = self.num_classes

        # move to device
        img0 = img0.to(self.device, non_blocking=True)
        m0 = m0.to(self.device, non_blocking=True).long()
        query_imgs = query_imgs.to(self.device, non_blocking=True)

        # init prob from support mask (one-hot + tiny smoothing)
        prob = torch.zeros((B, C, H, W), device=self.device, dtype=torch.float32)
        for b in range(B):
            oh = _one_hot_semantic(m0[b], C, ignore_index=self.ignore_index)  # (C,H,W)
            # smoothing avoids log(0)
            prob[b] = 0.98 * oh + 0.02 / C

        out_logits = torch.zeros((B, T, C, H, W), device=self.device, dtype=torch.float32)

        # prev image for flow
        prev_img = img0  # (B,3,H,W)

        for t in range(T):
            cur_img = query_imgs[:, t]  # (B,3,H,W)

            # RAFT easiest baseline: do per-item (B may be >1). This is baseline code; fine.
            prob_new = torch.zeros_like(prob)
            for b in range(B):
                im1 = self._prep_img_3hw(prev_img[b])
                im2 = self._prep_img_3hw(cur_img[b])
                flow = self._flow_12(im1, im2)  # (1,2,H,W)

                warped = _warp_with_flow(prob[b:b+1], flow)  # (1,C,H,W)
                warped = torch.clamp(warped, 1e-6, 1.0)

                # renormalize to sum=1
                warped = warped / (warped.sum(dim=1, keepdim=True) + 1e-6)
                prob_new[b:b+1] = warped

            prob = prob_new
            out_logits[:, t] = _prob_to_logit(prob)

            prev_img = cur_img

        return out_logits


# Optional: make it trainable like your XMemCalibWrapper
class RAFTFlowCalibWrapper(nn.Module):
    """
    Trainable baseline:
      - frozen RAFTFlowPropWrapper produces logits
      - small head refines logits (trainable)
    """
    def __init__(
        self,
        raft_prop: RAFTFlowPropWrapper,
        num_classes: int,
        head: str = "1x1",
        hidden: int = 64,
        dropout: float = 0.0,
        train_raft: bool = False,
    ):
        super().__init__()
        self.raft_prop = raft_prop
        self.num_classes = int(num_classes)

        # trainer compat
        self.frame_model = _EncoderProxy(self.raft_prop.raft)

        # freeze RAFT baseline (default)
        if not train_raft:
            for p in self.raft_prop.parameters():
                p.requires_grad = False
            self.raft_prop.eval()

        if head == "1x1":
            self.head = nn.Conv2d(num_classes, num_classes, 1, bias=True)
        elif head == "2layer":
            self.head = nn.Sequential(
                nn.Conv2d(num_classes, hidden, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=float(dropout)) if dropout > 0 else nn.Identity(),
                nn.Conv2d(hidden, num_classes, 1, bias=True),
            )
        else:
            raise ValueError(f"Unknown head='{head}'")

    def forward(self, support_img, support_mask, query_imgs, **kwargs):
        if any(p.requires_grad for p in self.raft_prop.parameters()):
            logits = self.raft_prop(support_img, support_mask, query_imgs, **kwargs)
        else:
            with torch.no_grad():
                logits = self.raft_prop(support_img, support_mask, query_imgs, **kwargs)

        B, T, C, H, W = logits.shape
        x = logits.reshape(B * T, C, H, W)
        x = self.head(x)
        x = x.reshape(B, T, C, H, W)
        return x


# --- replace RAFT with "support-copy" sanity baseline
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class _EncoderProxy(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder


class SupportCopyCalibWrapper(nn.Module):
    """
    Trainable sanity baseline:
      - builds logits from support mask for each query frame
      - passes through a tiny trainable head so loss.backward() works
    Output: (B,T,C,H,W)
    """
    def __init__(self, num_classes: int, ignore_index: int = 255, bg_index: int = 0, device="cuda"):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.bg_index = int(bg_index)
        self.device = torch.device(device)

        # trainer expects model.frame_model.encoder.parameters()
        self.frame_model = _EncoderProxy(nn.Identity())

        # tiny trainable head so gradients exist
        self.head = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=1, bias=True)

    def forward(self, support_img, support_mask, query_imgs, **kwargs):
        if support_mask.ndim == 4:   # (B,S,H,W)
            m0 = support_mask[:, 0]
        else:
            m0 = support_mask

        B, T, _, H, W = query_imgs.shape
        C = self.num_classes

        m0 = m0.to(self.device, non_blocking=True).long()
        if self.ignore_index is not None:
            m0 = m0.clone()
            m0[m0 == self.ignore_index] = self.bg_index
        m0 = torch.clamp(m0, 0, C - 1)

        oh = F.one_hot(m0, num_classes=C).permute(0, 3, 1, 2).float()  # (B,C,H,W)
        logits0 = oh * 20.0 - 10.0                                     # (B,C,H,W)

        # repeat for query frames
        x = logits0.unsqueeze(1).repeat(1, T, 1, 1, 1)                 # (B,T,C,H,W)

        # apply trainable head per-frame
        x = x.reshape(B * T, C, H, W)
        x = self.head(x)
        x = x.reshape(B, T, C, H, W)
        return x
