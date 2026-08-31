# VOS/src/networks/xmem_wrapper.py

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# XMem repo imports
# -------------------------
XMEM_ROOT = Path(__file__).resolve().parents[1] / "baselines" / "XMem"
sys.path.insert(0, str(XMEM_ROOT))

from model.network import XMem
from inference.inference_core import InferenceCore


# -------------------------
# Trainer-compat proxies
# -------------------------
class _EncoderProxy(nn.Module):
    """
    Make any baseline look like your SAM2-style model:
      model.frame_model.encoder.parameters()
    """
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder


# -------------------------
# Utils
# -------------------------
def _prob_to_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert probability to logits safely: log(p / (1 - p)).
    """
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)  # log(p) - log(1-p)


def _unique_present_classes(mask_hw: torch.Tensor, num_classes: int, bg_index: int = 0, ignore_index: int = 255) -> torch.Tensor:
    """
    mask_hw: (H,W) int labels
    returns: unique semantic ids excluding background and ignore_index and out-of-range
    """
    u = torch.unique(mask_hw)
    u = u[(u != int(bg_index)) & (u != int(ignore_index)) & (u >= 0) & (u < int(num_classes))]
    return u


def _norm_support(support_img, support_mask):
    """
    Normalize:
      support_img  -> (B,S,3,H,W)
      support_mask -> (B,S,H,W)
    Accepts variants:
      support_img:  (B,3,H,W) or (B,S,3,H,W)
      support_mask: (B,H,W) or (B,1,H,W) or (B,S,H,W) or (B,S,1,H,W)
    """
    if support_img.dim() == 4:
        support_img = support_img.unsqueeze(1)  # (B,1,3,H,W)
    elif support_img.dim() != 5:
        raise ValueError(f"support_img must be (B,3,H,W) or (B,S,3,H,W), got {tuple(support_img.shape)}")

    if support_mask.dim() == 3:
        support_mask = support_mask.unsqueeze(1)  # (B,1,H,W)
    elif support_mask.dim() == 4:
        # could be (B,1,H,W) or (B,S,H,W)
        if support_mask.shape[1] == 1 and support_img.shape[1] > 1:
            support_mask = support_mask.expand(-1, support_img.shape[1], -1, -1)
    elif support_mask.dim() == 5 and support_mask.shape[2] == 1:
        support_mask = support_mask[:, :, 0]  # (B,S,H,W)
    else:
        raise ValueError(f"support_mask has unsupported shape {tuple(support_mask.shape)}")

    return support_img, support_mask


# ============================================================================
# 1) Inference-only: XMem as SVSS (semantic classes treated as objects)
# ============================================================================
# ============================================================================
# 1) XMem: SVSS wrapper with streaming init_state/step (dense rollout compatible)
# ============================================================================
class XMemSVSSWrapper(nn.Module):
    """
    Treat XMem as an SVSS model (semantic classes treated as objects).

    Supports BOTH:
      (A) Tensor forward:
          forward(support_img, support_mask, query_imgs) -> logits (B,T,C,H,W)

      (B) Streaming rollout (for dense rollout / your trainer val stream):
          state = init_state(support_img, support_mask, support_indices, video_id)
          logits_t, logits_raw_t, state = step(query_img, state, query_index)

    Notes:
      - XMem is VOS. We adapt by treating each semantic class as an "object".
      - We remap present semantic IDs -> contiguous object IDs (1..K).
      - Multi-support: we inject all provided GT supports into XMem memory.
      - For compatibility with your raw-logits loss path:
            logits_raw_t == logits_t
    """

    def __init__(
        self,
        ckpt_path: str,
        num_classes: int = 10,
        device: str = "cuda",

        # XMem inference knobs
        mem_every: int = 5,
        enable_long_term: bool = True,
        max_mid_term_frames: int = 10,
        min_mid_term_frames: int = 5,
        max_long_term_elements: int = 10000,
        num_prototypes: int = 128,
        top_k: int = 30,
        deep_update_every: int = -1,

        # preprocessing
        expects_01: bool = True,
        imagenet_norm: bool = False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.device = torch.device(device)

        self.expects_01 = bool(expects_01)
        self.imagenet_norm = bool(imagenet_norm)

        # Config keys used by InferenceCore
        self.config = dict(
            mem_every=int(mem_every),
            deep_update_every=int(deep_update_every),

            enable_long_term=bool(enable_long_term),
            disable_long_term=not bool(enable_long_term),

            enable_long_term_count_usage=False,
            enable_long_term_usage=bool(enable_long_term),

            max_mid_term_frames=int(max_mid_term_frames),
            min_mid_term_frames=int(min_mid_term_frames),
            max_long_term_elements=int(max_long_term_elements),
            num_prototypes=int(num_prototypes),
            top_k=int(top_k),
        )

        self.net = XMem(self.config).to(self.device).eval()

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model", None) or ckpt.get("net", None) or ckpt.get("state_dict", None) or ckpt
        if isinstance(state, dict):
            new_state = {}
            for k, v in state.items():
                nk = k[len("module."):] if k.startswith("module.") else k
                new_state[nk] = v
            state = new_state

        missing, unexpected = self.net.load_state_dict(state, strict=False)
        if missing:
            print(f"[XMemSVSSWrapper] missing keys: {len(missing)} (ok if ckpt format differs)")
        if unexpected:
            print(f"[XMemSVSSWrapper] unexpected keys: {len(unexpected)}")

        # Trainer-compat: expose frame_model.encoder
        self.frame_model = _EncoderProxy(self.net)

        # per-video streaming cache: video_id -> dict(proc, sem_ids, obj_ids, H,W)
        self._streams = {}

    # -------------------------
    # preprocessing
    # -------------------------
    def _undo_imagenet_norm(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(-1, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(-1, 1, 1)
        if x.dim() == 4:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return x * std + mean

    def _prep_img(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (3,H,W) OR (1,3,H,W)
        returns: (3,H,W) in [0,1] if expects_01
        """
        if img.dim() == 4:
            img = img[0]
        if img.dim() != 3:
            raise ValueError(f"_prep_img expected (3,H,W), got {tuple(img.shape)}")
        if self.imagenet_norm:
            img = self._undo_imagenet_norm(img)
        if self.expects_01:
            img = torch.clamp(img, 0.0, 1.0)
        return img

    def clear_video(self, video_id: str):
        self._streams.pop(str(video_id), None)

    @staticmethod
    def _present_union(mask_bshw: torch.Tensor, num_classes: int) -> list:
        """
        mask_bshw: (1,S,H,W) long
        returns sorted semantic ids present in ANY support, excluding bg(0)
        """
        u = torch.unique(mask_bshw)
        u = u[(u > 0) & (u < int(num_classes))]
        return sorted([int(x) for x in u.detach().cpu().tolist()])

    def _onehot_for_sem_ids(self, mask_hw: torch.Tensor, sem_ids: list, H: int, W: int) -> torch.Tensor:
        """
        mask_hw: (H,W) long
        returns: (K,H,W) float onehot aligned with sem_ids
        """
        K = len(sem_ids)
        onehot = torch.zeros((K, H, W), device=self.device, dtype=torch.float32)
        for i, sem in enumerate(sem_ids):
            onehot[i] = (mask_hw == int(sem)).float()
        return onehot

    # -------------------------
    # streaming API (dense rollout)
    # -------------------------
    @torch.no_grad()
    def init_state(self, support_img, support_mask, support_indices=None, video_id: str = "b0"):
        """
        support_img:
          (B,3,H,W) or (B,S,3,H,W)
        support_mask:
          (B,H,W) or (B,S,H,W) or (B,S,1,H,W)
        support_indices:
          (B,) or (B,S) optional (used only for ordering supports)
        """
        support_img = support_img.to(self.device, non_blocking=True)
        support_mask = support_mask.to(self.device, non_blocking=True)

        # normalize shapes to (1,S,3,H,W) and (1,S,H,W)
        if support_img.ndim == 4:
            support_img = support_img.unsqueeze(1)  # (B,1,3,H,W)
        if support_mask.ndim == 3:
            support_mask = support_mask.unsqueeze(1)  # (B,1,H,W)
        elif support_mask.ndim == 5 and support_mask.shape[2] == 1:
            support_mask = support_mask[:, :, 0]  # (B,S,H,W)

        B, S, _, H, W = support_img.shape
        assert B == 1, "XMem streaming init_state expects batch_size=1"

        # order supports by support_indices if provided
        if support_indices is not None:
            if not torch.is_tensor(support_indices):
                support_indices = torch.as_tensor(support_indices, device=self.device, dtype=torch.long)
            support_indices = support_indices.to(self.device)
            if support_indices.ndim == 1:
                # (1,) -> expand to (1,S) if needed
                if S > 1:
                    support_indices = support_indices.view(1, 1).expand(1, S)
                else:
                    support_indices = support_indices.view(1, 1)
            idx = support_indices[0].detach().cpu().tolist()
            order = sorted(list(range(S)), key=lambda i: int(idx[i]))
        else:
            order = list(range(S))

        # union of present semantic classes across supports
        sem_ids = self._present_union(support_mask, self.num_classes)

        proc = InferenceCore(self.net, config=self.config)
        proc.clear_memory()

        if len(sem_ids) == 0:
            # empty clip => always background
            self._streams[str(video_id)] = dict(
                proc=proc, sem_ids=[], obj_ids=[], H=H, W=W, t_local=0
            )
            return {"video_id": str(video_id), "t_local": 0}

        K = len(sem_ids)
        obj_ids = list(range(1, K + 1))
        proc.set_all_labels(obj_ids)

        # inject each support frame as GT mask into memory
        for si in order:
            img_s = self._prep_img(support_img[0, si])     # (3,H,W)
            m_s   = support_mask[0, si].long()             # (H,W)
            onehot = self._onehot_for_sem_ids(m_s, sem_ids, H, W)  # (K,H,W)
            _ = proc.step(img_s, onehot, valid_labels=obj_ids, end=False)

        self._streams[str(video_id)] = dict(
            proc=proc, sem_ids=sem_ids, obj_ids=obj_ids, H=H, W=W, t_local=0
        )
        return {"video_id": str(video_id), "t_local": 0}

    @torch.no_grad()
    def step(self, query_img: torch.Tensor, state: dict, query_index=None):
        """
        query_img: (1,3,H,W)
        returns: logits (1,C,H,W), logits_raw (1,C,H,W), updated state
        """
        assert query_img.ndim == 4 and query_img.shape[0] == 1, "XMem.step expects (1,3,H,W)"
        video_id = str(state.get("video_id", "b0"))
        if video_id not in self._streams:
            raise RuntimeError(f"XMem.step called before init_state for video_id='{video_id}'")

        pack = self._streams[video_id]
        proc = pack["proc"]
        sem_ids = pack["sem_ids"]
        obj_ids = pack["obj_ids"]
        H, W = int(pack["H"]), int(pack["W"])

        if len(sem_ids) == 0:
            # all background
            C = self.num_classes
            logits = torch.zeros((1, C, H, W), device=self.device, dtype=torch.float32)
            logits[:, 0] = 10.0
            state["t_local"] = int(state.get("t_local", 0)) + 1
            return logits, logits, state

        img_t = self._prep_img(query_img[0])  # (3,H,W)

        prob = proc.step(img_t, mask=None, valid_labels=None, end=False)  # (1+K,H,W) prob

        # map back into semantic C channels
        C = self.num_classes
        full_prob = torch.zeros((C, H, W), device=self.device, dtype=prob.dtype)
        full_prob[0] = prob[0]
        for i, sem in enumerate(sem_ids):
            full_prob[int(sem)] = prob[1 + i]

        logits = _prob_to_logit(full_prob).unsqueeze(0).to(torch.float32)  # (1,C,H,W)

        pack["t_local"] = int(pack.get("t_local", 0)) + 1
        state["t_local"] = int(state.get("t_local", 0)) + 1
        return logits, logits, state  # logits_raw == logits for compatibility

    # -------------------------
    # tensor forward (non-stream)
    # -------------------------
    @torch.no_grad()
    def forward(
        self,
        support_img,
        support_mask,
        query_imgs,
        support_indices=None,
        query_indices=None,
        meta=None,
        return_raw: bool = False,
        **kwargs,
    ):
        support_img = support_img.to(self.device, non_blocking=True)
        support_mask = support_mask.to(self.device, non_blocking=True)
        query_imgs = query_imgs.to(self.device, non_blocking=True)

        B, T, _, H, W = query_imgs.shape
        C = self.num_classes
        out_logits = torch.zeros((B, T, C, H, W), device=self.device, dtype=torch.float32)

        for b in range(B):
            vid = f"b{b}"

            # slice supports to (1,...) for init_state
            sup_img_b = support_img[b:b+1]
            sup_msk_b = support_mask[b:b+1]

            sup_idx_b = None
            if support_indices is not None:
                if not torch.is_tensor(support_indices):
                    support_indices_t = torch.as_tensor(support_indices, device=self.device, dtype=torch.long)
                else:
                    support_indices_t = support_indices.to(self.device)
                sup_idx_b = support_indices_t[b:b+1]

            state = self.init_state(
                support_img=sup_img_b,
                support_mask=sup_msk_b,
                support_indices=sup_idx_b,
                video_id=vid,
            )

            for t in range(T):
                logits_t, _, state = self.step(query_imgs[b:b+1, t], state, query_index=None)
                out_logits[b, t] = logits_t[0]

        if return_raw:
            return out_logits, out_logits
        return out_logits

# ============================================================================
# 2) Trainable: frozen XMem + small refinement head
# ============================================================================
class XMemCalibWrapper(nn.Module):
    """
    Trainable wrapper:
      - XMemSVSSWrapper produces per-frame logits
      - a small trainable head refines them
      - output matches trainer: (B,T,C,H,W)

    return_raw compatibility:
      - returns (logits_refined, logits_xmem) if return_raw=True
        so your trainer can apply aux loss on "raw" logits consistently.
    """

    def __init__(
        self,
        xmem: XMemSVSSWrapper,
        num_classes: int = 10,
        head: str = "1x1",          # "1x1" or "2layer"
        hidden: int = 64,           # only used for "2layer"
        dropout: float = 0.0,
        train_xmem: bool = False,   # if False: freeze XMem, train only head
    ):
        super().__init__()
        self.xmem = xmem
        self.num_classes = int(num_classes)

        # Trainer-compat: expose frame_model.encoder
        self.frame_model = _EncoderProxy(self.xmem.net)

        # Freeze XMem if requested
        if not train_xmem:
            for p in self.xmem.parameters():
                p.requires_grad = False
            self.xmem.eval()

        # Small refinement head over logits (C,H,W)
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

    def forward(
        self,
        support_img,
        support_mask,
        query_imgs,
        support_indices=None,
        query_indices=None,
        meta=None,
        return_raw: bool = False,  # ✅ trainer compatibility
        **kwargs,
    ):
        # XMem logits (frozen unless train_xmem=True)
        if any(p.requires_grad for p in self.xmem.parameters()):
            xmem_logits = self.xmem(
                support_img, support_mask, query_imgs,
                support_indices=support_indices,
                query_indices=query_indices,
                meta=meta,
                return_raw=False,
            )
        else:
            with torch.no_grad():
                xmem_logits = self.xmem(
                    support_img, support_mask, query_imgs,
                    support_indices=support_indices,
                    query_indices=query_indices,
                    meta=meta,
                    return_raw=False,
                )

        B, T, C, H, W = xmem_logits.shape
        if C != self.num_classes:
            raise ValueError(f"Expected C={self.num_classes}, got {C}")

        x = xmem_logits.reshape(B * T, C, H, W)
        x = self.head(x)
        out = x.reshape(B, T, C, H, W)

        if return_raw:
            # "raw" = pre-head xmem logits
            return out, xmem_logits
        return out