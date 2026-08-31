# memory/memory_state.py
# DMM v1 for ABD-Net:
# - Anchor Memory (append-only, pinned)
# - Transient/Trusted Memory (evictable, Top-K by recency/FIFO)
# - Temporal PE (Δt embedding)
# - MemoryState (reset/add/export)  ✅ 02/11 tasks
#
# Drop-in replacement: adds
#   - write-time TopK (add(..., topk_tokens=0))
#   - detach toggle (detach_memory=True by default)
#   - PE auto-device alignment on add()
#   - vectorized get_memory() (faster)
#   - stats includes token counts
#
# Assumptions:
# - tokens are torch.Tensor with shape [N, D] or [1, N, D].

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from anchor_memory import AnchorMemory
from transient_memory import TransientMemory

Tensor = torch.Tensor


@dataclass
class MemItem:
    tokens: Tensor      # [N, D]
    t: int              # frame index
    is_anchor: bool     # pinned if True


class TemporalPE(nn.Module):
    """
    Simple learned temporal embedding for Δt.
    Applies to token features: tokens + PE(Δt).
    """
    def __init__(self, d_model: int, max_dt: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_dt = max_dt
        self.emb = nn.Embedding(2 * max_dt + 1, d_model)  # indices in [0..2*max_dt]

    def forward(self, dt: Union[int, Tensor]) -> Tensor:
        """
        dt: int or Tensor of ints. Clamp to [-max_dt, max_dt], shift by +max_dt.
        Returns: [D] if dt is int, else [..., D]
        """
        if isinstance(dt, int):
            dt_clamped = max(-self.max_dt, min(self.max_dt, dt))
            idx = dt_clamped + self.max_dt
            return self.emb.weight[idx]  # [D]
        dt = dt.to(torch.long)
        dt = torch.clamp(dt, -self.max_dt, self.max_dt)
        idx = dt + self.max_dt
        return self.emb(idx)  # [..., D]


class MemoryState:
    """
    Per-video memory state (AM + TM + TemporalPE).
    Stores tokens per frame, exports unified memory tokens for the reader.

    Supports multiple videos by keying internal states on `video_id`.
    If you don't pass video_id, it uses a single default slot.
    """
    def __init__(
        self,
        K: int,
        d_model: int,
        max_dt: int = 512,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        detach_memory: bool = True,
    ):
        self.K = int(K)
        self.d_model = int(d_model)
        self.detach_memory = bool(detach_memory)

        self.temporal_pe = TemporalPE(d_model=self.d_model, max_dt=max_dt)
        if device is not None:
            self.temporal_pe = self.temporal_pe.to(device)
        if dtype is not None:
            self.temporal_pe = self.temporal_pe.to(dtype=dtype)

        # video_id -> (AM, TM)
        self._am: Dict[str, AnchorMemory] = {}
        self._tm: Dict[str, TransientMemory] = {}

    @staticmethod
    def _normalize_tokens(tokens: Tensor) -> Tensor:
        """Accepts [N,D] or [1,N,D] and returns [N,D]."""
        if tokens.dim() == 3:
            assert tokens.shape[0] == 1, "MemoryState expects B=1 if tokens are batched."
            tokens = tokens[0]
        assert tokens.dim() == 2, f"Expected tokens [N,D], got shape {tuple(tokens.shape)}"
        return tokens

    @staticmethod
    def _topk_tokens_by_cosine_to_mean(tokens: Tensor, k: int, eps: float = 1e-6) -> Tensor:
        if k <= 0 or tokens.shape[0] <= k:
            return tokens

        # drop near-zero tokens first (important for mask-conditioned tokens)
        norms = tokens.norm(dim=1)
        keep = norms > eps
        if keep.sum().item() == 0:
            return tokens[: min(k, tokens.shape[0])]  # fallback
        tokens_nz = tokens[keep]
        if tokens_nz.shape[0] <= k:
            return tokens_nz

        x = tokens_nz - tokens_nz.mean(dim=0, keepdim=True)
        t = F.normalize(x, dim=1)
        g = F.normalize(t.mean(dim=0, keepdim=True), dim=1)
        scores = (t @ g.T).squeeze(1)
        idx = torch.topk(scores, k=k, largest=True).indices
        return tokens_nz.index_select(0, idx)

    def reset(self, video_id: str = "default") -> None:
        self._am[video_id] = AnchorMemory()
        self._tm[video_id] = TransientMemory(K=self.K)

    def add(
        self,
        tokens: Tensor,
        t: int,
        is_anchor: bool,
        video_id: str = "default",
        topk_tokens: int = 0,   # ✅ write-time TopK for TM
    ) -> None:
        """
        Add tokens for a frame at time t to AM (if anchor) or TM (if not).

        Args:
          tokens: [N,D] or [1,N,D]
          t: time index
          is_anchor: True -> AnchorMemory (pinned), False -> TransientMemory (evictable)
          topk_tokens: if >0 and is_anchor=False, store only top-k tokens by norm
        """
        if video_id not in self._am or video_id not in self._tm:
            self.reset(video_id)

        tokens = self._normalize_tokens(tokens)

        # Ensure temporal_pe lives on same device as incoming tokens (avoids per-call .to()).
        if self.temporal_pe.emb.weight.device != tokens.device:
            self.temporal_pe = self.temporal_pe.to(tokens.device)

        # Optional safety: ensure feature dim matches d_model
        assert tokens.shape[-1] == self.d_model, (
            f"tokens dim D={tokens.shape[-1]} != d_model={self.d_model}. "
            "Add a projection layer before MemoryState."
        )

        # ✅ write-time TopK for TM only
        if (not bool(is_anchor)) and int(topk_tokens) > 0:
            tokens = self._topk_tokens_by_cosine_to_mean(tokens, int(topk_tokens))

        tok_store = tokens.detach() if self.detach_memory else tokens
        # Robust scalar extraction
        if isinstance(t, torch.Tensor):
            t = int(t.item()) if t.numel() == 1 else int(t.view(-1)[0].item())
        else:
            t = int(t)

        if isinstance(is_anchor, torch.Tensor):
            is_anchor = bool(is_anchor.item()) if is_anchor.numel() == 1 else bool(is_anchor.view(-1)[0].item())
        else:
            is_anchor = bool(is_anchor)

        item = MemItem(tokens=tok_store, t=t, is_anchor=is_anchor)

        if item.is_anchor:
            self._am[video_id].add(item)
        else:
            self._tm[video_id].add(item)

    @torch.no_grad()
    def get_memory(
        self,
        t_now: int,
        video_id: str = "default",
        include_tm: bool = True,
        include_am: bool = True,
        device: Optional[torch.device] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Export unified memory tokens (AM ∪ TM) with Temporal PE applied.

        Returns:
          mem_tokens: [M, D] where M = sum over stored (N_i)
          mem_meta:
            - is_anchor: [M] bool
            - t: [M] int (original time index per token)
        """
        if video_id not in self._am or video_id not in self._tm:
            self.reset(video_id)

        items: List[MemItem] = []
        if include_am:
            items.extend(self._am[video_id].items)
        if include_tm:
            items.extend(self._tm[video_id].items)

        if len(items) == 0:
            dev = device or self.temporal_pe.emb.weight.device
            if self.temporal_pe.emb.weight.device != dev:
                self.temporal_pe = self.temporal_pe.to(dev)
            empty = torch.empty((0, self.d_model), device=dev)            
            meta = {
                "is_anchor": torch.empty((0,), dtype=torch.bool, device=dev),
                "t": torch.empty((0,), dtype=torch.long, device=dev),
            }
            return empty, meta

        # ---- vectorized export
        # tokens device from stored items
        dev = device or items[0].tokens.device
        if self.temporal_pe.emb.weight.device != dev:
            self.temporal_pe = self.temporal_pe.to(dev)

        tokens_list = [it.tokens.to(dev) for it in items]   # 👈 ensure on dev        
        counts = torch.tensor([x.shape[0] for x in tokens_list], device=dev)

        tokens = torch.cat(tokens_list, dim=0)  # [M,D]

        t_item = torch.tensor([it.t for it in items], device=dev, dtype=torch.long)
        a_item = torch.tensor([it.is_anchor for it in items], device=dev, dtype=torch.bool)

        t_tok = torch.repeat_interleave(t_item, counts)  # [M]
        a_tok = torch.repeat_interleave(a_item, counts)  # [M]

        dt = int(t_now) - t_tok  # [M]
        pe = self.temporal_pe(dt)  # [M,D]  (TemporalPE clamps internally)

        mem_tokens = tokens + pe
        mem_meta = {"is_anchor": a_tok, "t": t_tok}
        return mem_tokens, mem_meta

    def stats(self, video_id: str = "default") -> Dict[str, int]:
        if video_id not in self._am or video_id not in self._tm:
            return {"am_items": 0, "tm_items": 0, "am_tokens": 0, "tm_tokens": 0}

        am_items = self._am[video_id].items
        tm_items = self._tm[video_id].items
        am_tokens = sum(it.tokens.shape[0] for it in am_items)
        tm_tokens = sum(it.tokens.shape[0] for it in tm_items)

        return {
            "am_items": len(am_items),
            "tm_items": len(tm_items),
            "am_tokens": int(am_tokens),
            "tm_tokens": int(tm_tokens),
        }


# -----------------------
# Quick sanity test script
# -----------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    d_model = 256
    K = 3

    state = MemoryState(K=K, d_model=d_model, max_dt=64, device="cpu", detach_memory=True)
    state.reset("vid")

    for t in range(10):
        tokens = torch.randn(64, d_model)  # [N,D]
        is_anchor = (t == 0 or t == 5)

        # store only top-16 tokens in TM frames
        state.add(tokens=tokens, t=t, is_anchor=is_anchor, video_id="vid", topk_tokens=16)

        mem_tokens, mem_meta = state.get_memory(t_now=t, video_id="vid")
        st = state.stats("vid")
        print(
            f"t={t:02d}  AM={st['am_items']}({st['am_tokens']} tok)  "
            f"TM={st['tm_items']}({st['tm_tokens']} tok)  "
            f"mem={tuple(mem_tokens.shape)}  anchors_in_mem={int(mem_meta['is_anchor'].sum())}"
        )

    # Expected:
    # - AM increments at t=0 and t=5 => AM=2 at end
    # - TM never exceeds K=3
    # - TM token count reflects TopK (<= 16 per TM item)
