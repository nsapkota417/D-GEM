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

from memory import AnchorMemory, TransientMemory, MemItem

Tensor = torch.Tensor

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
        am_max_items: int = 8, 
        am_red_lambda: float = 0.5,
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

        self.am_max_items = int(am_max_items)
        self.am_red_lambda = float(am_red_lambda)

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

        # ---- diversity guard (cosine NMS)
        selected = []
        tokens_sel = tokens_nz.index_select(0, idx)
        tokens_sel = F.normalize(tokens_sel, dim=1)

        for i in range(tokens_sel.shape[0]):
            if len(selected) == 0:
                selected.append(i)
                continue
            sims = torch.stack([
                (tokens_sel[i] @ tokens_sel[j]).abs()
                for j in selected
            ])
            if sims.max().item() < 0.95:   # allow mild similarity
                selected.append(i)
            if len(selected) == k:
                break

        if len(selected) > 0:
            return tokens_nz.index_select(0, idx[selected])
        else:
            return tokens_nz.index_select(0, idx[:k])        

    def reset(self, video_id: str = "default") -> None:
        self._am[video_id] = AnchorMemory(max_items=self.am_max_items,
                                        redundancy_lambda=self.am_red_lambda)

        self._tm[video_id] = TransientMemory(K=self.K)

    def add(
        self,
        tokens: Tensor,
        t: int,
        is_anchor: bool,
        video_id: str = "default",
        topk_tokens: int = 0,
        class_id: Optional[int] = None,     # NEW
        w: Optional[Tensor] = None,          # NEW
    ) -> None:
        """
        Add tokens for a frame at time t to AM (if anchor) or TM (if not).

        Args:
          tokens: [N,D] or [1,N,D]
          t: time index
          is_anchor: True -> AnchorMemory (pinned), False -> TransientMemory (evictable)
          topk_tokens: if >0 and is_anchor=False, store only _topk_tokens_by_cosine_to_mean
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

        w_store = None if w is None else (w.detach() if self.detach_memory else w)
        cid = None if class_id is None else int(class_id)
        item = MemItem(tokens=tok_store, t=t, is_anchor=is_anchor, class_id=cid, w=w_store)

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

        tokens_list = [it.tokens.to(dev) for it in items]
        counts = torch.tensor([x.shape[0] for x in tokens_list], device=dev)

        tokens = torch.cat(tokens_list, dim=0)  # [M,D]

        t_item = torch.tensor([it.t for it in items], device=dev, dtype=torch.long)
        a_item = torch.tensor([it.is_anchor for it in items], device=dev, dtype=torch.bool)

        t_tok = torch.repeat_interleave(t_item, counts)  # [M]
        a_tok = torch.repeat_interleave(a_item, counts)  # [M]

        # build w/class_id aligned with token order
        w_list, cid_list = [], []
        for it in items:
            n = it.tokens.shape[0]
            w_list.append(torch.ones((n,), device=dev) if it.w is None else it.w.to(dev))
            cid = -1 if it.class_id is None else int(it.class_id)
            cid_list.append(torch.full((n,), cid, device=dev, dtype=torch.long))

        w_tok = torch.cat(w_list, dim=0)
        cid_tok = torch.cat(cid_list, dim=0)

        dt = int(t_now) - t_tok
        pe = self.temporal_pe(dt)
        mem_tokens = tokens + pe

        mem_meta = {"is_anchor": a_tok, "t": t_tok, "w": w_tok, "class_id": cid_tok}
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

    def add_anchor(
        self,
        tokens: Tensor,
        t: int,
        class_id: int,
        video_id: str = "default",
        w: Optional[Tensor] = None,
        conf: Optional[float] = None,
        pinned: bool = False,
        proto: Optional[Tensor] = None,
    ) -> None:
        if video_id not in self._am or video_id not in self._tm:
            self.reset(video_id)

        tokens = self._normalize_tokens(tokens)

        if self.temporal_pe.emb.weight.device != tokens.device:
            self.temporal_pe = self.temporal_pe.to(tokens.device)

        assert tokens.shape[-1] == self.d_model

        tok_store = tokens.detach() if self.detach_memory else tokens
        w_store = None if w is None else (w.detach() if self.detach_memory else w)
        if w_store is not None:
            w_store = w_store.view(-1)
            assert w_store.numel() == tok_store.shape[0], (
                f"w has {w_store.numel()} elems, but tokens have N={tok_store.shape[0]}"
            )

        proto_store = None if proto is None else (proto.detach() if self.detach_memory else proto)

        item = MemItem(
            tokens=tok_store,
            t=int(t),
            is_anchor=True,
            class_id=int(class_id),
            conf=None if conf is None else float(conf),
            pinned=bool(pinned),
            proto=proto_store,
            w=w_store,
        )
        self._am[video_id].add(item)

        # ---- ENFORCE CAP immediately (keep pinned if possible)
        while len(self._am[video_id].items) > self.am_max_items:
            items = self._am[video_id].items

            # candidates = non-pinned anchors first
            cand_idx = [i for i,it in enumerate(items) if not bool(getattr(it, "pinned", False))]

            # if everything is pinned (should be rare), fall back to evict oldest pinned
            if not cand_idx:
                cand_idx = list(range(len(items)))

            # evict policy: oldest-by-time among candidates
            ev_i = min(cand_idx, key=lambda i: int(items[i].t))
            del items[ev_i]




    def has_pinned_anchor(self, video_id: str, class_id: int) -> bool:
        if video_id not in self._am:
            return False
        for it in self._am[video_id].items:
            if (it.class_id == int(class_id)) and bool(getattr(it, "pinned", False)):
                return True
        return False

    def merge_anchor(self, video_id: str, item_idx: int,
                    tok_cond: torch.Tensor, w: torch.Tensor,
                    conf: float, proto: torch.Tensor, t: int):
        am = self._am[video_id]
        it = am.items[item_idx]          # MemItem

        # ---- DEBUG: save old proto
        old_proto = it.proto.detach().clone() if it.proto is not None else None

        # tok_cond: (1,k,D) -> (k,D)
        x_new = tok_cond.squeeze(0)

        # ---- merge
        it.tokens = torch.cat([it.tokens, x_new], dim=0)
        it.w      = torch.cat([it.w, w.to(device=it.w.device, dtype=it.w.dtype)], dim=0)

        # ---- prune back to K
        K = min(256, it.tokens.shape[0])
        idx = torch.topk(it.w, k=K, largest=True).indices
        it.tokens = it.tokens.index_select(0, idx)
        it.w      = it.w.index_select(0, idx)

        # ---- update proto/conf/time
        it.proto = (0.7 * it.proto + 0.3 * proto.detach()) if it.proto is not None else proto.detach()
        it.conf  = max(float(getattr(it, "conf", 0.0)), float(conf))
        it.t     = int(t)

        # # ---- DEBUG: compare old vs new proto
        # if old_proto is not None:
        #     dp = float(F.cosine_similarity(old_proto[None], it.proto.detach()[None]).item())
        #     print(f"[AM merge] cid={it.class_id} proto_cos(old,new)={dp:.4f} k={it.tokens.shape[0]}")