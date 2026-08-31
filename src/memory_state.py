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
import math 

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
        is_pseudo: bool = False,            # ✅ NEW

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
            is_pseudo=is_pseudo,            
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

    @staticmethod
    @torch.no_grad()
    def composite_reliability(
        logits_bchw: Tensor,                 # (B,C,H,W)
        prev_pred_bhw: Optional[Tensor] = None,
        mem_pred_bhw: Optional[Tensor] = None,
        ignore_index: int = 255,
        w_entropy: float = 0.45,
        w_margin: float = 0.35,
        w_temporal: float = 0.20,
        w_mem_agree: float = 0.00,
        eps: float = 1e-8,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
          rel_bhw : (B,H,W) in [0,1] (higher = more reliable)
          pred_bhw: (B,H,W) long
        Notes:
          - entropy term prefers low entropy (peaky)
          - margin term prefers p1-p2 high
          - temporal term prefers agreement with prev_pred (if provided)
          - mem_agree prefers agreement with mem_pred (optional)
        """
        assert logits_bchw.dim() == 4, f"logits must be (B,C,H,W), got {tuple(logits_bchw.shape)}"
        B, C, H, W = logits_bchw.shape
        p = torch.softmax(logits_bchw, dim=1)                     # (B,C,H,W)
        p_sorted, _ = torch.sort(p, dim=1, descending=True)
        p1 = p_sorted[:, 0]                                       # (B,H,W)
        p2 = p_sorted[:, 1] if C > 1 else torch.zeros_like(p1)
        pred = p.argmax(dim=1)                                    # (B,H,W)

        # ---- normalized entropy in [0,1]
        ent = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=1)      # (B,H,W)
        ent = ent / max(float(math.log(max(C, 2))), eps)                   # normalize by log(C)
        ent_term = (1.0 - ent).clamp(0.0, 1.0)

        # ---- margin in [0,1] (already bounded)
        margin_term = (p1 - p2).clamp(0.0, 1.0)

        # ---- temporal agreement in {0,1}
        if prev_pred_bhw is not None:
            prev = prev_pred_bhw.to(device=pred.device)
            if prev.dim() == 2:
                prev = prev.unsqueeze(0)
            if prev.shape != pred.shape:
                # best-effort: skip if shapes mismatch
                temporal_term = torch.ones_like(p1)
            else:
                temporal_term = (pred == prev).float()
        else:
            temporal_term = torch.ones_like(p1)

        # ---- memory agreement in {0,1}
        if mem_pred_bhw is not None:
            mp = mem_pred_bhw.to(device=pred.device)
            if mp.dim() == 2:
                mp = mp.unsqueeze(0)
            if mp.shape != pred.shape:
                mem_term = torch.ones_like(p1)
            else:
                mem_term = (pred == mp).float()
        else:
            mem_term = torch.ones_like(p1)

        # ---- weighted sum -> [0,1]
        w_sum = float(w_entropy + w_margin + w_temporal + w_mem_agree)
        if w_sum <= 0:
            rel = ent_term * 0.0 + 1.0
        else:
            rel = (
                w_entropy * ent_term +
                w_margin  * margin_term +
                w_temporal * temporal_term +
                w_mem_agree * mem_term
            ) / w_sum

        # ---- ignore_index handling (if provided via prev_pred as GT mask etc.)
        # We cannot know GT here; but if prev_pred_bhw contains ignore_index (rare), mask it out.
        rel = rel.clamp(0.0, 1.0)

        return rel, pred

    @staticmethod
    @torch.no_grad()
    def class_aware_reliability(
        reliability_bhw: Tensor,    # (B,H,W)
        pred_bhw: Tensor,           # (B,H,W) long
        num_classes: int,
        ignore_index: int = 255,
        min_pixels: int = 64,
    ) -> Tensor:
        """
        Returns r_c: (C,) mean reliability over pixels predicted as class c.
        If a class has < min_pixels, its score is 0.
        """
        assert reliability_bhw.dim() == 3 and pred_bhw.dim() == 3
        B, H, W = pred_bhw.shape
        C = int(num_classes)
        r_c = reliability_bhw.new_zeros((C,), dtype=torch.float32)

        # flatten across batch
        r = reliability_bhw.reshape(-1)
        p = pred_bhw.reshape(-1)

        valid = (p != ignore_index)
        r = r[valid]
        p = p[valid]

        for c in range(C):
            m = (p == c)
            if int(m.sum().item()) >= int(min_pixels):
                r_c[c] = r[m].mean()
        return r_c

    @staticmethod
    @torch.no_grad()
    def _cos_sim(a: Tensor, b: Tensor, eps: float = 1e-6) -> float:
        a = a.view(-1)
        b = b.view(-1)
        an = a / (a.norm() + eps)
        bn = b / (b.norm() + eps)
        return float((an * bn).sum().item())

    @torch.no_grad()
    def promote_to_am(
        self,
        video_id: str,
        t: int,
        class_id: int,
        tok_cond: Tensor,         # (1,k,D) or (k,D)
        w: Optional[Tensor],
        conf: float,
        proto: Optional[Tensor],
        min_conf: float = 0.80,
        merge_sim: float = 0.92,
        add_sim_max: float = 0.90,
        min_conf_delta: float = 0.03,
        debug: bool = False,
    ) -> str:
        """
        Diversity + consistency gate before promoting a pseudo-anchor into AM.
        Returns action: 'skip' | 'merge' | 'add'
        """
        if (not self._am) or (video_id not in self._am):
            self.reset(video_id)

        if float(conf) < float(min_conf):
            return "skip"

        if proto is None:
            # cheap proto from tokens if not provided
            x = self._normalize_tokens(tok_cond)
            proto = x.mean(dim=0)

        am_items = self._am[video_id].items
        cid = int(class_id)

        # ---- same-class anchors
        cls_pairs = [(i, it) for i, it in enumerate(am_items) if int(getattr(it, "class_id", -999)) == cid]

        # ---- 1) Merge if too similar to existing same-class anchor
        best_sim = -1.0
        best_idx = -1
        if cls_pairs:
            for i, it in cls_pairs:
                if it.proto is None:
                    continue
                s = self._cos_sim(proto, it.proto)
                if s > best_sim:
                    best_sim = s
                    best_idx = int(i)

        if best_idx >= 0 and best_sim >= float(merge_sim):
            old_conf = float(getattr(am_items[best_idx], "conf", 0.0) or 0.0)
            if float(conf) >= (old_conf + float(min_conf_delta)):
                self.merge_anchor(
                    video_id=video_id,
                    item_idx=best_idx,
                    tok_cond=tok_cond,
                    w=(w if w is not None else torch.ones((self._normalize_tokens(tok_cond).shape[0],), device=proto.device)),
                    conf=float(conf),
                    proto=proto,
                    t=int(t),
                )
                if debug:
                    print(f"[AM promote] MERGE cid={cid} sim={best_sim:.3f} conf={conf:.3f} old={old_conf:.3f}")
                return "merge"
            return "skip"

        # ---- 2) Global diversity: don't add if too similar to ANY existing anchor
        if am_items:
            gbest = -1.0
            for it in am_items:
                if it.proto is None:
                    continue
                gbest = max(gbest, self._cos_sim(proto, it.proto))
            if gbest >= float(add_sim_max):
                return "skip"

        # ---- 3) Add new anchor (cap enforcement happens inside add_anchor)
        tokens_norm = self._normalize_tokens(tok_cond)
        w_store = w
        if w_store is None:
            w_store = torch.ones((tokens_norm.shape[0],), device=tokens_norm.device, dtype=tokens_norm.dtype)

        self.add_anchor(
            tokens=tokens_norm,
            t=int(t),
            class_id=cid,
            video_id=video_id,
            w=w_store,
            conf=float(conf),
            pinned=False,
            proto=proto,
        )
        if debug:
            print(f"[AM promote] ADD cid={cid} conf={conf:.3f}")
        return "add"
