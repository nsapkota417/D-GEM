# memory/anchor_memory.py
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING
import torch
import torch.nn.functional as F
from dataclasses import dataclass

Tensor = torch.Tensor

@dataclass
class MemItem:
    tokens: Tensor
    t: int
    is_anchor: bool
    class_id: Optional[int] = None
    conf: Optional[float] = None
    pinned: bool = False
    proto: Optional[Tensor] = None
    w: Optional[Tensor] = None
    is_pseudo: bool = False   # ✅ NEW


class AnchorMemory:
    """
    Anchor Memory (AM):
      - supports pinning (never evict pinned)
      - fixed budget (max_items)
      - per-class filtering via item.class_id
      - optional diversity-aware eviction using item.proto
    Keeps `.items` list so MemoryState can `items.extend(self._am[vid].items)` unchanged.
    """

    def __init__(
        self,
        max_items: int = 8,                 # total anchors per video (across classes)
        redundancy_lambda: float = 0.5,      # how hard to penalize redundancy in eviction
        eps: float = 1e-6,
    ):
        self.items: List["MemItem"] = []
        self.max_items = int(max_items)
        self.redundancy_lambda = float(redundancy_lambda)
        self.eps = float(eps)

    def reset(self) -> None:
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)

    # -------------------------
    # Core API
    # -------------------------
    def add(self, item: "MemItem") -> None:
        """Add anchor item then enforce budget."""
        self.items.append(item)
        self.enforce_budget()

    def get_items(self, class_id: Optional[int] = None) -> List["MemItem"]:
        """Return anchors optionally filtered by class_id."""
        if class_id is None:
            return list(self.items)
        return [it for it in self.items if (it.class_id is not None and int(it.class_id) == int(class_id))]

    # -------------------------
    # Budget / eviction
    # -------------------------
    def enforce_budget(self) -> None:
        """Evict weakest NON-pinned anchors until len(items) <= max_items."""
        if self.max_items <= 0:
            return
        while len(self.items) > self.max_items:
            idx = self._pick_evict_index()
            if idx is None:
                # all pinned: hard-trim oldest pinned (or lowest conf)
                idx = min(range(len(self.items)), key=lambda i: int(self.items[i].t))
            self.items.pop(idx)


    def _pick_evict_index(self) -> Optional[int]:
        """
        Evict policy: lowest (conf - redundancy_penalty).
        Only considers non-pinned.

        Patch notes:
          - redundancy similarity excludes the candidate itself
          - avoids torch.tensor(scores) device issues (pure python min)
        """
        nonpinned = [(i, it) for i, it in enumerate(self.items) if not bool(getattr(it, "pinned", False))]
        if len(nonpinned) == 0:
            return None

        # base confidence + candidate protos
        confs: List[float] = []
        protos: List[Optional[Tensor]] = []
        for _, it in nonpinned:
            c = getattr(it, "conf", None)
            confs.append(float(c) if c is not None else 0.0)
            protos.append(getattr(it, "proto", None))

        # collect all protos once
        all_protos: List[Tensor] = []
        for it in self.items:
            p = getattr(it, "proto", None)
            if p is not None:
                all_protos.append(p)

        penalties = [0.0 for _ in nonpinned]
        if len(all_protos) >= 2:
            for j, p in enumerate(protos):
                if p is None:
                    continue
                # EXCLUDE candidate itself by object identity
                others = [q for q in all_protos if q is not p]
                if len(others) == 0:
                    continue
                P = F.normalize(torch.stack([q.detach() for q in others], dim=0), dim=1, eps=self.eps)  # [A-1,D]
                q = F.normalize(p.detach().unsqueeze(0), dim=1, eps=self.eps)                           # [1,D]
                penalties[j] = float((q @ P.T).max().item())

        scores = [confs[j] - self.redundancy_lambda * penalties[j] for j in range(len(nonpinned))]

        # evict minimum score (pure python; no device issues)
        j_min = min(range(len(scores)), key=lambda j: scores[j])
        return nonpinned[j_min][0]

    # -------------------------
    # Refresh / diversity helper
    # -------------------------
    def should_add_refresh(
        self,
        proto_new: Tensor,
        sim_max: float = 0.9,   # require diversity: max cosine sim must be <= this
    ) -> bool:
        """Return True if new anchor is diverse enough vs existing anchors (by proto)."""
        protos = [getattr(it, "proto", None) for it in self.items]
        protos = [p for p in protos if p is not None]
        if len(protos) == 0:
            return True
        P = torch.stack([p.detach() for p in protos], dim=0)  # [A,D]
        P = F.normalize(P, dim=1, eps=self.eps)
        q = F.normalize(proto_new.detach().unsqueeze(0), dim=1, eps=self.eps)  # [1,D]
        sim = (q @ P.T).max().item()
        return float(sim) <= float(sim_max)



class TransientMemory:
    """
    Evictable memory with a hard capacity K.
    v1 policy: keep most recent K items (FIFO/recency).
    """
    def __init__(self, K: int):
        assert K > 0, "K must be > 0"
        self.K = K
        self.items: List[MemItem] = []

    def reset(self) -> None:
        self.items.clear()

    def add(self, item: MemItem) -> None:
        self.items.append(item)
        self.prune()

    def prune(self) -> None:
        if len(self.items) > self.K:
            self.items = self.items[-self.K :]

    def __len__(self) -> int:
        return len(self.items)