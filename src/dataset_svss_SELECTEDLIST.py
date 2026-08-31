import os
import re
import random
import hashlib
from typing import List, Tuple, Optional, Dict
import math
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import json


class SVSSDataset(Dataset):

    def __init__(
        self,
        cfg,
        df: pd.DataFrame,
        split: str = "train",
        # If set (>=0), overrides cfg.t_query for val/test only.
        # -1 => all frames; >0 => that many evenly spaced frames.
        val_t_query: Optional[int] = None,
    ):
        self.cfg = cfg
        if self.cfg.data.name == "sarrarp50":
            self.FRAME_RE = re.compile(r"^(\d{9})\.png$")
        else:
            self.FRAME_RE = re.compile(r"^frame(\d+)\.png$")

        self.df = df
        self.split = str(split).lower()
        assert self.split in {"train", "val", "test"}, f"split must be train/val/test, got {split}"

        # --- config
        self.mask_col = str(self.cfg.data.mask_col)
        self.ignore_index = int(getattr(self.cfg.data, "ignore_index", 255))

        # train sampling controls
        self.train_t_query = int(getattr(self.cfg.train, "t_query", 10))
        self.train_jitter = int(getattr(self.cfg.train, "jitter", 0))

        # NEW: multi-support controls (train only by default)
        self.s_support = int(getattr(self.cfg.train, "s_support", 1))           # number of support bins
        self.support_jitter = int(getattr(self.cfg.train, "support_jitter", 0)) # jitter magnitude for support bins
        self.epoch_salt = int(getattr(self.cfg.train, "epoch_salt", 0))        # -----------------------------
        
        # NEW: Allowed-frame sampling
        # -----------------------------
        # Emulated deterministic allowed set per (video_src, video_clip).
        # Same across runs. Very fast (cached).
        self.allowed_max_items = int(getattr(self.cfg.train, "allowed_max_items", 50))
        self.allowed_seed = int(getattr(self.cfg.train, "allowed_seed", 1337))
        self.allowed_kwin = int(getattr(self.cfg.train, "allowed_kwin", 2))  # consecutive frames per bin/window
        self.allowed_strict_consecutive = bool(getattr(self.cfg.train, "allowed_strict_consecutive", True))
        # NOTE:
        #  - strict_consecutive=True tries to return real consecutive frame ids (i,i+1,...) but still in allowed
        #  - if impossible, falls back to neighbors in allowed-order (still temporally close)

        # per-video cache (per-worker process)
        self._allowed_cache: Dict[str, np.ndarray] = {}
        self._allowed_set_cache: Dict[str, set] = {}

        # val/test sampling controls
        if val_t_query is None:
            self.val_t_query = int(getattr(self.cfg.val, "val_t_query", -1))
        else:
            self.val_t_query = int(val_t_query)

        # --- required columns
        required = {"img", "video_src", "video_clip", self.mask_col}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        # --- build LUT for code->class mapping (fast, vectorized)
        self.lut = np.full(256, self.ignore_index, dtype=np.uint8)
        for k in range(13):
            self.lut[k] = np.uint8(k)
        self.lut[self.ignore_index] = np.uint8(self.ignore_index)

        # -------- RGB label mapping (EndoVis) --------
        self.rgb_to_id = None

        label_json = getattr(self.cfg.data, "label_json", None)
        code_to_class = getattr(self.cfg.data, "code_to_class", None)

        # Case 1: RGB labels via JSON (e.g., EndoVis)
        if label_json:
            with open(label_json, "r") as f:
                items = json.load(f)
            self.rgb_to_id = {
                tuple(map(int, it["color"])): int(it["classid"])
                for it in items
            }

        # Case 2: Grayscale labels via code_to_class (fallback)
        elif code_to_class is not None:
            self.lut = np.full(256, self.ignore_index, dtype=np.uint8)
            for k, v in dict(code_to_class).items():
                ki = int(k, 0) if isinstance(k, str) else int(k)
                vi = int(v)
                if 0 <= ki <= 255:
                    self.lut[ki] = np.uint8(vi)

        # --- group by clip: one dataset item = one clip episode
        self.groups = {}
        for (vs, vc), g in self.df.groupby(["video_src", "video_clip"], sort=True):
            g = g.copy()
            g["_k"] = g["img"].apply(self._frame_sort_key)
            g = g.sort_values("_k").drop(columns="_k").reset_index(drop=True)
            self.groups[(vs, vc)] = g

        self.clips = list(self.groups.keys())

        self.num_clips = len(self.clips)
        self.num_frames = sum(len(g) for g in self.groups.values())

        # ---- expected frames consumed per epoch (train-style sampling)
        if self.split in {"train", "test", "val"}:

            if self.split == "train":
                q_per_clip = int(self.train_t_query)          # TOTAL frames per clip
                s_per_clip = int(self.s_support)
            else:
                q_per_clip = int(self.val_t_query)
                s_per_clip = 1

            # handle "all frames" case
            per_clip_queries = []
            for g in self.groups.values():
                n = len(g)
                if n < 2:
                    per_clip_queries.append(0)
                    continue

                if q_per_clip < 0:
                    per_clip_queries.append(n - 1)
                else:
                    per_clip_queries.append(min(q_per_clip, n - 1))

            avg_q = float(np.mean(per_clip_queries)) if per_clip_queries else 0.0
            total_q = int(sum(per_clip_queries))
            total_s = int(self.num_clips * max(1, s_per_clip))

            label = "training" if self.split == "train" else ("val" if self.split == "val" else "test")

            print(
                f"  {label:<8} | "
                f"# clips = {self.num_clips:>4} | "
                f"support/clip = {max(1, s_per_clip):>2} | "
                f"total_support = {total_s:>5} | "
                f"query/clip = {avg_q:>5.1f} | "
                f"total_query = {total_q:>6} | "
                # f"Kwin = {int(self.allowed_kwin):>2}"
            )

    def __len__(self):
        return len(self.clips)

    def _get_rng(self, idx: int):
        """
        Per-worker, per-epoch RNG.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            seed = torch.initial_seed()
        else:
            seed = worker_info.seed
        return random.Random(seed + idx)

    # -----------------------------
    # Allowed-set helpers
    # -----------------------------
    def _video_key(self, vs: str, vc: str) -> str:
        return f"{vs}::{vc}"

    def _stable_video_seed(self, video_key: str) -> int:
        h = hashlib.sha1((video_key + f"__{self.allowed_seed}").encode("utf-8")).hexdigest()
        return int(h[:8], 16)  # 32-bit

    def _get_allowed_indices(self, vs: str, vc: str, n_frames: int) -> np.ndarray:
        """
        Deterministic allowed indices per (vs, vc), cached.
        Ensures 0 is included (so support can always use it).
        Size <= allowed_max_items (or n_frames if smaller).
        """
        vkey = self._video_key(vs, vc)
        if vkey in self._allowed_cache:
            return self._allowed_cache[vkey]

        N = int(n_frames)
        k = min(max(1, int(self.allowed_max_items)), N)

        rng = np.random.RandomState(self._stable_video_seed(vkey))
        # choose k unique indices
        arr = rng.choice(np.arange(N, dtype=np.int32), size=k, replace=False)
        arr = np.unique(arr)
        arr.sort()

        # force include 0 (support anchor)
        if 0 not in arr:
            arr = np.unique(np.concatenate([arr, np.asarray([0], dtype=np.int32)]))
            arr.sort()

        # clamp + final size control (keep 0, then earliest others)
        arr = arr[(arr >= 0) & (arr < N)]
        if arr.size > k:
            # keep 0 and k-1 others
            others = arr[arr != 0]
            others = others[: max(0, k - 1)]
            arr = np.unique(np.concatenate([np.asarray([0], dtype=np.int32), others]))
            arr.sort()

        self._allowed_cache[vkey] = arr
        if self.allowed_strict_consecutive:
            self._allowed_set_cache[vkey] = set(map(int, arr.tolist()))
        return arr

    def _pick_allowed_anchor_in_bin(
        self,
        allowed: np.ndarray,
        lo: int,
        hi: int,
        c: int,
        forbidden: Optional[set] = None,
        rng: Optional[random.Random] = None,
    ) -> int:
        forbidden = forbidden or set()

        L = int(np.searchsorted(allowed, lo, side="left"))
        R = int(np.searchsorted(allowed, hi, side="right"))

        # if we have candidates in-bin, sample randomly (preferred)
        if L < R:
            cand = [int(x) for x in allowed[L:R].tolist() if int(x) not in forbidden]
            if cand:
                if rng is None:
                    return cand[len(cand)//2]  # fallback deterministic
                return rng.choice(cand)

        # otherwise fall back to nearest globally (existing behavior)
        # (search outward for a non-forbidden allowed)
        if allowed.size > 0:
            p = int(np.searchsorted(allowed, c, side="left"))
            for step in range(0, allowed.size):
                for q in (p - step, p + step):
                    if 0 <= q < allowed.size:
                        ai = int(allowed[q])
                        if ai not in forbidden:
                            return ai

        return int(allowed[0]) if allowed.size else int(lo)

    def _expand_window(
        self,
        allowed: np.ndarray,
        vs: str,
        vc: str,
        anchor: int,
        K: int,
        n_frames: int,
        forbidden: Optional[set] = None,
    ) -> List[int]:
        """
        Return exactly K indices, all in allowed, avoiding forbidden when possible.
        Strategy:
          1) strict consecutive frame ids (anchor, anchor+1, ...) filtered by allowed_set
          2) fallback: K neighbors in allowed-order near anchor
          3) pad by repeating last if still short
        """
        forbidden = forbidden or set()
        K = max(1, int(K))
        N = int(n_frames)
        vkey = self._video_key(vs, vc)

        out: List[int] = []

        # (1) strict consecutive (frame-index space), but must be allowed
        if self.allowed_strict_consecutive:
            aset = self._allowed_set_cache.get(vkey, set(map(int, allowed.tolist())))
            # forward first
            for t in range(K):
                j = anchor + t
                if 0 <= j < N and (j in aset) and (j not in forbidden):
                    out.append(int(j))
                else:
                    break
            if len(out) < K:
                # try extend backward around anchor
                t = 1
                while len(out) < K and (anchor - t) >= 0:
                    j = anchor - t
                    if (j in aset) and (j not in forbidden):
                        out = [int(j)] + out
                    t += 1

        # (2) fallback to allowed-order neighbors
        if len(out) < K:
            # find anchor position in allowed (or nearest)
            pos = int(np.searchsorted(allowed, anchor, side="left"))
            pos = max(0, min(pos, allowed.size - 1))
            if int(allowed[pos]) != int(anchor):
                # nearest neighbor among pos-1,pos
                cand = [pos]
                if pos - 1 >= 0:
                    cand.append(pos - 1)
                best = cand[0]
                bestd = abs(int(allowed[best]) - anchor)
                for q in cand[1:]:
                    d = abs(int(allowed[q]) - anchor)
                    if d < bestd:
                        bestd = d
                        best = q
                pos = best

            # walk outward in allowed-order to collect K
            out2 = []
            left = pos
            right = pos + 1
            while len(out2) < K and (left >= 0 or right < allowed.size):
                if left >= 0:
                    ai = int(allowed[left])
                    if ai not in forbidden:
                        out2.append(ai)
                    left -= 1
                    if len(out2) >= K:
                        break
                if right < allowed.size:
                    ai = int(allowed[right])
                    if ai not in forbidden:
                        out2.append(ai)
                    right += 1

            out2 = sorted(set(out2))  # keep unique, sorted
            # if we got more than K due to sorting/uniques, take closest window around anchor
            if len(out2) > K:
                # take K closest to anchor
                out2.sort(key=lambda x: (abs(x - anchor), x))
                out2 = sorted(out2[:K])
            out = out2 if len(out2) >= len(out) else out

        # (3) final pad to exactly K
        if len(out) == 0:
            out = [int(anchor)]
        if len(out) < K:
            out = out + [int(out[-1])] * (K - len(out))
        elif len(out) > K:
            out = out[:K]

        return out

    def _sample_bins_windows_from_allowed(
        self,
        allowed: np.ndarray,
        vs: str,
        vc: str,
        n_frames: int,
        num_bins: int,
        jitter: int,
        Kwin: int,
        start_idx: int,
        idx: int,
        forbidden: Optional[set] = None,
    ) -> List[int]:
        """
        Sample num_bins windows from [start_idx..n-1], each window has Kwin frames,
        all constrained to allowed (and avoiding forbidden if provided).
        Returns flattened list length = num_bins * Kwin (fixed).
        """
        forbidden = forbidden or set()
        N = int(n_frames)
        num_bins = max(1, int(num_bins))
        Kwin = max(1, int(Kwin))
        start_idx = int(start_idx)
        end_idx = N - 1

        if start_idx > end_idx:
            start_idx = end_idx

        # build bins over [start_idx..end_idx] inclusive
        # use edges in integer space
        # edges length num_bins+1 in [start_idx..end_idx+1]
        edges = np.linspace(start_idx, end_idx + 1, num=num_bins + 1, endpoint=True).astype(int)

        rng = self._get_rng(idx + 100000 * int(self.epoch_salt))
        out_flat: List[int] = []

        for b in range(num_bins):
            lo = int(edges[b])
            hi = int(edges[b + 1] - 1)
            if hi < lo:
                hi = lo

            # candidate center
            c = (lo + hi) // 2
            if int(jitter) > 0 and hi > lo:
                j = rng.randint(-int(jitter), int(jitter))
                c = int(np.clip(c + j, lo, hi))

            anchor = self._pick_allowed_anchor_in_bin(
                allowed=allowed, lo=lo, hi=hi, c=c, forbidden=forbidden, rng=rng
            )

            win = self._expand_window(
                allowed=allowed,
                vs=vs,
                vc=vc,
                anchor=int(anchor),
                K=Kwin,
                n_frames=N,
                forbidden=forbidden,
            )

            # ensure each element is within [0..N-1] and allowed; if something slipped, snap back
            # (should be rare; but keep robust)
            allowed_set = None
            if self.allowed_strict_consecutive:
                allowed_set = self._allowed_set_cache.get(self._video_key(vs, vc), None)
            if allowed_set is None:
                allowed_set = set(map(int, allowed.tolist()))

            clean = []
            for w in win:
                w = int(max(0, min(N - 1, int(w))))
                if w not in allowed_set:
                    # snap to nearest allowed globally
                    p = int(np.searchsorted(allowed, w, side="left"))
                    if p <= 0:
                        w = int(allowed[0])
                    elif p >= allowed.size:
                        w = int(allowed[-1])
                    else:
                        a0, a1 = int(allowed[p - 1]), int(allowed[p])
                        w = a0 if abs(a0 - w) <= abs(a1 - w) else a1
                if w in forbidden:
                    # try nearest allowed not forbidden
                    w = self._pick_allowed_anchor_in_bin(allowed, 0, N - 1, w, forbidden)
                clean.append(int(w))

            # pad to fixed Kwin
            if len(clean) < Kwin:
                clean = clean + [int(clean[-1])] * (Kwin - len(clean))
            elif len(clean) > Kwin:
                clean = clean[:Kwin]

            out_flat.extend(clean)

        # fixed length: num_bins*Kwin
        if len(out_flat) != num_bins * Kwin:
            # enforce
            if len(out_flat) < num_bins * Kwin:
                out_flat = out_flat + [int(out_flat[-1])] * (num_bins * Kwin - len(out_flat))
            else:
                out_flat = out_flat[: num_bins * Kwin]

        return out_flat



    def _sample_total_frames_from_allowed(
        self,
        allowed: np.ndarray,
        vs: str,
        vc: str,
        n_frames: int,
        total_frames: int,     # <-- THIS is t_query or s_support now
        jitter: int,
        Kwin: int,
        start_idx: int,
        idx: int,
        forbidden: Optional[set] = None,
    ) -> List[int]:
        """
        Returns EXACTLY `total_frames` indices (fixed length),
        sampled as windows of size Kwin (for flow/consistency),
        but truncated/padded to match total_frames.

        All indices are from allowed, and avoid forbidden when possible.
        """
        forbidden = forbidden or set()
        total_frames = int(total_frames)
        if total_frames <= 0:
            return []

        Kwin = max(1, int(Kwin))
        # Number of windows needed to cover total_frames
        num_bins = int(math.ceil(total_frames / float(Kwin)))

        flat = self._sample_bins_windows_from_allowed(
            allowed=allowed,
            vs=vs,
            vc=vc,
            n_frames=n_frames,
            num_bins=num_bins,
            jitter=jitter,
            Kwin=Kwin,
            start_idx=start_idx,
            idx=idx,
            forbidden=forbidden,
        )

        # Now enforce EXACT total_frames
        if len(flat) >= total_frames:
            return flat[:total_frames]
        else:
            # pad by repeating last (keeps tensor shapes fixed)
            pad = [int(flat[-1])] * (total_frames - len(flat)) if flat else [start_idx] * total_frames
            return flat + pad



    # -----------------------------
    # __getitem__
    # -----------------------------
    def __getitem__(self, idx):
        vs, vc = self.clips[idx]
        g = self.groups[(vs, vc)]
        n = len(g)
        if n < 2:
            raise RuntimeError(f"{vs}/{vc} has <2 frames")

        allowed = self._get_allowed_indices(vs, vc, n)
        allowed_set = set(map(int, allowed.tolist()))

        Kwin = int(self.allowed_kwin)

        # -------------------------
        # Support indices (TOTAL frames now)
        # -------------------------
        if self.split == "train":
            S_total = max(1, int(self.s_support))          # <-- total support frames
            s_idx = self._sample_total_frames_from_allowed(
                allowed=allowed,
                vs=vs, vc=vc,
                n_frames=n,
                total_frames=S_total,
                jitter=int(self.support_jitter),
                Kwin=Kwin,
                start_idx=0,
                idx=idx,
                forbidden=set(),
            )
            # force include 0 if you want first frame annotated
            if 0 in set(map(int, allowed.tolist())) and 0 not in s_idx:
                s_idx[0] = 0
        else:
            # val/test: still 1 support frame total (or keep cfg if you want)
            s_idx = [0] if 0 in set(map(int, allowed.tolist())) else [int(allowed[0])]

        # -------------------------
        # Query indices (TOTAL frames now)
        # -------------------------
        if self.split == "train":
            T_total = int(self.train_t_query)              # <-- total query frames
            if T_total < 0:
                # "all allowed frames excluding support" (fixed-ish, but may vary)
                forbidden = set(s_idx)
                q_idx = [int(i) for i in allowed.tolist() if int(i) >= 1 and int(i) not in forbidden]
                # optionally cap to keep fixed shapes:
                # q_idx = q_idx[:max(1, self.allowed_max_items - 1)]
            else:
                forbidden = set(s_idx)
                q_idx = self._sample_total_frames_from_allowed(
                    allowed=allowed,
                    vs=vs, vc=vc,
                    n_frames=n,
                    total_frames=T_total,
                    jitter=int(self.train_jitter),
                    Kwin=Kwin,
                    start_idx=1,
                    idx=idx,
                    forbidden=forbidden,
                )

        # -------------------------
        # VAL / TEST: simple sampling over FULL video
        #   val_t_query == -1  -> all frames (1..n-1)
        #   val_t_query  >  0  -> evenly spaced val_t_query frames
        # -------------------------
        else:
            T_total = int(self.val_t_query)

            if T_total < 0:
                q_idx = list(range(1, n))  # entire video except frame 0
            else:
                q_idx = self._evenly_spaced_indices(n_frames=n, t_query=T_total)
                # (q_idx already lies in [1..n-1] and is sorted)

                # enforce fixed length for collate (rare corner cases)
                if len(q_idx) == 0:
                    q_idx = [1] if n > 1 else [0]
                if len(q_idx) < T_total:
                    q_idx = q_idx + [int(q_idx[-1])] * (T_total - len(q_idx))
                elif len(q_idx) > T_total:
                    q_idx = q_idx[:T_total]

        # =======================
        # (C) OPTIONAL SAFETY: enforce query disjoint from support after sampling
        # paste this RIGHT AFTER you finish building q_idx (end of query sampling block)
        # =======================
        sset = set(map(int, s_idx))
        q_idx = [int(i) for i in q_idx if int(i) not in sset]

        # keep fixed length if requested (train and val/test when val_t_query>0)
        target_q = None
        if self.split == "train" and int(self.train_t_query) > 0:
            target_q = int(self.train_t_query)
        elif self.split in {"val", "test"} and int(self.val_t_query) > 0:
            target_q = int(self.val_t_query)

        if target_q is not None:
            if len(q_idx) == 0:
                # fallback: pick something allowed >=1
                cand = [int(i) for i in allowed.tolist() if int(i) >= 1 and int(i) not in sset]
                if not cand:
                    cand = [int(allowed[-1])]
                q_idx = [cand[0]]
            if len(q_idx) < target_q:
                q_idx = q_idx + [int(q_idx[-1])] * (target_q - len(q_idx))
            elif len(q_idx) > target_q:
                q_idx = q_idx[:target_q]

        # -------------------------
        # Load supports
        # -------------------------
        sup_imgs, sup_msks = [], []
        for si in s_idx:
            sup_imgs.append(self._read_rgb(g.loc[int(si), "img"]))
            sup_msks.append(self._read_and_map_mask(g.loc[int(si), self.mask_col]))

        # -------------------------
        # Load queries
        # -------------------------
        q_imgs, q_msks = [], []
        for qi in q_idx:
            q_imgs.append(self._read_rgb(g.loc[int(qi), "img"]))
            q_msks.append(self._read_and_map_mask(g.loc[int(qi), self.mask_col]))

        # -------------------------
        # To torch
        # -------------------------
        sup_img_t = torch.stack(
            [torch.from_numpy(x).permute(2, 0, 1).float() / 255.0 for x in sup_imgs],
            dim=0,
        )  # (S*K,3,H,W) typically

        sup_msk_t = torch.stack(
            [torch.from_numpy(x.astype(np.int64)) for x in sup_msks],
            dim=0,
        )  # (S*K,H,W)

        q_imgs_t = torch.stack(
            [torch.from_numpy(x).permute(2, 0, 1).float() / 255.0 for x in q_imgs],
            dim=0,
        )  # (T*K,3,H,W) typically

        q_msks_t = torch.stack(
            [torch.from_numpy(x.astype(np.int64)) for x in q_msks],
            dim=0,
        )  # (T*K,H,W)

        # if getattr(self.cfg.train, "debug", False):
        #     print("ALLOWED size:", int(allowed.size), "allowed[:10]:", allowed[:10].tolist())
        #     print("Kwin:", Kwin)
        #     print("OUT support:", len(s_idx), s_idx[: min(20, len(s_idx))])
        #     print("OUT query  :", len(q_idx), q_idx[: min(20, len(q_idx))])


        if getattr(self.cfg.train, "debug", False) and idx == 0 and self.split == "train":
            allowed_dbg = self._get_allowed_indices(vs, vc, n)
            aset = set(map(int, allowed_dbg.tolist()))

            def _chunks(lst, k):
                return [lst[i:i+k] for i in range(0, len(lst), k)]

            print("\n================ DEBUG_ALLOWED ================")
            print("clip:", vs, vc, "n:", n)
            print("allowed_size:", len(allowed_dbg), "allowed[:20]:", allowed_dbg[:20].tolist())
            print("S_total:", int(self.s_support if self.split == "train" else 1), "Kwin:", int(self.allowed_kwin))
            print("T_total:", int(self.train_t_query if self.split == "train" else self.val_t_query), "Kwin:", int(self.allowed_kwin))

            bad_s = [i for i in s_idx if int(i) not in aset]
            bad_q = [i for i in q_idx if int(i) not in aset]
            print("support_not_in_allowed:", bad_s)
            print("query_not_in_allowed  :", bad_q)

            inter = sorted(set(map(int, s_idx)).intersection(set(map(int, q_idx))))
            print("support∩query:", inter)

            K = int(self.allowed_kwin)
            print("support_windows:", _chunks(list(map(int, s_idx)), K))
            print("query_windows  :", _chunks(list(map(int, q_idx)), K))

            def is_consecutive(win):
                return all(win[i+1] == win[i] + 1 for i in range(len(win)-1))

            if getattr(self.cfg.train, "allowed_strict_consecutive", True):
                sc_s = [is_consecutive(w) for w in _chunks(list(map(int, s_idx)), K)]
                sc_q = [is_consecutive(w) for w in _chunks(list(map(int, q_idx)), K)]
                print("strict_consecutive support windows:", sc_s)
                print("strict_consecutive query windows  :", sc_q)

            print("================================================\n")
        # ------------------------------------------------------------------------------


        return {
            "support_img": sup_img_t,         # (S*K,3,H,W)
            "support_mask": sup_msk_t,        # (S*K,H,W)
            "support_indices": s_idx,         # List[int] length S*K
            "query_imgs": q_imgs_t,           # (T*K,3,H,W) or variable if val_t_query<0
            "query_masks": q_msks_t,          # (T*K,H,W)
            "query_indices": q_idx,           # List[int]
            "video_src": vs,
            "video_clip": vc,
            "video_len": int(n),
        }

    # ----------------------- helpers -----------------------

    def _evenly_spaced_indices(self, n_frames: int, t_query: int) -> List[int]:
        if n_frames <= 1 or t_query <= 0:
            return []
        max_q = n_frames - 1
        t = min(int(t_query), max_q)

        xs = np.linspace(1, max_q, num=t, endpoint=True)
        idx = np.unique(np.rint(xs).astype(int))  # round, unique

        # if uniqueness collapsed (short clips), densify by adding neighbors
        if len(idx) < t:
            pool = np.arange(1, max_q + 1)
            missing = t - len(idx)
            extra = [p for p in pool if p not in set(idx)]
            idx = np.concatenate([idx, extra[:missing]])

        idx = np.sort(idx).tolist()
        return idx

    def _jitter_indices(self, idxs, n_frames, jitter, idx):
        max_q = n_frames - 1
        rng = self._get_rng(idx)
        out = []
        for i in idxs:
            j = i + rng.randint(-jitter, jitter)
            j = max(1, min(max_q, j))
            out.append(j)
        return sorted(set(out))

    def _jitter_indices_any(self, idxs, n_frames, jitter, lo, hi, idx):
        rng = self._get_rng(idx)
        out = []
        for i in idxs:
            j = int(i) + rng.randint(-jitter, jitter)
            j = max(int(lo), min(int(hi), j))
            out.append(j)
        return sorted(set(out))

    def _frame_sort_key(self, path: str) -> Tuple[int, str]:
        b = os.path.basename(path)
        m = self.FRAME_RE.search(b)
        return (int(m.group(1)), b) if m else (-1, b)

    def _read_rgb(self, path: str) -> np.ndarray:
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # resize only if explicitly provided
        rh = getattr(self.cfg.train, "resize_h", None)
        rw = getattr(self.cfg.train, "resize_w", None)

        if rh is not None and rw is not None:
            th, tw = int(rh), int(rw)
            if im.shape[0] != th or im.shape[1] != tw:
                im = self._resize_pad_fit(im, is_mask=False)

        return im

    def _read_and_map_mask(self, path: str) -> np.ndarray:
        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(path)

        # resize only if explicitly provided
        rh = getattr(self.cfg.train, "resize_h", None)
        rw = getattr(self.cfg.train, "resize_w", None)

        if rh is not None and rw is not None:
            th, tw = int(rh), int(rw)
            if m.shape[0] != th or m.shape[1] != tw:
                m = self._resize_pad_fit(m, is_mask=True)

        # ---- RGB masks (EndoVis)
        if m.ndim == 3 and self.rgb_to_id is not None:
            if m.shape[2] == 4:        # BGRA -> RGB
                rgb = cv2.cvtColor(m, cv2.COLOR_BGRA2RGB)
            else:                      # BGR  -> RGB
                rgb = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)

            out = np.full(rgb.shape[:2], self.ignore_index, dtype=np.uint8)

            key = ((rgb[..., 0].astype(np.int32) << 16) |
                   (rgb[..., 1].astype(np.int32) << 8)  |
                    rgb[..., 2].astype(np.int32))

            for (r, g, b), cid in self.rgb_to_id.items():
                k = (r << 16) | (g << 8) | b
                out[key == k] = np.uint8(cid)

            return out

        # ---- Grayscale masks (SAR / Cholec)
        gg = m[..., 0] if m.ndim == 3 else m
        gg = gg.astype(np.uint8)
        return self.lut[gg]

    def _stratified_indices(self, n_frames: int, t_query: int, bins: int = 3) -> List[int]:
        """
        Pick t_query indices from [1..n-1] with stratified coverage over time.
        """
        if n_frames <= 1 or t_query <= 0:
            return []
        max_q = n_frames - 1
        t = min(int(t_query), max_q)

        # split [1..max_q] into bins, sample ~t/bins per bin
        edges = np.linspace(1, max_q + 1, num=bins + 1, endpoint=True).astype(int)
        per = [t // bins] * bins
        for i in range(t % bins):
            per[i] += 1

        out = []
        for bi in range(bins):
            lo, hi = edges[bi], edges[bi + 1] - 1
            if lo > hi:
                continue
            k = min(per[bi], hi - lo + 1)
            xs = np.linspace(lo, hi, num=k, endpoint=True)
            idx = np.unique(np.rint(xs).astype(int)).tolist()
            out += idx

        out = sorted(set(out))
        # densify if uniqueness collapsed
        if len(out) < t:
            pool = [i for i in range(1, max_q + 1) if i not in set(out)]
            out += pool[: (t - len(out))]
        return sorted(out)[:t]

    def _resize_pad_fit(self, im: np.ndarray, is_mask: bool):
        """
        Fit within (target_h,target_w) preserving aspect ratio, then pad bottom/right.
        Returns resized+pad image (H,W[,C]).
        """
        th = int(getattr(self.cfg.train, "resize_h", 480))
        tw = int(getattr(self.cfg.train, "resize_w", 854))

        oh, ow = im.shape[:2]
        scale = min(tw / ow, th / oh)
        nh = max(1, int(round(oh * scale)))
        nw = max(1, int(round(ow * scale)))

        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        im_r = cv2.resize(im, (nw, nh), interpolation=interp)

        pad_h = th - nh
        pad_w = tw - nw
        if pad_h < 0 or pad_w < 0:
            raise ValueError(f"resize_pad_fit produced larger than target: {(nh,nw)} > {(th,tw)}")

        # pad masks with ignore_index, images with 0
        pad_val = self.ignore_index if is_mask else 0

        if im_r.ndim == 2:
            out = np.pad(
                im_r,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=pad_val,
            )
        else:
            out = np.pad(
                im_r,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=pad_val,
            )
        return out


def svss_collate(batch):
    """
    NOTE:
      With allowed windows, shapes are typically:
        support_img  : (B, S*K, 3, H, W)
        support_mask : (B, S*K, H, W)
        query_imgs   : (B, T*K, 3, H, W)
        query_masks  : (B, T*K, H, W)

      If cfg.val.val_t_query < 0, query length can vary across items.
      In that case, either use batch_size=1 for val/test OR set val_t_query > 0.
    """
    out = {}
    out["support_img"]  = torch.stack([b["support_img"] for b in batch], dim=0)
    out["support_mask"] = torch.stack([b["support_mask"] for b in batch], dim=0)
    out["query_imgs"]   = torch.stack([b["query_imgs"] for b in batch], dim=0)
    out["query_masks"]  = torch.stack([b["query_masks"] for b in batch], dim=0)

    # indices are fixed-length lists in train and in val/test when val_t_query > 0
    out["support_indices"] = torch.tensor([b["support_indices"] for b in batch], dtype=torch.long)
    out["query_indices"]   = torch.tensor([b["query_indices"] for b in batch], dtype=torch.long)

    out["video_src"]  = [b["video_src"] for b in batch]
    out["video_clip"] = [b["video_clip"] for b in batch]
    out["video_len"]  = torch.tensor([b["video_len"] for b in batch], dtype=torch.long)
    return out