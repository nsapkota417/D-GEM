# LAST WORKING VERSION THAT SUPPORTS BIN+JITTER YIELDING 5-10% improvement 


# svss_dataset.py
# Semi-supervised Video Semantic Segmentation (SVSS) dataset (no resizing)
#
# Train mode:
#   supports = S evenly spaced frames from 0..N-1 (optional jitter)   ✅ NEW
#   query    = T evenly spaced frames from 1..N-1 (optional jitter)
#
# Val/Test mode:
#   - default: full video (all frames 1..N-1), no jitter
#   - optional: use val_t_query > 0 to evaluate on evenly spaced subset (no jitter)
#
# Returns:
#   support_img:      (S,3,H,W)   float
#   support_mask:     (S,H,W)     long
#   support_indices:  List[int]   (len S)  frame indices in original clip
#   query_imgs:       (T,3,H,W)   float
#   query_masks:      (T,H,W)     long
#   query_indices:    List[int]   (len T)  frame indices in original clip
#
# Notes:
#   - All images/masks in a clip have identical (H,W)
#   - CSV has absolute paths
#   - mask_col points to the semantic mask to use (recommended: "watershed_mask")

import os
import re
import random
from typing import List, Tuple, Optional

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
        self.s_support = int(getattr(self.cfg.train, "s_support", 1))           # number of support frames
        self.support_jitter = int(getattr(self.cfg.train, "support_jitter", 0)) # optional jitter for supports

        # val/test sampling controls
        if val_t_query is None:
            self.val_t_query = int(getattr(self.cfg.val, "val_t_query", -1))
        else:
            self.val_t_query = int(val_t_query)

        # RNG for optional jitter (train only)
        # self.rng = random.Random(int(getattr(self.cfg.experiment, "seed", 0)))

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

        # import IPython; IPython.embed()
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
            tq2use = self.val_t_query
            if self.split == "train":
                tq2use = self.train_t_query

            per_clip_queries = []
            for g in self.groups.values():
                n = len(g)
                if n < 2:
                    per_clip_queries.append(0)
                    continue

                if tq2use < 0:
                    q = n - 1
                else:
                    q = min(tq2use, n - 1)
                per_clip_queries.append(q)

            support_per_clip = max(1, int(self.s_support)) if self.split == "train" else 1
            avg_q_per_clip = float(np.mean(per_clip_queries)) if len(per_clip_queries) else 0.0
            total_query = int(sum(per_clip_queries))
            total_support = int(self.num_clips * support_per_clip)
            print(
                f"  {self.split:<6} sampling | "
                f"# clips = {self.num_clips:>4} | "
                f"query/clip = {avg_q_per_clip:>6.1f} | "
                f"support/clip = {support_per_clip:>2} | "
                f"total query = {total_query:>5} "
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

    def __getitem__(self, idx):
        vs, vc = self.clips[idx]
        g = self.groups[(vs, vc)]
        n = len(g)
        if n < 2:
            raise RuntimeError(f"{vs}/{vc} has <2 frames")

        # -------------------------
        # Support indices (NEW)
        # -------------------------
        if self.split == "train":
            S = max(1, int(self.s_support))
            if S == 1:
                s_idx = [0]
            else:
                xs = np.linspace(0, n - 1, num=S, endpoint=True)
                s_idx = np.unique(np.rint(xs).astype(int)).tolist()
                if 0 not in s_idx:
                    s_idx = [0] + s_idx
                s_idx = sorted(s_idx)

                # densify if uniqueness collapsed (short clips)
                if len(s_idx) < S:
                    pool = np.arange(0, n)
                    missing = S - len(s_idx)
                    extra = [p for p in pool if p not in set(s_idx)]
                    s_idx = (s_idx + extra[:missing])
                s_idx = sorted(s_idx)[:S]

                # optional jitter (train only)
                if int(self.support_jitter) > 0:
                    s_idx = self._jitter_indices_any(s_idx, n_frames=n, jitter=int(self.support_jitter), lo=0, hi=n - 1, idx=idx)
        else:
            # keep classic behavior for val/test: only first frame support
            s_idx = [0]

        # Load supports
        sup_imgs, sup_msks = [], []
        for si in s_idx:
            sup_imgs.append(self._read_rgb(g.loc[si, "img"]))
            sup_msks.append(self._read_and_map_mask(g.loc[si, self.mask_col]))

        # -------------------------
        # Query indices (existing)
        # -------------------------
        if self.split == "train":
            t_query = int(self.train_t_query)
            jitter = int(self.train_jitter)

            if t_query < 0:
                q_idx = list(range(1, n))
            else:
                # q_idx = self._evenly_spaced_indices(n, t_query)
                q_idx = self._stratified_indices(n, t_query, bins=5)
                if jitter > 0:
                    # q_idx = self._jitter_indices(q_idx, n, jitter)
                    q_idx = self._jitter_indices(q_idx, n, jitter, idx)


                    
        else:
            t_query = int(self.val_t_query)
            if t_query < 0:
                q_idx = list(range(1, n))
            else:
                q_idx = self._evenly_spaced_indices(n, t_query)

        # -------------------------
        # NEW: make query disjoint from support (avoid anchor/query overlap)
        # -------------------------
        s_set = set(s_idx)
        q_idx = [i for i in q_idx if i not in s_set]

        # If we removed some and we need to refill (train or val/test with t_query>0)
        # Refill from candidates in [1..n-1] excluding supports and current queries.
        if t_query is not None and int(t_query) > 0:
            need = int(t_query) - len(q_idx)
            if need > 0:
                # candidate pool excludes supports (and excludes 0 by construction)
                candidates = [i for i in range(1, n) if i not in s_set]
                # also exclude already-selected queries
                q_set = set(q_idx)
                candidates = [i for i in candidates if i not in q_set]

                if candidates:
                    # Prefer evenly spaced refill if we can, else random.
                    # (even spacing is nicer for temporal coverage)
                    if need <= len(candidates):
                        xs = np.linspace(0, len(candidates) - 1, num=need, endpoint=True)
                        pick = np.unique(np.rint(xs).astype(int)).tolist()
                        # densify if uniqueness collapsed
                        if len(pick) < need:
                            extra = [j for j in range(len(candidates)) if j not in set(pick)]
                            pick = (pick + extra[: (need - len(pick))])
                        q_idx = sorted(q_idx + [candidates[j] for j in pick[:need]])
                    else:
                        q_idx = sorted(q_idx + candidates)


        # Load queries
        q_imgs, q_msks = [], []
        for qi in q_idx:
            q_imgs.append(self._read_rgb(g.loc[qi, "img"]))
            q_msks.append(self._read_and_map_mask(g.loc[qi, self.mask_col]))

        # -------------------------
        # To torch
        # -------------------------
        sup_img_t = torch.stack(
            [torch.from_numpy(x).permute(2, 0, 1).float() / 255.0 for x in sup_imgs],
            dim=0,
        )  # (S,3,H,W)

        sup_msk_t = torch.stack(
            [torch.from_numpy(x.astype(np.int64)) for x in sup_msks],
            dim=0,
        )  # (S,H,W)

        q_imgs_t = torch.stack(
            [torch.from_numpy(x).permute(2, 0, 1).float() / 255.0 for x in q_imgs],
            dim=0,
        )  # (T,3,H,W)

        q_msks_t = torch.stack(
            [torch.from_numpy(x.astype(np.int64)) for x in q_msks],
            dim=0,
        )  # (T,H,W)

        # if self.cfg.train.debug:
        if idx==0:
            print(f"\n video: {vc} | {self.split}")
            print("REQ s_support:", self.s_support, "OUT:", len(s_idx), s_idx)
            print("REQ t_query:", self.train_t_query, "OUT:", len(q_idx), q_idx)
            print("support_mask uniq per support:",
                [torch.unique(m).cpu().tolist()[:10] for m in sup_msk_t] if isinstance(sup_msk_t, list) else "tensor")

        # print(q_idx)
        return {
            "support_img": sup_img_t,         # (S,3,H,W)
            "support_mask": sup_msk_t,        # (S,H,W)
            "support_indices": s_idx,         # List[int]
            "query_imgs": q_imgs_t,           # (T,3,H,W)
            "query_masks": q_msks_t,          # (T,H,W)
            "query_indices": q_idx,           # List[int]
            "video_src": vs,
            "video_clip": vc,
            "video_len": int(n),   # total frames in the original clip
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

    # def _jitter_indices(self, idxs: List[int], n_frames: int, jitter: int) -> List[int]:
    def _jitter_indices(self, idxs, n_frames, jitter, idx):
        max_q = n_frames - 1
        rng = self._get_rng(idx)
        out = []
        for i in idxs:
            j = i + rng.randint(-jitter, jitter)
            j = max(1, min(max_q, j))
            out.append(j)
        return sorted(set(out))

    # def _jitter_indices_any(self, idxs: List[int], n_frames: int, jitter: int, lo: int, hi: int) -> List[int]:
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

        # import IPython; IPython.embed()

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
        g = m[..., 0] if m.ndim == 3 else m
        g = g.astype(np.uint8)
        return self.lut[g]

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

        # 🔑 critical fix: pad masks with ignore_index, images with 0
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
    # ---- one-time debug
    # if not getattr(svss_collate, "_dbg_printed", False):
    #     svss_collate._dbg_printed = True
    #     print("single item support_img shape:", batch[0]["support_img"].shape)
    #     print("single item support_mask shape:", batch[0]["support_mask"].shape)
    #     print("single item support_indices:", batch[0]["support_indices"])
    #     print("single item support_mask uniq:", torch.unique(batch[0]["support_mask"]).tolist()[:30])

    out = {}
    out["support_img"]  = torch.stack([b["support_img"] for b in batch], dim=0)   # (B,S,3,H,W)
    out["support_mask"] = torch.stack([b["support_mask"] for b in batch], dim=0)  # (B,S,H,W)
    out["query_imgs"]   = torch.stack([b["query_imgs"] for b in batch], dim=0)    # (B,T,3,H,W)
    out["query_masks"]  = torch.stack([b["query_masks"] for b in batch], dim=0)   # (B,T,H,W)

    out["support_indices"] = torch.tensor([b["support_indices"] for b in batch], dtype=torch.long)  # (B,S)
    out["query_indices"]   = torch.tensor([b["query_indices"] for b in batch], dtype=torch.long)    # (B,T)

    out["video_src"]  = [b["video_src"] for b in batch]
    out["video_clip"] = [b["video_clip"] for b in batch]
    out["video_len"] = torch.tensor([b["video_len"] for b in batch], dtype=torch.long)  # (B,)
    return out
