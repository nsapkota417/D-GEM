# =========================
# COPY-PASTE PATCHES
# =========================
# 1) Replace your SVSSDataset __init__ group construction block with the "FAST" one below
# 2) Replace your __getitem__ with the "FAST" one below (only uses lists, no pandas loc/iloc)
# 3) Replace svss_collate with "FAST" one below (stacks tensors directly)
# 4) Add worker_init_fn + DataLoader args (bottom)

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

        # multi-support controls
        self.s_support = int(getattr(self.cfg.train, "s_support", 1))
        self.support_jitter = int(getattr(self.cfg.train, "support_jitter", 0))
        self.epoch_salt = int(getattr(self.cfg.train, "epoch_salt", 0))

        # allowed-frame sampling
        self.allowed_max_items = int(getattr(self.cfg.train, "allowed_max_items", 50))
        self.allowed_seed = int(getattr(self.cfg.train, "allowed_seed", 1337))
        self.allowed_kwin = int(getattr(self.cfg.train, "allowed_kwin", 2))
        self.allowed_strict_consecutive = bool(getattr(self.cfg.train, "allowed_strict_consecutive", True))

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

        # --- build LUT for code->class mapping
        self.lut = np.full(256, self.ignore_index, dtype=np.uint8)
        for k in range(self.cfg.data.num_class):
            self.lut[k] = np.uint8(k)
        self.lut[self.ignore_index] = np.uint8(self.ignore_index)

        # -------- RGB label mapping (EndoVis) --------
        self.rgb_to_id = None
        label_json = getattr(self.cfg.data, "label_json", None)
        code_to_class = getattr(self.cfg.data, "code_to_class", None)

        if label_json:
            with open(label_json, "r") as f:
                items = json.load(f)
            self.rgb_to_id = {tuple(map(int, it["color"])): int(it["classid"]) for it in items}
        elif code_to_class is not None:
            self.lut = np.full(256, self.ignore_index, dtype=np.uint8)
            for k, v in dict(code_to_class).items():
                ki = int(k, 0) if isinstance(k, str) else int(k)
                vi = int(v)
                if 0 <= ki <= 255:
                    self.lut[ki] = np.uint8(vi)

        # ============================================================
        # FAST GROUPING: build per-clip lists ONCE (no pandas in __getitem__)
        # ============================================================
        self.clip_data = {}  # (vs,vc) -> dict(img_list, mask_list, n)
        for (vs, vc), g in self.df.groupby(["video_src", "video_clip"], sort=True):
            # sort by frame id
            g = g.copy()
            g["_k"] = g["img"].apply(self._frame_sort_key)
            g = g.sort_values("_k").drop(columns="_k").reset_index(drop=True)

            self.clip_data[(vs, vc)] = {
                "img": g["img"].tolist(),
                "mask": g[self.mask_col].tolist(),
                "labeled": [self._has_mask(path) for path in g[self.mask_col]],
                "n": int(len(g)),
            }

        self.clips = list(self.clip_data.keys())
        self.num_clips = len(self.clips)
        self.num_frames = sum(cd["n"] for cd in self.clip_data.values())

        # print quick stats
        if self.split in {"train", "val", "test"}:
            if self.split == "train":
                q_per_clip = int(self.train_t_query)
                s_per_clip = int(self.s_support)
            else:
                q_per_clip = int(self.val_t_query)
                s_per_clip = 1

            per_clip_queries = []
            for cd in self.clip_data.values():
                n = cd["n"]
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
            )

    def __len__(self):
        return len(self.clips)

    @staticmethod
    def _has_mask(path) -> bool:
        """Return whether a manifest entry refers to an available label mask.

        Sparse-video manifests may leave ``mask`` blank (or use ``-``) for
        unannotated frames.  Paths are checked when the mask is actually read,
        so remote or lazily mounted datasets remain supported.
        """
        if path is None or (isinstance(path, float) and np.isnan(path)):
            return False
        return str(path).strip() not in {"", "-", "none", "None", "nan", "NaN"}

    @staticmethod
    def _repeat_to_length(indices: List[int], length: int) -> List[int]:
        if not indices:
            return []
        return [int(indices[i % len(indices)]) for i in range(length)]

    def _get_rng(self, idx: int):
        worker_info = torch.utils.data.get_worker_info()
        seed = torch.initial_seed() if worker_info is None else worker_info.seed
        return random.Random(seed + idx)

    # -----------------------------
    # Allowed-set helpers
    # -----------------------------
    def _video_key(self, vs: str, vc: str) -> str:
        return f"{vs}::{vc}"

    def _stable_video_seed(self, video_key: str) -> int:
        h = hashlib.sha1((video_key + f"__{self.allowed_seed}").encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def _get_allowed_indices(self, vs: str, vc: str, n_frames: int) -> np.ndarray:
        vkey = self._video_key(vs, vc)
        if vkey in self._allowed_cache:
            return self._allowed_cache[vkey]

        N = int(n_frames)
        k = min(max(1, int(self.allowed_max_items)), N)

        rng = np.random.RandomState(self._stable_video_seed(vkey))
        arr = rng.choice(np.arange(N, dtype=np.int32), size=k, replace=False)
        arr = np.unique(arr)
        arr.sort()

        if 0 not in arr:
            arr = np.unique(np.concatenate([arr, np.asarray([0], dtype=np.int32)]))
            arr.sort()

        arr = arr[(arr >= 0) & (arr < N)]
        if arr.size > k:
            others = arr[arr != 0]
            others = others[: max(0, k - 1)]
            arr = np.unique(np.concatenate([np.asarray([0], dtype=np.int32), others]))
            arr.sort()

        self._allowed_cache[vkey] = arr
        # cache the set once (big win vs rebuilding every __getitem__)
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

        # candidates in-bin
        if L < R:
            sub = allowed[L:R]
            if forbidden:
                cand = [int(x) for x in sub if int(x) not in forbidden]
            else:
                cand = sub.astype(int).tolist()
            if cand:
                if rng is None:
                    return cand[len(cand) // 2]
                return rng.choice(cand)

        # nearest globally
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
        forbidden = forbidden or set()
        K = max(1, int(K))
        N = int(n_frames)
        vkey = self._video_key(vs, vc)

        out: List[int] = []
        aset = self._allowed_set_cache.get(vkey, set(map(int, allowed.tolist())))

        # (1) strict consecutive ids but allowed
        if self.allowed_strict_consecutive:
            for t in range(K):
                j = anchor + t
                if 0 <= j < N and (j in aset) and (j not in forbidden):
                    out.append(int(j))
                else:
                    break
            if len(out) < K:
                t = 1
                while len(out) < K and (anchor - t) >= 0:
                    j = anchor - t
                    if (j in aset) and (j not in forbidden):
                        out = [int(j)] + out
                    t += 1

        # (2) allowed-order neighbors
        if len(out) < K:
            pos = int(np.searchsorted(allowed, anchor, side="left"))
            pos = max(0, min(pos, allowed.size - 1))

            if int(allowed[pos]) != int(anchor):
                cand = [pos] + ([pos - 1] if pos - 1 >= 0 else [])
                best = cand[0]
                bestd = abs(int(allowed[best]) - anchor)
                for q in cand[1:]:
                    d = abs(int(allowed[q]) - anchor)
                    if d < bestd:
                        bestd, best = d, q
                pos = best

            out2 = []
            left, right = pos, pos + 1
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

            out2 = sorted(set(out2))
            if len(out2) > K:
                out2.sort(key=lambda x: (abs(x - anchor), x))
                out2 = sorted(out2[:K])
            if len(out2) >= len(out):
                out = out2

        # (3) pad/crop
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
        forbidden = forbidden or set()
        N = int(n_frames)
        num_bins = max(1, int(num_bins))
        Kwin = max(1, int(Kwin))
        start_idx = int(start_idx)
        end_idx = N - 1
        if start_idx > end_idx:
            start_idx = end_idx

        edges = np.linspace(start_idx, end_idx + 1, num=num_bins + 1, endpoint=True).astype(int)
        rng = self._get_rng(idx + 100000 * int(self.epoch_salt))

        vkey = self._video_key(vs, vc)
        allowed_set = self._allowed_set_cache.get(vkey, set(map(int, allowed.tolist())))

        out_flat: List[int] = []
        for b in range(num_bins):
            lo = int(edges[b])
            hi = int(edges[b + 1] - 1)
            if hi < lo:
                hi = lo

            c = (lo + hi) // 2
            if int(jitter) > 0 and hi > lo:
                j = rng.randint(-int(jitter), int(jitter))
                c = int(np.clip(c + j, lo, hi))

            anchor = self._pick_allowed_anchor_in_bin(allowed, lo, hi, c, forbidden, rng)
            win = self._expand_window(allowed, vs, vc, int(anchor), Kwin, N, forbidden)

            clean = []
            for w in win:
                w = int(max(0, min(N - 1, int(w))))
                if w not in allowed_set:
                    p = int(np.searchsorted(allowed, w, side="left"))
                    if p <= 0:
                        w = int(allowed[0])
                    elif p >= allowed.size:
                        w = int(allowed[-1])
                    else:
                        a0, a1 = int(allowed[p - 1]), int(allowed[p])
                        w = a0 if abs(a0 - w) <= abs(a1 - w) else a1
                if w in forbidden:
                    w = self._pick_allowed_anchor_in_bin(allowed, 0, N - 1, w, forbidden)
                clean.append(int(w))

            if len(clean) < Kwin:
                clean = clean + [int(clean[-1])] * (Kwin - len(clean))
            elif len(clean) > Kwin:
                clean = clean[:Kwin]

            out_flat.extend(clean)

        need = num_bins * Kwin
        if len(out_flat) < need:
            out_flat = out_flat + [int(out_flat[-1])] * (need - len(out_flat))
        elif len(out_flat) > need:
            out_flat = out_flat[:need]

        return out_flat

    def _sample_total_frames_from_allowed(
        self,
        allowed: np.ndarray,
        vs: str,
        vc: str,
        n_frames: int,
        total_frames: int,
        jitter: int,
        Kwin: int,
        start_idx: int,
        idx: int,
        forbidden: Optional[set] = None,
    ) -> List[int]:
        forbidden = forbidden or set()
        total_frames = int(total_frames)
        if total_frames <= 0:
            return []
        Kwin = max(1, int(Kwin))
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

        if len(flat) >= total_frames:
            return flat[:total_frames]
        pad = [int(flat[-1])] * (total_frames - len(flat)) if flat else [start_idx] * total_frames
        return flat + pad

    # -----------------------------
    # __getitem__ (FAST: no pandas access)
    # -----------------------------
    def __getitem__(self, idx):
        vs, vc = self.clips[idx]
        cd = self.clip_data[(vs, vc)]
        img_list = cd["img"]
        msk_list = cd["mask"]
        labeled = cd["labeled"]
        n = cd["n"]
        if n < 2:
            raise RuntimeError(f"{vs}/{vc} has <2 frames")

        labeled_idx = [i for i, has_label in enumerate(labeled) if has_label]
        if not labeled_idx:
            raise RuntimeError(
                f"{vs}/{vc} has no annotated frames. Add at least one mask "
                f"path in the '{self.mask_col}' column."
            )

        # stream toggles
        stream_train = bool(getattr(self.cfg.train, "use_stream", False))
        stream_val   = bool(getattr(self.cfg.val,   "use_stream", False))
        stream_query = (self.split == "train" and stream_train) or (self.split in {"val", "test"} and stream_val)

        dbg = bool(self.cfg.train.debug[0])
        dbg_first_only = bool(getattr(self.cfg.train, "debug_first_only", True))
        do_dbg = dbg and ((not dbg_first_only) or (idx == 0))

        # allowed
        allowed = self._get_allowed_indices(vs, vc, n)
        vkey = self._video_key(vs, vc)
        allowed_set = self._allowed_set_cache[vkey]
        Kwin = int(self.allowed_kwin)

        # Support frames must have ground-truth masks.  In sparse settings they
        # are selected from all annotated frames, rather than from the full
        # video timeline.
        if self.split == "train":
            S_total = max(1, int(self.s_support))
            rng = self._get_rng(idx + 100000 * int(self.epoch_salt))
            ordered_labels = list(labeled_idx)
            rng.shuffle(ordered_labels)
            s_idx = self._repeat_to_length(ordered_labels, S_total)
        else:
            S_total = max(1, int(self.s_support))
            s_idx = self._repeat_to_length(labeled_idx, S_total)

        # query
        if self.split == "train":
            T_total = int(self.train_t_query)
            support_set = set(map(int, s_idx))
            query_candidates = [i for i in labeled_idx if i not in support_set]
            if not query_candidates:
                raise RuntimeError(
                    f"{vs}/{vc} needs at least two annotated frames for "
                    "training: one support frame and one supervised query frame."
                )
            if T_total < 0:
                q_idx = query_candidates
            else:
                rng = self._get_rng(idx + 200000 * int(self.epoch_salt))
                q_idx = self._repeat_to_length(
                    rng.sample(query_candidates, k=min(T_total, len(query_candidates))),
                    T_total,
                )
        else:
            T_total = int(self.val_t_query)
            query_candidates = [i for i in labeled_idx if i not in set(map(int, s_idx))]
            if T_total < 0:
                q_idx = query_candidates
            else:
                q_idx = self._repeat_to_length(query_candidates, T_total)

        # enforce disjoint + fixed length
        sset = set(map(int, s_idx))
        q_idx = [int(i) for i in q_idx if int(i) not in sset]

        target_q = None
        if self.split == "train" and int(self.train_t_query) > 0:
            target_q = int(self.train_t_query)
        elif self.split in {"val", "test"} and int(self.val_t_query) > 0:
            target_q = int(self.val_t_query)

        if target_q is not None:
            if len(q_idx) == 0:
                cand = [int(i) for i in labeled_idx if int(i) not in sset]
                if not cand:
                    raise RuntimeError(
                        f"{vs}/{vc} has no annotated query frame after selecting supports."
                    )
                q_idx = [cand[0]]
            if len(q_idx) < target_q:
                q_idx = q_idx + [int(q_idx[-1])] * (target_q - len(q_idx))
            elif len(q_idx) > target_q:
                q_idx = q_idx[:target_q]

        # load support tensors
        sup_imgs, sup_msks = [], []
        for si in s_idx:
            p_img = img_list[int(si)]
            p_msk = msk_list[int(si)]
            sup_imgs.append(self._read_rgb(p_img))
            sup_msks.append(self._read_and_map_mask(p_msk))

        sup_img_t = torch.stack(
            [torch.from_numpy(x).permute(2, 0, 1).float().div_(255.0) for x in sup_imgs], dim=0
        )
        sup_msk_t = torch.stack(
            [torch.from_numpy(x.astype(np.int64)) for x in sup_msks], dim=0
        )

        # streaming: return paths only for query rollout
        if stream_query:
            # ----------------------------
            # SPLIT-AWARE rollout controls
            # ----------------------------
            if self.split == "train":
                roll_cfg = self.cfg.train
            else:  # val / test
                roll_cfg = self.cfg.val

            T_roll = int(getattr(roll_cfg, "rollout_len", -1))
            mode   = str(getattr(roll_cfg, "rollout_mode", "full"))

            # ----------------------------
            # Rollout index selection
            # ----------------------------
            if mode == "full" or T_roll < 0:
                query_roll_idx = list(range(1, n))

            elif mode == "window":
                T_roll = max(1, min(T_roll, n - 1))

                # split-aware start_idx (default=1)
                if self.split == "train":
                    start_idx = int(getattr(self.cfg.train, "rollout_start_idx", 1))
                else:  # val/test
                    start_idx = int(getattr(self.cfg.val, "rollout_start_idx", 1))

                # clamp
                start_idx = max(1, min(start_idx, n - T_roll))

                if self.split in {"val", "test"}:
                    # deterministic val: always same continuous window
                    start = start_idx
                else:
                    # train: random window, but respecting start_idx floor
                    rng = self._get_rng(idx + 100000 * int(self.epoch_salt))
                    hi = (n - T_roll)
                    start = rng.randint(start_idx, hi) if hi > start_idx else start_idx

                query_roll_idx = list(range(start, start + T_roll))

            elif mode == "allowed":
                roll = [int(i) for i in allowed.tolist() if int(i) >= 1]
                if T_roll > 0 and len(roll) > T_roll:
                    rng = self._get_rng(idx + 100000 * int(self.epoch_salt))
                    start = rng.randint(0, len(roll) - T_roll)
                    roll = roll[start:start + T_roll]
                query_roll_idx = roll

            else:
                raise ValueError(f"unknown rollout_mode={mode}")

            # ----------------------------
            # Supervised query frames MUST be inside rollout
            # ----------------------------
            if self.split == "train":
                q_sup = [int(i) for i in q_idx if int(i) in set(query_roll_idx)]
                if not q_sup:
                    raise RuntimeError(
                        f"{vs}/{vc} selected no annotated query frame in the "
                        "rollout. Use rollout_mode: full or increase rollout_len."
                    )

            else:  # val/test
                q_sup = [int(i) for i in q_idx if int(i) in set(query_roll_idx)]

            q_img_paths = [str(img_list[i]) for i in query_roll_idx]
            q_msk_paths = [str(msk_list[i]) for i in query_roll_idx]

            return {
                "support_img": sup_img_t,
                "support_mask": sup_msk_t,

                "support_indices": torch.as_tensor(list(map(int, s_idx)), dtype=torch.long),
                "query_roll_indices": torch.as_tensor(query_roll_idx, dtype=torch.long),
                "query_indices": torch.as_tensor(q_sup, dtype=torch.long),

                "query_img_paths": q_img_paths,
                "query_mask_paths": q_msk_paths,

                "video_src": vs,
                "video_clip": vc,
                "video_len": int(n),
            }

        # non-streaming: load query tensors
        q_imgs, q_msks = [], []
        for qi in q_idx:
            q_imgs.append(self._read_rgb(img_list[int(qi)]))
            q_msks.append(self._read_and_map_mask(msk_list[int(qi)]))

        q_imgs_t = torch.stack(
            [torch.from_numpy(x).permute(2, 0, 1).float().div_(255.0) for x in q_imgs], dim=0
        )
        q_msks_t = torch.stack(
            [torch.from_numpy(x.astype(np.int64)) for x in q_msks], dim=0
        )

        return {
            "support_img": sup_img_t,
            "support_mask": sup_msk_t,
            "support_indices": torch.as_tensor(list(map(int, s_idx)), dtype=torch.long),

            "query_imgs": q_imgs_t,
            "query_masks": q_msks_t,
            "query_indices": torch.as_tensor(list(map(int, q_idx)), dtype=torch.long),

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
        idx = np.unique(np.rint(xs).astype(int))
        if len(idx) < t:
            pool = np.arange(1, max_q + 1)
            missing = t - len(idx)
            extra = [p for p in pool if p not in set(idx)]
            idx = np.concatenate([idx, extra[:missing]])
        return np.sort(idx).tolist()

    def _frame_sort_key(self, path: str) -> Tuple[int, str]:
        b = os.path.basename(path)
        m = self.FRAME_RE.search(b)
        return (int(m.group(1)), b) if m else (-1, b)

    def _read_rgb(self, path: str) -> np.ndarray:
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

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

        rh = getattr(self.cfg.train, "resize_h", None)
        rw = getattr(self.cfg.train, "resize_w", None)
        if rh is not None and rw is not None:
            th, tw = int(rh), int(rw)
            if m.shape[0] != th or m.shape[1] != tw:
                m = self._resize_pad_fit(m, is_mask=True)

        if m.ndim == 3 and self.rgb_to_id is not None:
            rgb = cv2.cvtColor(m, cv2.COLOR_BGRA2RGB) if m.shape[2] == 4 else cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
            out = np.full(rgb.shape[:2], self.ignore_index, dtype=np.uint8)
            key = ((rgb[..., 0].astype(np.int32) << 16) |
                   (rgb[..., 1].astype(np.int32) << 8) |
                    rgb[..., 2].astype(np.int32))
            for (r, g, b), cid in self.rgb_to_id.items():
                k = (r << 16) | (g << 8) | b
                out[key == k] = np.uint8(cid)
            return out

        gg = m[..., 0] if m.ndim == 3 else m
        gg = gg.astype(np.uint8)
        return self.lut[gg]

    def _resize_pad_fit(self, im: np.ndarray, is_mask: bool):
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
        pad_val = self.ignore_index if is_mask else 0
        if im_r.ndim == 2:
            return np.pad(im_r, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=pad_val)
        return np.pad(im_r, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=pad_val)


# =========================
# FAST COLLATE (NON-STREAM)
# =========================
def svss_collate(batch):
    out = {}
    out["support_img"]  = torch.stack([b["support_img"] for b in batch], dim=0)
    out["support_mask"] = torch.stack([b["support_mask"] for b in batch], dim=0)
    out["query_imgs"]   = torch.stack([b["query_imgs"] for b in batch], dim=0)
    out["query_masks"]  = torch.stack([b["query_masks"] for b in batch], dim=0)

    # indices are already tensors in dataset
    out["support_indices"] = torch.stack([b["support_indices"] for b in batch], dim=0)
    out["query_indices"]   = torch.stack([b["query_indices"]   for b in batch], dim=0)

    out["video_src"]  = [b["video_src"] for b in batch]
    out["video_clip"] = [b["video_clip"] for b in batch]
    out["video_len"]  = torch.as_tensor([b["video_len"] for b in batch], dtype=torch.long)
    return out


def svss_collate_stream(batch):
    if len(batch) == 0:
        return {}

    need = ["query_img_paths", "query_mask_paths", "query_roll_indices",
            "support_img", "support_mask", "support_indices", "query_indices"]
    missing = [k for k in need if k not in batch[0]]
    if missing:
        keys0 = list(batch[0].keys())
        raise KeyError(
            f"svss_collate_stream missing keys={missing}. sample keys={keys0}. "
            f"Fix dataset __getitem__ for streaming."
        )

    out = {}

    # list-of-lists for paths
    out["query_img_paths"]  = [b["query_img_paths"]  for b in batch]
    out["query_mask_paths"] = [b["query_mask_paths"] for b in batch]

    # stack tensors
    out["query_roll_indices"] = torch.stack([b["query_roll_indices"] for b in batch], dim=0)  # (B,T_rollout)
    out["query_indices"]      = torch.stack([b["query_indices"]      for b in batch], dim=0)  # (B,t_query)

    out["support_img"]        = torch.stack([b["support_img"]        for b in batch], dim=0)  # (B,S,3,H,W)
    out["support_mask"]       = torch.stack([b["support_mask"]       for b in batch], dim=0)  # (B,S,H,W)
    out["support_indices"]    = torch.stack([b["support_indices"]    for b in batch], dim=0)  # (B,S)

    # pass-through meta (optional)
    out["video_src"]  = [b.get("video_src", "") for b in batch]
    out["video_clip"] = [b.get("video_clip", "") for b in batch]
    out["video_len"]  = torch.as_tensor([b.get("video_len", 0) for b in batch], dtype=torch.long)

    return out

# =========================
# OPENCV WORKER INIT (KEY)
# =========================
def worker_init_fn(_):
    import cv2
    cv2.setNumThreads(0)          # prevents oversubscription
    cv2.ocl.setUseOpenCL(False)   # optional


# =========================
# DATALOADER RECOMMENDED
# =========================
# loader = DataLoader(
#     dataset,
#     batch_size=BS,
#     shuffle=(split=="train"),
#     num_workers=8,
#     pin_memory=True,
#     persistent_workers=True,
#     prefetch_factor=4,
#     worker_init_fn=worker_init_fn,
#     collate_fn=svss_collate,  # or svss_collate_stream
# )
