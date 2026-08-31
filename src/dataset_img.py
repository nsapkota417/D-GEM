import os
import cv2
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ImageSegDataset(Dataset):
    def __init__(self, cfg, df: pd.DataFrame):
        self.cfg = cfg
        self.df = df.reset_index(drop=True)

        self.img_col = str(self.cfg.data.img_col)
        self.mask_col = str(self.cfg.data.mask_col)

        # in __init__
        self.modality = self.cfg.data.modality

        if self.modality:
            if self.modality == "swir_img":
                self.img_cols = ["swir_img"]
                self.mask_col = "swir_mask"
            elif self.modality == "wl_img":
                self.img_cols = ["wl_img"]
                self.mask_col = "wl_mask"
            elif self.modality == "both_on_wl_img":
                self.img_cols = ["swir_img", "wl_img"]
                self.mask_col = "wl_mask"
            elif self.modality == "both_on_swir_img":
                self.img_cols = ["swir_img", "wl_img"]
                self.mask_col = "swir_mask"
        else:
            self.img_cols = [self.cfg.data.img_col]
            self.mask_col = self.cfg.data.mask_col

        self.ignore_index = self.cfg.data.ignore_index

        required = set(self.img_cols + [self.mask_col])
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # grayscale code -> class LUT
        self.lut = np.full(256, self.ignore_index, dtype=np.uint8)
        for k in range(self.cfg.data.num_class):
            self.lut[k] = np.uint8(k)
        self.lut[self.ignore_index] = np.uint8(self.ignore_index)

        # optional RGB label mapping
        self.rgb_to_id = None
        label_json = getattr(self.cfg.data, "label_json", None)
        code_to_class = getattr(self.cfg.data, "code_to_class", None)

        if label_json:
            with open(label_json, "r") as f:
                items = json.load(f)
            self.rgb_to_id = {
                tuple(map(int, it["color"])): int(it["classid"])
                for it in items
            }
        elif code_to_class is not None:
            self.lut = np.full(256, self.ignore_index, dtype=np.uint8)
            for k, v in dict(code_to_class).items():
                ki = int(k, 0) if isinstance(k, str) else int(k)
                vi = int(v)
                if 0 <= ki <= 255:
                    self.lut[ki] = np.uint8(vi)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_paths = [row[c] for c in self.img_cols]
        mask_path = row[self.mask_col]

        # Read one or both modalities
        # imgs = [self._read_rgb(p) for p in img_paths]

        imgs = []

        for col, path in zip(self.img_cols, img_paths):
            if col == "swir_img" and len(self.img_cols) == 1:
                # swir only: read as RGB -> 3 channels
                im = self._read_rgb(path)

            elif col == "swir_img":
                # swir + wl: keep swir as 1 channel
                im = self._read_gray(path)

            else:
                # wl image: always RGB -> 3 channels
                im = self._read_rgb(path)

            imgs.append(im)

        # swir only: 3ch
        # wl only: 3ch
        # swir + wl: 1ch + 3ch = 4ch
        img = np.concatenate(imgs, axis=2)

        mask = self._read_and_map_mask(mask_path)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
        mask_t = torch.from_numpy(mask.astype(np.int64))

        sample = {
            "image": img_t,
            "mask": mask_t,
            "img_path": img_paths[0] if len(img_paths) == 1 else img_paths,
            "mask_path": mask_path,
        }

        if "video_src" in row:
            sample["video_src"] = row["video_src"]
        if "video_clip" in row:
            sample["video_clip"] = row["video_clip"]
        if "rep" in row:
            sample["rep"] = row["rep"]

        return sample

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
                # im = self._resize_pad_fit(im, is_mask=False)
                im = self._center_crop(im)

        return im

    def _read_gray(self, path: str) -> np.ndarray:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(path)

        im = im[..., None]  # [H, W] -> [H, W, 1]

        rh = getattr(self.cfg.train, "resize_h", None)
        rw = getattr(self.cfg.train, "resize_w", None)
        if rh is not None and rw is not None:
            th, tw = int(rh), int(rw)
            if im.shape[0] != th or im.shape[1] != tw:
                # im = self._resize_pad_fit(im, is_mask=False)
                im = self._center_crop(im)
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
                # m = self._resize_pad_fit(m, is_mask=True)
                m = self._center_crop(m)

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


    def _center_crop(self, im: np.ndarray):
        th = int(getattr(self.cfg.train, "resize_h", 896))
        tw = int(getattr(self.cfg.train, "resize_w", 1120))

        h, w = im.shape[:2]

        if h < th or w < tw:
            raise ValueError(
                f"Image too small ({h}x{w}) for center crop ({th}x{tw})"
            )

        top = (h - th) // 2
        left = (w - tw) // 2

        return im[top:top + th, left:left + tw]


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
        pad_val = self.ignore_index if is_mask else 0

        if im_r.ndim == 2:
            return np.pad(
                im_r,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=pad_val,
            )

        return np.pad(
            im_r,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=pad_val,
        )

def image_collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "img_path": [b["img_path"] for b in batch],
        "mask_path": [b["mask_path"] for b in batch],
        "video_src": [b.get("video_src", "") for b in batch],
        "video_clip": [b.get("video_clip", "") for b in batch],
        "rep": [b.get("rep", "-") for b in batch],
    }