import cv2
import json
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ImageSegDataset(Dataset):
    def __init__(self, cfg, df: pd.DataFrame, is_train: bool = True):
        self.cfg = cfg
        self.df = df.reset_index(drop=True)
        self.is_train = is_train

        self.img_col = str(self.cfg.data.img_col)
        self.mask_col = str(self.cfg.data.mask_col)
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
                raise ValueError(f"Unknown modality: {self.modality}")
        else:
            self.img_cols = [self.cfg.data.img_col]
            self.mask_col = self.cfg.data.mask_col

        self.ignore_index = int(self.cfg.data.ignore_index)

        required = set(self.img_cols + [self.mask_col])
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.target_h = int(getattr(self.cfg.train, "resize_h", 896))
        self.target_w = int(getattr(self.cfg.train, "resize_w", 1120))

        # Augmentation probabilities
        self.hflip_prob = float(
            getattr(self.cfg.train, "hflip_prob", 0.5)
        )
        self.rot90_prob = float(
            getattr(self.cfg.train, "rot90_prob", 0.35)
        )
        self.photo_prob = float(
            getattr(self.cfg.train, "photo_prob", 0.8)
        )

        # Resolution and aspect-ratio variation
        self.scale_min = float(
            getattr(self.cfg.train, "aug_scale_min", 0.75)
        )
        self.scale_max = float(
            getattr(self.cfg.train, "aug_scale_max", 1.25)
        )
        self.aspect_min = float(
            getattr(self.cfg.train, "aug_aspect_min", 0.80)
        )
        self.aspect_max = float(
            getattr(self.cfg.train, "aug_aspect_max", 1.25)
        )

        # Grayscale code -> class LUT
        self.lut = np.full(256, self.ignore_index, dtype=np.uint8)
        for k in range(self.cfg.data.num_class):
            self.lut[k] = np.uint8(k)
        self.lut[self.ignore_index] = np.uint8(self.ignore_index)

        # Optional RGB label mapping
        self.rgb_to_id = None
        label_json = getattr(self.cfg.data, "label_json", None)
        code_to_class = getattr(self.cfg.data, "code_to_class", None)

        if label_json:
            with open(label_json, "r") as f:
                items = json.load(f)

            self.rgb_to_id = {
                tuple(map(int, item["color"])): int(item["classid"])
                for item in items
            }

        elif code_to_class is not None:
            self.lut = np.full(
                256, self.ignore_index, dtype=np.uint8
            )

            for key, value in dict(code_to_class).items():
                key = int(key, 0) if isinstance(key, str) else int(key)
                value = int(value)

                if 0 <= key <= 255:
                    self.lut[key] = np.uint8(value)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_paths = [row[col] for col in self.img_cols]
        mask_path = row[self.mask_col]

        imgs = []

        for col, path in zip(self.img_cols, img_paths):
            if col == "swir_img" and len(self.img_cols) > 1:
                image = self._read_gray(path)
                is_rgb = False
            else:
                image = self._read_rgb(path)
                is_rgb = True

            if self.is_train:
                image = self._photometric_augment(
                    image,
                    is_rgb=is_rgb,
                )

            imgs.append(image)

        # Validate that modalities have matching spatial dimensions
        spatial_shapes = {img.shape[:2] for img in imgs}
        if len(spatial_shapes) != 1:
            raise ValueError(
                f"Modalities have different shapes: {spatial_shapes}"
            )

        # SWIR only: 3 channels
        # WL only: 3 channels
        # SWIR + WL: 1 + 3 = 4 channels
        image = np.concatenate(imgs, axis=2)

        # Mask is mapped to class IDs before augmentation
        mask = self._read_and_map_mask(mask_path)

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image/mask shape mismatch: "
                f"{image.shape[:2]} vs {mask.shape[:2]}"
            )

        if self.is_train:
            image, mask = self._train_spatial_augment(
                image,
                mask,
            )
        else:
            image = self._resize_pad_fit(
                image,
                is_mask=False,
            )
            mask = self._resize_pad_fit(
                mask,
                is_mask=True,
            )

        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        image_t = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
        )

        mask_t = torch.from_numpy(mask.astype(np.int64))

        sample = {
            "image": image_t,
            "mask": mask_t,
            "img_path": (
                img_paths[0]
                if len(img_paths) == 1
                else img_paths
            ),
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
        image = cv2.imread(path, cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(path)

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _read_gray(self, path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(path)

        return image[..., None]

    def _read_and_map_mask(self, path: str) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if mask is None:
            raise FileNotFoundError(path)

        if mask.ndim == 3 and self.rgb_to_id is not None:
            if mask.shape[2] == 4:
                rgb = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGB)
            else:
                rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            output = np.full(
                rgb.shape[:2],
                self.ignore_index,
                dtype=np.uint8,
            )

            key = (
                (rgb[..., 0].astype(np.int32) << 16)
                | (rgb[..., 1].astype(np.int32) << 8)
                | rgb[..., 2].astype(np.int32)
            )

            for (r, g, b), class_id in self.rgb_to_id.items():
                rgb_key = (r << 16) | (g << 8) | b
                output[key == rgb_key] = np.uint8(class_id)

            return output

        gray = mask[..., 0] if mask.ndim == 3 else mask
        gray = gray.astype(np.uint8)

        return self.lut[gray]

    def _train_spatial_augment(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ):
        # Left-right flip
        if np.random.rand() < self.hflip_prob:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)

        # Simulate unexpected 90°, 180°, or 270° orientation
        if np.random.rand() < self.rot90_prob:
            k = np.random.choice([1, 2, 3])
            image = np.rot90(image, k=k)
            mask = np.rot90(mask, k=k)

        # Simulate resolution and aspect-ratio variation
        image, mask = self._random_scale_aspect(
            image,
            mask,
        )

        # Produce the fixed training tensor size
        image, mask = self._random_crop_or_pad(
            image,
            mask,
        )

        return (
            np.ascontiguousarray(image),
            np.ascontiguousarray(mask),
        )

    def _random_scale_aspect(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ):
        scale = np.random.uniform(
            self.scale_min,
            self.scale_max,
        )

        # Log-uniform aspect-ratio jitter
        log_aspect = np.random.uniform(
            math.log(self.aspect_min),
            math.log(self.aspect_max),
        )
        aspect = math.exp(log_aspect)

        scale_x = scale * math.sqrt(aspect)
        scale_y = scale / math.sqrt(aspect)

        new_h = max(
            1,
            int(round(self.target_h * scale_y)),
        )
        new_w = max(
            1,
            int(round(self.target_w * scale_x)),
        )

        image = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = cv2.resize(
            mask,
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST,
        )

        if image.ndim == 2:
            image = image[..., None]

        return image, mask

    def _random_crop_or_pad(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ):
        h, w = image.shape[:2]

        # Random crop when larger than target
        if h > self.target_h:
            top = np.random.randint(
                0,
                h - self.target_h + 1,
            )
        else:
            top = 0

        if w > self.target_w:
            left = np.random.randint(
                0,
                w - self.target_w + 1,
            )
        else:
            left = 0

        crop_h = min(h, self.target_h)
        crop_w = min(w, self.target_w)

        image = image[
            top:top + crop_h,
            left:left + crop_w,
        ]
        mask = mask[
            top:top + crop_h,
            left:left + crop_w,
        ]

        h, w = image.shape[:2]

        pad_h = self.target_h - h
        pad_w = self.target_w - w

        # Random padding position
        pad_top = (
            np.random.randint(0, pad_h + 1)
            if pad_h > 0
            else 0
        )
        pad_left = (
            np.random.randint(0, pad_w + 1)
            if pad_w > 0
            else 0
        )

        pad_bottom = pad_h - pad_top
        pad_right = pad_w - pad_left

        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )

        mask = cv2.copyMakeBorder(
            mask,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=self.ignore_index,
        )

        if image.ndim == 2:
            image = image[..., None]

        return image, mask

    def _photometric_augment(
        self,
        image: np.ndarray,
        is_rgb: bool,
    ):
        if np.random.rand() >= self.photo_prob:
            return image

        output = image.astype(np.float32)

        # Brightness and contrast
        contrast = np.random.uniform(0.75, 1.25)
        brightness = np.random.uniform(-25.0, 25.0)

        output = output * contrast + brightness
        output = np.clip(output, 0, 255)

        # Gamma variation
        gamma = np.random.uniform(0.75, 1.35)
        output = 255.0 * np.power(output / 255.0, gamma)

        # Saturation and hue for RGB only
        if is_rgb and output.shape[2] == 3:
            rgb_uint8 = np.clip(
                output,
                0,
                255,
            ).astype(np.uint8)

            hsv = cv2.cvtColor(
                rgb_uint8,
                cv2.COLOR_RGB2HSV,
            ).astype(np.float32)

            hsv[..., 1] *= np.random.uniform(0.70, 1.30)
            hsv[..., 0] += np.random.uniform(-8.0, 8.0)

            hsv[..., 0] = np.mod(hsv[..., 0], 180)
            hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)

            output = cv2.cvtColor(
                hsv.astype(np.uint8),
                cv2.COLOR_HSV2RGB,
            ).astype(np.float32)

        # Mild blur
        if np.random.rand() < 0.20:
            kernel = int(np.random.choice([3, 5]))
            output = cv2.GaussianBlur(
                output,
                (kernel, kernel),
                sigmaX=0,
            )

            if output.ndim == 2:
                output = output[..., None]

        # Mild sensor noise
        if np.random.rand() < 0.20:
            noise_std = np.random.uniform(2.0, 8.0)
            noise = np.random.normal(
                0,
                noise_std,
                output.shape,
            )
            output += noise

        return np.clip(output, 0, 255).astype(np.uint8)

    def _resize_pad_fit(
        self,
        image: np.ndarray,
        is_mask: bool,
    ):
        original_h, original_w = image.shape[:2]

        scale = min(
            self.target_w / original_w,
            self.target_h / original_h,
        )

        new_h = max(
            1,
            int(round(original_h * scale)),
        )
        new_w = max(
            1,
            int(round(original_w * scale)),
        )

        interpolation = (
            cv2.INTER_NEAREST
            if is_mask
            else cv2.INTER_LINEAR
        )

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=interpolation,
        )

        if resized.ndim == 2 and not is_mask:
            resized = resized[..., None]

        pad_h = self.target_h - new_h
        pad_w = self.target_w - new_w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        pad_value = self.ignore_index if is_mask else 0

        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_value,
        )

        if padded.ndim == 2 and not is_mask:
            padded = padded[..., None]

        return padded


def image_collate(batch):
    return {
        "image": torch.stack(
            [item["image"] for item in batch],
            dim=0,
        ),
        "mask": torch.stack(
            [item["mask"] for item in batch],
            dim=0,
        ),
        "img_path": [
            item["img_path"] for item in batch
        ],
        "mask_path": [
            item["mask_path"] for item in batch
        ],
        "video_src": [
            item.get("video_src", "")
            for item in batch
        ],
        "video_clip": [
            item.get("video_clip", "")
            for item in batch
        ],
        "rep": [
            item.get("rep", "-")
            for item in batch
        ],
    }