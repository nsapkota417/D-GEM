import os
import cv2
import numpy as np
from tqdm import tqdm

ROOT = "/users/nsapkota/VOS/data/datasets/cholecseg8k"

global_vals = set()
clip_vals = {}

for video_src in sorted(os.listdir(ROOT)):
    src_path = os.path.join(ROOT, video_src)
    if not os.path.isdir(src_path):
        continue

    for clip in sorted(os.listdir(src_path)):
        clip_path = os.path.join(src_path, clip)
        if not os.path.isdir(clip_path):
            continue

        mask_dir = os.path.join(clip_path, "watershed_mask")
        if not os.path.isdir(mask_dir):
            continue

        local = set()

        # iterate files (no tqdm per clip; too much overhead)
        for fname in os.listdir(mask_dir):
            if not fname.lower().endswith(".png"):
                continue

            p = os.path.join(mask_dir, fname)
            m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if m is None:
                continue

            g = m[..., 0] if m.ndim == 3 else m  # single channel
            u = np.unique(g)
            local.update(u.tolist())
            global_vals.update(u.tolist())

        clip_vals[f"{video_src}/{clip}"] = local

print("Global unique watershed values:", sorted(global_vals))
print("Num global values:", len(global_vals))
