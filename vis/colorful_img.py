from pathlib import Path
from datetime import datetime
import shutil
import re

import numpy as np
from PIL import Image


CODE_TO_CLASS = {
    0: 0,  # background
    1: 1,  # grasper
    2: 2,  # intraoperative_ultrasound
    3: 3,  # kidney
    4: 4,  # perinephric_fat
    5: 5,  # scissor
    6: 6,  # suction
    7: 7,  # tumor
    8: 8,  # clip
}


GLOBAL_PALETTE = np.array([
    [0, 0, 0],        # background
    [230, 25, 75],    # grasper
    [60, 180, 75],    # intraoperative_ultrasound
    [0, 130, 200],    # kidney
    [245, 130, 48],   # perinephric_fat
    [145, 30, 180],   # scissor
    [70, 240, 240],   # suction
    [240, 50, 230],   # tumor
    [210, 245, 60],   # clip
], dtype=np.uint8)


def build_index_to_file_map(img_dir, recursive=True):
    img_dir = Path(img_dir)
    exts = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]

    glob_fn = img_dir.rglob if recursive else img_dir.glob
    index_to_file = {}

    for ext in exts:
        for p in sorted(glob_fn(ext)):
            nums = re.findall(r"\d+", p.stem)

            for n in nums:
                idx = int(n)

                # keep first match if duplicate index exists
                if idx not in index_to_file:
                    index_to_file[idx] = p

    return index_to_file


def remap_mask_values(mask, code_to_class):
    remapped = np.zeros_like(mask, dtype=np.uint8)

    for src_val, class_id in code_to_class.items():
        remapped[mask == src_val] = class_id

    return remapped


def colorize_mask_dir(
    img_dir,
    results_dir,
    start_idx,
    end_idx,
    to_use="gt",
    code_to_class=CODE_TO_CLASS,
    palette=GLOBAL_PALETTE,
    recursive=True,
):
    img_dir = Path(img_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(results_dir) / f"{to_use}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    index_to_file = build_index_to_file_map(
        img_dir=img_dir,
        recursive=recursive,
    )

    num_classes = len(palette)

    for idx in range(start_idx, end_idx + 1):
        img_path = index_to_file.get(idx)

        if img_path is None:
            print(f"Missing index {idx}")
            continue

        save_path = results_dir / img_path.name

        if to_use == "img":
            shutil.copy2(img_path, save_path)
            continue

        mask = np.array(Image.open(img_path))

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        mask = remap_mask_values(mask, code_to_class)

        if mask.max() >= num_classes:
            print(
                f"Skipping {img_path.name}: "
                f"max class {mask.max()} >= palette size {num_classes}"
            )
            continue

        color_mask = palette[mask]
        Image.fromarray(color_mask).save(save_path)

    print(f"Saved results to: {results_dir}")


# ------------------------------------------------------
# NE
# paths = {
#     "img": "/groups/dchen/mxs/processed_data/CNI/images/NE",
#     "gt": "/groups/dchen/bz/data/childrensHospital/masks/NE/all",
#     "preds": "/users/nsapkota/VOS/results/cnh_ne_nm/dv3_vit16sp_nm_cnh_ne/cnh_ne_r2_seg_0425_201427/preds",
#     "sam" : ''
# }


# PE
paths = {
    "img": "/groups/dchen/mxs/processed_data/CNI/images/Partial Nephrectomy video/",
    "gt": "/groups/dchen/mxs/PE_dgem_task/masks/all_NE_ordering/",
    "preds": "/users/nsapkota/VOS/results/cnh_pe/cnh/dv3_vits16plus/seg_0425_125407/preds",
    "sam" : '/groups/dchen/nick/dgem/sam3_pe_mapped'
}

to_use_all = [
    # "img",
    # "gt",
    # "preds",
    "sam"
]

kw = "dgem_pe"

for to_use in to_use_all:
    colorize_mask_dir(
        img_dir=paths[to_use],
        results_dir=f"/users/nsapkota/VOS/outputs_pe/{kw}_{to_use}",
        start_idx=1,
        end_idx=500,
        to_use=to_use,
    )