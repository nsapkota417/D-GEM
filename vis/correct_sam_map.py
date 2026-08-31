from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np
from PIL import Image


SAM3_TO_GLOBAL = {0: 0, 1: 2, 2: 3, 3: 4, 4: 5, 5: 7, 6: 1, 7: 6, 8: 8}


def remap_one(args):
    src_path, dst_path, mapping = args

    mask = np.array(Image.open(src_path))

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    out = np.zeros_like(mask, dtype=np.uint8)

    for src, dst in mapping.items():
        out[mask == src] = dst

    Image.fromarray(out).save(dst_path)

    return src_path.name


def remap_mask_dir_fast(
    src_dir,
    dst_dir,
    mapping,
    recursive=True,
    num_workers=None,
):
    src_dir = Path(src_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst_dir = Path(dst_dir) / f"remapped_{timestamp}"
    dst_dir.mkdir(parents=True, exist_ok=True)

    exts = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]

    files = []
    for ext in exts:
        if recursive:
            files.extend(src_dir.rglob(ext))
        else:
            files.extend(src_dir.glob(ext))

    files = sorted(files)

    print(f"Found {len(files)} files")

    tasks = []
    for src_path in files:
        rel_path = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        tasks.append((src_path, dst_path, mapping))

    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 2)

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for i, name in enumerate(ex.map(remap_one, tasks), 1):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(files)}")

    print(f"Saved remapped masks to: {dst_dir}")


remap_mask_dir_fast(
    src_dir="/groups/dchen/mxs/PE_dgem_task/PE_iter1_surround16_sam3/",
    dst_dir="/groups/dchen/nick/dgem/sam3_pe_mapped",
    mapping=SAM3_TO_GLOBAL,
    recursive=False,
    num_workers=16,
)