def infer_mapping_from_dirs_fast(
    pred_dir,
    gt_dir,
    num_pred_cls,
    num_gt_cls,
    start_idx,
    end_idx,
    recursive=True,
    stride=100,
    downsample=4,
):
    import re
    from pathlib import Path

    import numpy as np
    from PIL import Image

    def build_index_map(root):
        root = Path(root)
        exts = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
        glob_fn = root.rglob if recursive else root.glob

        out = {}

        for ext in exts:
            for p in sorted(glob_fn(ext)):
                nums = re.findall(r"\d+", p.stem)

                if not nums:
                    continue

                # filename only has frame index
                idx = int(nums[-1])

                if idx not in out:
                    out[idx] = p

        return out

    pred_map = build_index_map(pred_dir)
    gt_map = build_index_map(gt_dir)

    overlap = np.zeros(
        (num_pred_cls, num_gt_cls),
        dtype=np.int64,
    )

    used = 0

    for idx in range(start_idx, end_idx + 1, stride):
        p_path = pred_map.get(idx)
        g_path = gt_map.get(idx)

        if p_path is None or g_path is None:
            continue

        pred = np.array(Image.open(p_path))
        gt = np.array(Image.open(g_path))

        if pred.ndim == 3:
            pred = pred[:, :, 0]

        if gt.ndim == 3:
            gt = gt[:, :, 0]

        if downsample > 1:
            pred = pred[::downsample, ::downsample]
            gt = gt[::downsample, ::downsample]

        valid = (
            (pred >= 0) & (pred < num_pred_cls) &
            (gt >= 0) & (gt < num_gt_cls)
        )

        pred = pred[valid].astype(np.int64)
        gt = gt[valid].astype(np.int64)

        pair_id = pred * num_gt_cls + gt

        counts = np.bincount(
            pair_id,
            minlength=num_pred_cls * num_gt_cls,
        )

        overlap += counts.reshape(num_pred_cls, num_gt_cls)
        used += 1

    score = overlap.astype(float)

    # Avoid mapping foreground predictions to background
    for p in range(1, num_pred_cls):
        score[p, 0] = 0

    mapping = {
        p: int(np.argmax(score[p]))
        for p in range(num_pred_cls)
    }

    return mapping, score, used


mapping, score, used = infer_mapping_from_dirs_fast(
    pred_dir="/groups/dchen/mxs/PE_dgem_task/PE_iter1_surround16_sam3/",
    gt_dir="/groups/dchen/mxs/PE_dgem_task/masks/all_NE_ordering/",
    num_pred_cls=9,
    num_gt_cls=9,
    start_idx=0,
    end_idx=35000,
    stride=100,
    downsample=4,
)

print("used frames:", used)
print("mapping:", mapping)
print("score matrix:")
print(score)