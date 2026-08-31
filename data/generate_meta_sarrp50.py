# make_csvs.py
# Creates: train.csv and val.csv
# Columns: img, mask, video_src, video_clip
# Also prints frame counts per video and split summaries

import csv
from pathlib import Path
from typing import Dict

IMG_DIR = "frames_1hz"
MSK_DIR = "segmentation"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def _count_frames(frames_dir: Path) -> int:
    return sum(
        1 for f in frames_dir.iterdir()
        if f.is_file() and _is_image(f)
    )

def generate_split_csv(root: str, split: str, out_csv: str):
    root = Path(root).resolve()
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split folder: {split_dir}")

    rows = []
    frame_stats: Dict[str, int] = {}   # video_clip -> frame count

    for video_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        video_clip = video_dir.name    # video_XX
        video_src  = ""                # empty by design

        frames_dir = video_dir / IMG_DIR
        masks_dir  = video_dir / MSK_DIR

        if not frames_dir.exists():
            print(f"[WARN] Missing frames dir: {frames_dir}")
            continue
        if not masks_dir.exists():
            print(f"[WARN] Missing masks dir:  {masks_dir}")
            continue

        # count frames for stats
        frame_stats[video_clip] = _count_frames(frames_dir)

        mask_map = {
            m.stem: m
            for m in masks_dir.iterdir()
            if m.is_file() and _is_image(m)
        }

        for img_path in sorted(
            f for f in frames_dir.iterdir()
            if f.is_file() and _is_image(f)
        ):
            msk_path = mask_map.get(img_path.stem)
            if msk_path is None:
                continue

            rows.append(
                (str(img_path), str(msk_path), video_src, video_clip)
            )

    out_csv = Path(out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img", "mask", "video_src", "video_clip"])
        w.writerows(rows)

    print(f"\n[OK] {split}: {len(rows)} rows -> {out_csv}")

    # ---- Print frame statistics
    print(f"[Frame counts | {split}]")
    for vc in sorted(frame_stats):
        print(f"{vc:>10} | {frame_stats[vc]:5d} frames")

    print(f"[Summary | {split}]")
    print(f"Videos : {len(frame_stats)}")
    print(f"Frames : {sum(frame_stats.values())}")

    return rows, frame_stats

if __name__ == "__main__":
    root = "/users/nsapkota/VOS/data/datasets/sarrarp50"
    dest = "/users/nsapkota/VOS/data/meta"

    generate_split_csv(
        root,
        "train",
        Path(dest) / "sarrarp50_train.csv",
    )

    generate_split_csv(
        root,
        "val",
        Path(dest) / "sarrarp50_test.csv",
    )  # don't name it test.csv unless it's truly test