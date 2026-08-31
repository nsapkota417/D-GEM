# make_endovis_csvs.py
# Creates TRAIN / VAL CSVs for EndoVis
# Rule: VAL is explicitly defined, everything else = TRAIN
# Also prints frame counts per video + split summaries
# Columns: img, mask, video_src, video_clip

import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple

IMG_DIR = "left_frames"
MSK_DIR = "labels"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ----- Explicit VAL set (only this matters)
VAL_SEQS: Set[str] = {
    "seq_3",
    "seq_12",
    "seq_13",
    "seq_15",
}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def list_present_sequences(root: Path) -> List[str]:
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and p.name.startswith("seq_")
    )

def count_frames(frames_dir: Path) -> int:
    return sum(
        1 for f in frames_dir.iterdir()
        if f.is_file() and is_image(f)
    )

def generate_csvs(
    root: str,
    dest: str,
    val_seqs: Set[str] = VAL_SEQS,
) -> Tuple[int, int]:
    root = Path(root).resolve()
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    present = set(list_present_sequences(root))
    missing_val = sorted(val_seqs - present)
    if missing_val:
        print(f"[WARN] Val sequences not found under root: {missing_val}")

    train_rows, val_rows = [], []
    frame_stats: Dict[str, int] = {}  # seq_name -> num_frames

    for seq_dir in sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith("seq_")
    ):
        seq_name = seq_dir.name
        frames_dir = seq_dir / IMG_DIR
        masks_dir  = seq_dir / MSK_DIR

        if not frames_dir.exists() or not masks_dir.exists():
            print(f"[WARN] Skipping {seq_name} (missing '{IMG_DIR}' or '{MSK_DIR}')")
            continue

        # count frames for stats
        frame_stats[seq_name] = count_frames(frames_dir)

        # build mask lookup by stem
        mask_map = {
            m.stem: m
            for m in masks_dir.iterdir()
            if m.is_file() and is_image(m)
        }

        is_val = seq_name in val_seqs

        for img in sorted(
            f for f in frames_dir.iterdir()
            if f.is_file() and is_image(f)
        ):
            msk = mask_map.get(img.stem)
            if msk is None:
                continue

            row = (
                str(img),
                str(msk),
                "",         # video_src
                seq_name,   # video_clip
            )

            if is_val:
                val_rows.append(row)
            else:
                train_rows.append(row)

    def write_csv(name: str, rows: List[tuple]):
        out = dest / name
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["img", "mask", "video_src", "video_clip"])
            w.writerows(rows)
        print(f"[OK] {name}: {len(rows)} rows -> {out}")

    write_csv("endovis_train.csv", train_rows)
    write_csv("endovis_test.csv", val_rows)

    # print split lists
    print("\n[Split]")
    print("VAL  =", sorted(val_seqs))
    print("TRAIN=", sorted(present - val_seqs))

    # frame counts per video
    print("\n[Frame counts per video]")
    for seq in sorted(frame_stats):
        split = "VAL" if seq in val_seqs else "TRAIN"
        print(f"{seq:>6} | {split:5} | {frame_stats[seq]:5d} frames")

    # summary totals
    val_frames = sum(frame_stats[s] for s in val_seqs if s in frame_stats)
    train_frames = sum(v for s, v in frame_stats.items() if s not in val_seqs)

    print("\n[Summary]")
    print(f"VAL videos   : {sum(1 for s in val_seqs if s in frame_stats)}")
    print(f"TRAIN videos : {sum(1 for s in frame_stats if s not in val_seqs)}")
    print(f"VAL frames   : {val_frames}")
    print(f"TRAIN frames : {train_frames}")

    return len(train_rows), len(val_rows)

if __name__ == "__main__":
    generate_csvs(
        "/users/nsapkota/VOS/data/datasets/endovis",
        "/users/nsapkota/VOS/data/meta",
    )