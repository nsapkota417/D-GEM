import os
import csv
import re

# NE
# img_dir = "/groups/dchen/mxs/processed_data/CNI/images/NE"
# mask_dir = "/groups/dchen/bz/data/childrensHospital/masks/NE/all"
# output_csv = "/users/nsapkota/VOS/data/meta/cnh_ne.csv"
# indices_txt = "/users/nsapkota/VOS/data/datasets/cnh/indices.txt"

# PE
img_dir = "/groups/dchen/mxs/processed_data/CNI/images/Partial Nephrectomy video"
mask_dir = "/groups/dchen/mxs/PE_dgem_task/masks/all_NE_ordering"
output_csv = "/users/nsapkota/VOS/data/meta/cnh_pe.csv"
indices_txt = "/users/nsapkota/VOS/data/datasets/cnh_pe/indices.txt"

# -------------------------------
# 1. Load frame indices from txt
# -------------------------------
with open(indices_txt, "r") as f:
    indices = set(int(line.strip()) for line in f if line.strip())

# -------------------------------
# 2. Generate CSV
# -------------------------------
rows = []

for root, _, files in os.walk(img_dir):
    for file in files:
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(root, file)
        rel_path = os.path.relpath(img_path, img_dir)
        mask_rel_path = os.path.splitext(rel_path)[0] + ".png"
        mask_path = os.path.join(mask_dir, mask_rel_path)

        # import IPython; IPython.embed()

        if not os.path.exists(mask_path):
            print(f'skipping {img_path}')
            continue

        video_clip = os.path.basename(root)

        # -------------------------------
        # 3. Extract frame index
        # -------------------------------
        rep_val = "-"
        match = re.search(r'\d+', file)

        if match:
            frame_id = int(match.group())
            if frame_id in indices:
                rep_val = "rep"

        rows.append([
            img_path,
            mask_path,
            "",
            video_clip,
            rep_val
        ])

# -------------------------------
# 4. Write CSV
# -------------------------------
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["img", "mask", "video_src", "video_clip", "rep"])
    writer.writerows(rows)

print(f"Saved {len(rows)} rows")