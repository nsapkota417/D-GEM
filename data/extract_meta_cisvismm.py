from pathlib import Path
import re
import pandas as pd
import os

def get_frame_id(path):
    path = Path(path)
    nums = re.findall(r"\d+", path.stem)
    return nums[-1] if nums else None


def index_files(folder, pattern):
    folder = Path(folder)
    files = {}

    for path in folder.glob(pattern):
        frame_id = get_frame_id(path)
        if frame_id is not None:
            files[frame_id] = path.resolve()

    return files


def make_csv(
    swir_img_video_dir,
    wl_img_video_dir,
    swir_gts_video_dir,
    wl_gts_video_dir,
    out_csv,
):
    swir_img_video_dir = Path(swir_img_video_dir)
    wl_img_video_dir = Path(wl_img_video_dir)
    swir_gts_video_dir = Path(swir_gts_video_dir)
    wl_gts_video_dir = Path(wl_gts_video_dir)

    print(swir_img_video_dir, '\n', wl_img_video_dir, '\n', swir_gts_video_dir, '\n', wl_gts_video_dir)

    video_src = swir_img_video_dir.name

    swir_imgs = index_files(swir_img_video_dir, "frame_*.jpg")
    wl_imgs = index_files(wl_img_video_dir, "frame_*.jpg")

    swir_masks = index_files(swir_gts_video_dir, "mask_frame_*.png")
    wl_masks = index_files(wl_gts_video_dir, "mask_frame_*.png")

    common_frame_ids = sorted(
        set(swir_imgs)
        & set(wl_imgs)
        & set(swir_masks)
        & set(wl_masks),
        key=lambda x: int(x),
    )

    rows = []

    for frame_id in common_frame_ids:
        rows.append({
            # "image_id": f"{video_src}_{frame_id}",
            "swir_img": str(swir_imgs[frame_id]),
            "wl_img": str(wl_imgs[frame_id]),
            "swir_mask": str(swir_masks[frame_id]),
            "wl_mask": str(wl_masks[frame_id]),
            "video_src": video_src,
            "frame_id": frame_id,
        })

    df = pd.DataFrame(
        rows,
        columns=[
            # "image_id",
            "swir_img",
            "wl_img",
            "swir_mask",
            "wl_mask",
            "video_src",
            "frame_id",
        ],
    )

    df.to_csv(out_csv, index=False)

    print(f"Saved {len(df)} rows to {out_csv}")
    return df


SUR_TYPES = [
    '01_ureter_surgery_1min_URETER',         
    '02_ureter_surgery_15min_URETER',
    '03_bile_duct_surgery_10min_BILE_DUCT_CYSTIC_ARTERY',
    '04_thoracic_surgery_4min_THORACIC_DUCT', 
]

IMG_PATH = "/groups/dchen/bz/data/CisionVision/images/JHU_Lab_0826/"

SWIR_GT_PATH = '/groups/dchen/bz/Representative/output_jhulab_new'
WL_GT_PATH = '/groups/dchen/bz/Representative/output_jhulab_white_light_new'

DEST_PATH = '/users/nsapkota/VOS/data/datasets/cisvismm'
# swir = "/groups/dchen/bz/data/CisionVision/images/JHU_Lab_0826/01_ureter_surgery_1min_URETER_swir"
# wl = "/groups/dchen/bz/data/CisionVision/images/JHU_Lab_0826/01_ureter_surgery_1min_URETER_white_light"

# swir_gt = '/groups/dchen/bz/Representative/output_jhulab_new/01_ureter_surgery_1min'
# wl_gt = '/groups/dchen/bz/Representative/output_jhulab_white_light_new/01_ureter_surgery_1min'

for stype in SUR_TYPES:
    mask_sub = '_'.join(stype.split('_')[:4])
    if 'bile' in stype:
        mask_sub = '_'.join(stype.split('_')[:5])

    df = make_csv(
        swir_img_video_dir=os.path.join(IMG_PATH, stype+'_swir'),
        wl_img_video_dir=os.path.join(IMG_PATH, stype+'_white_light'),
        swir_gts_video_dir=os.path.join(SWIR_GT_PATH, mask_sub),
        wl_gts_video_dir=os.path.join(WL_GT_PATH, mask_sub),
        out_csv=os.path.join(DEST_PATH, "_".join(stype.split("_")[i] for i in [1, 3])+'.csv')
    )