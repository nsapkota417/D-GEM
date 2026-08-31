import os
import pandas as pd


TEST_VIDEOS = {"video43", "video48", "video52", "video55"}


def build_and_split(root_dir):

    rows = []

    for video_src in sorted(os.listdir(root_dir)):

        vsrc_path = os.path.join(root_dir, video_src)

        if not (os.path.isdir(vsrc_path) and video_src.startswith("video")):
            continue

        for video_clip in sorted(os.listdir(vsrc_path)):

            vclip_path = os.path.join(vsrc_path, video_clip)

            if not os.path.isdir(vclip_path):
                continue

            img_dir = os.path.join(vclip_path, "img")

            if not os.path.isdir(img_dir):
                continue

            mask_dir = os.path.join(vclip_path, "mask")
            color_dir = os.path.join(vclip_path, "color_mask")
            water_dir = os.path.join(vclip_path, "watershed_mask")

            for fname in sorted(os.listdir(img_dir)):

                if not fname.endswith(".png"):
                    continue

                img_path = os.path.join(img_dir, fname)

                base = fname.replace("_endo.png", "")

                rows.append({
                    "img": img_path,
                    "mask": os.path.join(mask_dir, base + "_endo_mask.png"),
                    "color_mask": os.path.join(color_dir, base + "_endo_color_mask.png"),
                    "watershed_mask": os.path.join(water_dir, base + "_endo_watershed_mask.png"),
                    "video_src": video_src,
                    "video_clip": video_clip,
                })

    df = pd.DataFrame(rows)

    # Split
    train_df = df[~df["video_src"].isin(TEST_VIDEOS)].reset_index(drop=True)
    test_df  = df[df["video_src"].isin(TEST_VIDEOS)].reset_index(drop=True)

    return train_df, test_df


if __name__ == "__main__":

    ROOT = "/users/nsapkota/VOS/data/datasets/cholecseg8k"

    train_df, test_df = build_and_split(ROOT)

    print("Train samples:", len(train_df))
    print("Test samples :", len(test_df))

    train_df.to_csv(os.path.join('/users/nsapkota/VOS/data/meta', "cholecseg8k_meta_train.csv"), index=False)
    test_df.to_csv(os.path.join('/users/nsapkota/VOS/data/meta', "cholecseg8k_meta_test.csv"), index=False)

    print("Saved: train.csv, test.csv")