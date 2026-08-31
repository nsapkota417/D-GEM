import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _get_default_font():
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20
        )
    except Exception:
        return ImageFont.load_default()


def _ensure_rgb(img_pil):
    arr = np.array(img_pil)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return arr.astype(np.uint8)


def _ensure_mask(arr):
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def _mask_to_color(mask_np, color_map):
    h, w = mask_np.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in color_map.items():
        out[mask_np == cls] = color
    return out


def _fit_to_size(img_np, target_w, target_h, is_mask=False):
    pil = Image.fromarray(img_np)
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    return np.array(pil.resize((target_w, target_h), resample=resample))


def _make_row_strip(
    img_np,
    gt_np,
    pred_np,
    filename,
    rep_value="-",
    panel_w=320,
    panel_h=240,
    header_h=34,
    gap=8,
    color_map=None,
    font=None,
):
    if color_map is None:
        color_map = {
            0: (0, 0, 0),
            1: (220, 20, 60),
            2: (0, 128, 0),
            3: (30, 144, 255),
            4: (255, 215, 0),
            5: (138, 43, 226),
            6: (0, 255, 127),
            7: (255, 140, 0),
        }

    if font is None:
        font = _get_default_font()

    gt_color = _mask_to_color(gt_np, color_map)
    pred_color = _mask_to_color(pred_np, color_map)

    img_vis = _fit_to_size(img_np, panel_w, panel_h, is_mask=False)
    gt_vis = _fit_to_size(gt_color, panel_w, panel_h, is_mask=True)
    pred_vis = _fit_to_size(pred_color, panel_w, panel_h, is_mask=True)

    canvas_w = panel_w * 3 + gap * 4
    canvas_h = header_h + panel_h + gap * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    header_text = f"{filename} | rep={rep_value}"
    draw.text((gap, 6), header_text, fill=(0, 0, 0), font=font)

    x1 = gap
    x2 = gap * 2 + panel_w
    x3 = gap * 3 + panel_w * 2
    y = header_h

    canvas.paste(Image.fromarray(img_vis), (x1, y))
    canvas.paste(Image.fromarray(gt_vis), (x2, y))
    canvas.paste(Image.fromarray(pred_vis), (x3, y))

    draw.text((x1 + 8, y + 8), "IMG", fill=(255, 255, 255), font=font)
    draw.text((x2 + 8, y + 8), "GT", fill=(255, 255, 255), font=font)
    draw.text((x3 + 8, y + 8), "PRED", fill=(255, 255, 255), font=font)

    draw.rectangle([x1, y, x1 + panel_w - 1, y + panel_h - 1], outline=(0, 0, 0), width=2)
    draw.rectangle([x2, y, x2 + panel_w - 1, y + panel_h - 1], outline=(0, 0, 0), width=2)
    draw.rectangle([x3, y, x3 + panel_w - 1, y + panel_h - 1], outline=(0, 0, 0), width=2)

    return np.array(canvas)


def export_vis_results_pages(
    csv_path=None,
    df=None,
    save_root="vis_results",
    subdir_name="grids",
    img_col="img",
    gt_col="mask",
    pred_col="pred_path",
    rep_col="rep",
    max_images=None,
    color_map=None,
    rows_per_page=10,
    panel_w=320,
    panel_h=240,
):
    """
    Saves paged visualization images.
    Each page contains rows_per_page rows x 3 panels:
      [IMG | GT | PRED]
    with filename + rep status in header for each row.

    Output goes to:
      save_root / subdir_name / <timestamped_run_folder> / page_XXX.png
    """

    if df is None:
        if csv_path is None:
            raise ValueError("Provide either df or csv_path")
        df = pd.read_csv(csv_path)
    else:
        df = df.copy()

    if max_images is not None:
        df = df.iloc[:max_images].copy()

    save_root = Path(save_root)
    page_dir = save_root / subdir_name 
    page_dir.mkdir(parents=True, exist_ok=True)

    if color_map is None:
        color_map = {
            0: (0, 0, 0),
            1: (220, 20, 60),
            2: (0, 128, 0),
            3: (30, 144, 255),
            4: (255, 215, 0),
            5: (138, 43, 226),
            6: (0, 255, 127),
            7: (255, 140, 0),
        }

    font = _get_default_font()

    row_strips = []
    missing_files = []
    page_idx = 0
    saved_examples = 0

    def flush_page(row_strips_, page_idx_):
        if len(row_strips_) == 0:
            return
        page_np = np.concatenate(row_strips_, axis=0)
        page_path = page_dir / f"page_{page_idx_:03d}.png"
        Image.fromarray(page_np).save(page_path)
        print(f"Saved: {page_path}")

    for _, row in df.iterrows():
        img_path = Path(row[img_col])
        gt_path = Path(row[gt_col])
        pred_path = Path(row[pred_col])

        if not img_path.exists():
            missing_files.append(str(img_path))
            continue
        if not gt_path.exists():
            missing_files.append(str(gt_path))
            continue
        if not pred_path.exists():
            missing_files.append(str(pred_path))
            continue

        img = _ensure_rgb(Image.open(img_path).convert("RGB"))
        gt = _ensure_mask(np.array(Image.open(gt_path)))
        pred = _ensure_mask(np.array(Image.open(pred_path)))

        filename = img_path.name
        rep_value = "Yes" if (rep_col in row and str(row[rep_col]).strip().lower() == "rep") else "No"

        row_strip = _make_row_strip(
            img_np=img,
            gt_np=gt,
            pred_np=pred,
            filename=filename,
            rep_value=str(rep_value),
            panel_w=panel_w,
            panel_h=panel_h,
            color_map=color_map,
            font=font,
        )

        row_strips.append(row_strip)
        saved_examples += 1

        if len(row_strips) == rows_per_page:
            flush_page(row_strips, page_idx)
            row_strips = []
            page_idx += 1

    if len(row_strips) > 0:
        flush_page(row_strips, page_idx)

    print(f"\nDone. Saved {saved_examples} examples into paged grids: {page_dir}")

    if missing_files:
        print(f"Missing files: {len(missing_files)}")
        for p in missing_files[:10]:
            print(p)

    return page_dir


def select_examples_by_thresholds(
    csv_path,
    num_classes=8,
    per_class_bad_threshold=None,
    overall_good_threshold=None,
    max_examples_per_class=10,
    max_good_examples=20,
):
    df = pd.read_csv(csv_path)
    out = {}

    if per_class_bad_threshold is not None:
        bad_per_class = {}
        for c in range(num_classes):
            col = f"class_{c}_miou"
            if col not in df.columns:
                continue

            sub = df[df[col].notna() & (df[col] < per_class_bad_threshold)].copy()
            sub = sub.sort_values(by=col, ascending=True)

            if len(sub) > max_examples_per_class:
                sub = sub.head(max_examples_per_class)

            bad_per_class[c] = sub

        out["bad_per_class"] = bad_per_class

    if overall_good_threshold is not None:
        if "miou" not in df.columns:
            raise ValueError("CSV must contain 'miou' column")

        good_df = df[df["miou"].notna() & (df["miou"] > overall_good_threshold)].copy()
        good_df = good_df.sort_values(by="miou", ascending=False)

        if len(good_df) > max_good_examples:
            good_df = good_df.head(max_good_examples)

        out["good_overall"] = good_df

    return out


def plot_threshold_examples(
    csv_path,
    save_root,
    img_col="img",
    gt_col="mask",
    pred_col="pred_path",
    rep_col="rep",
    num_classes=8,
    per_class_bad_threshold=None,
    overall_good_threshold=None,
    max_examples_per_class=10,
    max_good_examples=20,
    color_map=None,
    rows_per_page=10,
    panel_w=320,
    panel_h=240,
):
    """
    Wrapper:
    - for each class, pick worst examples below threshold
    - optionally also pick best examples above overall threshold
    - save results in class-specific folders as paged grids
    """

    results = select_examples_by_thresholds(
        csv_path=csv_path,
        num_classes=num_classes,
        per_class_bad_threshold=per_class_bad_threshold,
        overall_good_threshold=overall_good_threshold,
        max_examples_per_class=max_examples_per_class,
        max_good_examples=max_good_examples,
    )

    output_dirs = {}

    if "bad_per_class" in results:
        for c, subdf in results["bad_per_class"].items():
            if subdf.empty:
                print(f"No bad examples found for class {c}")
                continue

            subdir_name = f"class_{c}/bad_lt_{str(per_class_bad_threshold).replace('.', 'p')}"
            out_dir = export_vis_results_pages(
                df=subdf,
                save_root=save_root,
                subdir_name=subdir_name,
                img_col=img_col,
                gt_col=gt_col,
                pred_col=pred_col,
                rep_col=rep_col,
                color_map=color_map,
                rows_per_page=rows_per_page,
                panel_w=panel_w,
                panel_h=panel_h,
            )
            output_dirs[f"class_{c}_bad"] = out_dir

    if "good_overall" in results:
        good_df = results["good_overall"]
        if not good_df.empty:
            subdir_name = f"overall_good/gt_{str(overall_good_threshold).replace('.', 'p')}"
            out_dir = export_vis_results_pages(
                df=good_df,
                save_root=save_root,
                subdir_name=subdir_name,
                img_col=img_col,
                gt_col=gt_col,
                pred_col=pred_col,
                rep_col=rep_col,
                color_map=color_map,
                rows_per_page=rows_per_page,
                panel_w=panel_w,
                panel_h=panel_h,
            )
            output_dirs["overall_good"] = out_dir
        else:
            print("No overall-good examples found")

    return output_dirs


# -------------------------
# Example usage
# -------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = "/users/nsapkota/VOS/results/nch/cnh/dv3_vits16plus/seg_0405_175207/metrices/metrics_0000.csv"
save_root = f"/users/nsapkota/VOS/results/nch/cnh/dv3_vits16plus/seg_0405_175207/vis_{timestamp}"

output_dirs = plot_threshold_examples(
    csv_path=csv_path,
    save_root=save_root,
    img_col="img",
    gt_col="mask",
    pred_col="pred_path",
    rep_col="rep",
    num_classes=8,
    per_class_bad_threshold=20.0,
    overall_good_threshold=91.0,
    max_examples_per_class=10,
    max_good_examples=20,
    rows_per_page=10,
    panel_w=320,
    panel_h=240,
)

print(output_dirs)
