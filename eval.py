import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable, Union

import numpy as np
import torch
from PIL import Image


# ============================================================
# Mapping: accept either
#   (A) code_to_class dict (grayscale code -> class id), OR
#   (B) JSON file / dict describing either:
#       - {"code_to_class": {...}}  (grayscale)
#       - {...}                    (grayscale)
#       - [ { "color":[R,G,B], "classid":k, ... }, ... ] (palette RGB)
# ============================================================
MappingInput = Union[Dict[Any, Any], str, Path]


def _load_mapping_obj(mapping: MappingInput) -> Any:
    if isinstance(mapping, (str, Path)):
        mp = Path(mapping)
        if not mp.exists():
            raise FileNotFoundError(f"Mapping not found: {mp}")
        with open(mp, "r") as f:
            return json.load(f)
    if isinstance(mapping, dict) or isinstance(mapping, list):
        return mapping
    raise TypeError(f"Unsupported mapping type: {type(mapping)}")


def _mapping_kind(obj: Any) -> str:
    """
    Returns:
      "palette_rgb" if obj is list of dicts with color/classid
      "grayscale_codes" otherwise (dict forms)
    """
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict) and "color" in obj[0] and "classid" in obj[0]:
        return "palette_rgb"
    return "grayscale_codes"


def _as_int_int_dict(d: Dict[Any, Any]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for k, v in d.items():
        out[int(k)] = int(v)
    return out


def load_grayscale_code_to_class(mapping: MappingInput) -> Dict[int, int]:
    obj = _load_mapping_obj(mapping)
    if isinstance(obj, dict) and "code_to_class" in obj and isinstance(obj["code_to_class"], dict):
        return _as_int_int_dict(obj["code_to_class"])
    if isinstance(obj, dict):
        return _as_int_int_dict(obj)
    raise ValueError("Grayscale mapping must be a dict or {'code_to_class': dict}.")


def load_palette_rgb_to_class(mapping: MappingInput) -> Dict[int, int]:
    """
    Returns rgb_int -> classid, where rgb_int=(R<<16)|(G<<8)|B
    Accepts:
      - JSON list [{color:[R,G,B], classid:int}, ...]
      - dict containing that list under common keys (rare; handled lightly)
    """
    obj = _load_mapping_obj(mapping)
    if isinstance(obj, dict):
        # If someone wraps it, try common keys
        for k in ("labels", "classes", "palette", "mapping"):
            if k in obj and isinstance(obj[k], list):
                obj = obj[k]
                break

    if not (isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict)):
        raise ValueError("Palette mapping must be a JSON list of objects with 'color' and 'classid'.")

    rgb2cls: Dict[int, int] = {}
    for item in obj:
        if "color" not in item or "classid" not in item:
            raise ValueError("Each palette item must contain keys: 'color' and 'classid'")
        r, g, b = item["color"]
        cid = int(item["classid"])
        key = (int(r) << 16) | (int(g) << 8) | int(b)
        rgb2cls[key] = cid
    return rgb2cls


# ============================================================
# Remapping
# ============================================================
def remap_grayscale_codes(mask: torch.Tensor, code_to_class: Dict[int, int], ignore_index: int = 255) -> torch.Tensor:
    out = torch.full_like(mask, ignore_index)
    for code, cls in code_to_class.items():
        out[mask == int(code)] = int(cls)
    return out


def rgb_mask_to_class(mask_rgb: np.ndarray, rgb_to_class: Dict[int, int], ignore_index: int = 255) -> np.ndarray:
    # Accept RGB or RGBA
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA mask (H,W,3/4), got {mask_rgb.shape}")

    if mask_rgb.shape[2] == 4:
        mask_rgb = mask_rgb[..., :3]  # drop alpha

    r = mask_rgb[..., 0].astype(np.int64)
    g = mask_rgb[..., 1].astype(np.int64)
    b = mask_rgb[..., 2].astype(np.int64)
    key = (r << 16) | (g << 8) | b

    out = np.full((mask_rgb.shape[0], mask_rgb.shape[1]), ignore_index, dtype=np.int64)
    for rgb_int, cid in rgb_to_class.items():
        out[key == rgb_int] = int(cid)
    return out


def remap_mask_numpy(arr: np.ndarray, mapping_obj: Any, ignore_index: int = 255) -> np.ndarray:
    """
    Remap a single mask loaded as numpy.

    If mapping is:
      - palette_rgb: expects arr is RGB (H,W,3). If arr is 2D, we assume it's already ids.
      - grayscale_codes: expects arr is 2D codes. If arr is RGB, we take arr[...,0] (best-effort).
    """
    kind = _mapping_kind(mapping_obj)
    if kind == "palette_rgb":
        rgb2cls = load_palette_rgb_to_class(mapping_obj)
        if arr.ndim == 2:
            return arr.astype(np.int64)
        return rgb_mask_to_class(arr, rgb2cls, ignore_index=ignore_index)

    # grayscale mapping
    code2cls = load_grayscale_code_to_class(mapping_obj)
    if arr.ndim == 3:
        arr = arr[..., 0]
    # vectorize via torch-style approach but in numpy
    out = np.full(arr.shape[:2], ignore_index, dtype=np.int64)
    for code, cls in code2cls.items():
        out[arr == code] = cls
    return out


# ============================================================
# IO
# ============================================================
def _load_png_as_numpy(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


# ============================================================
# mIoU (per-frame) + phases (phases by folder count)
# ============================================================
def _miou_per_frame_and_phases(
    pred_thw: torch.Tensor,          # (T,H,W) long (semantic ids)
    tgt_thw: torch.Tensor,           # (T,H,W) long (semantic ids)
    num_classes: int,
    ignore_index: int = 255,
    include_bg: bool = True,
    p1_frac: float = 0.2,
    p3_frac: float = 0.2,
) -> Tuple[float, float, float, float, List[float]]:
    assert pred_thw.ndim == 3 and tgt_thw.ndim == 3
    T = int(pred_thw.shape[0])
    if T == 0:
        nan = float("nan")
        return nan, nan, nan, nan, []

    device = pred_thw.device
    C = int(num_classes)
    eps = 1e-12

    cls = torch.arange(C, device=device) if include_bg else torch.arange(1, C, device=device)

    miou_t: List[float] = []
    for t in range(T):
        p = pred_thw[t].reshape(-1)
        y = tgt_thw[t].reshape(-1)

        valid = (y != ignore_index)
        if not bool(valid.any()):
            miou_t.append(float("nan"))
            continue

        p = p[valid].clamp(min=0, max=C - 1)
        y = y[valid].clamp(min=0, max=C - 1)

        k = (y * C + p).to(torch.int64)
        conf = torch.bincount(k, minlength=C * C).reshape(C, C).float()

        tp = conf.diag()
        fp = conf.sum(0) - tp
        fn = conf.sum(1) - tp
        denom = tp + fp + fn

        iou = (tp + eps) / (denom + eps)
        valid_cls = denom[cls] > 0
        miou_t.append(float(iou[cls][valid_cls].mean().item()) if bool(valid_cls.any()) else float("nan"))

    x = torch.tensor(miou_t, device=device, dtype=torch.float32)
    good = torch.isfinite(x)
    miou_all = float(x[good].mean().item()) if bool(good.any()) else float("nan")

    # phases by folder count (rank-based rel)
    rel = (torch.arange(T, device=device, dtype=torch.float32) + 0.5) / T

    p1 = float(max(0.0, min(0.49, p1_frac)))
    p3 = float(max(0.0, min(0.49, p3_frac)))
    mid_end = 1.0 - p3
    if mid_end <= p1:
        p1, mid_end = 1 / 3, 2 / 3

    p1_idx = (rel < p1)
    p2_idx = (rel >= p1) & (rel < mid_end)
    p3_idx = (rel >= mid_end)

    def masked_nanmean(mask: torch.Tensor) -> float:
        m = mask & good
        return float(x[m].mean().item()) if bool(m.any()) else float("nan")

    return miou_all, masked_nanmean(p1_idx), masked_nanmean(p2_idx), masked_nanmean(p3_idx), miou_t


# ============================================================
# Frame-id matching (filenames can differ)
# ============================================================
_DIGITS = re.compile(r"\d+")


def _digit_groups(name: str) -> List[str]:
    return _DIGITS.findall(name)


def _candidate_extractors() -> List[Tuple[str, Callable[[str], Optional[int]]]]:
    def first_group(s: str) -> Optional[int]:
        g = _digit_groups(s)
        return int(g[0]) if g else None

    def last_group(s: str) -> Optional[int]:
        g = _digit_groups(s)
        return int(g[-1]) if g else None

    def longest_group(s: str) -> Optional[int]:
        g = _digit_groups(s)
        if not g:
            return None
        return int(max(g, key=len))

    def last_longest_group(s: str) -> Optional[int]:
        g = _digit_groups(s)
        if not g:
            return None
        maxlen = max(len(x) for x in g)
        cand = [x for x in g if len(x) == maxlen]
        return int(cand[-1])

    def second_last_group(s: str) -> Optional[int]:
        g = _digit_groups(s)
        return int(g[-2]) if len(g) >= 2 else None

    return [
        ("first_group", first_group),
        ("last_group", last_group),
        ("longest_group", longest_group),
        ("last_longest_group", last_longest_group),
        ("second_last_group", second_last_group),
    ]


def _build_id_map(files: List[Path], extractor: Callable[[str], Optional[int]]) -> Tuple[Dict[int, Path], int]:
    m: Dict[int, Path] = {}
    collisions = 0
    for p in files:
        fid = extractor(p.name)
        if fid is None:
            continue
        if fid in m and m[fid] != p:
            collisions += 1
            continue
        m[fid] = p
    return m, collisions


def _choose_best_extractor(gt_files: List[Path], pr_files: List[Path]) -> Tuple[str, Callable[[str], Optional[int]]]:
    best = None
    for name, ex in _candidate_extractors():
        gt_map, gt_col = _build_id_map(gt_files, ex)
        pr_map, pr_col = _build_id_map(pr_files, ex)
        common = set(gt_map) & set(pr_map)
        score = (len(common), -(gt_col + pr_col), len(gt_map) + len(pr_map))
        if best is None or score > best[0]:
            best = (score, name, ex)
    assert best is not None
    return best[1], best[2]


# ============================================================
# Main evaluator
#   - phases by folder COUNT
#   - optional mapping for gt/pred (dict OR json)
# ============================================================
def eval_masks_folder_phase_by_count(
    root: str | Path,
    num_classes: int,
    gt_dirname: str,
    pred_dirname: str,
    ignore_index: int = 255,
    include_bg: bool = True,
    p1_frac: float = 0.2,
    p3_frac: float = 0.2,
    device: str = "cpu",
    exts: Tuple[str, ...] = (".png",),
    verbose_match: bool = True,

    # Mapping options: either dict or json path
    gt_mapping: Optional[MappingInput] = None,
    pred_mapping: Optional[MappingInput] = None,
    strict_range_warn: bool = True,
) -> Dict[str, Any]:
    """
    Layout:
      Root/video_folder/<gt_dirname>/*.png
      Root/video_folder/<pred_dirname>/*.png

    Matching:
      - names can differ; matched by numeric id in filename (auto heuristic).

    Phases:
      - defined by folder length (count of matched frames).

    Mapping:
      - gt_mapping/pred_mapping can be:
          * grayscale code->class dict (or json dict)
          * palette RGB list json (like you pasted)
      - If mapping is provided, masks are remapped to semantic ids before mIoU.
      - Unknown colors/codes -> ignore_index.
    """
    root = Path(root)
    assert root.exists(), f"Root not found: {root}"

    gt_map_obj = _load_mapping_obj(gt_mapping) if gt_mapping is not None else None
    pr_map_obj = _load_mapping_obj(pred_mapping) if pred_mapping is not None else None

    results: Dict[str, Any] = {"videos": {}, "dataset": {}}
    vid_all, vid_p1, vid_p2, vid_p3 = [], [], [], []

    for vdir in sorted([p for p in root.iterdir() if p.is_dir()]):
        gt_dir = vdir / gt_dirname
        pr_dir = vdir / pred_dirname
        if not gt_dir.is_dir() or not pr_dir.is_dir():
            continue

        gt_files = [p for p in gt_dir.iterdir() if p.suffix.lower() in exts]
        pr_files = [p for p in pr_dir.iterdir() if p.suffix.lower() in exts]
        if len(gt_files) == 0 or len(pr_files) == 0:
            print(f"[SKIP] {vdir.name}: empty gt/pred")
            continue

        ex_name, extractor = _choose_best_extractor(gt_files, pr_files)
        gt_id2p, gt_col = _build_id_map(gt_files, extractor)
        pr_id2p, pr_col = _build_id_map(pr_files, extractor)

        common_ids = sorted(set(gt_id2p) & set(pr_id2p))
        if len(common_ids) == 0:
            print(f"[WARN] {vdir.name}: no matching ids (best extractor={ex_name})")
            continue

        gt_frames: List[torch.Tensor] = []
        pr_frames: List[torch.Tensor] = []

        for fid in common_ids:
            gt_arr = _load_png_as_numpy(gt_id2p[fid])
            pr_arr = _load_png_as_numpy(pr_id2p[fid])

            if gt_map_obj is not None:
                gt_arr = remap_mask_numpy(gt_arr, gt_map_obj, ignore_index=ignore_index)
            else:
                # if RGB but no mapping given, best-effort: use first channel
                if gt_arr.ndim == 3:
                    gt_arr = gt_arr[..., 0]
                gt_arr = gt_arr.astype(np.int64)

            if pr_map_obj is not None:
                pr_arr = remap_mask_numpy(pr_arr, pr_map_obj, ignore_index=ignore_index)
            else:
                if pr_arr.ndim == 3:
                    pr_arr = pr_arr[..., 0]
                pr_arr = pr_arr.astype(np.int64)

            gt_frames.append(torch.from_numpy(gt_arr).to(torch.int64))
            pr_frames.append(torch.from_numpy(pr_arr).to(torch.int64))

        gt_stack = torch.stack(gt_frames, dim=0).to(device=device)
        pr_stack = torch.stack(pr_frames, dim=0).to(device=device)

        miou_all, miou_p1, miou_p2, miou_p3, _ = _miou_per_frame_and_phases(
            pred_thw=pr_stack,
            tgt_thw=gt_stack,
            num_classes=num_classes,
            ignore_index=ignore_index,
            include_bg=include_bg,
            p1_frac=p1_frac,
            p3_frac=p3_frac,
        )

        if strict_range_warn:
            allowed = set(range(num_classes)) | {ignore_index}
            gt_u = set(torch.unique(gt_stack).detach().cpu().tolist())
            pr_u = set(torch.unique(pr_stack).detach().cpu().tolist())
            bad_gt = sorted([x for x in gt_u if x not in allowed])[:10]
            bad_pr = sorted([x for x in pr_u if x not in allowed])[:10]
            if bad_gt:
                print(f"[WARN] {vdir.name}: GT has labels outside 0..{num_classes-1}/ignore (examples): {bad_gt}")
            if bad_pr:
                print(f"[WARN] {vdir.name}: PRED has labels outside 0..{num_classes-1}/ignore (examples): {bad_pr}")

        if verbose_match:
            gt_kind = _mapping_kind(gt_map_obj) if gt_map_obj is not None else "none"
            pr_kind = _mapping_kind(pr_map_obj) if pr_map_obj is not None else "none"
            print(
                f"VAL {vdir.name} — matched {gt_stack.shape[0]} frames "
                f"(extractor={ex_name}, collisions gt/pred={gt_col}/{pr_col}, map gt={gt_kind}, pred={pr_kind}) | "
                f"mIoU: {miou_all:.2f} | Phases: {miou_p1:.2f}/{miou_p2:.2f}/{miou_p3:.2f}"
            )
        else:
            print(
                f"VAL {vdir.name} — {gt_stack.shape[0]} frames | "
                f"mIoU: {miou_all:.2f} | Phases: {miou_p1:.2f}/{miou_p2:.2f}/{miou_p3:.2f}"
            )

        results["videos"][vdir.name] = {
            "matched_frames": int(gt_stack.shape[0]),
            "extractor": ex_name,
            "gt_collisions": gt_col,
            "pred_collisions": pr_col,
            "miou": float(miou_all),
            "miou_p1": float(miou_p1),
            "miou_p2": float(miou_p2),
            "miou_p3": float(miou_p3),
        }

        if np.isfinite(miou_all): vid_all.append(miou_all)
        if np.isfinite(miou_p1):  vid_p1.append(miou_p1)
        if np.isfinite(miou_p2):  vid_p2.append(miou_p2)
        if np.isfinite(miou_p3):  vid_p3.append(miou_p3)

    def mean_safe(xs: List[float]) -> float:
        xs = [float(x) for x in xs if np.isfinite(x)]
        return float(np.mean(xs)) if xs else float("nan")

    results["dataset"] = {
        "num_videos": len(results["videos"]),
        "miou": mean_safe(vid_all),
        "miou_p1": mean_safe(vid_p1),
        "miou_p2": mean_safe(vid_p2),
        "miou_p3": mean_safe(vid_p3),
        "agg": "macro over folders; phases by folder length",
        "gt_mapping_used": (gt_map_obj is not None),
        "pred_mapping_used": (pr_map_obj is not None),
    }

    print(
        f"\nDATASET — {results['dataset']['num_videos']} folders | "
        f"mIoU: {100*results['dataset']['miou']:.2f} | "
        f"Phases: {100*results['dataset']['miou_p1']:.2f}/"
        f"{100*results['dataset']['miou_p2']:.2f}/"
        f"{100*results['dataset']['miou_p3']:.2f}\n"
    )

    return results


# ============================================================
# Example usage
# ============================================================
# results = eval_masks_folder_phase_by_count(
#     root="/path/to/Root",
#     num_classes=12,
#     gt_dirname="segmentation",
#     pred_dirname="pred_mask",
#     gt_mapping="/path/to/gt_palette.json",  # your JSON list
#     pred_mapping=None,                      # if pred already 0..C-1
#     device="cuda",
# )
# =========================
# Example usage:
# =========================
# results = eval_sam3_results_phase_by_count(
#     root="/path/to/Root",
#     num_classes=13,
#     gt_dirname="segmentation",
#     pred_dirname="pred_mask",
#     device="cuda",
# )

# Example:
names = [
    "endovis", 
    # "cholecseg8k", 
    # "sarrarp50"
]

for name in names:
    pred_dirname = "pred_mask"
    gt_mapping = None
    pred_mapping = None
    gt_dirname = None
    num_classes = None

    if name == "cholecseg8k":
        gt_dirname = "watershed_mask"
        num_classes = 13
        # watershed grayscale codes -> class ids (apply to GT)
        gt_mapping = {
            50: 0,    # background
            0:  0,    # sometimes background is 0
            11: 1,    # abdominal wall
            21: 2,    # liver
            13: 3,    # GI tract
            12: 4,    # fat
            31: 5,    # grasper
            23: 6,    # connective tissue
            24: 7,    # blood
            25: 8,    # cystic duct
            32: 9,    # L-hook
            22: 10,   # gallbladder
            33: 11,   # hepatic vein
            5:  12,   # liver ligament
            255: 255, # ignore
        }

        # pred_mapping stays None if your pred masks are already 0..12
        # If pred is ALSO in watershed codes, set pred_mapping = gt_mapping

    # elif name == "sarrarp50":
    #     gt_dirname = "segmentation"
    #     num_classes = 10

    #     # If GT is already 0..9, you do NOT need a mapping.
    #     # Only set gt_mapping if GT is coded weirdly.
    #     gt_mapping = None

    #     # If PRED is already 0..9, keep pred_mapping None.
    #     pred_mapping = None

    elif name == "endovis":
        gt_dirname = "labels"
        num_classes = 12

        # This JSON looks like your RGB palette list; apply to GT.
        gt_mapping = "/users/nsapkota/VOS/data/datasets/endovis/labels.json"
        pred_mapping = gt_mapping # None  # if pred already 0..11

    else:
        raise ValueError(f"Unknown dataset name: {name}")

    root = f"/groups/dchen/mxs/sam3_svss/predictions_every20/{name}"

    results = eval_masks_folder_phase_by_count(
        root=root,
        num_classes=num_classes,
        gt_dirname=gt_dirname,
        pred_dirname=pred_dirname,
        gt_mapping=gt_mapping,
        pred_mapping=pred_mapping,
        device="cpu",
    )