"""Standalone image and video inference for trained D-GEM checkpoints.

Video mode takes annotated supports from ``--support-csv`` and processes every
frame in ``--test-csv``. Test masks are optional: when present they are used
only for reporting metrics, never to initialize the video memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from networks.dinov3_seg import DINOv3ViTSeg
from networks.svsswrapper import SVSSWrapper
from utils import nested_dotdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D-GEM inference from CSV manifests.")
    parser.add_argument("-cfg", "--config", required=True)
    parser.add_argument("--task-type", choices=("image", "video"), default=None)
    parser.add_argument("--support-csv", help="Annotated supports for video mode.")
    parser.add_argument("--test-csv", required=True, help="Frames to process.")
    parser.add_argument("--weights", required=True, help="Trained checkpoint.")
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--save-preds", action="store_true", help="Write predicted PNG masks.")
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = base.copy()
    for key, value in override.items():
        merged[key] = deep_merge(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    return merged


def load_cfg(args: argparse.Namespace):
    config_path = Path(args.config).expanduser()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    model_config = config.pop("model_config", None)
    if model_config:
        model_path = config_path.parent / model_config
        with model_path.open(encoding="utf-8") as handle:
            config = deep_merge(config, yaml.safe_load(handle) or {})
    if args.task_type:
        config.setdefault("data", {})["task_type"] = args.task_type
    return nested_dotdict(config)


def has_mask(value: Any) -> bool:
    return value is not None and str(value).strip() not in {"", "-", "none", "None", "nan", "NaN"}


def normalize_manifest(df: pd.DataFrame, mask_col: str, require_video: bool) -> pd.DataFrame:
    required = {"img"}
    if require_video:
        required.add("video_src")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if mask_col not in df.columns:
        df = df.copy()
        df[mask_col] = "-"
    if "video_src" not in df.columns:
        df = df.copy()
        df["video_src"] = "images"
    if "video_clip" not in df.columns:
        df = df.copy()
        df["video_clip"] = df["video_src"]
    df = df.copy()
    df["video_src"] = df["video_src"].fillna("").astype(str)
    df["video_clip"] = df["video_clip"].fillna("").astype(str)
    sort_col = "frame_idx" if "frame_idx" in df.columns else "img"
    return df.sort_values(["video_src", "video_clip", sort_col]).reset_index(drop=True)


def build_lut(cfg) -> np.ndarray:
    lut = np.full(256, int(cfg.data.ignore_index), dtype=np.uint8)
    for source, target in dict(cfg.data.code_to_class).items():
        source, target = int(source), int(target)
        if 0 <= source <= 255:
            lut[source] = target
    return lut


def read_image(path: str, cfg) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return resize_pad_fit(image, int(cfg.train.resize_h), int(cfg.train.resize_w), False, 0)


def read_mask(path: str, cfg, lut: np.ndarray) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = resize_pad_fit(mask, int(cfg.train.resize_h), int(cfg.train.resize_w), True, int(cfg.data.ignore_index))
    return lut[mask.astype(np.uint8)]


def resize_pad_fit(image: np.ndarray, height: int, width: int, is_mask: bool, pad_value: int) -> np.ndarray:
    old_height, old_width = image.shape[:2]
    scale = min(width / old_width, height / old_height)
    new_height, new_width = max(1, round(old_height * scale)), max(1, round(old_width * scale))
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    pad_height, pad_width = height - new_height, width - new_width
    if resized.ndim == 2:
        return np.pad(resized, ((0, pad_height), (0, pad_width)), constant_values=pad_value)
    return np.pad(resized, ((0, pad_height), (0, pad_width), (0, 0)), constant_values=pad_value)


def to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)


def build_model(cfg, device: torch.device):
    model_name = str(cfg.train.model)
    if "dv3" not in model_name:
        raise ValueError("Standalone inference currently supports DINOv3/D-GEM checkpoints only.")
    backbone = DINOv3ViTSeg(
        model_name=f"facebook/dinov3-{model_name.split('_')[-1]}-pretrain-lvd1689m",
        num_classes=int(cfg.data.num_class), pt_encoder=bool(cfg.train.pt_encoder),
        ft_encoder=bool(cfg.train.ft_encoder), in_chans=int(getattr(cfg.data, "num_ch", 3)),
    ).to(device)
    if str(cfg.data.task_type).lower() == "image":
        return backbone
    t = cfg.train
    return SVSSWrapper(
        backbone, patch_size=16, K=t.K, max_dt=t.max_dt, use_memory=t.use_memory,
        use_am=t.use_am, use_tm=t.use_tm, write_topk_patch_tokens=t.write_topk_patch_tokens,
        read_topk_mem_tokens=t.read_topk_mem_tokens, alpha_am=t.alpha_am, alpha_tm=t.alpha_tm,
        learnable_alpha=t.learnable_alpha, tm_warmup=t.tm_warmup,
        skip_tm_t0=bool(getattr(t, "skip_tm_t0", True)), gate_mode=t.gate_mode,
        gate_conf_thr=t.gate_conf_thr, gate_ent_thr=t.gate_ent_thr,
        allow_pseudo_anchors=bool(getattr(t, "allow_pseudo_anchors", False)),
        pseudo_use_fused_logits=bool(getattr(t, "pseudo_use_fused_logits", True)),
        pseudo_every=int(getattr(t, "pseudo_every", 1)), pseudo_warmup=int(getattr(t, "pseudo_warmup", 0)),
        pseudo_tau=float(getattr(t, "pseudo_tau", .8)), pseudo_q99_thr=float(getattr(t, "pseudo_q99_thr", .9)),
        pseudo_mean_in_thr=float(getattr(t, "pseudo_mean_in_thr", .7)),
        pseudo_min_area=float(getattr(t, "pseudo_min_area", .001)), pseudo_max_area=float(getattr(t, "pseudo_max_area", .2)),
        pseudo_streak_req=int(getattr(t, "pseudo_streak_req", 1)), pseudo_k_am=int(getattr(t, "pseudo_k_am", 128)),
        pseudo_max_per_class=int(getattr(t, "pseudo_max_per_class", 1)),
        pseudo_conf_scale=float(getattr(t, "pseudo_conf_scale", .25)), pseudo_w_scale=float(getattr(t, "pseudo_w_scale", .5)),
        use_mem_attention=t.use_mem_attention, attn_sharp=t.attn_sharp, attn_topk_am=t.attn_topk_am,
        attn_topk_tm=t.attn_topk_tm, am_attn_beta=t.am_attn_beta, bg_index=0,
        ignore_index=cfg.data.ignore_index, am_max_items=t.am_max_items, am_red_lambda=t.am_red_lambda,
        enable_am_refresh=bool(getattr(t, "enable_am_refresh", False)),
        am_refresh_sim_max=float(getattr(t, "am_refresh_sim_max", .9)), am_max_per_class=int(getattr(t, "am_max_per_class", 3)),
        use_abs_time=bool(getattr(t, "use_abs_time", True)), max_time_index=int(getattr(t, "max_time_index", 4096)),
        detach_memory=bool(getattr(t, "detach_memory", True)), debug=False,
    ).to(device)


def load_weights(model, path: str, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    state = {key.removeprefix("module.").removeprefix("model."): value for key, value in state.items()}
    # An image-only checkpoint stores DINOv3 keys directly (for example,
    # ``encoder.*``).  Video inference wraps that same network under
    # ``frame_model``; remap compatible keys so an image-trained encoder and
    # decoder can be used with training-free D-GEM memory at inference time.
    model_keys = set(model.state_dict())
    if "frame_model.encoder.cls_token" in model_keys and not any(
        key.startswith("frame_model.") for key in state
    ):
        state = {
            f"frame_model.{key}" if f"frame_model.{key}" in model_keys else key: value
            for key, value in state.items()
        }
    incompatible = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {path}")
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"Checkpoint compatibility: {len(incompatible.missing_keys)} missing, {len(incompatible.unexpected_keys)} unexpected keys")


def miou(pred: np.ndarray, target: np.ndarray, classes: int, ignore_index: int) -> float | None:
    valid = target != ignore_index
    if not valid.any():
        return None
    scores = []
    for class_id in range(classes):
        predicted, actual = pred[valid] == class_id, target[valid] == class_id
        union = np.logical_or(predicted, actual).sum()
        if union:
            scores.append(np.logical_and(predicted, actual).sum() / union)
    return float(np.mean(scores) * 100) if scores else None


def save_prediction(prediction: np.ndarray, row: pd.Series, output_dir: Path) -> str:
    video = str(row.get("video_src", "images")).replace("/", "_")
    clip = str(row.get("video_clip", "")).replace("/", "_")
    path = output_dir / "predictions" / video / clip / f"{Path(row.img).stem}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction.astype(np.uint8)).save(path)
    return str(path)


def run_image(model, test_df, cfg, lut, device, output_dir, save_preds):
    rows = []
    for _, row in test_df.iterrows():
        image = to_tensor(read_image(row.img, cfg), device)
        with torch.inference_mode():
            logits = model(image)
            logits = logits[0] if isinstance(logits, (tuple, list)) else logits
        prediction = logits.argmax(1).squeeze(0).cpu().numpy()
        score = miou(prediction, read_mask(row[cfg.data.mask_col], cfg, lut), cfg.data.num_class, cfg.data.ignore_index) if has_mask(row[cfg.data.mask_col]) else None
        rows.append({"img": row.img, "video_src": row.video_src, "pred_path": save_prediction(prediction, row, output_dir) if save_preds else None, "miou": score})
    return rows


def run_video(model, support_df, test_df, cfg, lut, device, output_dir, save_preds):
    groups = {key: group for key, group in support_df.groupby(["video_src", "video_clip"], sort=False)}
    rows = []
    for key, frames in test_df.groupby(["video_src", "video_clip"], sort=False):
        supports = groups.get(key)
        if supports is None:
            raise ValueError(f"No support frames in --support-csv for {key}.")
        supports = supports[supports[cfg.data.mask_col].map(has_mask)]
        if supports.empty:
            raise ValueError(f"No labeled support masks in --support-csv for {key}.")
        support_images = torch.cat([to_tensor(read_image(row.img, cfg), device) for _, row in supports.iterrows()]).unsqueeze(0)
        support_masks = torch.from_numpy(np.stack([read_mask(row[cfg.data.mask_col], cfg, lut) for _, row in supports.iterrows()])).unsqueeze(0).to(device)
        indices = supports["frame_idx"].to_numpy(dtype=np.int64) if "frame_idx" in supports else np.arange(len(supports))
        state = model.init_state(support_img=support_images, support_mask=support_masks, support_indices=torch.as_tensor(indices, device=device).unsqueeze(0), video_id=f"infer_{key[0]}_{key[1]}")
        for position, (_, row) in enumerate(frames.iterrows()):
            index = int(row.frame_idx) if "frame_idx" in frames else position
            with torch.inference_mode():
                logits, _, state = model.step(query_img=to_tensor(read_image(row.img, cfg), device), state=state, query_index=torch.tensor([index], device=device))
            prediction = logits.argmax(1).squeeze(0).cpu().numpy()
            score = miou(prediction, read_mask(row[cfg.data.mask_col], cfg, lut), cfg.data.num_class, cfg.data.ignore_index) if has_mask(row[cfg.data.mask_col]) else None
            rows.append({"img": row.img, "video_src": row.video_src, "video_clip": row.video_clip, "pred_path": save_prediction(prediction, row, output_dir) if save_preds else None, "miou": score})
        if hasattr(model, "clear_video"):
            model.clear_video(f"infer_{key[0]}_{key[1]}")
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)
    task_type = str(cfg.data.task_type).lower()
    if task_type == "video" and not args.support_csv:
        raise ValueError("Video inference requires --support-csv with annotated frames.")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_df = normalize_manifest(pd.read_csv(args.test_csv), str(cfg.data.mask_col), require_video=task_type == "video")
    support_df = normalize_manifest(pd.read_csv(args.support_csv), str(cfg.data.mask_col), require_video=True) if args.support_csv else None
    model = build_model(cfg, device)
    load_weights(model, args.weights, device)
    model.eval()
    lut = build_lut(cfg)
    rows = run_video(model, support_df, test_df, cfg, lut, device, output_dir, args.save_preds) if task_type == "video" else run_image(model, test_df, cfg, lut, device, output_dir, args.save_preds)
    report = pd.DataFrame(rows)
    report.to_csv(output_dir / "report.csv", index=False)
    scores = report.miou.dropna()
    summary = {"task_type": task_type, "processed_frames": len(report), "evaluated_frames": len(scores), "mean_frame_miou": float(scores.mean()) if len(scores) else None, "predictions_saved": bool(args.save_preds)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
