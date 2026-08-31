import argparse
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from networks.dinov3_seg import DINOv3ViTSeg
from utils import nested_dotdict

try:
    import wandb
except ImportError:
    wandb = None

IMAGE_EXT = ".png"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract video frames, run segmentation inference, save masks, "
            "and create a side-by-side result video."
        )
    )

    parser.add_argument(
        "-cfg",
        "--config",
        required=True,
        help="Path to the base YAML configuration.",
    )

    # Optional overrides for cfg.inf
    parser.add_argument(
        "--weights",
        default=None,
        help="Override inf.weights.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Override inf.video.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override inf.results_dir.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Override inf.run_name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override inf.batch_size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override inf.num_workers.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Override inf.gpu.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Override inf.overwrite and replace existing results.",
    )

    return parser.parse_args()


def load_config(path: Path):
    with path.open("r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)

    if not isinstance(cfg_dict, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")

    if "inf" not in cfg_dict:
        raise ValueError(
            "The YAML configuration must contain an 'inf' section."
        )

    return nested_dotdict(cfg_dict)


def require_inf_value(cfg, name: str):
    value = getattr(cfg.inf, name, None)

    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required YAML value: inf.{name}")

    return value


def get_target_hw(cfg) -> Tuple[int, int]:
    # Inference-specific overrides, with fallback to training resolution.
    height = int(
        getattr(
            cfg.inf,
            "resize_h",
            getattr(cfg.train, "resize_h", 896),
        )
    )
    width = int(
        getattr(
            cfg.inf,
            "resize_w",
            getattr(cfg.train, "resize_w", 1120),
        )
    )

    return height, width


def center_crop(
    image: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    h, w = image.shape[:2]

    if h < target_h or w < target_w:
        raise ValueError(
            f"Frame is too small ({h}x{w}) for center crop "
            f"({target_h}x{target_w})."
        )

    top = (h - target_h) // 2
    left = (w - target_w) // 2

    return image[
        top : top + target_h,
        left : left + target_w,
    ]


def resize_pad_fit(
    image: np.ndarray,
    target_h: int,
    target_w: int,
    pad_value=0,
    return_meta: bool = False,
):
    """Resize with preserved aspect ratio, then center-pad to model size.

    When ``return_meta`` is True, also return the valid, non-padded region
    as ``(top, left, height, width)``.
    """
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_h = target_h - new_h
    pad_w = target_w - new_w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value,
    )

    if return_meta:
        valid_region = (pad_top, pad_left, new_h, new_w)
        return padded, valid_region

    return padded


def preprocess_frame(
    image: np.ndarray,
    target_h: int,
    target_w: int,
    return_meta: bool = False,
):
    """Center-crop normal-size frames; resize+pad smaller frames.

    The returned valid region identifies pixels originating from the video,
    excluding any padding introduced for the model input.
    """
    h, w = image.shape[:2]

    if h >= target_h and w >= target_w:
        processed = center_crop(
            image,
            target_h,
            target_w,
        )
        valid_region = (0, 0, target_h, target_w)

        if return_meta:
            return processed, valid_region

        return processed

    return resize_pad_fit(
        image,
        target_h,
        target_w,
        pad_value=0,
        return_meta=return_meta,
    )


def crop_valid_region(
    image: np.ndarray,
    valid_region: Tuple[int, int, int, int],
) -> np.ndarray:
    """Remove model-input padding using ``(top, left, height, width)``."""
    top, left, height, width = map(int, valid_region)
    return image[top : top + height, left : left + width]


def extract_video_frames(
    video_path: Path,
    frames_dir: Path,
) -> Dict[str, float]:
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))

    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    source_width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    source_height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    frame_count = 0

    try:
        while True:
            ok, frame_bgr = capture.read()

            if not ok:
                break

            output_path = (
                frames_dir
                / f"frame_{frame_count:06d}{IMAGE_EXT}"
            )

            if not cv2.imwrite(
                str(output_path),
                frame_bgr,
            ):
                raise RuntimeError(
                    f"Failed to save frame: {output_path}"
                )

            frame_count += 1

    finally:
        capture.release()

    if frame_count == 0:
        raise RuntimeError(
            f"No frames were extracted from: {video_path}"
        )

    return {
        "fps": fps,
        "frame_count": frame_count,
        "source_width": source_width,
        "source_height": source_height,
    }


class VideoFrameDataset(Dataset):
    def __init__(
        self,
        frame_paths: Iterable[Path],
        cfg,
    ):
        self.frame_paths = list(frame_paths)
        self.target_h, self.target_w = get_target_hw(cfg)
        self.num_channels = int(
            getattr(cfg.data, "num_ch", 3)
        )

        if self.num_channels != 3:
            raise ValueError(
                "This video inference script currently expects "
                "a 3-channel RGB model, but "
                f"cfg.data.num_ch={self.num_channels}."
            )

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, index):
        path = self.frame_paths[index]

        image_bgr = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:
            raise FileNotFoundError(path)

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        image_rgb, valid_region = preprocess_frame(
            image_rgb,
            self.target_h,
            self.target_w,
            return_meta=True,
        )

        image_tensor = (
            torch.from_numpy(
                np.ascontiguousarray(image_rgb)
            )
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
        )

        return {
            "image": image_tensor,
            "frame_name": path.name,
            "valid_region": valid_region,
        }


def collate_frames(batch):
    return {
        "image": torch.stack(
            [item["image"] for item in batch],
            dim=0,
        ),
        "frame_name": [
            item["frame_name"]
            for item in batch
        ],
        "valid_region": [
            item["valid_region"]
            for item in batch
        ],
    }


def build_model(cfg, device: torch.device):
    model_name = str(cfg.train.model)

    if "dv3" not in model_name.lower():
        raise ValueError(
            "This inference script is configured for the "
            "DINOv3 image segmentation model. Received "
            f"cfg.train.model={model_name!r}."
        )

    model = DINOv3ViTSeg(
        model_name=(
            f"facebook/dinov3-"
            f"{model_name.split('_')[-1]}"
            f"-pretrain-lvd1689m"
        ),
        num_classes=int(cfg.data.num_class),
        pt_encoder=bool(cfg.train.pt_encoder),
        ft_encoder=bool(cfg.train.ft_encoder),
        in_chans=int(cfg.data.num_ch),
    )

    return model.to(device)


def clean_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    removable_prefixes = (
        "module.",
        "model.",
    )

    for key, value in state_dict.items():
        new_key = key
        changed = True

        while changed:
            changed = False

            for prefix in removable_prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        cleaned[new_key] = value

    return cleaned


def load_weights(
    model: torch.nn.Module,
    weights_path: Path,
    device: torch.device,
):
    checkpoint = torch.load(
        weights_path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Checkpoint does not contain a valid "
            "state dictionary."
        )

    state_dict = clean_state_dict(state_dict)

    incompatible = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(f"Loaded checkpoint: {weights_path}")

    if incompatible.missing_keys:
        print(
            f"Warning: "
            f"{len(incompatible.missing_keys)} missing keys"
        )
        print(
            "  "
            + "\n  ".join(
                incompatible.missing_keys[:20]
            )
        )

    if incompatible.unexpected_keys:
        print(
            f"Warning: "
            f"{len(incompatible.unexpected_keys)} "
            f"unexpected keys"
        )
        print(
            "  "
            + "\n  ".join(
                incompatible.unexpected_keys[:20]
            )
        )


def forward_logits(
    model: torch.nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    output = model(images)

    if isinstance(output, (tuple, list)):
        output = output[0]

    return output


def default_palette(num_classes: int) -> np.ndarray:
    # Class 0 remains black.
    palette = np.zeros(
        (max(num_classes, 1), 3),
        dtype=np.uint8,
    )

    for class_id in range(1, num_classes):
        palette[class_id] = np.array(
            [
                (37 * class_id + 67) % 256,
                (17 * class_id + 149) % 256,
                (29 * class_id + 211) % 256,
            ],
            dtype=np.uint8,
        )

    return palette


def save_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    masks_dir: Path,
    device: torch.device,
    num_classes: int,
):
    model.eval()

    use_autocast = device.type == "cuda"
    processed = 0
    start = time.time()

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_autocast,
            ):
                logits = forward_logits(
                    model,
                    images,
                )

            predictions = (
                logits.argmax(dim=1)
                .cpu()
                .numpy()
            )

            for prediction, frame_name, valid_region in zip(
                predictions,
                batch["frame_name"],
                batch["valid_region"],
            ):
                # Remove any padding before saving the predicted mask.
                prediction = crop_valid_region(
                    prediction,
                    valid_region,
                )
                if (
                    prediction.min() < 0
                    or prediction.max() >= num_classes
                ):
                    raise ValueError(
                        f"Prediction values for {frame_name} "
                        f"are outside [0, {num_classes - 1}]."
                    )

                mask_path = (
                    masks_dir
                    / Path(frame_name).with_suffix(
                        IMAGE_EXT
                    )
                )

                if not cv2.imwrite(
                    str(mask_path),
                    prediction.astype(np.uint8),
                ):
                    raise RuntimeError(
                        f"Failed to save mask: {mask_path}"
                    )

                processed += 1

            elapsed = time.time() - start
            fps = (
                processed / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"\rInference: "
                f"{processed:,}/{len(loader.dataset):,} frames "
                f"| {fps:.2f} frames/s",
                end="",
                flush=True,
            )

    print()


def create_side_by_side_video(
    frames_dir: Path,
    masks_dir: Path,
    output_path: Path,
    fps: float,
    target_h: int,
    target_w: int,
    num_classes: int,
):
    frame_paths = sorted(
        frames_dir.glob(f"*{IMAGE_EXT}")
    )

    if not frame_paths:
        raise RuntimeError(
            f"No extracted frames found in: {frames_dir}"
        )

    palette_rgb = default_palette(num_classes)

    # Determine the padding-free display size from the first frame.
    first_frame = cv2.imread(
        str(frame_paths[0]),
        cv2.IMREAD_COLOR,
    )

    if first_frame is None:
        raise FileNotFoundError(frame_paths[0])

    first_processed, first_valid_region = preprocess_frame(
        first_frame,
        target_h,
        target_w,
        return_meta=True,
    )
    first_processed = crop_valid_region(
        first_processed,
        first_valid_region,
    )

    display_h, display_w = first_processed.shape[:2]

    # Many video codecs require even dimensions. At most one valid pixel is
    # removed from the bottom/right; no padding is reintroduced.
    display_h -= display_h % 2
    display_w -= display_w % 2

    if display_h <= 0 or display_w <= 0:
        raise ValueError(
            f"Invalid padding-free video size: {display_h}x{display_w}"
        )

    output_size = (
        display_w * 2,
        display_h,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        output_size,
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Unable to create output video: {output_path}"
        )

    try:
        for index, frame_path in enumerate(frame_paths):
            frame_bgr = cv2.imread(
                str(frame_path),
                cv2.IMREAD_COLOR,
            )

            mask_path = masks_dir / frame_path.name

            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if frame_bgr is None:
                raise FileNotFoundError(frame_path)

            if mask is None:
                raise FileNotFoundError(mask_path)

            frame_bgr, valid_region = preprocess_frame(
                frame_bgr,
                target_h,
                target_w,
                return_meta=True,
            )
            frame_bgr = crop_valid_region(
                frame_bgr,
                valid_region,
            )

            # Saved masks are already padding-free. Match the codec-safe size.
            frame_bgr = frame_bgr[:display_h, :display_w]
            mask = mask[:display_h, :display_w]

            if frame_bgr.shape[:2] != mask.shape[:2]:
                raise ValueError(
                    f"Frame/mask shape mismatch for {frame_path.name}: "
                    f"{frame_bgr.shape[:2]} vs {mask.shape[:2]}"
                )

            clipped_mask = np.clip(
                mask.astype(np.int64),
                0,
                num_classes - 1,
            )

            mask_rgb = palette_rgb[clipped_mask]

            mask_bgr = cv2.cvtColor(
                mask_rgb,
                cv2.COLOR_RGB2BGR,
            )

            cv2.putText(
                frame_bgr,
                "Input frame",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                mask_bgr,
                "Predicted mask",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            side_by_side = np.concatenate(
                [frame_bgr, mask_bgr],
                axis=1,
            )

            writer.write(side_by_side)

            if (
                (index + 1) % 100 == 0
                or index + 1 == len(frame_paths)
            ):
                print(
                    f"\rVideo: "
                    f"{index + 1:,}/{len(frame_paths):,} "
                    f"frames",
                    end="",
                    flush=True,
                )

    finally:
        writer.release()

    print()

def main():


    args = parse_args()

    config_path = (
        Path(args.config)
        .expanduser()
        .resolve()
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    cfg = load_config(config_path)

    # -------------------------------------------------
    # Override cfg.inf using command-line arguments
    # -------------------------------------------------
    if args.weights is not None:
        cfg.inf.weights = args.weights

    if args.video is not None:
        cfg.inf.video = args.video

    if args.results_dir is not None:
        cfg.inf.results_dir = args.results_dir

    if args.run_name is not None:
        cfg.inf.run_name = args.run_name

    if args.batch_size is not None:
        cfg.inf.batch_size = args.batch_size

    if args.num_workers is not None:
        cfg.inf.num_workers = args.num_workers

    if args.gpu is not None:
        cfg.inf.gpu = args.gpu

    if args.overwrite:
        cfg.inf.overwrite = True

    # Initialize W&B after cfg exists and overrides are applied.
    wandb_run = None

    if wandb is not None:
        wandb_run = wandb.init(
            project="demo_prep",
            config=dict(cfg),
            job_type="inference",
            name=f"DEMO_INF_{cfg.inf.run_name}",
        )


    weights_path = (
        Path(require_inf_value(cfg, "weights"))
        .expanduser()
        .resolve()
    )

    video_path = (
        Path(require_inf_value(cfg, "video"))
        .expanduser()
        .resolve()
    )

    results_root = (
        Path(require_inf_value(cfg, "results_dir"))
        .expanduser()
        .resolve()
    )

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}"
        )

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    gpu_index = int(
        getattr(cfg.inf, "gpu", 0)
    )

    batch_size = int(
        getattr(cfg.inf, "batch_size", 1)
    )

    num_workers = int(
        getattr(cfg.inf, "num_workers", 2)
    )

    overwrite = bool(
        getattr(cfg.inf, "overwrite", False)
    )

    save_frames = bool(
        getattr(cfg.inf, "save_frames", True)
    )

    save_masks = bool(
        getattr(cfg.inf, "save_masks", True)
    )

    save_video = bool(
        getattr(cfg.inf, "save_video", True)
    )

    if batch_size < 1:
        raise ValueError(
            "inf.batch_size must be at least 1"
        )

    if num_workers < 0:
        raise ValueError(
            "inf.num_workers cannot be negative"
        )

    device = torch.device(
        f"cuda:{gpu_index}"
        if torch.cuda.is_available()
        else "cpu"
    )

    run_name = str(
        getattr(
            cfg.inf,
            "run_name",
            video_path.stem,
        )
    ).strip()

    if not run_name:
        run_name = video_path.stem

    run_dir = results_root / run_name
    frames_dir = run_dir / "frames"
    masks_dir = run_dir / "masks"

    output_video_path = (
        run_dir
        / f"{video_path.stem}_side_by_side.mp4"
    )

    if run_dir.exists():
        if overwrite:
            shutil.rmtree(run_dir)
        else:
            raise FileExistsError(
                f"Results directory already exists: "
                f"{run_dir}\n"
                "Set inf.overwrite: true to replace it."
            )

    frames_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    masks_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(f"Video       : {video_path}")
    print(f"Weights     : {weights_path}")
    print(f"Config      : {config_path}")
    print(f"Results     : {run_dir}")
    print(f"Device      : {device}")
    print(f"Batch size  : {batch_size}")
    print(f"Workers     : {num_workers}")
    print("=" * 80)

    total_start = time.time()

    print("\n[1/4] Extracting video frames...")

    metadata = extract_video_frames(
        video_path,
        frames_dir,
    )

    print(
        f"Extracted {metadata['frame_count']:,} "
        f"frames at {metadata['fps']:.3f} FPS"
    )

    frame_paths = sorted(
        frames_dir.glob(f"*{IMAGE_EXT}")
    )

    dataset = VideoFrameDataset(
        frame_paths,
        cfg,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=(
            2 if num_workers > 0 else None
        ),
        drop_last=False,
        collate_fn=collate_frames,
    )

    print("\n[2/4] Building model...")

    model = build_model(
        cfg,
        device,
    )

    print(
        "\n[3/4] Loading weights and "
        "running inference..."
    )

    load_weights(
        model,
        weights_path,
        device,
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    save_predictions(
        model=model,
        loader=loader,
        masks_dir=masks_dir,
        device=device,
        num_classes=int(cfg.data.num_class),
    )

    if save_video:
        print(
            "\n[4/4] Creating side-by-side video..."
        )

        target_h, target_w = get_target_hw(cfg)

        create_side_by_side_video(
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            output_path=output_video_path,
            fps=float(metadata["fps"]),
            target_h=target_h,
            target_w=target_w,
            num_classes=int(cfg.data.num_class),
        )

    else:
        print(
            "\n[4/4] Side-by-side video "
            "saving disabled."
        )

    # Frames and masks are required as temporary files.
    # Delete them only after optional video generation.
    if not save_frames:
        shutil.rmtree(frames_dir)

    if not save_masks:
        shutil.rmtree(masks_dir)

    elapsed = time.time() - total_start

    print("\n" + "=" * 80)
    print("Inference complete")

    if save_frames:
        print(f"Frames              : {frames_dir}")
    else:
        print("Frames              : not retained")

    if save_masks:
        print(
            f"Segmentation masks  : {masks_dir}"
        )
    else:
        print(
            "Segmentation masks  : not retained"
        )

    if save_video:
        print(
            f"Side-by-side video  : "
            f"{output_video_path}"
        )
    else:
        print(
            "Side-by-side video  : disabled"
        )

    print(
        f"Total time          : {elapsed:.2f} seconds"
    )
    print("=" * 80)

    if wandb_run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()


# python inference.py --config configs/inference.yaml