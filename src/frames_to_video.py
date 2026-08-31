import re
from pathlib import Path

import cv2


# ============================================================
# CONFIGURATION
# ============================================================

FRAMES_DIR = Path("/groups/dchen/bz/data/CisionVision/cisionvision/task_106_cityscapes/imgsFine/leftImg8bit/default")
OUTPUT_VIDEO = Path("/groups/dchen/nick/demo/test_video.mp4")

FPS = 30.0

# Options:
#   "error"  -> require all frames to already have same size
#   "resize" -> stretch every frame to target size
#   "crop"   -> center-crop every frame to target size
#   "pad"    -> preserve aspect ratio and add black padding
RESIZE_MODE = "error"

# Set both to None to use the first frame's resolution.
TARGET_HEIGHT = None
TARGET_WIDTH = None

OVERWRITE = True


# ============================================================
# IMPLEMENTATION
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def natural_sort_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def collect_frames(frames_dir: Path):
    frame_paths = [
        path
        for path in frames_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    frame_paths.sort(key=natural_sort_key)

    if not frame_paths:
        raise FileNotFoundError(
            f"No image frames found in: {frames_dir}"
        )

    return frame_paths


def center_crop(
    image,
    target_height: int,
    target_width: int,
):
    height, width = image.shape[:2]

    if height < target_height or width < target_width:
        raise ValueError(
            f"Image size {height}x{width} is smaller than "
            f"requested crop {target_height}x{target_width}"
        )

    top = (height - target_height) // 2
    left = (width - target_width) // 2

    return image[
        top : top + target_height,
        left : left + target_width,
    ]


def resize_with_padding(
    image,
    target_height: int,
    target_width: int,
):
    height, width = image.shape[:2]

    scale = min(
        target_width / width,
        target_height / height,
    )

    resized_width = max(
        1,
        int(round(width * scale)),
    )
    resized_height = max(
        1,
        int(round(height * scale)),
    )

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_height = target_height - resized_height
    pad_width = target_width - resized_width

    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    return cv2.copyMakeBorder(
        resized,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def prepare_frame(
    image,
    target_height: int,
    target_width: int,
    mode: str,
):
    height, width = image.shape[:2]

    if height == target_height and width == target_width:
        return image

    if mode == "error":
        raise ValueError(
            f"Inconsistent frame size: {height}x{width}; "
            f"expected {target_height}x{target_width}"
        )

    if mode == "resize":
        return cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )

    if mode == "crop":
        return center_crop(
            image,
            target_height,
            target_width,
        )

    if mode == "pad":
        return resize_with_padding(
            image,
            target_height,
            target_width,
        )

    raise ValueError(
        f"Unsupported RESIZE_MODE: {mode}"
    )


def frames_to_video():
    frames_dir = FRAMES_DIR.expanduser().resolve()
    output_path = OUTPUT_VIDEO.expanduser().resolve()

    if not frames_dir.is_dir():
        raise NotADirectoryError(
            f"Frames directory not found: {frames_dir}"
        )

    if output_path.exists():
        if OVERWRITE:
            output_path.unlink()
        else:
            raise FileExistsError(
                f"Output video already exists: {output_path}"
            )

    if FPS <= 0:
        raise ValueError("FPS must be greater than zero")

    frame_paths = collect_frames(frames_dir)

    first_frame = cv2.imread(
        str(frame_paths[0]),
        cv2.IMREAD_COLOR,
    )

    if first_frame is None:
        raise FileNotFoundError(frame_paths[0])

    first_height, first_width = first_frame.shape[:2]

    target_height = (
        int(TARGET_HEIGHT)
        if TARGET_HEIGHT is not None
        else first_height
    )

    target_width = (
        int(TARGET_WIDTH)
        if TARGET_WIDTH is not None
        else first_width
    )

    if target_height <= 0 or target_width <= 0:
        raise ValueError(
            "TARGET_HEIGHT and TARGET_WIDTH must be positive"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        FPS,
        (target_width, target_height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Unable to create video: {output_path}"
        )

    try:
        for index, frame_path in enumerate(frame_paths):
            frame = cv2.imread(
                str(frame_path),
                cv2.IMREAD_COLOR,
            )

            if frame is None:
                raise FileNotFoundError(
                    f"Unable to read frame: {frame_path}"
                )

            frame = prepare_frame(
                frame,
                target_height=target_height,
                target_width=target_width,
                mode=RESIZE_MODE,
            )

            writer.write(frame)

            print(
                f"\rWriting frame "
                f"{index + 1:,}/{len(frame_paths):,}",
                end="",
                flush=True,
            )

    finally:
        writer.release()

    duration = len(frame_paths) / FPS

    print()
    print("=" * 70)
    print(f"Input frames : {frames_dir}")
    print(f"Frames       : {len(frame_paths):,}")
    print(f"Resolution   : {target_width}x{target_height}")
    print(f"FPS          : {FPS:g}")
    print(f"Duration     : {duration:.2f} seconds")
    print(f"Output video : {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    frames_to_video()