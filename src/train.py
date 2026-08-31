"""Unified D-GEM training entry point.

The selected workflow is controlled by ``data.task_type`` in the YAML config:
``image`` for independent image segmentation and ``video`` for D-GEM's
sequential video workflow.
"""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path

import yaml


WORKFLOWS = {
    "image": Path(__file__).with_name("workflows") / "train_image.py",
    "video": Path(__file__).with_name("workflows") / "train_video.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train D-GEM using the workflow selected in a YAML config."
    )
    parser.add_argument(
        "-cfg",
        "--config",
        required=True,
        help="Path to a YAML config containing data.task_type.",
    )
    return parser.parse_known_args()[0]


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("Config must contain a top-level 'data' mapping.")

    task_type = str(data.get("task_type", "")).lower()
    workflow = WORKFLOWS.get(task_type)
    if workflow is None:
        choices = ", ".join(WORKFLOWS)
        raise ValueError(
            f"Unsupported data.task_type={task_type!r}. Choose one of: {choices}."
        )

    runpy.run_path(str(workflow), run_name="__main__")


if __name__ == "__main__":
    main()
