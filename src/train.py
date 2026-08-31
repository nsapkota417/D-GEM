"""Unified D-GEM training entry point.

The selected workflow is controlled by ``data.task_type`` in the YAML config:
``image`` for independent image segmentation and ``video`` for D-GEM's
sequential video workflow.
"""

from __future__ import annotations

import argparse
import runpy
import sys
import tempfile
from pathlib import Path
from typing import Any

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
    parser.add_argument(
        "-m_cfg",
        "--model-config",
        default=None,
        help="Optional model YAML overriding the model_config declared in the base config.",
    )
    parser.add_argument(
        "--task-type",
        choices=tuple(WORKFLOWS),
        help="Override data.task_type in the base config.",
    )
    parser.add_argument(
        "--data-csv",
        help="Use one CSV manifest for both training and evaluation.",
    )
    parser.add_argument("--train-csv", help="CSV manifest for training.")
    parser.add_argument("--test-csv", help="CSV manifest for evaluation.")
    memory_group = parser.add_mutually_exclusive_group()
    memory_group.add_argument(
        "--use-memory", dest="use_memory", action="store_true", help="Enable D-GEM memory."
    )
    memory_group.add_argument(
        "--no-memory", dest="use_memory", action="store_false", help="Disable D-GEM memory."
    )
    parser.set_defaults(use_memory=None)
    return parser.parse_known_args()[0]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested mappings without discarding base configuration sections."""
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_model_config(
    config: dict[str, Any], config_path: Path, cli_model_config: str | None
) -> dict[str, Any]:
    model_config = cli_model_config or config.get("model_config")
    if not model_config:
        return config

    model_path = Path(model_config).expanduser()
    if not model_path.is_absolute():
        model_path = (Path.cwd() if cli_model_config else config_path.parent) / model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"Model config file not found: {model_path}")

    with model_path.open("r", encoding="utf-8") as model_file:
        model_config_data = yaml.safe_load(model_file) or {}
    if not isinstance(model_config_data, dict):
        raise ValueError(f"Model config must be a mapping: {model_path}")

    merged = deep_merge(config, model_config_data)
    merged.pop("model_config", None)
    return merged


def workflow_argv(merged_config: dict[str, Any]) -> tuple[list[str], Path]:
    """Give the legacy workflow an ephemeral, already-merged config file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="dgem_merged_", delete=False, encoding="utf-8"
    ) as temp_file:
        yaml.safe_dump(merged_config, temp_file, sort_keys=False)
        merged_path = Path(temp_file.name)

    argv = [sys.argv[0]]
    skip_next = False
    for argument in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if argument in {"-cfg", "--config"}:
            argv.extend([argument, str(merged_path)])
            skip_next = True
        elif argument.startswith("--config="):
            argv.append(f"--config={merged_path}")
        elif argument in {
            "-m_cfg", "--model-config", "--task-type", "--data-csv", "--train-csv", "--test-csv"
        }:
            skip_next = True
        elif argument.startswith(("--model-config=", "--task-type=", "--data-csv=", "--train-csv=", "--test-csv=")):
            continue
        elif argument in {"--use-memory", "--no-memory"}:
            continue
        else:
            argv.append(argument)
    return argv, merged_path


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    if not isinstance(config, dict):
        raise ValueError("Base config must be a mapping.")

    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("Config must contain a top-level 'data' mapping.")

    if args.task_type:
        data["task_type"] = args.task_type
    if args.data_csv:
        data["train"] = args.data_csv
        data["test"] = args.data_csv
    if args.train_csv:
        data["train"] = args.train_csv
    if args.test_csv:
        data["test"] = args.test_csv

    task_type = str(data.get("task_type", "")).lower()
    workflow = WORKFLOWS.get(task_type)
    if workflow is None:
        choices = ", ".join(WORKFLOWS)
        raise ValueError(
            f"Unsupported data.task_type={task_type!r}. Choose one of: {choices}."
        )

    merged_config = resolve_model_config(config, config_path, args.model_config)
    if args.use_memory is not None:
        merged_config.setdefault("train", {})["use_memory"] = args.use_memory
    original_argv = sys.argv
    sys.argv, merged_path = workflow_argv(merged_config)
    try:
        runpy.run_path(str(workflow), run_name="__main__")
    finally:
        sys.argv = original_argv
        merged_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
