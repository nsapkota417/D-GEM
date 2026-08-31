from copy import deepcopy
from contextlib import contextmanager
import io
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "DiceCELoss",
    "DotDict",
    "banner",
    "build_optimizer",
    "nested_dotdict",
    "run_training",
]

def fmt_metric(metric):
    return int(round(float(metric) * 100))


def maybe_save_best(trainer, metric, best_metric, epoch, tag="best"):
    if metric > best_metric:
        best_metric = metric

        if trainer.cfg.experiment.project_path:
            save_tag = f"{tag}_ep{epoch:03d}_met{fmt_metric(metric):04d}"
            ckpt_path = trainer.save_model(save_tag)

            print(
                f"✅ Saved {save_tag} model to: {ckpt_path} "
                f"(metric={best_metric:.2f})"
            )

    return best_metric


def run_training(
    cfg,
    trainer,
    train_loader,
    val_loader,
    wandb_run=None,
    topk: int = 5,
):
    epochs = int(cfg.train.epochs)
    val_every = int(getattr(cfg.val, "val_every", 1))
    topk = int(topk)

    eval_only = bool(
        getattr(cfg.train, "eval_only", False) or getattr(cfg.val, "eval_only", False)
    )

    def should_val(ep: int) -> bool:
        return (
            ep == 0
            or ep == epochs - 1
            or ep % val_every == 0
        )

    def upd_topk(lst, row):
        lst.append(row)
        lst.sort(key=lambda x: x["val_metric"], reverse=True)
        return lst[:topk]

    def safe_stats(lst, key: str):
        vals = [x[key] for x in lst if x[key] == x[key]]
        if not vals:
            return float("nan"), float("nan")
        return float(max(vals)), float(sum(vals) / len(vals))

    best_overall = -1e9
    topk_list = []
    last_val_metric = float("nan")

    if eval_only:
        ep = 0

        val_loss, val_metric, vmp1, vmp2, vmp3 = trainer.validate(
            val_loader,
            epoch=ep,
            log_run=True,
        )

        m = float(val_metric)
        p1 = float(vmp1)
        p2 = float(vmp2)
        p3 = float(vmp3)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "val_loss": float(val_loss),
                    "val_metric": m,
                },
                step=ep,
            )

        topk_list = [{
            "epoch": int(ep),
            "val_loss": float(val_loss),
            "val_metric": m,
            "p1": p1,
            "p2": p2,
            "p3": p3,
        }]

        best_overall = m

    else:
        for ep in range(epochs):
            train_loss, train_metric = trainer.train(
                train_loader,
                ep,
                log_run=True,
            )

            if should_val(ep):
                val_loss, val_metric, vmp1, vmp2, vmp3 = trainer.validate(
                    val_loader,
                    epoch=ep,
                    log_run=True,
                )

                m = float(val_metric)
                last_val_metric = m
                p1 = float(vmp1)
                p2 = float(vmp2)
                p3 = float(vmp3)

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train_loss": float(train_loss),
                            "train_metric": float(train_metric),
                            "val_loss": float(val_loss),
                            "val_metric": m,
                        },
                        step=ep,
                    )

                # Save immediately whenever validation finds new best
                best_overall = maybe_save_best(
                    trainer,
                    m,
                    best_overall,
                    epoch=ep,
                    tag="best",
                )

                topk_list = upd_topk(
                    topk_list,
                    {
                        "epoch": int(ep),
                        "val_loss": float(val_loss),
                        "val_metric": m,
                        "p1": p1,
                        "p2": p2,
                        "p3": p3,
                    },
                )

            else:
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train_loss": float(train_loss),
                            "train_metric": float(train_metric),
                        },
                        step=ep,
                    )

        if topk_list and getattr(cfg.experiment, "project_path", None):
            save_tag = f"last_ep{ep:03d}_met{fmt_metric(last_val_metric):04d}"
            ckpt_path = trainer.save_model(save_tag)
            print(f"✅ Saved last model to: {ckpt_path}")

    best_topk_overall, avg_topk_overall = safe_stats(
        topk_list,
        "val_metric",
    )
    best_topk_p1, avg_topk_p1 = safe_stats(topk_list, "p1")
    best_topk_p2, avg_topk_p2 = safe_stats(topk_list, "p2")
    best_topk_p3, avg_topk_p3 = safe_stats(topk_list, "p3")

    return {
        "best_overall": float(best_overall),
        "topk": topk_list,
        "topk_avg_overall": avg_topk_overall,
        "topk_best_overall": best_topk_overall,
        "topk_avg_p1": avg_topk_p1,
        "topk_best_p1": best_topk_p1,
        "topk_avg_p2": avg_topk_p2,
        "topk_best_p2": best_topk_p2,
        "topk_avg_p3": avg_topk_p3,
        "topk_best_p3": best_topk_p3,
    }



class DotDict(dict):
    """Dictionary with attribute-style access; use ``nested_dotdict`` recursively."""

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as error:
            raise AttributeError(item) from error
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.items():
            setattr(result, k, deepcopy(v, memo))
        return result


def nested_dotdict(value):
    """Recursively convert dictionaries to :class:`DotDict` instances."""
    if not isinstance(value, dict):
        return value
    return DotDict({key: nested_dotdict(item) for key, item in value.items()})


@contextmanager
def banner(width=120, char="=", pad=2, top=False, bottom=True):
    if width <= pad * 2:
        raise ValueError("Width must be greater than twice the padding.")

    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer

    try:
        yield
    finally:
        sys.stdout = old_stdout

        text = buffer.getvalue().rstrip()
        if not text:
            return

        lines = text.split("\n")
        content_width = width - pad * 2

        border = char * width

        if top:
            print(border)

        for line in lines:
            truncated = line[:content_width]
            print(" " * pad + truncated.ljust(content_width) + " " * pad)

        # Bottom border
        if bottom:
            print(border)

def build_optimizer(cfg, model):
    enc_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("encoder."):
            enc_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if enc_params:
        param_groups.append({"params": enc_params, "lr": float(cfg.train.lr_enc)})
    if other_params:
        param_groups.append({"params": other_params, "lr": float(cfg.train.lr_dec)})

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=float(cfg.train.wd),
    )
    return optimizer

class DiceCELoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, dice_weight=0.5, include_bg=False):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dice_weight = float(dice_weight)
        self.include_bg = bool(include_bg)

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)

        if self.dice_weight <= 0:
            return ce

        probs = torch.softmax(logits, dim=1)

        valid = (targets != self.ignore_index)
        t = targets.clone()
        t[~valid] = 0

        onehot = F.one_hot(t, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        valid = valid.unsqueeze(1)
        probs = probs * valid
        onehot = onehot * valid

        intersection = (probs * onehot).sum((0,2,3))
        union = probs.sum((0,2,3)) + onehot.sum((0,2,3))

        dice_per_class = 1 - (2 * intersection + 1e-6) / (union + 1e-6)

        if self.include_bg:
            dice = dice_per_class.mean()
        else:
            dice = dice_per_class[1:].mean()

        return ce + self.dice_weight * dice
