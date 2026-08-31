import warnings
warnings.filterwarnings("ignore")

import os, yaml, argparse, random, time
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import wandb
from datetime import datetime
from pathlib import Path

from networks.dinov3_seg import DINOv3ViTSeg
from networks.svsswrapper import SVSSWrapper
from transformers import Sam2VideoModel, Sam2VideoProcessor
from networks.sam2_wrapper import SAM2SVSSWrapper

from utils import nested_dotdict, run_training, banner, build_optimizer, DiceCELoss
from trainer_img import Trainer, inference_with_miou

from dataset_video import (
    SVSSDataset,
    svss_collate,
    svss_collate_stream,
    worker_init_fn,
)
from dataset_image import ImageSegDataset, image_collate


start = time.time()
default_config = "/users/nsapkota/VOS/cfg/data/cholecseg8k.yaml"

parser = argparse.ArgumentParser()
parser.add_argument("-cfg", "--config", default=default_config)
parser.add_argument("-m_cfg", "--model_config", default=None)
args = parser.parse_args()


# -----------------------------
# load config
# -----------------------------
with open(args.config, "r") as f:
    cfg_dict = yaml.load(f, Loader=yaml.FullLoader)

if args.model_config:
    with open(args.model_config, "r") as f:
        model_cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
    cfg_dict.update(model_cfg_dict)

cfg = nested_dotdict(cfg_dict)


# -----------------------------
# task type
# -----------------------------
cfg.data.task_type = getattr(cfg.data, "task_type", "image")
is_image_task = str(cfg.data.task_type).lower() == "image"

if cfg.val.eval_only:
    cfg.train.bs = cfg.val.bs

# over-wride config:
cfg.data.num_ch = (
    3 if cfg.data.modality in ["swir_img", "wl_img"]
    else 4
)

cfg.data.img_col = (
    "wl_img" if cfg.data.modality in ["wl_img", "both_on_wl_img"]
    else "swir_img"
)

cfg.data.mask_col = (
    "wl_mask" if cfg.data.modality in ["wl_img", "both_on_wl_img"]
    else "swir_mask"
)

rep_col = (
    "wl_rep" if cfg.data.modality in ["wl_img", "both_on_wl_img"]
    else "swir_rep"
)

# -----------------------------
# debug mode
# -----------------------------
if cfg.experiment.debug:
    cfg.experiment.name = ""
    cfg.experiment.project_path = ""
    cfg.train.epochs = min(100, int(cfg.train.epochs))
    cfg.train.num_workers = min(0, int(getattr(cfg.train, "num_workers", 2)))
    cfg.val.wandb_vis = False


# -----------------------------
# force eval-only for SAM2
# -----------------------------
if "sam2" in cfg.train.model:
    cfg.val.eval_only = True

if "dv3" not in cfg.train.model:
    cfg.train.use_raw_logits = False

if cfg.val.eval_only:
    cfg.train.epochs = 0


# -----------------------------
# seed
# -----------------------------
seed = cfg.experiment.seed or random.randint(0, 2**32 - 1)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
cfg.experiment.seed = seed

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# -----------------------------
# device
# -----------------------------
device_idx = int(getattr(cfg.train, "gpu", 0))
device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")


# -----------------------------
# run name
# -----------------------------
suffix = getattr(
    cfg.experiment,
    "SUFFX",
    getattr(cfg.experiment, "SUFFIX", ""),
)

run_hp = (
    f"{suffix}"
    # f"_{cfg.train.resize_h}p"
    f"_pt{str(cfg.train.pt_encoder)[0]}"
    f"_ft{str(cfg.train.ft_encoder)[0]}"
)

run_hp += (
    f"_bs{cfg.train.bs}"
)

mem_hp = (
    f"_amtm{str(cfg.train.use_am)[0]}"
    f"{str(cfg.train.use_tm)[0]}"
    f"_pa.{str(cfg.train.allow_pseudo_anchors)[0]}"
    f"_gate.{cfg.train.gate_mode}"
) if (not is_image_task and cfg.train.use_memory) else ""

run_name = run_hp + mem_hp + "_"

if cfg.val.eval_only:
    run_name += f'{cfg.val.val_rows}'


# -----------------------------
# checkpoint + result dirs
# only create train ckpt dirs if training
# -----------------------------
if (not cfg.val.eval_only) and cfg.train.save_every >= 0:
    # ckpt_root = os.path.join(cfg.experiment.project_path, run_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_root = os.path.join(
        cfg.experiment.project_path,
        f"{run_name}_{timestamp}",
    )    
    results_root = os.path.join(ckpt_root, "results")
    os.makedirs(ckpt_root, exist_ok=True)
    # os.makedirs(results_root, exist_ok=True)

    cfg.train.save_every = int(getattr(cfg.train, "save_every", 5))
    cfg.val.save_preds_every = int(getattr(cfg.val, "save_preds_every", 20))
    cfg.val.save_preds_max_batches = int(getattr(cfg.val, "save_preds_max_batches", 2))
    cfg.val.save_preds_max_frames = int(getattr(cfg.val, "save_preds_max_frames", 50))

    cfg.train.model_save_path = ckpt_root
    cfg.val.results_root = results_root


# -----------------------------
# wandb only when requested
# -----------------------------
wandb_run = None
if cfg.experiment.name:
    wandb_run = wandb.init(
        project=cfg.experiment.name,
        config=cfg_dict,
        name=run_name,
        notes=getattr(cfg.experiment, "notes", ""),
    )


with banner(top=True):
    print("\n")
    print(f"EXPERIMENT -- {run_name}")

with banner():
    print(f"cfg: {args.config}")
    print("\n")
    pprint(cfg)
    print("\n")


# -----------------------------
# helpers
# -----------------------------
def normalize_df(df: pd.DataFrame, img_col='img') -> pd.DataFrame:

    df = df.copy()

    if "video_src" not in df.columns:
        df["video_src"] = ""
    if "video_clip" not in df.columns:
        df["video_clip"] = ""
    
    if img_col not in df.columns:
        df[img_col] = ""

    df["video_src"] = df["video_src"].fillna("").astype(str)
    df["video_clip"] = df["video_clip"].fillna("").astype(str)
    df[img_col] = df[img_col].fillna("").astype(str)

    df = df.sort_values(
        by=[img_col],
        kind="mergesort"
    ).reset_index(drop=True)

    return df

def select_by_frame_ranges(df, col, ranges):
    mask = False

    for r in ranges:
        if len(r) == 1:
            start = r[0]
            mask = mask | (df[col] >= start)
        else:
            start, end = r
            mask = mask | df[col].between(
                start, end, inclusive="both"
            )

    return df[mask].copy()

def select_df_by_video_range(df: pd.DataFrame, video_range):
    """
    YAML:
      val_rows: [0, 5]

    Now means:
      select videos with video-index 0,1,2,3,4
      based on first appearance order in CSV.
    """
    if video_range == -1:
        return df

    start_vid, end_vid = video_range

    video_ids = (
        df["video_src"]
        .fillna("")
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    selected_videos = video_ids[start_vid:end_vid]

    return df[df["video_src"].astype(str).isin(selected_videos)].copy()

def spaced_consecutive_sample_df(df, run=3, max_samples=30):
    n = len(df)

    if n == 0:
        return df

    num_anchors = max(1, max_samples // run)
    anchors = np.linspace(0, n - 1, num_anchors, dtype=int)

    idxs = []
    for start_idx in anchors:
        for i in range(run):
            idx = start_idx + i
            if idx < n:
                idxs.append(idx)

    idxs = idxs[:max_samples]

    return df.iloc[idxs]


def image_to_svss_collate(batch):
    """
    Adapter so image dataset can use support/query-shaped tensors.

    Produces:
      support_img      : (B, 1, 3, H, W)
      support_mask     : (B, 1, H, W)
      query_imgs       : (B, 1, 3, H, W)
      query_masks      : (B, 1, H, W)
      support_indices  : (B, 1)
      query_indices    : (B, 1)
    """

    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)

    images_5d = images.unsqueeze(1)
    masks_4d = masks.unsqueeze(1)

    out = {
        "image": images,
        "mask": masks,

        "support_img": images_5d,
        "support_mask": masks_4d,
        "support_indices": torch.zeros((len(batch), 1), dtype=torch.long),

        "query_imgs": images_5d,
        "query_masks": masks_4d,
        "query_indices": torch.zeros((len(batch), 1), dtype=torch.long),

        "video_src": [b.get("video_src", "") for b in batch],
        "video_clip": [b.get("video_clip", "") for b in batch],
        "video_len": torch.ones(len(batch), dtype=torch.long),

        "img_path": [b.get("img_path", "") for b in batch],
        "mask_path": [b.get("mask_path", "") for b in batch],
        "rep": [b.get("rep", "-") for b in batch],
    }

    return out


# -----------------------------
# common loader params
# -----------------------------
batch_size = int(getattr(cfg.train, "bs", 1))
num_workers = int(getattr(cfg.train, "num_workers", 2))

train_loader = None
val_loader = None


# -----------------------------
# data
# important:
# skip train/val dataset construction in eval-only
# -----------------------------
if not cfg.val.eval_only:

    # load safely, exclude pre-defined bad frames, select only represented frames
    train_df = normalize_df(pd.read_csv(cfg.data.train), img_col=cfg.data.img_col)
    train_df = select_by_frame_ranges(train_df, "frame_id", ranges=cfg.data.include_indices)
    if cfg.data.use_rep_only:
        train_df = train_df.loc[train_df[rep_col] == "rep"]

    # load safely, exclude pre-defined bad frames, select only represented frames
    test_df = normalize_df(pd.read_csv(cfg.data.test), img_col=cfg.data.img_col)
    test_df = select_by_frame_ranges(test_df, "frame_id", ranges=cfg.data.include_indices)
    test_df = test_df.loc[test_df[rep_col] != "rep"].copy()

    if cfg.train.sample_in_video:
        def sample_evenly(df, x):
            idx = np.linspace(0, len(df) - 1, x, dtype=int)
            return df.iloc[idx].copy()

        sampled_train_df = sample_evenly(
            train_df, 
            x=cfg.train.t_query
        )

        test_df = train_df.loc[
            ~train_df.index.isin(sampled_train_df.index)
        ].copy()

        train_df = sampled_train_df

    test_df = spaced_consecutive_sample_df(
        test_df,
        run=3,
        max_samples=cfg.val.max_samples,
    )

    with banner():
        print(f"train: {len(train_df)} | val: {len(test_df)}")

    if is_image_task:
        train_ds = ImageSegDataset(
            cfg=cfg,
            df=train_df,
        )

        val_ds = ImageSegDataset(
            cfg=cfg,
            df=test_df,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
            prefetch_factor=1 if num_workers > 0 else None,
            worker_init_fn=worker_init_fn,
            drop_last=False,
            collate_fn=image_to_svss_collate,
        )

        val_batch_size = int(getattr(cfg.val, "batch_size", batch_size))

        val_loader = DataLoader(
            val_ds,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            worker_init_fn=worker_init_fn,
            drop_last=False,
            collate_fn=image_to_svss_collate,
        )

    else:
        train_ds = SVSSDataset(
            cfg=cfg,
            df=train_df,
            split="train",
        )

        val_ds = SVSSDataset(
            cfg=cfg,
            df=test_df,
            split="val",
            val_t_query=cfg.val.val_t_query,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
            prefetch_factor=1 if num_workers > 0 else None,
            worker_init_fn=worker_init_fn,
            drop_last=False,
            collate_fn=svss_collate_stream if cfg.train.use_stream else svss_collate,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            worker_init_fn=worker_init_fn,
            drop_last=False,
            collate_fn=svss_collate_stream if cfg.val.use_stream else svss_collate,
        )


with banner(top=True):
    print(f"Run: {run_name} | Device: {device}")
    print("\n")


# -----------------------------
# model
# -----------------------------
if "sam2" in cfg.train.model:
    model_hf = Sam2VideoModel.from_pretrained(
        f"facebook/sam2.1-hiera-{cfg.train.model.split('_')[-1]}",
    ).to(device, dtype=torch.bfloat16)

    if hasattr(model_hf, "config") and hasattr(model_hf.config, "pred_obj_scores"):
        model_hf.config.pred_obj_scores = True

    if hasattr(model_hf, "pred_obj_scores"):
        model_hf.pred_obj_scores = True

    processor = Sam2VideoProcessor.from_pretrained(
        "facebook/sam2.1-hiera-small",
    )

    model = SAM2SVSSWrapper(
        model=model_hf,
        processor=processor,
        num_classes=cfg.data.num_class,
        ignore_index=cfg.data.ignore_index,
        include_bg_channel=True,
        torch_dtype=torch.bfloat16,
        use_enhanced_prompting=getattr(cfg.train, "sam2_enhanced", False),
        topk_per_class=getattr(cfg.train, "sam2_topk_per_class", 3),
        reprompt_every=getattr(cfg.train, "sam2_reprompt_every", 10),
        cc_min_area=getattr(cfg.train, "sam2_cc_min_area", 25),
        reprompt_cc_min_area=getattr(cfg.train, "sam2_reprompt_cc_min_area", 50),
        reprompt_prob_thresh=getattr(cfg.train, "sam2_reprompt_prob_thresh", 0.35),
    )

else:
    backbone = DINOv3ViTSeg(
        model_name=(
            f"facebook/dinov3-"
            f"{cfg.train.model.split('_')[-1]}-pretrain-lvd1689m"
        ),
        num_classes=cfg.data.num_class,
        pt_encoder=cfg.train.pt_encoder,
        ft_encoder=cfg.train.ft_encoder,
        in_chans=cfg.data.num_ch,
    ).to(device)

    if is_image_task:
        model = backbone
    else:
        model = SVSSWrapper(
            backbone,
            patch_size=16,

            K=cfg.train.K,
            max_dt=cfg.train.max_dt,

            use_memory=cfg.train.use_memory,
            use_am=cfg.train.use_am,
            use_tm=cfg.train.use_tm,

            write_topk_patch_tokens=cfg.train.write_topk_patch_tokens,
            read_topk_mem_tokens=cfg.train.read_topk_mem_tokens,

            alpha_am=cfg.train.alpha_am,
            alpha_tm=cfg.train.alpha_tm,
            learnable_alpha=cfg.train.learnable_alpha,

            tm_warmup=cfg.train.tm_warmup,
            skip_tm_t0=bool(getattr(cfg.train, "skip_tm_t0", True)),

            gate_mode=cfg.train.gate_mode,
            gate_conf_thr=cfg.train.gate_conf_thr,
            gate_ent_thr=cfg.train.gate_ent_thr,

            allow_pseudo_anchors=bool(
                getattr(cfg.train, "allow_pseudo_anchors", False)
            ),
            pseudo_use_fused_logits=bool(
                getattr(cfg.train, "pseudo_use_fused_logits", True)
            ),
            pseudo_every=int(getattr(cfg.train, "pseudo_every", 10)),
            pseudo_warmup=int(getattr(cfg.train, "pseudo_warmup", 0)),

            pseudo_tau=float(getattr(cfg.train, "pseudo_tau", 0.92)),
            pseudo_q99_thr=float(getattr(cfg.train, "pseudo_q99_thr", 0.97)),
            pseudo_mean_in_thr=float(getattr(cfg.train, "pseudo_mean_in_thr", 0.90)),
            pseudo_min_area=float(getattr(cfg.train, "pseudo_min_area", 0.001)),
            pseudo_max_area=float(getattr(cfg.train, "pseudo_max_area", 0.20)),

            pseudo_streak_req=int(getattr(cfg.train, "pseudo_streak_req", 2)),

            pseudo_k_am=int(getattr(cfg.train, "pseudo_k_am", 128)),
            pseudo_max_per_class=int(getattr(cfg.train, "pseudo_max_per_class", 1)),
            pseudo_conf_scale=float(getattr(cfg.train, "pseudo_conf_scale", 0.25)),
            pseudo_w_scale=float(getattr(cfg.train, "pseudo_w_scale", 0.50)),

            use_mem_attention=cfg.train.use_mem_attention,
            attn_sharp=cfg.train.attn_sharp,
            attn_topk_am=cfg.train.attn_topk_am,
            attn_topk_tm=cfg.train.attn_topk_tm,
            am_attn_beta=cfg.train.am_attn_beta,

            bg_index=int(getattr(cfg.data, "bg_index", 0)),
            ignore_index=cfg.data.ignore_index,

            am_max_items=cfg.train.am_max_items,
            am_red_lambda=cfg.train.am_red_lambda,

            enable_am_refresh=bool(
                getattr(cfg.train, "enable_am_refresh", False)
            ),
            am_refresh_sim_max=float(
                getattr(cfg.train, "am_refresh_sim_max", 0.90)
            ),
            am_max_per_class=int(getattr(cfg.train, "am_max_per_class", 3)),

            use_abs_time=bool(getattr(cfg.train, "use_abs_time", True)),
            max_time_index=int(getattr(cfg.train, "max_time_index", 4096)),

            debug=bool(cfg.train.debug[1]),
            dbg_level=-1,
            dbg_every=10,
            dbg_first_video_only=True,
            detach_memory=bool(getattr(cfg.train, "detach_memory", True)),
        ).to(device)

    if cfg.train.ft_encoder:
        if is_image_task:
            for p in model.encoder.parameters():
                p.requires_grad = False
        else:
            for p in model.frame_model.encoder.parameters():
                p.requires_grad = False


# -----------------------------
# force eval-only model state
# before optimizer/criterion
# -----------------------------
if cfg.val.eval_only:
    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False


# -----------------------------
# model summary only for train mode
# -----------------------------
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

if not cfg.val.eval_only:
    with banner():
        total_params = sum(p.numel() for p in model.parameters())

        print("\n")
        print("💫 Model Summary 💫")
        print(f"Network         : {cfg.train.model}")
        print(f"Task type       : {'image' if is_image_task else 'video'}")
        print(f"Parameters      : {total_params:,} ({total_params / 1e6:.2f} M)")
        # print(f"Trainable params: {train_params}")
        print(f"PT Encoder      : {cfg.train.pt_encoder}")
        print(f"FT Encoder      : {cfg.train.ft_encoder}")


# -----------------------------
# loss / optimizer / scheduler
# important:
# Trainer requires criterion, even in eval-only
# but optimizer/scheduler can be skipped
# -----------------------------
criterion = DiceCELoss(
    num_classes=cfg.data.num_class,
    ignore_index=cfg.data.ignore_index,
    dice_weight=cfg.train.dice_weight,
    include_bg=cfg.train.dice_inc_bg,
)

optimizer = None
lr_scheduler = None

if not cfg.val.eval_only:
    criterion = criterion.to(device)
    optimizer = build_optimizer(cfg, model)

    if getattr(cfg.train, "use_scheduler", False):
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.train.epochs),
            eta_min=float(getattr(cfg.train, "min_lr", 1e-6)),
        )


# -----------------------------
# trainer
# -----------------------------
trainer = Trainer(
    cfg=cfg,
    model=model,
    device=device,
    device_idx=device_idx,
    criterion=criterion,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
)

# -----------------------------
# eval mode
# -----------------------------
if cfg.val.eval_only:
    # inf_df = normalize_df(pd.read_csv(cfg.data.test))
    inf_df = normalize_df(
        pd.read_csv(cfg.data.test),
        img_col=cfg.data.img_col,
    )

    val_rows = cfg.val.val_rows

    if val_rows != -1:
        inf_df = select_df_by_video_range(inf_df, val_rows)

    with banner():
        print(f"inf_df: {len(inf_df)}")

    inf_ds = ImageSegDataset(
        cfg=cfg,
        df=inf_df,
    )

    inf_batch_size = int(getattr(cfg.val, "bs", batch_size))

    inf_loader = DataLoader(
        inf_ds,
        batch_size=inf_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
        drop_last=False,
        collate_fn=image_to_svss_collate,
    )

    ckpt = torch.load(cfg.train.ckpt_to_use, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    missing, unexpected = model.load_state_dict(ckpt, strict=False)

    print(f"✅ Loaded checkpoint: {cfg.train.ckpt_to_use}")

    if len(missing) > 0:
        print(f"⚠️ Missing keys: {len(missing)}")

    if len(unexpected) > 0:
        print(f"⚠️ Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    ets = datetime.now().strftime("%m%d_%H%M%S")

    res_dir = Path(cfg.val.res_dir) / f"{suffix}_{ets}"
    res_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        results = inference_with_miou(
            cfg=cfg,
            inf_df=inf_df,
            trainer=trainer,
            model=model,
            data_loader=inf_loader,
            device=device,
            save_dir=res_dir,
        )


# -----------------------------
# train mode
# -----------------------------
else:
    stats = run_training(
        cfg=cfg,
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        wandb_run=wandb_run,
        topk=5,
        run_name=run_name,
    )

    with banner(top=True, bottom=False):
        print("Top-5 (val) epochs (ranked by overall mIoU):\n")

        for r in stats["topk"]:
            print(
                f"  epoch {r['epoch']:>3d} | "
                f"mIoU: overall {r['val_metric']:.2f} -- "
                f"Phases {r['p1']:.2f} / "
                f"{r['p2']:.2f} / "
                f"{r['p3']:.2f}"
            )

        print("\n" + "_" * 75 + "\n")

        print(
            f"  Overall mIoU | Best: {stats['topk_best_overall']:.2f} "
            f"| Avg: {stats['topk_avg_overall']:.2f}\n"
            f"  Phase mIoU   | Best: "
            f"{stats['topk_best_p1']:.2f} / "
            f"{stats['topk_best_p2']:.2f} / "
            f"{stats['topk_best_p3']:.2f} | Avg: "
            f"{stats['topk_avg_p1']:.2f} / "
            f"{stats['topk_avg_p2']:.2f} / "
            f"{stats['topk_avg_p3']:.2f}\n"
        )


# -----------------------------
# finish
# -----------------------------
elapsed = time.time() - start
hrs, rem = divmod(elapsed, 3600)
mins, secs = divmod(rem, 60)
tstr = f"{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"

print("\n", f"  {run_name}")

with banner(top=False):
    print("\n", "--" * 18, f"** END. [{tstr}] **", "--" * 18, "\n")

if wandb_run is not None:
    wandb.finish()
