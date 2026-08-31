import warnings
warnings.filterwarnings("ignore")

import os, yaml, argparse, random, time
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import wandb
from networks.dinov3_seg import DINOv3ViTSeg
from networks.svsswrapper import SVSSWrapper
from transformers import Sam2VideoModel, Sam2VideoProcessor
from networks.sam2_wrapper import SAM2SVSSWrapper
from networks.xmem_wrapper import XMemSVSSWrapper, XMemCalibWrapper
from networks.stm_wrapper import STMBaselineWrapper, STMCalibWrapper
from networks.raft_wrapper import RAFTFlowPropWrapper, RAFTFlowCalibWrapper, SupportCopyCalibWrapper


from utils import nested_dotdict, run_training, banner, build_optimizer, DiceCELoss
from trainer_video import Trainer

from dataset_video import SVSSDataset, svss_collate, svss_collate_stream, worker_init_fn

start = time.time()
default_config = "/users/nsapkota/VOS/cfg/data/cholecseg8k.yaml"

parser = argparse.ArgumentParser()
parser.add_argument("-cfg", "--config", default=default_config)
args = parser.parse_args()

# ---- load config
with open(args.config, "r") as f:
    cfg_dict = yaml.load(f, Loader=yaml.FullLoader)

cfg = nested_dotdict(cfg_dict)

# ---- debug mode
if cfg.experiment.debug:
    cfg.experiment.name = ""
    cfg.train.model_path = ""
    cfg.train.epochs = min(100, int(cfg.train.epochs))
    cfg.train.num_workers = min(0, int(getattr(cfg.train, "num_workers", 2)))
    cfg.val.wandb_vis = False

if 'sam2' in cfg.train.model:
    cfg.train.eval_only = True

if 'dv3' not in cfg.train.model:
    cfg.train.use_raw_logits = False

if cfg.train.eval_only:
    cfg.train.epochs = 0

# ---- seed
seed = cfg.experiment.seed or random.randint(0, 2**32 - 1)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
cfg.experiment.seed = seed

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# ---- device
device_idx = int(getattr(cfg.train, "gpu", 0))
device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")

ts = time.strftime("%m%d_%H%M%S")

# ---- wandb and logging
run_hp = (
    f'{cfg.experiment.SUFFX}'
    f'_{cfg.train.resize_h}p'
    # f'sd.{seed}'
    f'_{cfg.data.name[0].upper()}'
    # f'_{cfg.train.model}'
    # f'_pt{str(cfg.train.pt_encoder)[0]}'
    f'_ft{str(cfg.train.ft_encoder)[0]}'
    # f"_elr.{int(cfg.train.lr_enc)}"
    # f"_dlr.{int(cfg.train.lr_dec)}"
    # f"_wd.{cnf.train.wd}"
    # f'_dlw{cfg.train.dice_weight}'
    # f'_ibg{str(cfg.train.dice_inc_bg)[0]}'
    # f'_bs{cfg.train.bs}'
    f'_ds{str(cfg.train.use_stream)[0]}'
    f'_tqr{cfg.train.t_query}.{cfg.train.rollout_len}'
    f'_sf{cfg.train.s_support}'
    f'_vqr{cfg.val.rollout_len}'
    # f'_ep.{cnf.train.epochs}'
    # f'_amp.{str(cnf.train.amp)[0]}'
)

mem_hp = (
    # f"_mem{str(cfg.train.use_memory)[0]}"
    f"_amtm{str(cfg.train.use_am)[0]}"
    f"{str(cfg.train.use_tm)[0]}"
    f"_pa.{str(cfg.train.allow_pseudo_anchors)[0]}"
    f"_gate.{cfg.train.gate_mode}"
) if cfg.train.use_memory else ""


run_name = run_hp + mem_hp + '_'


# -------------------------------
# Checkpoint + results directories
# -------------------------------
ckpt_root = os.path.join(cfg.train.model_path, run_name)          # weights go here
results_root = os.path.join(cfg.train.model_path, "results", run_name)  # preds/GT go here
os.makedirs(ckpt_root, exist_ok=True)
os.makedirs(results_root, exist_ok=True)

# knobs (safe defaults)
cfg.train.save_every = int(getattr(cfg.train, "save_every", 5))          # save weights every N epochs
cfg.val.save_preds_every = int(getattr(cfg.val, "save_preds_every", 20)) # dump preds every N epochs
cfg.val.save_preds_max_batches = int(getattr(cfg.val, "save_preds_max_batches", 2))  # limit IO
cfg.val.save_preds_max_frames = int(getattr(cfg.val, "save_preds_max_frames", 50))  # limit frames/video

# store paths on cfg so trainer/utils can use them
cfg.train.ckpt_root = ckpt_root
cfg.val.results_root = results_root





wandb_run = None
if cfg.experiment.name:
    wandb_run = wandb.init(
        project=cfg.experiment.name,
        config=cfg_dict,
        name=run_name,
        notes=getattr(cfg.experiment, "notes", ""),
    )

with banner(top=True):
    print('\n')
    print(f"EXPERIMENT -- {run_name}")

with banner():
    print(f"cfg: {args.config}")
    print('\n')
    pprint(cfg)
    print('\n')

# ---- data
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if "video_src" not in df.columns:
        raise ValueError("Video manifests require a 'video_src' column.")

    df = df.copy()
    df["video_src"] = df["video_src"].fillna("").astype(str)
    if "video_clip" not in df.columns:
        # A manifest with one clip per video needs only video_src.
        df["video_clip"] = df["video_src"]
    else:
        df["video_clip"] = df["video_clip"].fillna("").astype(str)
    return df

train_df = normalize_df(pd.read_csv(cfg.data.train))
test_df  = normalize_df(pd.read_csv(cfg.data.test))

# if "sam" in cfg.train.model:    
#     test_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)


# ---------------- Debug mode ----------------
# if getattr(cfg.experiment, "debug", False):
#     train_df = train_df.iloc[:10].reset_index(drop=True)
#     test_df  = test_df.iloc[:10].reset_index(drop=True)

train_ds = SVSSDataset(
    cfg=cfg,
    df=train_df,
    split="train",
)

# ---- VAL OPTION A (recommended): full video eval (all frames 1..N-1)
val_ds = SVSSDataset(
    cfg=cfg,
    df=test_df,
    split="val",
    val_t_query=cfg.val.val_t_query,   # ✅ all frames
)

batch_size = int(getattr(cfg.train, "batch_size", 1))
num_workers = int(getattr(cfg.train, "num_workers", 2))

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
    collate_fn=svss_collate_stream if cfg.train.use_stream else svss_collate
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
    collate_fn=svss_collate_stream if cfg.val.use_stream else svss_collate
)

with banner(top=True):
    print(f"Run: {run_name} | Device: {device} ")
    print('\n')


# ---- model / loss / opt
if "sam2" in cfg.train.model:
    # facebook/sam2.1-hiera-tiny
    # facebook/sam2.1-hiera-base
    model_hf = Sam2VideoModel.from_pretrained(
        f"facebook/sam2.1-hiera-{cfg.train.model.split('_')[-1]}",
    ).to("cuda", dtype=torch.bfloat16)

    # ---- FORCE object score logits ON (prevents KeyError)
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

elif "xmem" in cfg.train.model:
    xmem = XMemSVSSWrapper(
        ckpt_path="/users/nsapkota/VOS/src/baselines/XMem/XMem.pth",
        num_classes=cfg.data.num_class,
        device=device,
        mem_every=5,
        enable_long_term=True,
        expects_01=True,
        imagenet_norm=False,
    ).cuda()

    model = XMemCalibWrapper(
        xmem=xmem,
        num_classes=cfg.data.num_class,
        head="2layer",   # or "1x1"
        hidden=64,
        dropout=0.1,
        train_xmem=False,
    ).cuda().train()

elif 'stm' in cfg.train.model:
# 1) build frozen STM core
    stm_core = STMBaselineWrapper(
        num_classes=cfg.data.num_class,
        ignore_index=cfg.data.ignore_index,
        bg_index=0,
        device=device,

        # surgical-friendly defaults
        memorize_all_support=True,
        online_update=False,
        write_stride=2,
        keep_last=20,
        conf_thr=0.60,
    )

    stm_core.load_pretrained('/users/nsapkota/VOS/src/baselines/STM/STM_weights.pth')

    # 2) wrap with trainable calibration head
    model = STMCalibWrapper(
        stm=stm_core,
        num_classes=cfg.data.num_class,
        head="1x1",          # or "2layer"
        hidden=64,
        dropout=0.0,
        train_stm=False,    # keep STM frozen (recommended)
    )

    model = model.to(device)
    model.train()

    # load pretrained STM weights
    # model.eval()

elif 'raft' in cfg.train.model:

    model = SupportCopyCalibWrapper(
        num_classes=cfg.data.num_class,
        ignore_index=cfg.data.ignore_index,
        device=device,
        bg_index=0,
    ).to(device)

else:
    # facebook/dinov3-vits16plus-pretrain-lvd1689m
    # facebook/dinov3-vitb16-pretrain-lvd1689m
    # facebook/dinov3-vitl16-pretrain-lvd1689m
    # facebook/dinov3-vith14-pretrain-lvd1689m   

    backbone = DINOv3ViTSeg(
        model_name=f"facebook/dinov3-{cfg.train.model.split('_')[-1]}-pretrain-lvd1689m",
        num_classes=cfg.data.num_class,
        pt_encoder=cfg.train.pt_encoder,
        ft_encoder=cfg.train.ft_encoder,
    ).to(device)

    # -------------------------------
    # ABD wrapper (memory + reliability + attention)
    #   Supports multi-support + absolute time PE
    # -------------------------------
    model = SVSSWrapper(
        backbone,                                   # base DINOv3 segmentation model
        patch_size=16,                              # ViT patch size

        # ---- memory capacity
        K=cfg.train.K,                              # TM keeps last K frames
        max_dt=cfg.train.max_dt,                    # temporal PE window (relative)

        # ---- memory switches
        use_memory=cfg.train.use_memory,            # enable memory
        use_am=cfg.train.use_am,                    # use anchor memory
        use_tm=cfg.train.use_tm,                    # use transient memory

        # ---- write / read compression
        write_topk_patch_tokens=cfg.train.write_topk_patch_tokens,  # TM token budget
        read_topk_mem_tokens=cfg.train.read_topk_mem_tokens,        # global read pruning

        # ---- asymmetric fusion
        alpha_am=cfg.train.alpha_am,                # anchor memory weight
        alpha_tm=cfg.train.alpha_tm,                # transient memory weight
        learnable_alpha=cfg.train.learnable_alpha,

        # ---- TM warmup / duplication control
        tm_warmup=cfg.train.tm_warmup,              # start TM after warmup
        skip_tm_t0=bool(getattr(cfg.train, "skip_tm_t0", True)),

        # ---- reliability gate (TM)
        gate_mode=cfg.train.gate_mode,              # off | conf | ent | conf+ent
        gate_conf_thr=cfg.train.gate_conf_thr,      # confidence threshold
        gate_ent_thr=cfg.train.gate_ent_thr,        # entropy threshold

        # ---- PSEUDO-ANCHORS (AM from high-quality predictions)  ✅ NEW
        allow_pseudo_anchors=bool(getattr(cfg.train, "allow_pseudo_anchors", False)),
        pseudo_use_fused_logits=bool(getattr(cfg.train, "pseudo_use_fused_logits", True)),
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

        # ---- memory attention
        use_mem_attention=cfg.train.use_mem_attention,  # enable attention read
        attn_sharp=cfg.train.attn_sharp,                # attention sharpness
        attn_topk_am=cfg.train.attn_topk_am,
        attn_topk_tm=cfg.train.attn_topk_tm,
        am_attn_beta=cfg.train.am_attn_beta,            # AM weight bias inside attention

        # ---- label handling
        bg_index=int(getattr(cfg.data, "bg_index", 0)),
        ignore_index=cfg.data.ignore_index,

        # ---- AM budget / eviction
        am_max_items=cfg.train.am_max_items,            # total anchors per video (e.g., 8)
        am_red_lambda=cfg.train.am_red_lambda,          # redundancy penalty (e.g., 0.5)

        # ---- Multi-support / evolving anchors
        enable_am_refresh=bool(getattr(cfg.train, "enable_am_refresh", False)),
        am_refresh_sim_max=float(getattr(cfg.train, "am_refresh_sim_max", 0.90)),
        am_max_per_class=int(getattr(cfg.train, "am_max_per_class", 3)),

        # ---- Temporal PE correctness with evenly-spaced supports
        use_abs_time=bool(getattr(cfg.train, "use_abs_time", True)),
        max_time_index=int(getattr(cfg.train, "max_time_index", 4096)),

        # ---- misc
        debug=bool(cfg.train.debug[1]),
        dbg_level=-1,        # start with 2
        dbg_every=10,        # print every 10 frames
        dbg_first_video_only=True,
        detach_memory=bool(getattr(cfg.train, "detach_memory", True)),
    ).to(device)


    # If we plan to finetune later, start frozen (warmup)
    if cfg.train.ft_encoder:
        for p in model.frame_model.encoder.parameters():
            p.requires_grad = False


train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
with banner():
    total_params = sum(p.numel() for p in model.parameters())
    print('\n')
    print("💫 Model Summary 💫")
    print(f"Network      : {cfg.train.model}")
    print(f"Parameters   : {total_params:,} ({total_params/1e6:.2f} M)")
    print(f"Trainable params: {train_params}")
    print(f"PT Encoder   : {cfg.train.pt_encoder}")
    print(f"FT Encoder   : {cfg.train.ft_encoder}")


ignore_index = int(getattr(cfg.train, "ignore_index", 255))

criterion = DiceCELoss(
    num_classes=cfg.data.num_class,
    ignore_index=cfg.data.ignore_index,
    dice_weight=cfg.train.dice_weight,   # 0.1 or 0.5
    include_bg=cfg.train.dice_inc_bg,
).to(device)

wd = float(getattr(cfg.train, "wd", 1e-4))

optimizer = build_optimizer(cfg, model)


lr_scheduler = None
if getattr(cfg.train, "use_scheduler", False):
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.train.epochs),
        eta_min=float(getattr(cfg.train, "min_lr", 1e-6)),
    )

trainer = Trainer(
    cfg=cfg,
    model=model,
    device=device,
    device_idx=device_idx,
    criterion=criterion,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
)


# ---- train
stats = run_training(
    cfg=cfg,
    trainer=trainer,
    train_loader=train_loader,
    val_loader=val_loader,
    wandb_run=wandb_run,
    topk=5,
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

elapsed = time.time() - start
hrs, rem = divmod(elapsed, 3600)
mins, secs = divmod(rem, 60)
tstr = f"{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"

print('\n', f"  {run_name}")

with banner(top=False):
    print('\n', '--'*18, f"** END. [{tstr}] **", '--'*18, '\n')

if wandb_run is not None:
    wandb.finish()
