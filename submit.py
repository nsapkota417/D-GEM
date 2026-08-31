import datetime
import pathlib 
from src.job import Job
from itertools import product
import random

# ========================================================================== #
# * ### * ### * ### *          Submission Code           * ### * ### * ### * #
# ========================================================================== #

SRC_PATH = str(pathlib.Path().absolute())
NUM_GPUS = 1
seed = random.randint(1, 1000)

DT = datetime.datetime.now()
DATE = f'{DT.month:02d}{DT.day:02d}'

current_time = datetime.datetime.now()
timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")

if __name__ == '__main__':

    for seed in [417]:
        config_common = {}

        project = "ssvss_baselines_vis"  # ssvss_baselines_sam_20-60-20, ssvss_baselines_roll_out, ssvss_baselines_roll_out_debug

        notes = "Phases corrected; TM updated supports class-level memory. AM supports multi-frame anchors"

        epochs = 50
        amp = True
        unfreeze_epoch = 3

        # options:

        datasets = [
            'cholecseg8k',
            'endovis',
            'sarrarp50',
        ]

        model_options = [
            'dv3_vits16plus',
            # 'dv3_vitl16',
            # 'dv3_vitb16'
            # 'sam2.1_large',
            # 'sam2.1_tiny',
            # 'sam2.1_small',

            # 'stm'
            # 'xmem'
        ]

        use_stream = False
        SUFFX = model_options[0] # 'DRPv3' if  use_stream else 'NoDRP'

        ft_encoder_flags = [
            True,
            # False
        ]

        memory_am_tm_comb = [
            [False, False, False],  # No Memory
            # [True, True, False],      # AM Only
            # [True, True, True],     # All Memory
            # [True, False, True],    # TM Only
        ]

        # [s_support, enable_am_refresh, am_refresh_sim_max, am_max_items, am_max_per_class, allow_pseudo_anchors]
        am_knobs_combo = [
            [1, False, 0.95, 8, 1, False],
            # [3, True, 0.95, 8, 1, False],
            # [3, True, 0.95, 12, 2, True],
            # [10, True, 0.90, 18, 3, True],
        ]

        gate_modes = [
            'off',
            # 'conf',
            # 'ent',
            # 'conf+ent'
        ]

        gpu = '@@crc_gpu'  # '' -- '@@crc_gpu' -- '@@csecri' -- '@@csecri-v100' -- '@qa-v100-001'

        # sam2_enhanced, sam2_reprompt_every
        sam2_options_list = [
            # [True, 3],
            [False, 0]
        ]

        enc_lrs = [
            0.00002,

            # 0.00005,        # default
            # 0.0005,
        ]

        dec_lrs = [
            0.0002,       # DV3 (0.0002, 0.0005)

            # 0.0005,
            # 0.002,
            # 0.0002,
            # 0.0005,
            # 0.005
            # 0.00002
        ]

        dice_weight = [
            0.1,

            # 0,
            # 0.2,
            # 0.5
        ]

        dice_inc_bg_opt = [
            False, 
        ]

        t_query_list = [
            10,                 # default
            # 3,

            # 16,                 
            # 20,

            # 32,
            # 40,
            # 80
        ]

        for dw, tq, memflags, gate_mode, ft_encoder, dice_inc_bg, enc_lr, dec_lr, model, dataset, am_knobs, sam2_opt in product(
            dice_weight, t_query_list, memory_am_tm_comb, gate_modes, ft_encoder_flags, dice_inc_bg_opt,
            enc_lrs, dec_lrs, model_options, datasets, am_knobs_combo, sam2_options_list
        ):
            if dataset == 'sarrarp50':
                num_ch = 3
                num_class = 10
                code_to_class = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,255:255}
                label_json = ''
                mask_col = 'mask'
                val_t_query = 60
                # resize_h, resize_w = 160, 288 
                # resize_h, resize_w = 224, 400 
                # resize_h, resize_w = None, None 
                resize_h, resize_w = 720, 1280

            elif dataset == 'endovis':
                num_ch = 3
                num_class = 12
                code_to_class = {}
                label_json = '/users/nsapkota/VOS/data/datasets/endovis/labels.json'
                mask_col = 'mask'
                val_t_query = 60 # if 'csecri' in gpu else -1
                # resize_h, resize_w = 160, 208 
                resize_h, resize_w = 640, 800  # 640×800

            else:
                num_ch = 3
                num_class = 13
                code_to_class = {
                    50: 0, 0: 0, 11: 1, 21: 2, 13: 3, 12: 4, 31: 5, 23: 6, 24: 7,
                    25: 8, 32: 9, 22: 10, 33: 11, 5: 12, 255: 255
                }
                label_json = ''
                mask_col = 'watershed_mask'
                val_t_query = 80
                # resize_h, resize_w = 160, 288
                # resize_h, resize_w = 224, 400 
                resize_h, resize_w = 480, 854 

            # if ('v100' not in gpu) and ('ka' in model): 
            #     gpu = '@@crc_gpu'
            #     print(f'Forcing {model} training on {gpu} clusters.')

            # get initial configs (will be merged inside Job)
            config = f'cfg/data/{dataset}.yaml'
            # model_config = f"configs/model/{model.split('_')[0]}_cnf.yaml"
            
            eval_only = True if 'sam' in model else False
            
            job_name = str(dataset[0]).upper() + '_' + str(model[0]) + str(ft_encoder)[0] + str(tq) + str(memflags[1])[0] + str(memflags[2])[0] + gate_mode[0] 
            job_name = job_name.replace('db', '')
            job_name = job_name.replace('_', '')
            print(job_name)

            # if not ft_encoder:
            #     epochs = 30

            # flat changes_d overrides everything (train + model)
            changes_d = {**config_common, **{
                # -------------------------
                # experiment
                # -------------------------
                "experiment.seed": seed,
                "experiment.name": project,
                "experiment.debug": False,
                "experiment.notes": notes,
                "experiment.SUFFX": SUFFX,

                # -------------------------
                # data
                # -------------------------
                "data.name": dataset,
                "data.root": f"/users/nsapkota/VOS/data/datasets/{dataset}",
                "data.train": f"/users/nsapkota/VOS/data/meta/{dataset}_train.csv",
                "data.test": f"/users/nsapkota/VOS/data/meta/{dataset}_test.csv",
                "data.bg_index": 0,
                "data.mask_col": mask_col,
                "data.num_ch": num_ch,
                "data.num_class": num_class,
                "data.ignore_index": 255,

                # optional: label mapping (keep if you want reproducibility)
                "data.code_to_class": code_to_class,
                "data.label_json" : label_json,

                # -------------------------
                # training setup
                # -------------------------
                "train.eval_only": eval_only,
                "train.debug" : [False, False],
                "train.epochs": epochs,
                "train.bs": 1,
                "train.num_workers": 8,
                "train.model_path": "",
                "train.t_query": tq,
                "train.s_support": am_knobs[0],
                "train.resize_h" : resize_h,
                "train.resize_w" : resize_w,

                "train.jitter": 8,
                "train.allowed_max_items": 500,
                "train.allowed_seed": 1337,
                "train.allowed_kwin": 2,
                "train.allowed_strict_consecutive": True,
                "train.support_jitter": 8,      # match train.jitter (or 0)
                "train.epoch_salt": 5,          # default only; override in Trainer each epoch

                # -------------------------
                # streaming + rollout
                # -------------------------
                "train.use_raw_logits": True,
                "train.aux_raw_w" : 0.2,
                "train.use_stream": use_stream,            # enable streaming mode
                "train.stream_query": True,          # dataset returns query paths (not tensors)
                "train.rollout_mode": "window",        # full | window | allowed
                "train.rollout_len": 40,             # -1 = full video rollout
                "train.rollout_start_idx": 1,        # exclude support frame (t=0)

                "val.use_stream": False,
                "val.rollout_mode": 'window',
                "val.rollout_len": 50,
                "val.val_t_query": 50,
                "val.rollout_start_idx" : 1,
                # -------------------------
                # sparse supervision
                # -------------------------
                "train.loss_on_query_indices_only": True,  # loss + mIoU only on t_query frames

                # Hyper-parameters

                "train.wd": 1e-4,
                "train.lr_enc": enc_lr,
                "train.lr_dec": dec_lr,
                "train.amp": amp,
                "train.dice_weight": dw,
                "train.dice_inc_bg":dice_inc_bg,

                # backbone / finetune
                "train.model": model,

                "train.pt_encoder": True,
                "train.ft_encoder": ft_encoder,
                "train.unfreeze_epoch": unfreeze_epoch,

                # ---- SAM2 (only used when train.model is sam2.*)
                "train.sam2_enhanced": sam2_opt[0],
                "train.sam2_reprompt_every": sam2_opt[1],
                "train.sam2_topk_per_class": 3,
                "train.sam2_cc_min_area": 25,
                "train.sam2_reprompt_cc_min_area": 50,
                "train.sam2_reprompt_prob_thresh": 0.35,

                # -------------------------
                # memory switches
                # -------------------------
                "train.use_memory": memflags[0],
                "train.use_am": memflags[1],
                "train.use_tm": memflags[2],
                "train.K": 20,
                "train.max_dt": 64,

                # -------------------------
                # AM (Anchor Memory)
                # -------------------------
                "train.am_max_items": am_knobs[3],
                "train.am_red_lambda": 0.5,
                "train.am_attn_beta": 2.0,
                "train.am_keep_k": 256,
                "train.attn_topk_am": 256,              # read top-k for AM
                "train.enable_am_refresh": am_knobs[1],     # IMPORTANT for multi-anchors
                "train.am_refresh_sim_max": am_knobs[2],
                "train.am_max_per_class": am_knobs[4],

                # -------------------------
                # TM (Transient Memory)
                # -------------------------
                "train.attn_topk_tm": 128,              # read top-k for TM
                "train.write_topk_patch_tokens": 128,
                "train.read_topk_mem_tokens": 0,
                "train.tm_warmup": 1,

                # -------------------------
                # memory attention + fusion
                # -------------------------
                "train.use_mem_attention": True,
                "train.attn_sharp": 80,
                "train.alpha_am": 0.2,
                "train.alpha_tm": 0.15,
                "train.learnable_alpha": True,

                # -------------------------
                # reliability gate (TM writes)
                # -------------------------
                "train.gate_mode": gate_mode,
                "train.gate_conf_thr": 0.12,
                "train.gate_ent_thr": 0.99,

                # -------------------------
                # misc / time / detach
                # -------------------------
                "train.skip_tm_t0": True,
                "train.use_abs_time": True,
                "train.max_time_index": 4096,
                "train.detach_memory": True,

                # -------------------------
                # Pseudo-Anchor gate (AM from predictions) — NEW
                # -------------------------
                "train.allow_pseudo_anchors": am_knobs[5],
                "train.pseudo_use_fused_logits": am_knobs[5],

                "train.pseudo_every": 5,
                "train.pseudo_warmup": 10,

                "train.pseudo_tau": 0.9,
                "train.pseudo_q99_thr": 0.98,
                "train.pseudo_mean_in_thr": 0.92,
                "train.pseudo_min_area": 0.002,
                "train.pseudo_max_area": 0.12,

                "train.pseudo_streak_req": 2,

                "train.pseudo_k_am": 128,
                "train.pseudo_max_per_class": 1,
                "train.pseudo_conf_scale": 0.20,
                "train.pseudo_w_scale": 0.4,
                # -------------------------
                # validation
                # -------------------------
                "val.val_every": 5,
                "val.wandb_vis": False,
                "val.wandb_vis_every": 50,
                "val.wandb_vis_max": 2,
                "val.wandb_vis_frames": 3,
            }}

            # model config overrides
       
            changes_d.update({
                    # "model_params.use_resconv": use_resConv,
                    # "model_params.num_heads": 3,
                    # "model_params.dropout_rate": 0.1,
                    # "model_params.use_postnorm": use_postNorm,
                    # "model_params.use_triton":use_triton,
                    # "model_params.num_registers":num_reg,
                    # "model_params.use_kan": use_kan,
                    # "model_params.register_stages": reg_st,
                    # "model_params.use_register_norm":use_register_norm,
                    # "model_params.kan_group_size":kan_group_size,
                    # "model_params.kan_poly_order":kan_poly_order,
                })
            
            job = Job(
                job_name=job_name, 
                src_path=SRC_PATH, 
                config_file=config, 
                # model_config_file=model_config,
                changes_d=changes_d,
                num_gpus=NUM_GPUS, 
                gpu=gpu
            )
            
            job.submit(n=1)