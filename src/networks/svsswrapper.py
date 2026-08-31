import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from memory_state import MemoryState


class SVSSWrapper(nn.Module):
    """
    ABD v1 wrapper for SVSS Trainer API:
      out = model(support_img, support_mask, query_imgs) -> (B,T,C,H,W)
    """
    def __init__(
        self,
        frame_model: nn.Module,
        patch_size: int = 16,
        K: int = 3,
        max_dt: int = 64,

        # ---- Memory toggles
        use_memory: bool = True,
        use_am: bool = True,
        use_tm: bool = True,

        # ---- Semantic setup
        bg_index: int = 0,

        # ---- TopK controls
        write_topk_patch_tokens: int = 0,  # 0 disables; else store only top-k tokens per TM frame
        read_topk_mem_tokens: int = 0,     # 0 disables; else keep top-k tokens on read (global budget)

        # ---- Fusion control (separate weights)
        alpha_am: float = 0.35,            # AnchorMemory weight
        alpha_tm: float = 0.15,            # TransientMemory weight
        learnable_alpha: bool = True,      # learn alpha via sigmoid(logit)

        # ---- Misc
        tm_warmup: int = 2,                # start TM writes at t >= tm_warmup
        skip_tm_t0: bool = True,           # avoid duplicating AM with TM at t=0
        detach_memory: bool = True,        # detach memory tokens from graph
        debug: bool = False,

        
        # ---- Debug controls
        dbg_level: int = 0,              # 0 off; 1 key; 2 verbose; 3 very verbose
        dbg_every: int = 10,             # print every N frames in step()
        dbg_first_video_only: bool = True,  # only for first video/forward call
        # ---- Reliability gate (02/12)
        gate_mode: str = "conf+ent",       # "off" | "conf" | "ent" | "conf+ent"
        gate_conf_thr: float = 0.60,       # accept if mean max-softmax >= thr
        gate_ent_thr: float = 1.20,        # accept if mean entropy <= thr

        # ---- Memory attention read (02/12)
        use_mem_attention: bool = True,    # if False -> fallback to mean (v1 behavior)
        attn_topk_am: int = 256,            # max AM tokens used per read
        attn_topk_tm: int = 128,            # max TM tokens used per read
        attn_sharp: float = 60.0,           # inverse temperature (higher = sharper attention)

        # ---- Loss / mask handling
        ignore_index: int = 255,

        # ---- AnchorMemory specific
        am_max_items: int = 8,              # total anchors per video
        am_red_lambda: float = 0.5,         # redundancy penalty strength
        am_attn_beta: float = 2.0,          # weight for AM attention logits

        # ---- Anchor refresh / multi-support
        enable_am_refresh: bool = False,    # allow anchors from extra annotated frames
        am_refresh_sim_max: float = 0.90,   # diversity threshold (cos sim)
        am_max_per_class: int = 3,          # cap anchors per class (incl pinned)

        # ---- Pseudo-anchors (OFF by default)
        allow_pseudo_anchors: bool = False,
        pseudo_use_fused_logits: bool = True,     # use final logits (better) vs logits_raw
        pseudo_every: int = 8,                    # attempt write every N frames
        pseudo_warmup: int = 0,                   # don't write pseudo anchors before this many steps

        # gating
        pseudo_tau: float = 0.92,                 # pixel prob threshold for region
        pseudo_q99_thr: float = 0.97,             # require 99th percentile prob >= this
        pseudo_mean_in_thr: float = 0.90,         # mean prob inside mask >= this
        pseudo_min_area: float = 0.001,           # min fraction of pixels in region
        pseudo_max_area: float = 0.25,            # max fraction (avoid huge noisy regions)

        # stability / rate limiting
        pseudo_streak_req: int = 2,               # must pass gate this many consecutive checks before write

        # storage
        pseudo_k_am: int = 128,                   # how many patch tokens to store per pseudo-anchor
        pseudo_max_per_class: int = 1,            # cap pseudo anchors per class
        pseudo_conf_scale: float = 0.30,          # lower "conf" so eviction prefers GT anchors
        pseudo_w_scale: float = 0.50,             # downweight contribution on read (via w)

        # ---- Temporal position handling (NEW)
        use_abs_time: bool = True,           # use absolute frame indices if provided
        max_time_index: int = 4096,          # clamp time index for PE lookup
    ):

        super().__init__()    

        self.frame_model = frame_model
        self.patch_size = int(patch_size)
        self.bg_index = int(bg_index)

        # switches
        self.use_memory = bool(use_memory)
        self.use_am = bool(use_am)
        self.use_tm = bool(use_tm)
        self.use_abs_time = bool(use_abs_time)
        self.max_time_index = int(max_time_index)
        
        # topk knobs
        self.write_topk_patch_tokens = int(write_topk_patch_tokens)
        self.read_topk_mem_tokens = int(read_topk_mem_tokens)

        # fusion knobs
        self.learnable_alpha = learnable_alpha

        if learnable_alpha:
            # store as logits so actual alpha is sigmoid(logit) in (0,1)
            def inv_sigmoid(x, eps=1e-6):
                x = float(max(eps, min(1 - eps, x)))
                return math.log(x / (1 - x))

            self.alpha_am_logit = nn.Parameter(torch.tensor(inv_sigmoid(alpha_am)))
            self.alpha_tm_logit = nn.Parameter(torch.tensor(inv_sigmoid(alpha_tm)))
        else:
            self.alpha_am = float(alpha_am)
            self.alpha_tm = float(alpha_tm)

        # misc
        self.tm_warmup = int(tm_warmup)
        self.skip_tm_t0 = bool(skip_tm_t0)
        self.debug = bool(debug)

        self.dbg_level = int(dbg_level)
        self.dbg_every = int(max(1, dbg_every))
        self.dbg_first_video_only = bool(dbg_first_video_only)
        self._dbg_vid = None
        self._dbg_step_count = {}
        # DINO hidden dim -> memory d_model
        d_model = self.frame_model.encoder.config.hidden_size

        # memory
        self.am_attn_beta = float(am_attn_beta)
        self.mem = MemoryState(
            K=K, d_model=d_model, max_dt=max_dt, detach_memory=detach_memory,
            am_max_items=am_max_items,
            am_red_lambda=am_red_lambda,
        )
        
        self.enable_am_refresh = bool(enable_am_refresh)
        self.am_refresh_sim_max = float(am_refresh_sim_max)
        self.am_max_per_class = int(am_max_per_class)

        # reliability gate
        self.gate_mode = str(gate_mode)
        self.gate_conf_thr = float(gate_conf_thr)
        self.gate_ent_thr = float(gate_ent_thr)

        # svsswrapper.py inside __init__ (after reliability gate)

        self.allow_pseudo_anchors = bool(allow_pseudo_anchors)
        self.pseudo_use_fused_logits = bool(pseudo_use_fused_logits)
        self.pseudo_every = int(pseudo_every)
        self.pseudo_warmup = int(pseudo_warmup)

        self.pseudo_tau = float(pseudo_tau)
        self.pseudo_q99_thr = float(pseudo_q99_thr)
        self.pseudo_mean_in_thr = float(pseudo_mean_in_thr)
        self.pseudo_min_area = float(pseudo_min_area)
        self.pseudo_max_area = float(pseudo_max_area)

        self.pseudo_streak_req = int(pseudo_streak_req)

        self.pseudo_k_am = int(pseudo_k_am)
        self.pseudo_max_per_class = int(pseudo_max_per_class)
        self.pseudo_conf_scale = float(pseudo_conf_scale)
        self.pseudo_w_scale = float(pseudo_w_scale)


        # memory attention
        self.use_mem_attention = bool(use_mem_attention)
        self.attn_topk_am = int(attn_topk_am)
        self.attn_topk_tm = int(attn_topk_tm)
        self.attn_sharp = float(attn_sharp)
        self.ignore_index = int(ignore_index)
        assert self.attn_sharp > 0, "attn_sharp must be > 0"

        self._dbg_calls = 0
        self.debug_max_calls = 3   # print only for first 3 forward() calls
        self._dbg_runtime = False   # will be set per forward() call
   
    def _pad_to_patch(self, x):
        # x: (B,3,H,W)
        B, C, H, W = x.shape
        ph = (self.patch_size - (H % self.patch_size)) % self.patch_size
        pw = (self.patch_size - (W % self.patch_size)) % self.patch_size
        if ph == 0 and pw == 0:
            return x, (0, 0)
        x = F.pad(x, (0, pw, 0, ph), mode="constant", value=0.0)  # pad W then H
        return x, (ph, pw)

    def _unpad(self, logits, ph_pw):
        ph, pw = ph_pw
        if ph == 0 and pw == 0:
            return logits
        return logits[..., : logits.shape[-2] - ph, : logits.shape[-1] - pw]

    @staticmethod
    def _topk_mem_tokens(mem_tokens: torch.Tensor, k: int) -> torch.Tensor:
        """
        mem_tokens: (M,D). Keeps top-k by token L2 norm.
        """
        if k <= 0 or mem_tokens.shape[0] <= k:
            return mem_tokens
        scores = mem_tokens.norm(dim=1)  # (M,)
        idx = torch.topk(scores, k=k, largest=True).indices
        return mem_tokens.index_select(0, idx)

    @staticmethod
    def _softmax_conf_and_entropy(logits: torch.Tensor):
        """
        logits: (1,C,H,W)
        returns: (conf, ent) scalars
        """
        p = torch.softmax(logits, dim=1)                    # (1,C,H,W)
        pmax = p.max(dim=1).values                          # (1,H,W)
        conf = pmax.mean()                                  # scalar
        ent = -(p * (p.clamp_min(1e-8).log())).sum(dim=1)    # (1,H,W)
        ent = ent.mean()                                    # scalar
        return conf, ent

    @staticmethod
    def _cosine_scores(query_tokens: torch.Tensor, mem_tokens: torch.Tensor) -> torch.Tensor:
        """
        query_tokens: (1,N,D)
        mem_tokens:   (M,D)
        returns:      (M,) cosine similarity of each mem token to query prototype
        """
        q = query_tokens.mean(dim=1)            # (1,D)
        q = F.normalize(q, dim=1)               # (1,D)
        m = F.normalize(mem_tokens, dim=1)      # (M,D)
        return (m @ q.t()).squeeze(1)           # (M,)

    def _attn_ctx_tokens(
        self,
        query_tokens: torch.Tensor,      # (1,N,D)
        mem_tokens: torch.Tensor,        # (M,D)
        topk: int,
        dbg_tag: str = "",
        mem_w: torch.Tensor = None,      # (M,) optional token weights
        beta: float = 2.0,               # weight bias strength
    ) -> torch.Tensor:

        if mem_tokens.numel() == 0:
            return query_tokens.new_zeros(query_tokens.shape)

        # ---- prune memory tokens (proto-based) BEFORE attention
        if topk > 0 and mem_tokens.shape[0] > topk:
            q_proto = F.normalize(query_tokens.mean(dim=1), dim=1)  # (1,D)
            m_norm = F.normalize(mem_tokens, dim=1)                 # (M,D)
            prune_scores = (m_norm @ q_proto.t()).squeeze(1)        # (M,)
            idx = torch.topk(prune_scores, k=topk, largest=True).indices
            mem_tokens = mem_tokens.index_select(0, idx)
            if mem_w is not None:
                mem_w = mem_w.index_select(0, idx)

        # ---- cosine attention
        Q = F.normalize(query_tokens.squeeze(0), dim=1)  # (N,D)
        K = F.normalize(mem_tokens, dim=1)               # (M,D)

        scores = Q @ K.t()                               # (N,M)

        # ---- mask-conditioned bias (AM strengthening)
        if mem_w is not None:
            # normalize weights for stability
            w = mem_w.clamp_min(1e-3)
            w = w / (w.mean() + 1e-6)

            bias = beta * torch.log(w)
            bias = bias.clamp(min=-5.0, max=5.0)     # prevent domination

            scores = scores + bias.unsqueeze(0)      # (N,M)

        scores = scores.clamp(min=-10.0, max=10.0)
        W = torch.softmax(scores * self.attn_sharp, dim=1)

        if self._dbg_runtime and (dbg_tag != ""):
            self._dbg_attn_stats(scores, W, tag=dbg_tag)

        ctx = W @ mem_tokens                             # (N,D)
        return ctx.unsqueeze(0)

    def _topk_by_query(self, query_tokens: torch.Tensor, mem_tokens: torch.Tensor, k: int, return_idx=False):
        if k <= 0 or mem_tokens.shape[0] <= k:
            return mem_tokens
        scores = self._cosine_scores(query_tokens, mem_tokens)  # (M,)
        idx = torch.topk(scores, k=k, largest=True).indices
        if return_idx: return mem_tokens.index_select(0, idx), idx
        return mem_tokens.index_select(0, idx)

    def _mask_to_patch_weights(self, mask_hw: torch.Tensor, Hp: int, Wp: int) -> torch.Tensor:
        """
        Accepts mask in:
        (H,W) or (1,H,W) or (B,H,W) or (B,1,H,W)
        Returns:
        (B, N, 1) where N=(Hp/ps)*(Wp/ps)
        """
        ps = self.patch_size
        assert Hp % ps == 0 and Wp % ps == 0

        x = mask_hw

        # ---- normalize to (B,1,H,W)
        if x.ndim == 1:
            raise ValueError(f"_mask_to_patch_weights got 1D mask {tuple(x.shape)} (flattened?)")
        if x.ndim == 2:            # (H,W)
            x = x[None, None]
        elif x.ndim == 3:          # (1,H,W) or (B,H,W)
            x = x[:, None]
        elif x.ndim == 4:          # (B,1,H,W) or (B,C,H,W)
            if x.shape[1] != 1:
                raise ValueError(f"_mask_to_patch_weights expects 1 channel, got {tuple(x.shape)}")
        else:
            raise ValueError(f"_mask_to_patch_weights got {tuple(x.shape)}")

        # ---- fg weights (B,1,H,W)
        if x.dtype in (torch.int32, torch.int64, torch.uint8):
            fg = ((x != self.bg_index) & (x != self.ignore_index)).float()
        else:
            fg = x.clamp(0, 1).float()

        # ---- resize to padded image size
        fg_up = F.interpolate(fg, size=(Hp, Wp), mode="nearest")  # (B,1,Hp,Wp)

        # ---- pool to patch grid and flatten -> (B,N,1)
        w = F.avg_pool2d(fg_up, kernel_size=ps, stride=ps)        # (B,1,Hp/ps,Wp/ps)
        w = w.flatten(2).transpose(1, 2)                          # (B,N,1)
        return w


    def _dbg_attn_stats(self, scores_nm: torch.Tensor, W_nm: torch.Tensor, tag: str = ""):
        """
        scores_nm: (N,M)
        W_nm:      (N,M)
        Prints aggregated stats without huge overhead.
        """
        with torch.no_grad():
            # per-token (over M)
            w_max = W_nm.max(dim=1).values            # (N,)
            w_ent = -(W_nm.clamp_min(1e-8).log() * W_nm).sum(dim=1)  # (N,)

            # score spread per token
            s_std = scores_nm.std(dim=1)              # (N,)
            s_rng = (scores_nm.max(dim=1).values - scores_nm.min(dim=1).values)  # (N,)

            M = W_nm.shape[1]
            uni = 1.0 / float(M)

            if self.debug:
                print(
                    f"[Attn{tag}] N={W_nm.shape[0]} M={M} uni={uni:.5f} | "
                    f"w_max mean={w_max.mean().item():.4f} p95={w_max.quantile(0.95).item():.4f} max={w_max.max().item():.4f} | "
                    f"ent mean={w_ent.mean().item():.3f} | "
                    f"score std mean={s_std.mean().item():.4f} p95={s_std.quantile(0.95).item():.4f} | "
                    f"score rng mean={s_rng.mean().item():.4f} p95={s_rng.quantile(0.95).item():.4f}"
                )

    def _am_add_from_frame(self, video_id: str, img_1: torch.Tensor, mask_1: torch.Tensor, t_sup: int,
                        pin_first: bool = False):
        # img_1:  (1,3,H,W)
        # mask_1: (1,H,W) or (1,1,H,W)

        # --- normalize mask to (1,H,W)
        if mask_1.ndim == 4 and mask_1.shape[1] == 1:
            mask_1 = mask_1[:, 0]  # (1,H,W)

        if self._dbg_runtime:
            print(
                "AM dbg | mask_1 dtype:",
                mask_1.dtype,
                "min/max:",
                float(mask_1.min()),
                float(mask_1.max()),
                "uniq:",
                torch.unique(mask_1).detach().cpu().tolist()[:30]
            )


        # --- pad image to patch multiple
        img_pad, (ph, pw) = self._pad_to_patch(img_1)
        Hp0, Wp0 = img_pad.shape[-2:]

        sup_tokens = self.frame_model.encode_patch_tokens(img_pad, pre_norm=True)  # (1,N,D)

        # --- pad mask with background to match padded image
        sup_m = mask_1.to(sup_tokens.device)  # (1,H,W)
        if ph != 0 or pw != 0:
            sup_m = F.pad(sup_m, (0, pw, 0, ph), mode="constant", value=self.bg_index)  # (1,Hp0,Wp0)

        if self._dbg_runtime:
            print("AM dbg | bg_index:", self.bg_index, "ignore_index:", self.ignore_index)
            print("AM dbg | mask_1 uniq prepad:", torch.unique(mask_1).detach().cpu().tolist()[:30])
            print("AM dbg | sup_m uniq postpad:", torch.unique(sup_m).detach().cpu().tolist()[:30])

        # now safe: sup_m spatial == (Hp0,Wp0)
        uniq = torch.unique(sup_m)
        uniq = uniq[(uniq != self.bg_index) & (uniq != self.ignore_index)]

        if self._dbg_runtime:
            print("AM dbg | uniq fg:", uniq.detach().cpu().tolist())

        # am_items = self.mem._am[video_id].items

        for cid in uniq.tolist():
            cid = int(cid)

            # per-class weights -> patch weights
            m_c = (sup_m == cid).float()
            w_sup = self._mask_to_patch_weights(m_c, Hp=Hp0, Wp=Wp0)  # (1,N,1)

            # pick top-k patches
            k_am = 256
            w_flat = w_sup.squeeze(0).squeeze(-1)
            k = min(k_am, w_flat.numel())
            idx = torch.topk(w_flat, k=k, largest=True).indices

            tok_sel = sup_tokens.index_select(1, idx)       # (1,k,D)
            w_sel   = w_sup.index_select(1, idx)            # (1,k,1)
            w_store = w_sel.squeeze(0).squeeze(-1)          # (k,)
            tok_cond = tok_sel * w_sel                      # (1,k,D)

            proto = (tok_sel.squeeze(0) * w_store[:, None]).sum(dim=0) / (w_store.sum() + 1e-6)
            conf  = float(w_store.mean().detach().cpu())

            # pin only the very first anchor per class
            pinned = False
            if pin_first:
                pinned = not self.mem.has_pinned_anchor(video_id, cid)

            # --- refresh policy for later supports (diverse + cap)
            if not pinned:
                # am_items is the live list inside MemoryState
                am_items = self.mem._am[video_id].items

                # build (index, item) pairs for this class
                cls_pairs = [(i, it) for i, it in enumerate(am_items) if it.class_id == cid]

                # per-class cap (count anchors, not supports)
                if len(cls_pairs) >= self.am_max_per_class:
                    continue

                # try merging into most similar existing anchor
                if len(cls_pairs) > 0:
                    protos = [(i, it.proto) for i, it in cls_pairs if it.proto is not None]
                    if len(protos) > 0:
                        idxs = [i for i, _ in protos]
                        P = torch.stack([p.detach() for _, p in protos], dim=0)   # (K,D)
                        P = F.normalize(P, dim=1)
                        q = F.normalize(proto.detach().unsqueeze(0), dim=1)       # (1,D)

                        sims = (q @ P.T).squeeze(0)                               # (K,)
                        j = int(torch.argmax(sims).item())
                        best_sim = float(sims[j].item())
                        target_idx = int(idxs[j])                                 # SAFE integer index

                        if best_sim > self.am_refresh_sim_max:
                            self.mem.merge_anchor(
                                video_id=video_id,
                                item_idx=target_idx,
                                tok_cond=tok_cond,
                                w=w_store,
                                conf=conf,
                                proto=proto,
                                t=int(t_sup),
                            )
                            if self._dbg_runtime:
                                print(
                                    f"AM dbg | merged cid={cid} "
                                    f"sim={best_sim:.3f} into item_idx={target_idx}"
                                )
                            continue   # IMPORTANT: do NOT add a new anchor

                # # otherwise keep your “cap per class”
                # if len(cls_pairs) >= self.am_max_per_class:                    
                #     continue

            if self._dbg_runtime:
                print("AM dbg | adding cid:", cid, "k_sel:", tok_sel.shape[1], "w_mean:", float(w_store.mean()))

            self.mem.add_anchor(
                tok_cond, t=int(t_sup), class_id=cid, video_id=video_id,
                w=w_store, conf=conf, pinned=pinned, proto=proto
            )
            
            if self._dbg_runtime:
                st = self.mem.stats(video_id)
                print("AM dbg | after add -> items:", st.get("am_items"), "tokens:", st.get("am_tokens"))

    def _as_long_tensor(self, x, device):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.to(device=device, dtype=torch.long, non_blocking=True)
        return torch.as_tensor(x, device=device, dtype=torch.long)
    # -----------------------------
    # Debug helpers (controlled by dbg_level)
    # -----------------------------
    def _dbg_enabled_for(self, video_id: str) -> bool:
        if int(getattr(self, "dbg_level", 0)) <= 0:
            return False
        if not bool(getattr(self, "dbg_first_video_only", True)):
            return True
        if self._dbg_vid is None:
            self._dbg_vid = str(video_id)
            return True
        return str(video_id) == str(self._dbg_vid)

    def _dbg_should_print_step(self, video_id: str, t_local: int) -> bool:
        if not self._dbg_enabled_for(video_id):
            return False
        every = int(getattr(self, "dbg_every", 10))
        if every <= 1:
            return True
        return (int(t_local) % every) == 0

    def _dbg(self, level: int, msg: str, video_id: str = "", t_local: int = -1, force: bool = False) -> None:
        if force:
            print(msg)
            return
        if int(getattr(self, "dbg_level", 0)) < int(level):
            return
        if video_id and (not self._dbg_enabled_for(video_id)):
            return
        if (t_local is not None) and (t_local >= 0) and (not self._dbg_should_print_step(video_id, t_local)):
            return
        prefix = ""
        if video_id:
            prefix += f"[DBG vid={video_id}] "
        if (t_local is not None) and (t_local >= 0):
            prefix += f"t={int(t_local):04d} "
        print(prefix + str(msg))

    def _dbg_mem_comp(self, mem_meta: dict) -> str:
        try:
            is_a = mem_meta.get("is_anchor", None)
            cid  = mem_meta.get("class_id", None)
            if is_a is None:
                return "mem_meta(no is_anchor)"
            am_tok = int(is_a.sum().item())
            tm_tok = int((~is_a).sum().item())
            if cid is None:
                return f"AMtok={am_tok} TMtok={tm_tok}"
            uniq = cid.detach().cpu().tolist()
            uniq = sorted(set(int(x) for x in uniq if int(x) >= 0))
            if len(uniq) > 12:
                uniq = uniq[:12] + ["..."]
            return f"AMtok={am_tok} TMtok={tm_tok} uniq_cid={uniq}"
        except Exception as e:
            return f"mem_comp_err={type(e).__name__}"



    def _first_int(self, x, default: int = 0) -> int:
        if x is None:
            return int(default)
        if torch.is_tensor(x):
            if x.numel() == 0:
                return int(default)
            return int(x.view(-1)[0].item())
        return int(x)

    @torch.no_grad()
    def init_state(self, support_img, support_mask, support_indices=None, video_id: str = "b0"):
        """
        Build AM (pinned + optional refresh) and reset TM for a new clip.
        Streaming assumes B=1.
        """
        # normalize supports to (B,S,*,H,W)
        if support_img.dim() == 4:
            support_img = support_img.unsqueeze(1)   # (B,1,3,H,W)
        if support_mask.dim() == 3:
            support_mask = support_mask.unsqueeze(1) # (B,1,H,W)
        if support_mask.dim() == 5 and support_mask.shape[2] == 1:
            support_mask = support_mask[:, :, 0]     # (B,S,H,W)

        device = support_img.device
        B = support_img.shape[0]
        assert B == 1, "Streaming init_state expects batch_size=1"
        S = support_img.shape[1]

        support_indices = self._as_long_tensor(support_indices, device)
        if support_indices is not None:
            if support_indices.dim() == 1:
                support_indices = support_indices.view(1, -1)
            if support_indices.shape[1] == 1 and S > 1:
                support_indices = support_indices.expand(1, S)

        # reset memory for this clip
        self.mem.reset(video_id)

        # AM init/refresh from GT supports
        if self.use_memory and self.use_am:
            img0  = support_img[0:1, 0]
            mask0 = support_mask[0:1, 0]
            t0 = int(support_indices[0, 0].item()) if support_indices is not None else 0
            self._am_add_from_frame(video_id, img0, mask0, t_sup=t0, pin_first=True)

            if self.enable_am_refresh and S > 1:
                for s in range(1, S):
                    imgs  = support_img[0:1, s]
                    masks = support_mask[0:1, s]
                    ts = int(support_indices[0, s].item()) if support_indices is not None else int(s)
                    self._am_add_from_frame(video_id, imgs, masks, t_sup=ts, pin_first=False)

        # ---- Debug: confirm AM/TM init
        if int(getattr(self, "dbg_level", 0)) > 0 and self._dbg_enabled_for(video_id):
            st = self.mem.stats(video_id)
            self._dbg(1, f"init_state: {st}", video_id=video_id, t_local=0, force=True)
            if int(getattr(self, "dbg_level", 0)) >= 2:
                try:
                    am_items = self.mem._am[video_id].items
                    pinned = [(int(getattr(it, "class_id", -1)), bool(getattr(it, "pinned", False)), int(getattr(it, "t", -1)), float(getattr(it, "conf", 0.0) or 0.0)) for it in am_items]
                    self._dbg(2, f"AM items (cid,pinned,t,conf): {pinned}", video_id=video_id, t_local=0, force=True)
                except Exception as _e:
                    pass

        return {
            "video_id": video_id,
            "t_local": 0,
            "pseudo_streak": {},     # cid -> consecutive passes
        }


    def step(self, query_img: torch.Tensor, state: dict, query_index=None):
        """
        Process a single query frame (1,3,H,W) with persistent memory.
        Returns: logits (1,C,H,W), updated state
        """


        assert query_img.dim() == 4 and query_img.shape[0] == 1, "step expects (1,3,H,W)"
        device = query_img.device
        video_id = state.get("video_id", "b0")
        t_local = int(state.get("t_local", 0))



        # absolute time for TemporalPE if provided
        if (query_index is not None) and self.use_abs_time:
            qi = self._as_long_tensor(query_index, device)
            t_now = self._first_int(qi, default=t_local)
            t_now = max(0, min(int(t_now), self.max_time_index))
        else:
            t_now = t_local

        # print("[PA CFG]",
        #     self.allow_pseudo_anchors,
        #     self.pseudo_every,
        #     self.pseudo_streak_req)

        # ---- Debug: step header
        if int(getattr(self, "dbg_level", 0)) > 0 and self._dbg_should_print_step(video_id, t_local):
            qi_str = None
            try:
                qi_str = int(query_index.item()) if torch.is_tensor(query_index) and query_index.numel()==1 else query_index
            except Exception:
                qi_str = query_index
            self._dbg(1, f"step start: t_local={t_local} t_now={t_now} query_index={qi_str}", video_id=video_id, t_local=t_local)

        # pad to patch multiple
        frame_pad, ph_pw = self._pad_to_patch(query_img)
        Hp, Wp = frame_pad.shape[-2:]

        raw_ptok = self.frame_model.encode_patch_tokens(frame_pad, pre_norm=True)   # (1,N,D)

        # raw logits for gating (no memory)
        logits_raw = self.frame_model.decode_from_patch_tokens(raw_ptok, Hp, Wp)
        logits_raw = self._unpad(logits_raw, ph_pw)                                  # (1,C,H,W)

        # read memory (AM/TM unified)
        if self.use_memory:
            mem_tokens, mem_meta = self.mem.get_memory(
                t_now=t_now,
                video_id=video_id,
                include_am=self.use_am,
                include_tm=self.use_tm,
            )
            if self.read_topk_mem_tokens > 0 and mem_tokens.numel() > 0:
                mem_tokens, idx = self._topk_by_query(raw_ptok, mem_tokens, self.read_topk_mem_tokens, return_idx=True)
                for key in ("is_anchor", "class_id", "w", "t"):
                    if key in mem_meta and mem_meta[key] is not None:
                        mem_meta[key] = mem_meta[key].index_select(0, idx)

        else:
            mem_tokens = raw_ptok.new_empty((0, raw_ptok.shape[-1]))
            mem_meta = {}


        # ---- Debug: memory composition after read
        if int(getattr(self, "dbg_level", 0)) > 0 and self._dbg_should_print_step(video_id, t_local):
            try:
                st = self.mem.stats(video_id)
                self._dbg(1, f"mem stats: {st}", video_id=video_id, t_local=t_local)
                if mem_meta:
                    self._dbg(2, f"mem comp: {self._dbg_mem_comp(mem_meta)}", video_id=video_id, t_local=t_local)
            except Exception:
                pass

        # INSIDE STEP
        # fuse
        if (not self.use_memory) or (mem_tokens.numel() == 0):
            fused_ptok = raw_ptok
        else:
            # separate ramps: AM can come in early; TM starts after tm_warmup and ramps slower
            alpha_scale_am = min(1.0, max(0.0, (t_local - 0) / 5.0))
            alpha_scale_tm = min(1.0, max(0.0, (t_local - self.tm_warmup) / 25.0))  # slower ramp

            if self.learnable_alpha:
                alpha_am = torch.sigmoid(self.alpha_am_logit)
                alpha_tm = torch.sigmoid(self.alpha_tm_logit)
            else:
                alpha_am = self.alpha_am
                alpha_tm = self.alpha_tm

            if self._dbg_should_print_step(video_id, t_local):
                if self.learnable_alpha:
                    aam = float(torch.sigmoid(self.alpha_am_logit).detach().cpu())
                    atm = float(torch.sigmoid(self.alpha_tm_logit).detach().cpu())
                else:
                    aam, atm = float(self.alpha_am), float(self.alpha_tm)
                self._dbg(
                    1,
                    f"fusion: s_am={alpha_scale_am:.3f} s_tm={alpha_scale_tm:.3f} "
                    f"alpha_am={aam:.3f} alpha_tm={atm:.3f}",
                    video_id,
                    t_local,
                )

            is_a = mem_meta["is_anchor"]                       # (M,)
            am = mem_tokens[is_a]                              # (Ma,D)

            cid_tok = mem_meta.get("class_id", None)           # (M,)
            tm_mask = ~is_a
            tm_all = mem_tokens[tm_mask]                       # (Mt,D)
            tm_cid = cid_tok[tm_mask] if cid_tok is not None else None

            pred_cls = logits_raw.detach().mean(dim=(0,2,3)).argmax().item()

            tm_use = tm_all
            if (tm_cid is not None) and (pred_cls != self.bg_index):
                keep = (tm_cid == int(pred_cls))
                if keep.any():
                    tm_use = tm_all[keep]

            fused_ptok = raw_ptok

            if self.use_mem_attention:
                ctx_am, ctx_tm = None, None

                # AM
                if am.numel() > 0:
                    w_am = mem_meta["w"][is_a] if "w" in mem_meta else None
                    ctx_am = self._attn_ctx_tokens(
                        raw_ptok, am,
                        topk=self.attn_topk_am,
                        dbg_tag="",
                        mem_w=w_am,
                        beta=self.am_attn_beta,
                    )
                    fused_ptok = fused_ptok + (alpha_am * alpha_scale_am) * ctx_am

                # TM
                if tm_use.numel() > 0:
                    if tm_use.shape[0] > self.attn_topk_tm:
                        tm_use = self._topk_by_query(raw_ptok, tm_use, self.attn_topk_tm)
                    ctx_tm = self._attn_ctx_tokens(raw_ptok, tm_use, topk=0, dbg_tag="")

                    if self._dbg_should_print_step(video_id, t_local):
                        with torch.no_grad():
                            def n(x): return float(x.norm(dim=-1).mean().detach().cpu())
                            c_am = (alpha_am * alpha_scale_am) * ctx_am if ctx_am is not None else None
                            c_tm = (alpha_tm * alpha_scale_tm) * ctx_tm if ctx_tm is not None else None

                            msg = f"BEFORE TM FUSION, contrib_norm: "
                            msg += f"am={n(c_am):.2f}" if c_am is not None else "am=NA"
                            msg += " | "
                            msg += f"tm={n(c_tm):.2f}" if c_tm is not None else "tm=NA"
                            self._dbg(1, msg, video_id, t_local)   

                    fused_ptok = fused_ptok + (alpha_tm * alpha_scale_tm) * ctx_tm

                    if self._dbg_should_print_step(video_id, t_local):
                        with torch.no_grad():
                            def n(x): return float(x.norm(dim=-1).mean().detach().cpu())
                            c_am = (alpha_am * alpha_scale_am) * ctx_am if ctx_am is not None else None
                            c_tm = (alpha_tm * alpha_scale_tm) * ctx_tm if ctx_tm is not None else None

                            msg = f"AFTER TM FUSION, contrib_norm: "
                            msg += f"am={n(c_am):.2f}" if c_am is not None else "am=NA"
                            msg += " | "
                            msg += f"tm={n(c_tm):.2f}" if c_tm is not None else "tm=NA"
                            self._dbg(1, msg, video_id, t_local)

                # ---- Debug
                if self._dbg_should_print_step(video_id, t_local):
                    with torch.no_grad():
                        def n(x):
                            return float(x.norm(dim=-1).mean().detach().cpu())

                        msg = f"ctx_norm: raw={n(raw_ptok):.2f}"
                        if ctx_am is not None:
                            msg += f" am={n(ctx_am):.2f}"
                            d_am = float((ctx_am - raw_ptok).norm(dim=-1).mean().detach().cpu())
                            msg += f" d_am={d_am:.2f}"
                        else:
                            msg += " am=NA d_am=NA"

                        if ctx_tm is not None:
                            msg += f" tm={n(ctx_tm):.2f}"
                            d_tm = float((ctx_tm - raw_ptok).norm(dim=-1).mean().detach().cpu())
                            msg += f" d_tm={d_tm:.2f}"
                        else:
                            msg += " tm=NA d_tm=NA"

                        self._dbg(1, msg, video_id, t_local)

            else:
                # fallback mean fusion
                if am.numel() > 0:
                    fused_ptok = fused_ptok + (alpha_am * alpha_scale) * am.mean(dim=0, keepdim=True).unsqueeze(1)
                if tm_use.numel() > 0:
                    fused_ptok = fused_ptok + (alpha_tm * alpha_scale) * tm_use.mean(dim=0, keepdim=True).unsqueeze(1)

        # decode final logits
        logits = self.frame_model.decode_from_patch_tokens(fused_ptok, Hp, Wp)
        logits = self._unpad(logits, ph_pw)                    # (1,C,H,W)

        # ============================================================
        # PSEUDO-ANCHORS: add AM anchors from highly reliable predictions
        # ============================================================
        if (
            self.allow_pseudo_anchors
            and self.use_memory and self.use_am
            and (t_local >= self.pseudo_warmup)
            and (self.pseudo_every > 0)
            and (t_local % self.pseudo_every == 0)
        ):
            with torch.no_grad():
                # choose logits source for pseudo-labeling
                logits_src = logits if self.pseudo_use_fused_logits else logits_raw
                p = torch.softmax(logits_src.detach(), dim=1)  # (1,C,H,W)
                Cnum = p.shape[1]

                # pick a couple likely foreground classes (global)
                cls_scores = p.mean(dim=(0,2,3))
                cls_scores[self.bg_index] = 0.0
                topc = 2
                cls_ids = torch.topk(cls_scores, k=min(topc, Cnum-1), largest=True).indices.tolist()

                # helper: count pseudo anchors already stored for a class
                def _pseudo_count(video_id: str, cid: int) -> int:
                    try:
                        items = self.mem._am[video_id].items
                        return sum(1 for it in items if int(getattr(it, "class_id", -1)) == int(cid) and bool(getattr(it, "is_pseudo", False)))
                    except Exception:
                        return 0

                streak = state.get("pseudo_streak", {})
                for cid in cls_ids:
                    cid = int(cid)
                    if cid == self.bg_index:
                        continue

                    # cap pseudo anchors per class
                    if _pseudo_count(video_id, cid) >= self.pseudo_max_per_class:
                        streak[cid] = 0
                        continue

                    pc = p[:, cid]  # (1,H,W)
                    pcf = pc.flatten()

                    q99 = float(torch.quantile(pcf, 0.99))
                    if q99 < self.pseudo_q99_thr:
                        streak[cid] = 0
                        continue

                    tau = self.pseudo_tau
                    m = (pc > tau)  # bool (1,H,W)
                    area = float(m.float().mean().item())
                    if area < self.pseudo_min_area or area > self.pseudo_max_area:
                        streak[cid] = 0
                        continue

                    mean_in = float(pc[m].mean().item()) if bool(m.any()) else 0.0
                    if mean_in < self.pseudo_mean_in_thr:
                        streak[cid] = 0
                        continue

                    # stability: must pass gate multiple consecutive checks
                    streak[cid] = int(streak.get(cid, 0)) + 1
                    if streak[cid] < self.pseudo_streak_req:
                        continue

                    # ----- build patch weights on PADDED frame
                    # raw_ptok corresponds to frame_pad (Hp,Wp), but pc is unpadded (H,W).
                    pc_pad = pc
                    ph, pw = ph_pw  # from _pad_to_patch(query_img)
                    if ph != 0 or pw != 0:
                        pc_pad = F.pad(pc_pad, (0, pw, 0, ph), mode="constant", value=0.0)  # (1,Hp,Wp)

                    m_pad = (pc_pad > tau).float()  # (1,Hp,Wp)
                    w_sup = self._mask_to_patch_weights(m_pad, Hp=Hp, Wp=Wp)  # (1,N,1)

                    # select top-k patch tokens
                    w_flat = w_sup.squeeze(0).squeeze(-1)
                    k = min(self.pseudo_k_am, w_flat.numel())
                    idx = torch.topk(w_flat, k=k, largest=True).indices

                    tok_sel = raw_ptok.index_select(1, idx)    # (1,k,D)
                    w_sel   = w_sup.index_select(1, idx)       # (1,k,1)
                    w_store = w_sel.squeeze(0).squeeze(-1)     # (k,)
                    tok_cond = (tok_sel * w_sel).squeeze(0)    # (k,D)

                    # proto + weak confidence (for eviction preference)
                    proto = (tok_sel.squeeze(0) * w_store[:, None]).sum(dim=0) / (w_store.sum() + 1e-6)
                    conf  = float(w_store.mean().detach().cpu()) * self.pseudo_conf_scale
                    w_store = (w_store * self.pseudo_w_scale).detach()

                    self.mem.add_anchor(
                        tok_cond.detach(),
                        t=int(t_now),
                        class_id=int(cid),
                        video_id=video_id,
                        w=w_store,
                        conf=conf,
                        pinned=False,
                        proto=proto.detach(),
                        is_pseudo=True,     # ✅ requires MemoryState.add_anchor change
                    )

                    if self.debug and (self.dbg_level >= 0):
                        print(
                            f"\n PSEUDO: [PA+] vid={video_id} t={t_now:04d} "
                            f"class={cid} "
                            f"area={area:.3f} "
                            f"q99={q99:.3f} "
                            f"mean={mean_in:.3f} "
                            f"conf={conf:.3f}"
                        )

                    # after successful write, reset streak so you don't spam anchors
                    streak[cid] = 0

                state["pseudo_streak"] = streak




        # TM write (uses logits_raw)
        if self.use_memory and self.use_tm:
            do_write = ((not self.skip_tm_t0) or (t_local > 0)) and (t_local >= self.tm_warmup)
            if do_write:
                warm_T = 3
                if t_local < warm_T:
                    accept = True
                    conf = ent = None
                else:
                    if self.gate_mode == "off":
                        accept = True
                    else:
                        conf, ent = self._softmax_conf_and_entropy(logits_raw.detach())
                        Cnum = int(logits_raw.shape[1])
                        ent_max = float(math.log(max(2, Cnum)))  # safety

                        # Interpret gate_ent_thr as NORMALIZED entropy threshold in [0,1]
                        ent_norm = float(ent.item()) / (ent_max + 1e-12)
                        ent_thr_norm = float(self.gate_ent_thr)

                        if self.gate_mode == "conf":
                            accept = bool(conf >= self.gate_conf_thr)
                        elif self.gate_mode == "ent":
                            accept = bool(ent_norm <= ent_thr_norm)
                        else:  # "conf+ent"
                            accept = bool((conf >= self.gate_conf_thr) and (ent_norm <= ent_thr_norm))

                # ---- Debug: gating decision
                if int(getattr(self, "dbg_level", 0)) > 0 and self._dbg_should_print_step(video_id, t_local):
                    try:
                        if conf is None or ent is None:
                            self._dbg(2, f"TM gate: do_write={do_write} accept={accept} (warmup) mode={self.gate_mode}",
                                    video_id=video_id, t_local=t_local)
                        else:
                            self._dbg(
                                2,
                                f"TM gate: do_write={do_write} accept={accept} "
                                f"conf={float(conf):.3f} ent={float(ent):.3f} entN={float(ent_norm):.3f} thrN={float(ent_thr_norm):.3f} "
                                f"mode={self.gate_mode}",
                                video_id=video_id, t_local=t_local
                            )
                    except Exception:
                        pass

                # INSIDE STEP FUNCTION
                if accept:
                    p = torch.softmax(logits_raw.detach(), dim=1)   # (1,C,H,W)
                    Cnum = p.shape[1]
                    cls_scores = p.mean(dim=(0,2,3))
                    cls_scores[self.bg_index] = 0.0

                    topc = 2
                    cls_ids = torch.topk(cls_scores, k=min(topc, Cnum-1), largest=True).indices.tolist()

                    tau0, tau1 = 0.08, 0.20
                    r = min(1.0, float(t_local) / 50.0)
                    tau_c = tau0 * (1 - r) + tau1 * r

                    for cid in cls_ids:
                        pc = p[:, cid]  # (1,H,W)
                        pc_mean_prob = float(pc.mean().item())
                        pcf = pc.flatten()

                        q95 = float(torch.quantile(pcf, 0.95))
                        q98 = float(torch.quantile(pcf, 0.98))
                        q99 = float(torch.quantile(pcf, 0.99))
                        fg_tau = float((pc > tau_c).float().mean())

                        if self._dbg_runtime and self._dbg_should_print_step(video_id, t_local):
                            print(
                                f"[TM stats] cid={cid} mean={pc_mean_prob:.3f} "
                                f"q95={q95:.3f} q98={q98:.3f} q99={q99:.3f} "
                                f"tau={tau_c:.3f} fg>tau={fg_tau:.4f}"
                            )

                        if q98 < (tau_c + 0.02):
                            continue
                        if fg_tau < 0.001:
                            continue




                        pc_w = ((pc - tau_c) / (1.0 - tau_c)).clamp(0, 1)
                        pc_w = pc_w ** 0.5

                        w_tm = self._mask_to_patch_weights(pc_w, Hp=Hp, Wp=Wp)  # (1,N,1)

                        k_fg = 128
                        w_flat = w_tm.squeeze(0).squeeze(-1)
                        k_fg = min(k_fg, w_flat.numel())
                        idx = torch.topk(w_flat, k=k_fg, largest=True).indices

                        tm_tokens_sel = raw_ptok.index_select(1, idx)
                        w_sel = w_tm.index_select(1, idx)
                        tm_tokens_cond = (tm_tokens_sel * w_sel).detach()
                        w_store = w_sel.detach().squeeze(0).squeeze(-1)

                        if int(getattr(self, "dbg_level", 0)) >= 2 and self._dbg_should_print_step(video_id, t_local):
                            self._dbg(2, f"TM w stats: mean={float(w_store.mean()):.4f} max={float(w_store.max()):.4f}",
                                    video_id=video_id, t_local=t_local)

                        self.mem.add(
                            tm_tokens_cond,
                            t=t_now,
                            is_anchor=False,
                            video_id=video_id,
                            topk_tokens=self.write_topk_patch_tokens,
                            class_id=int(cid),
                            w=w_store,
                        )

                        if int(getattr(self, "dbg_level", 0)) >= 2 and self._dbg_should_print_step(video_id, t_local):
                            st = self.mem.stats(video_id)
                            self._dbg(2, f"TM write: cid={cid} pc_mean_prob={pc_mean_prob:.3f} k_fg={k_fg} | {st}",
                                    video_id=video_id, t_local=t_local)

        state["t_local"] = t_local + 1
        # ------------------------------------------------------------
        # DEBUG: Anchor Memory composition (once per clip start/end)
        # ------------------------------------------------------------
        # if self.debug and (self.dbg_level >= 1) and t_now in (0, T-1):
        # def _count_am(video_id):
        #     try:
        #         items = self.mem._am[video_id].items
        #     except Exception:
        #         return 0, 0, 0

        #     n_gt = sum(
        #         it.is_anchor and not getattr(it, "is_pseudo", False)
        #         for it in items
        #     )
        #     n_pa = sum(getattr(it, "is_pseudo", False) for it in items)
        #     return n_gt, n_pa, len(items)

        # n_gt, n_pa, n_tot = _count_am(video_id)
        # print(
        #     f"[AM] vid={video_id} t={t_now:04d} "
        #     f"GT={n_gt} PA={n_pa} TOTAL={n_tot}"
        # )

        # return logits, state
        return logits, logits_raw, state

    def forward(
        self,
        support_img,
        support_mask,
        query_imgs,
        support_indices=None,
        query_indices=None,
        return_raw: bool = False,
    ):
        """
        Consistent with step():

        - forward() becomes a thin wrapper:
            state = init_state(...)
            for t: logits_t, state = step(query_img, state, query_index)
        - Works for:
            support_img : (B,3,H,W) or (B,S,3,H,W)
            support_mask: (B,H,W) or (B,S,H,W) or with singleton channel dims
            query_imgs  : (B,T,3,H,W)
            support_indices: (B,) or (B,S)
            query_indices  : (B,T) (absolute frame ids) or None
        - Returns:
            (B,T,C,H,W)
        """
        self._dbg_calls += 1
        dbg_on = bool(self.debug) and (self._dbg_calls <= int(getattr(self, "debug_max_calls", 3)))
        self._dbg_runtime = dbg_on

        device = query_imgs.device

        # -----------------------------
        # Normalize support shapes
        # -----------------------------
        # support_img -> (B,S,3,H,W)
        if support_img.dim() == 4:
            support_img = support_img.unsqueeze(1)
        elif support_img.dim() != 5:
            raise ValueError(f"support_img must be (B,3,H,W) or (B,S,3,H,W), got {tuple(support_img.shape)}")

        # support_mask -> (B,S,H,W)
        # allow: (B,H,W), (B,1,H,W), (B,S,H,W), (B,S,1,H,W)
        if support_mask.dim() == 3:
            support_mask = support_mask.unsqueeze(1)                  # (B,1,H,W)
        elif support_mask.dim() == 4:
            if support_mask.shape[1] == 1 and support_img.shape[1] > 1:
                support_mask = support_mask.unsqueeze(1)              # (B,1,H,W) -> (B,1,1,H,W) (rare)
                support_mask = support_mask[:, 0]                     # back to (B,1,H,W)
            # else: (B,S,H,W) already fine
        elif support_mask.dim() == 5 and support_mask.shape[2] == 1:
            support_mask = support_mask[:, :, 0]                      # (B,S,H,W)
        else:
            raise ValueError(f"support_mask has unsupported shape {tuple(support_mask.shape)}")

        # if support_mask is (B,1,H,W) but support_img is (B,S,3,H,W), expand mask across S
        if support_mask.dim() == 4 and support_mask.shape[1] == 1 and support_img.shape[1] > 1:
            support_mask = support_mask.expand(-1, support_img.shape[1], -1, -1)

        support_mask = support_mask.long()

        # -----------------------------
        # Normalize indices
        # -----------------------------
        def _to_long(x):
            if x is None:
                return None
            if not torch.is_tensor(x):
                x = torch.as_tensor(x, device=device)
            return x.to(device=device, dtype=torch.long, non_blocking=True)

        support_indices = _to_long(support_indices)   # (B,) or (B,S) or None
        query_indices   = _to_long(query_indices)     # (B,T) or None

        B, T, _, H, W = query_imgs.shape
        outs = []
        outs_raw = []  # only used if return_raw
        # -----------------------------
        # Per-sample streaming via init_state/step
        # -----------------------------
        for b in range(B):
            # slice supports for this sample: -> (1,S,3,H,W), (1,S,H,W)
            sup_img_b  = support_img[b:b+1]          # (1,S,3,H,W)
            sup_msk_b  = support_mask[b:b+1]         # (1,S,H,W)

            # slice support_indices for this sample
            sup_idx_b = None
            if support_indices is not None:
                if support_indices.dim() == 1:
                    sup_idx_b = support_indices[b:b+1]          # (1,)
                    # if multi-support, expand to (1,S) so init_state can align times
                    if sup_img_b.shape[1] > 1:
                        sup_idx_b = sup_idx_b.view(1, 1).expand(1, sup_img_b.shape[1])
                else:
                    sup_idx_b = support_indices[b:b+1]          # (1,S)

            # NOTE: init_state expects B==1; we loop over b
            video_id = f"b{b}"

            if dbg_on:
                try:
                    print(f"[FWD] b={b} video_id={video_id}")
                    print("  sup_img:", tuple(sup_img_b.shape), "sup_msk:", tuple(sup_msk_b.shape))
                    if sup_idx_b is not None:
                        print("  sup_idx:", tuple(sup_idx_b.shape), sup_idx_b.view(-1)[:10].tolist())
                    if query_indices is not None:
                        print("  q_idx:", tuple(query_indices[b].shape), query_indices[b, :min(10, T)].tolist())
                except Exception:
                    pass

            state = self.init_state(
                support_img=sup_img_b,
                support_mask=sup_msk_b,
                support_indices=sup_idx_b,
                video_id=video_id,
            )

            logits_seq = []
            logits_raw_seq = [] if return_raw else None

            for t in range(T):
                img_t = query_imgs[b:b+1, t]  # (1,3,H,W)

                qi_t = None
                if query_indices is not None:
                    qi_t = query_indices[b, t:t+1]   # (1,) absolute idx -> step() uses it as t_now

                logits_t, logits_raw_t, state = self.step(
                    query_img=img_t,
                    state=state,
                    query_index=qi_t,
                )  # logits_t: (1,C,H,W), logits_raw_t: (1,C,H,W)

                logits_seq.append(logits_t)
                if return_raw:
                    logits_raw_seq.append(logits_raw_t)



            outs.append(torch.stack(logits_seq, dim=1))  # (1,T,C,H,W)
            if return_raw:
                outs_raw.append(torch.stack(logits_raw_seq, dim=1))  # (1,T,C,H,W)


        out = torch.cat(outs, dim=0)  # (B,T,C,H,W)
        if return_raw:
            out_raw = torch.cat(outs_raw, dim=0)  # (B,T,C,H,W)
            return out, out_raw
        return out

    def _norm_index_1(self, x, device):
        # x can be None, python int, 0-d tensor, (1,) tensor
        if x is None:
            return None
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, device=device)
        x = x.to(device=device, dtype=torch.long)
        if x.numel() == 0:
            return None
        return int(x.view(-1)[0].item())