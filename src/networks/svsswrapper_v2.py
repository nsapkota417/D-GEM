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

        return {"video_id": video_id, "t_local": 0}


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

        # fuse
        if (not self.use_memory) or (mem_tokens.numel() == 0):
            fused_ptok = raw_ptok
        else:
            alpha_scale = min(1.0, max(0.0, (t_local - self.tm_warmup) / 5.0))

            if self.learnable_alpha:
                alpha_am = torch.sigmoid(self.alpha_am_logit)
                alpha_tm = torch.sigmoid(self.alpha_tm_logit)
            else:
                alpha_am = self.alpha_am
                alpha_tm = self.alpha_tm

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
                    fused_ptok = fused_ptok + (alpha_am * alpha_scale) * ctx_am

                # TM
                if tm_use.numel() > 0:
                    if tm_use.shape[0] > self.attn_topk_tm:
                        tm_use = self._topk_by_query(raw_ptok, tm_use, self.attn_topk_tm)
                    ctx_tm = self._attn_ctx_tokens(raw_ptok, tm_use, topk=0, dbg_tag="")
                    fused_ptok = fused_ptok + (alpha_tm * alpha_scale) * ctx_tm
            else:
                # fallback mean fusion
                if am.numel() > 0:
                    fused_ptok = fused_ptok + (alpha_am * alpha_scale) * am.mean(dim=0, keepdim=True).unsqueeze(1)
                if tm_use.numel() > 0:
                    fused_ptok = fused_ptok + (alpha_tm * alpha_scale) * tm_use.mean(dim=0, keepdim=True).unsqueeze(1)

        # decode final logits
        logits = self.frame_model.decode_from_patch_tokens(fused_ptok, Hp, Wp)
        logits = self._unpad(logits, ph_pw)                    # (1,C,H,W)

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
                        Cnum = logits_raw.shape[1]
                        ent_max = float(math.log(Cnum))
                        ent_thr = min(float(self.gate_ent_thr), 0.98 * ent_max)

                        if self.gate_mode == "conf":
                            accept = bool(conf >= self.gate_conf_thr)
                        elif self.gate_mode == "ent":
                            accept = bool(ent <= ent_thr)
                        else:
                            accept = bool((conf >= self.gate_conf_thr) and (ent <= ent_thr))

                if accept:
                    p = torch.softmax(logits_raw.detach(), dim=1)   # (1,C,H,W)
                    Cnum = p.shape[1]
                    cls_scores = p.mean(dim=(0,2,3))
                    cls_scores[self.bg_index] = 0.0

                    topc = 1
                    cls_ids = torch.topk(cls_scores, k=min(topc, Cnum-1), largest=True).indices.tolist()
                    tau_c = 0.70

                    for cid in cls_ids:
                        pc = p[:, cid]  # (1,H,W)
                        if pc.mean().item() < tau_c:
                            continue

                        pc = ((pc - tau_c) / (1.0 - tau_c)).clamp(0, 1)
                        pc = pc ** 2
                        w_tm = self._mask_to_patch_weights(pc, Hp=Hp, Wp=Wp)  # (1,N,1)

                        k_fg = 128
                        w_flat = w_tm.squeeze(0).squeeze(-1)
                        k_fg = min(k_fg, w_flat.numel())
                        idx = torch.topk(w_flat, k=k_fg, largest=True).indices

                        tm_tokens_sel = raw_ptok.index_select(1, idx)
                        w_sel = w_tm.index_select(1, idx)
                        tm_tokens_cond = tm_tokens_sel * w_sel

                        tm_tokens_cond = tm_tokens_cond.detach()
                        w_store = w_sel.detach().squeeze(0).squeeze(-1)


                        self.mem.add(
                            tm_tokens_cond,
                            t=t_now,
                            is_anchor=False,
                            video_id=video_id,
                            topk_tokens=self.write_topk_patch_tokens,
                            class_id=int(cid),
                            w=w_store,
                        )
        
        state["t_local"] = t_local + 1
        return logits, state

    def forward(
        self,
        support_img,
        support_mask,
        query_imgs,
        support_indices=None,
        query_indices=None,
    ):        
        """
        support_img:  (B,3,H,W)
        support_mask: (B,H,W)   semantic labels (0..C-1) or ignore_index
        query_imgs:   (B,T,3,H,W)
        returns:      (B,T,C,H,W)
        """
        
        self._dbg_calls += 1
        dbg_on = self.debug and (self._dbg_calls <= self.debug_max_calls)
        self._dbg_runtime = dbg_on

        # --- normalize indices to tensors on correct device
        device = query_imgs.device


        # ---- normalize support shapes (avoid accidental squeeze bugs)
        # support_img expected: (B,S,3,H,W) or (B,3,H,W)
        # support_mask expected: (B,S,H,W) or (B,H,W)

        if support_img.dim() == 5 and support_mask.dim() == 3:
            # someone squeezed S out of mask; restore it
            support_mask = support_mask.unsqueeze(1)   # (B,1,H,W)

        if support_img.dim() == 4 and support_mask.dim() == 4:
            if support_mask.shape[1] == 1:
                support_mask = support_mask[:, 0]  # (B,H,W)


        # also normalize indices
        if support_indices is not None:
            if torch.is_tensor(support_indices) and support_indices.dim() == 1 and support_img.dim() == 5:
                # (B,) -> (B,1)
                support_indices = support_indices.unsqueeze(1)


        if support_indices is not None and not torch.is_tensor(support_indices):
            support_indices = torch.as_tensor(support_indices, device=device)
        if query_indices is not None and not torch.is_tensor(query_indices):
            query_indices = torch.as_tensor(query_indices, device=device)

        if torch.is_tensor(support_indices):
            support_indices = support_indices.to(device=device, dtype=torch.long, non_blocking=True)
        if torch.is_tensor(query_indices):
            query_indices = query_indices.to(device=device, dtype=torch.long, non_blocking=True)

        B, T, _, H, W = query_imgs.shape

        outs = []

        multi_support = (support_img.dim() == 5)  # (B,S,3,H,W)
        S = support_img.shape[1] if multi_support else 1

        for b in range(B):
            # ============================================================
            # Per-video loop (MemoryState is per video_id)
            #   1) Initialize Anchor Memory (AM) from support annotations
            #   2) For each query frame:
            #        - Encode patch tokens
            #        - Compute raw logits (for gating + class choice)
            #        - Read AM/TM memory and fuse into query tokens
            #        - Decode fused tokens -> logits
            #        - Update TM from gated predictions (class-aware)
            # ============================================================
            video_id = f"b{b}"

            self.mem.reset(video_id)
            if dbg_on:
                print("support_img shape:", tuple(support_img.shape))
                if support_indices is not None:
                    print("support_indices:", support_indices[b].tolist())
            # -------------------------------
            # AM init / refresh (GT only)
            #   - Always pin first anchor per class from first support
            #   - Optionally refresh anchors from additional annotated supports
            #   - Diversity + per-class cap prevents flooding
            # -------------------------------
            # ---- Add support frame as per-class pinned anchors at t=0
            if self.use_memory and self.use_am:

                if dbg_on:
                    print("forward support_mask full uniq:", torch.unique(support_mask[b:b+1]).detach().cpu().tolist()[:30])
                    if multi_support:
                        print("forward mask0 uniq:", torch.unique(support_mask[b:b+1,0]).detach().cpu().tolist()[:30])
                    else:
                        print("forward mask0 uniq:", torch.unique(support_mask[b:b+1]).detach().cpu().tolist()[:30])

                if multi_support:
                    # Always use first support as pinned init
                    img0  = support_img[b:b+1, 0]
                    mask0 = support_mask[b:b+1, 0]

                    if dbg_on:
                        print("forward support_mask slice uniq:", torch.unique(mask0).detach().cpu().tolist()[:30])

                    if support_indices is not None:
                        t0 = int(support_indices[b, 0].item())
                    else:
                        t0 = 0

                    self._am_add_from_frame(video_id, img0, mask0, t_sup=t0, pin_first=True)


                    # Optionally refresh from additional supports
                    if self.enable_am_refresh:
                        for s in range(1, S):
                            imgs  = support_img[b:b+1, s]
                            masks = support_mask[b:b+1, s]
                            t_sup = int(support_indices[b, s].item()) if support_indices is not None else int(s)

                            if dbg_on:
                                print(f"[AM refresh] support s={s} t_sup={t_sup}")

                            self._am_add_from_frame(video_id, imgs, masks, t_sup=t_sup, pin_first=False)
                else:
                    # single support
                    img0  = support_img[b:b+1]
                    mask0 = support_mask[b:b+1]
                    
                    if support_indices is not None:
                        if support_indices.dim() == 2: t0 = int(support_indices[b, 0].item())
                        else: t0 = int(support_indices[b].item())
                    else:
                        t0 = 0                    
                    
                    self._am_add_from_frame(video_id, img0, mask0, t_sup=t0, pin_first=True)


                if dbg_on:
                    st = self.mem.stats(video_id)
                    print(
                        f"[AM init/refresh] "
                        f"items={st['am_items']} "
                        f"tokens={st['am_tokens']} "
                        f"max_items={self.mem.am_max_items}"
                    )


            logits_list = []
            for t in range(T):
                frame = query_imgs[b:b + 1, t]  # (1,3,H,W)

                # absolute frame index (critical for sparse sampling)
                if (query_indices is not None) and self.use_abs_time:
                    t_now = int(query_indices[b, t].item())
                    t_now = max(0, min(t_now, self.max_time_index))
                else:
                    t_now = int(t)



                frame_pad, ph_pw = self._pad_to_patch(frame)
                Hp, Wp = frame_pad.shape[-2:]

                raw_ptok = self.frame_model.encode_patch_tokens(frame_pad, pre_norm=True)  # (1,N,D)

                # ---- Raw logits for gating (no memory). This avoids a feedback loop where bad memory -> worse logits -> no writes.
                logits_raw = self.frame_model.decode_from_patch_tokens(raw_ptok, Hp, Wp)  # (1,C,Hp,Wp)
                logits_raw = self._unpad(logits_raw, ph_pw)                               # (1,C,H,W)

                # -------------------------------
                # Memory read
                #   - Export unified tokens (AM ∪ TM) + metadata
                #   - Optional global pruning by query similarity
                # -------------------------------
                if self.use_memory:
                    mem_tokens, mem_meta = self.mem.get_memory(
                        t_now=t_now,
                        video_id=video_id,
                        include_am=self.use_am,
                        include_tm=self.use_tm,
                    )
                    if self.read_topk_mem_tokens > 0:
                        mem_tokens = self._topk_by_query(raw_ptok, mem_tokens, self.read_topk_mem_tokens)
                else:
                    mem_tokens = raw_ptok.new_empty((0, raw_ptok.shape[-1]))
                    mem_meta = {}

                # -------------------------------
                # Fuse memory into query tokens
                #   - Split memory into AM vs TM via is_anchor
                #   - TM is class-filtered using current predicted class
                #   - Attention-based context (preferred) or mean fallback
                # -------------------------------
                if (not self.use_memory) or (mem_tokens.numel() == 0):
                    fused_ptok = raw_ptok
                else:
                    # Warmup scale for memory fusion (prevents early poisoning)
                    # alpha_scale = min(1.0, max(0.0, (t - 2) / 5.0))  # ramp from t=2 to t=7
                    alpha_scale = min(1.0, max(0.0, (t - self.tm_warmup)/5.0))
                    # ---- Get learnable alphas
                    if self.learnable_alpha:
                        alpha_am = torch.sigmoid(self.alpha_am_logit)
                        alpha_tm = torch.sigmoid(self.alpha_tm_logit)
                    else:
                        alpha_am = self.alpha_am
                        alpha_tm = self.alpha_tm

                    if dbg_on and t in (3,7,9):
                        print(f"[AlphaLearn] t={t} am={alpha_am.item():.4f} tm={alpha_tm.item():.4f}")


                    assert "is_anchor" in mem_meta, "Memory meta missing is_anchor"
                    is_a = mem_meta["is_anchor"]  # [M] bool
                    am = mem_tokens[is_a]         # [Ma,D]

                    cid_tok = mem_meta.get("class_id", None)
                    tm_mask = ~is_a
                    tm_all = mem_tokens[tm_mask]
                    tm_cid = cid_tok[tm_mask] if cid_tok is not None else None
                    pred_cls = logits_raw.detach().mean(dim=(0,2,3)).argmax().item()

                    tm_use = tm_all
                    if (tm_cid is not None) and (pred_cls != self.bg_index):
                        keep = (tm_cid == int(pred_cls))
                        if keep.any():
                            tm_use = tm_all[keep]

                    fused_ptok = raw_ptok

                    if self.use_mem_attention:
                        # ---------- AM read ----------
                        if am.numel() > 0:
                            tag_am = f":AM t={t}" if self._dbg_runtime and (t in (3, 7, 9)) else ""

                            w_am = mem_meta["w"][is_a] if "w" in mem_meta else None

                            if dbg_on and t in (3,7) and (w_am is not None):
                                print(f"[AM read] tokens={am.shape[0]} w_mean={w_am.mean().item():.3f}")

                            ctx_am = self._attn_ctx_tokens(
                                raw_ptok,
                                am,
                                topk=self.attn_topk_am,
                                dbg_tag=tag_am,
                                mem_w=w_am,      # <-- NEW
                                beta=self.am_attn_beta
                            )  # (1,N,D)

                            fused_ptok = fused_ptok + (alpha_am * alpha_scale) * ctx_am


                        # ---------- TM read (prune candidates first) ----------
                        if tm_use.numel() > 0:
                            if tm_use.shape[0] > self.attn_topk_tm:
                                tm_use = self._topk_by_query(raw_ptok, tm_use, self.attn_topk_tm)

                            tag_tm = f":TM t={t}" if self._dbg_runtime and (t in (3, 7, 9)) else ""
                            ctx_tm = self._attn_ctx_tokens(raw_ptok, tm_use, topk=0, dbg_tag=tag_tm)
                            fused_ptok = fused_ptok + (alpha_tm * alpha_scale) * ctx_tm

                    else:
                        # mean fusion fallback
                        if am.numel() > 0:
                            fused_ptok = fused_ptok + (alpha_am * alpha_scale) * am.mean(dim=0, keepdim=True).unsqueeze(1)
                        if tm_use.numel() > 0:
                            fused_ptok = fused_ptok + (alpha_tm * alpha_scale) * tm_use.mean(dim=0, keepdim=True).unsqueeze(1)

                logits = self.frame_model.decode_from_patch_tokens(fused_ptok, Hp, Wp)  # (1,C,Hp,Wp)
                logits = self._unpad(logits, ph_pw)                                      # (1,C,H,W)
                logits_list.append(logits)

                # -------------------------------
                # TM write (prediction-driven, class-aware)
                #   - Gate writes using confidence/entropy (+ warmup)
                #   - Select top classes in frame (excluding background)
                #   - For each selected class, write weighted top-k patch tokens
                # -------------------------------
                # ---- TM update (MASK-GUIDED): store RAW tokens * soft foreground prob
                if self.use_memory and self.use_tm:
                    do_write = ((not self.skip_tm_t0) or (t > 0)) and (t >= self.tm_warmup)

                    if do_write:

                        # ---- Gate warmup: always accept early frames so TM can populate
                        # (Prevents deadlock early training; adjust 8->10 if needed)
                        warm_T = 3   # only first 3 frames always write
                        if t < warm_T:
                            accept = True
                            conf = ent = None
                        else:
                            accept = None

                        if accept is None:  # means warmup not active (t >= warm_T)
                            if self.gate_mode == "off":
                                accept = True
                                conf = ent = None
                            else:
                                conf, ent = self._softmax_conf_and_entropy(logits_raw.detach())

                                Cnum = logits_raw.shape[1]
                                ent_max = float(math.log(Cnum))
                                ent_thr = float(self.gate_ent_thr)  # user-controlled
                                # (optional safety) cap to max entropy:
                                ent_thr = min(ent_thr, 0.98 * ent_max)

                                if self.gate_mode == "conf":
                                    accept = bool(conf >= self.gate_conf_thr)
                                elif self.gate_mode == "ent":
                                    accept = bool(ent <= ent_thr)
                                else:  # "conf+ent"
                                    accept = bool((conf >= self.gate_conf_thr) and (ent <= ent_thr))


                        if dbg_on and (t in (2, 3, 7, 9)):
                            if conf is None or ent is None:
                                print(f"[GateWarm] t={t:02d} accept={int(accept)}")
                            else:
                                print(f"[Gate] t={t:02d} accept={int(accept)} conf={conf.item():.3f} ent={ent.item():.3f}")

                        if accept:
                            # soft foreground weight from prediction (assumes background class = 0)
                            p = torch.softmax(logits_raw.detach(), dim=1)   # (1,C,H,W)
                            Cnum = p.shape[1]

                            # choose which classes to write this frame (excluding bg/ignore)
                            cls_scores = p.mean(dim=(0, 2, 3))   # compute first
                            bg = self.bg_index
                            cls_scores[bg] = 0.0                # then suppress background
                            topc = 1                                       # write top-c predicted classes (tune 1–3)
                            cls_ids = torch.topk(cls_scores, k=min(topc, Cnum-1), largest=True).indices.tolist()

                            tau_c = 0.70                                    # per-class confidence threshold (tune 0.6–0.8)

                            for cid in cls_ids:
                                pc = p[:, cid]                               # (1,H,W)
                                if pc.mean().item() < tau_c:
                                    continue

                                # per-class weights -> patch weights
                                pc = ((pc - tau_c) / (1.0 - tau_c)).clamp(0, 1)
                                pc = pc ** 2
                                w_tm = self._mask_to_patch_weights(pc, Hp=Hp, Wp=Wp)  # (1,N,1)

                                # select top patches for this class
                                k_fg = 128                                   # per-class; tune 64–256
                                w_flat = w_tm.squeeze(0).squeeze(-1)         # (N,)
                                k_fg = min(k_fg, w_flat.numel())
                                idx = torch.topk(w_flat, k=k_fg, largest=True).indices

                                tm_tokens_sel = raw_ptok.index_select(1, idx)   # (1,k,D)
                                w_sel = w_tm.index_select(1, idx)               # (1,k,1)
                                tm_tokens_cond = tm_tokens_sel * w_sel          # (1,k,D)

                                # ---- store class_id + optional weights
                                self.mem.add(
                                    tm_tokens_cond,
                                    t=t_now,
                                    is_anchor=False,
                                    video_id=video_id,
                                    topk_tokens=self.write_topk_patch_tokens,
                                    class_id=int(cid),                           # <-- NEW (see MemoryState change below)
                                    w=w_sel.squeeze(0).squeeze(-1)               # <-- NEW (optional but useful)
                                )

            outs.append(torch.cat(logits_list, dim=0).unsqueeze(0))  # (1,T,C,H,W)

            if dbg_on:
                st = self.mem.stats(video_id)

                if self.learnable_alpha:
                    amv = float(torch.sigmoid(self.alpha_am_logit).detach().cpu())
                    tmv = float(torch.sigmoid(self.alpha_tm_logit).detach().cpu())
                else:
                    amv = float(self.alpha_am)
                    tmv = float(self.alpha_tm)

                print(
                    f"[SVSSWrapper] {video_id}: "
                    f"AM={st.get('am_items', 0)}({st.get('am_tokens', 0)} tok) "
                    f"TM={st.get('tm_items', 0)}({st.get('tm_tokens', 0)} tok) "
                    f"use_am={self.use_am} use_tm={self.use_tm} "
                    f"write_topk={self.write_topk_patch_tokens} read_topk={self.read_topk_mem_tokens} "
                    f"alpha_am={amv:.4f} alpha_tm={tmv:.4f} tm_warmup={self.tm_warmup}"
                )
            
                print(f"[AlphaScale] t=0->{min(1.0,max(0.0,(0-2)/5.0)):.2f}, t=3->{min(1.0,max(0.0,(3-2)/5.0)):.2f}, t=7->{min(1.0,max(0.0,(7-2)/5.0)):.2f}")


        return torch.cat(outs, dim=0)  # (B,T,C,H,W)

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