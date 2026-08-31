import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2


class SAM2SVSSWrapper(nn.Module):
    """
    HuggingFace-Transformers SAM2 baseline wrapper (video), with a switch:

      - legacy (use_enhanced_prompting=False): your original behavior
          * 1 union box per semantic class from the first support frame
          * no re-prompting (pure propagate)
          * bg = 1 - max(fg)

      - enhanced (use_enhanced_prompting=True): stronger baseline
          * Top-K connected-component boxes per class (multi-instance prompting)
          * periodic re-prompting to reduce drift
          * bg = 1 - max(fg)

    forward(support_img, support_mask, query_imgs) -> logits (B,T,C,H,W)
    """

    def __init__(
        self,
        model,                    # transformers.Sam2VideoModel
        processor,                # transformers.Sam2VideoProcessor
        num_classes: int,
        ignore_index: int = 255,
        include_bg_channel: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        max_vision_features_cache_size: int = 2,

        # ---- mode switch
        use_enhanced_prompting: bool = False,

        # ---- enhanced knobs (ignored in legacy mode)
        topk_per_class: int = 3,              # prompt up to K instances per class
        cc_min_area: int = 25,                # drop tiny components
        reprompt_every: int = 10,             # 0 disables; else refresh prompts every K frames
        reprompt_topk_per_class: int = 1,     # cheaper refresh than initial prompting
        reprompt_cc_min_area: int = 50,       # more strict on refresh
        reprompt_prob_thresh: float = 0.35,   # binarize semantic probs for refresh boxes
        max_objects: int = 64,                # safety cap to avoid OOM/slowdowns
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.include_bg_channel = bool(include_bg_channel)
        self.torch_dtype = torch_dtype
        self.max_vision_features_cache_size = int(max_vision_features_cache_size)

        self.use_enhanced_prompting = bool(use_enhanced_prompting)

        self.topk_per_class = int(topk_per_class)
        self.cc_min_area = int(cc_min_area)
        self.reprompt_every = int(reprompt_every)
        self.reprompt_topk_per_class = int(reprompt_topk_per_class)
        self.reprompt_cc_min_area = int(reprompt_cc_min_area)
        self.reprompt_prob_thresh = float(reprompt_prob_thresh)
        self.max_objects = int(max_objects)

        # baseline: frozen
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def forward(
        self,
        support_img,
        support_mask,
        query_imgs,
        query_indices=None,
        support_indices=None,
        meta=None,
    ):
        """
        support_img:  (B,3,H,W) OR (B,1,3,H,W)
        support_mask: (B,H,W) long labels in {0..C-1} plus ignore_index
        query_imgs:   (B,T,3,H,W) float
        return logits: (B,T,C,H,W)
        """
        # --- handle (B,1,3,H,W) safely (DO NOT squeeze channel by accident)
        if support_img.ndim == 5:
            support_img = support_img.squeeze(1)
        assert support_img.ndim == 4, f"support_img shape={tuple(support_img.shape)}"
        assert support_mask.ndim == 3, f"support_mask shape={tuple(support_mask.shape)}"
        assert query_imgs.ndim == 5, f"query_imgs shape={tuple(query_imgs.shape)}"

        device = query_imgs.device
        B, T, _, H, W = query_imgs.shape
        C = self.num_classes

        out_prob = torch.zeros((B, T, C, H, W), device=device, dtype=torch.float32)

        use_cuda = (device.type == "cuda")
        autocast_ctx = (
            torch.autocast("cuda", dtype=self.torch_dtype)
            if use_cuda else
            torch.autocast("cpu")
        )

        with autocast_ctx:
            for b in range(B):
                # 1) Build frames: [support] + query
                frames = [self._chw_to_pil_rgb(support_img[b])]
                frames += [self._chw_to_pil_rgb(query_imgs[b, t]) for t in range(T)]

                # 2) Init HF session
                session = self.processor.init_video_session(
                    video=frames,
                    inference_device=device,
                    dtype=self.torch_dtype,
                    max_vision_features_cache_size=self.max_vision_features_cache_size,
                )

                # 3) Present semantic classes in support (exclude bg=0 and ignore)
                sup = support_mask[b]
                present = torch.unique(sup)
                present = [
                    int(x) for x in present.tolist()
                    if x != 0 and x != self.ignore_index and 0 <= int(x) < C
                ]
                if len(present) == 0:
                    out_prob[b, :, 0] = 1.0
                    continue

                # -------------------------
                # Initial prompting (legacy vs enhanced)
                # -------------------------
                obj_ids, boxes, cls_for_oid = [], [], {}
                oid = 1

                if not self.use_enhanced_prompting:
                    # ---------- LEGACY ----------
                    # one union box per class
                    for cls in present:
                        bin_mask = (sup == cls)
                        box = self._mask_to_box_xyxy(bin_mask)
                        if box is None:
                            continue
                        obj_ids.append(oid)
                        boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
                        cls_for_oid[oid] = cls
                        oid += 1
                else:
                    # ---------- ENHANCED ----------
                    # Top-K CC boxes per class
                    for cls in present:
                        bin_mask = (sup == cls)
                        cls_boxes = self._mask_to_boxes_topk_cc(
                            bin_mask,
                            topk=self.topk_per_class,
                            min_area=self.cc_min_area,
                        )
                        for box in cls_boxes:
                            if len(obj_ids) >= self.max_objects:
                                break
                            obj_ids.append(oid)
                            boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
                            cls_for_oid[oid] = cls
                            oid += 1
                        if len(obj_ids) >= self.max_objects:
                            break

                if len(obj_ids) == 0:
                    out_prob[b, :, 0] = 1.0
                    continue

                # 4) Add prompts to session on frame 0
                self.processor.add_inputs_to_inference_session(
                    inference_session=session,
                    frame_idx=0,
                    obj_ids=obj_ids,
                    input_boxes=[boxes],  # batch=1
                    clear_old_inputs=True,
                )

                # 5) Propagate + optional periodic re-prompting
                reprompt_every = (
                    max(0, int(self.reprompt_every))
                    if self.use_enhanced_prompting
                    else 0
                )

                for sam2_out in self.model.propagate_in_video_iterator(
                    session,
                    start_frame_idx=0,
                    show_progress_bar=False,
                ):
                    frame_idx = int(sam2_out.frame_idx)  # 0..T
                    if frame_idx == 0:
                        continue
                    t = frame_idx - 1
                    if t < 0 or t >= T:
                        continue

                    masks = self.processor.post_process_masks(
                        [sam2_out.pred_masks],
                        original_sizes=[[session.video_height, session.video_width]],
                        binarize=False,
                    )[0].to(device=device, dtype=torch.float32)

                    # normalize to (N,H,W)
                    if masks.ndim == 4 and masks.shape[1] == 1:
                        masks_nhw = masks[:, 0]
                    elif masks.ndim == 3:
                        masks_nhw = masks
                    else:
                        masks_nhw = masks.squeeze()

                    # safety: object count alignment
                    if masks_nhw.ndim != 3 or masks_nhw.shape[0] != len(session.obj_ids):
                        continue

                    # object -> semantic max-fusion
                    for i, oid_i in enumerate(session.obj_ids):
                        oid_i = int(oid_i)
                        cls = cls_for_oid.get(oid_i, None)
                        if cls is None:
                            continue
                        out_prob[b, t, cls] = torch.maximum(
                            out_prob[b, t, cls],
                            masks_nhw[i].clamp(0.0, 1.0),
                        )

                    # ---------- periodic re-prompt to reduce drift (ENHANCED only) ----------
                    if reprompt_every > 0 and ((t + 1) % reprompt_every == 0) and (t + 1) < T:
                        new_obj_ids, new_boxes, new_cls_for_oid = [], [], {}
                        new_oid = 1

                        for cls in present:
                            pm = out_prob[b, t, cls]  # (H,W)
                            bin_pred = (pm >= self.reprompt_prob_thresh)

                            cls_boxes = self._mask_to_boxes_topk_cc(
                                bin_pred,
                                topk=self.reprompt_topk_per_class,
                                min_area=self.reprompt_cc_min_area,
                            )
                            for box in cls_boxes:
                                if len(new_obj_ids) >= self.max_objects:
                                    break
                                new_obj_ids.append(new_oid)
                                new_boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
                                new_cls_for_oid[new_oid] = cls
                                new_oid += 1
                            if len(new_obj_ids) >= self.max_objects:
                                break

                        if len(new_obj_ids) > 0:
                            self.processor.add_inputs_to_inference_session(
                                inference_session=session,
                                frame_idx=frame_idx,  # current session frame (support+query timeline)
                                obj_ids=new_obj_ids,
                                input_boxes=[new_boxes],
                                clear_old_inputs=True,
                            )
                            cls_for_oid = new_cls_for_oid  # replace mapping for subsequent frames

                # 6) background channel (safer with overlaps): 1 - max(fg)
                if self.include_bg_channel and C > 1:
                    fg = out_prob[b, :, 1:].amax(dim=1)  # (T,H,W)
                    out_prob[b, :, 0] = (1.0 - fg).clamp(0.0, 1.0)

        # probs -> logits (for CE-style eval pipelines)
        eps = 1e-6
        out_prob = out_prob.clamp(eps, 1.0 - eps)
        logits = torch.log(out_prob)
        return logits

    # ------------------------- helpers -------------------------

    @staticmethod
    def _chw_to_pil_rgb(img_chw: torch.Tensor) -> Image.Image:
        x = img_chw.detach()
        if x.dtype != torch.uint8:
            x = x.float()
            if x.max().item() <= 1.5:
                x = x * 255.0
            x = x.clamp(0.0, 255.0).to(torch.uint8)
        x = x.permute(1, 2, 0).contiguous().cpu().numpy()
        return Image.fromarray(x, mode="RGB")

    @staticmethod
    def _mask_to_box_xyxy(bin_mask: torch.Tensor):
        """
        Returns [x0,y0,x1,y1] where x1,y1 are EXCLUSIVE.
        """
        if bin_mask.dtype != torch.bool:
            bin_mask = bin_mask.bool()
        ys, xs = torch.where(bin_mask)
        if ys.numel() == 0:
            return None
        x0 = int(xs.min().item())
        y0 = int(ys.min().item())
        x1 = int(xs.max().item()) + 1
        y1 = int(ys.max().item()) + 1
        H, W = bin_mask.shape[-2], bin_mask.shape[-1]
        x1 = min(x1, W)
        y1 = min(y1, H)
        return [x0, y0, x1, y1]

    @staticmethod
    def _mask_to_boxes_topk_cc(bin_mask: torch.Tensor, topk: int = 3, min_area: int = 25):
        """
        Returns up to topk boxes (x0,y0,x1,y1) EXCLUSIVE, for largest connected components.
        Requires cv2.
        """
        if topk <= 0:
            return []

        if bin_mask.dtype != torch.bool:
            bin_mask = bin_mask > 0

        m = bin_mask.detach().cpu().numpy().astype(np.uint8)
        if m.ndim != 2:
            m = m.squeeze()
        if m.size == 0:
            return []

        num, labels = cv2.connectedComponents(m, connectivity=8)

        if num <= 1:
            box = SAM2SVSSWrapper._mask_to_box_xyxy(bin_mask)
            return [] if box is None else [box]

        boxes, areas = [], []
        for cid in range(1, num):
            ys, xs = np.where(labels == cid)
            if ys.size == 0:
                continue
            area = int(ys.size)
            if area < int(min_area):
                continue
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
            boxes.append([x0, y0, x1, y1])
            areas.append(area)

        if not boxes:
            box = SAM2SVSSWrapper._mask_to_box_xyxy(bin_mask)
            return [] if box is None else [box]

        order = np.argsort(np.array(areas))[::-1][:topk]
        return [boxes[i] for i in order]