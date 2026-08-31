
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

class SAM2SVSSWrapper(nn.Module):
    """
    HuggingFace-Transformers SAM2 baseline wrapper (video).

    forward(support_img, support_mask, query_imgs) -> logits (B,T,C,H,W)

    - Builds a HF inference_session over frames: [support] + query frames
    - Prompts objects on frame 0 using bounding boxes from support_mask per semantic class
    - Propagates masks through the video via model.propagate_in_video_iterator(session)
    - Maps SAM2 object masks -> semantic channels (max over objects assigned to same class)
    """
    def __init__(
        self,
        model,                    # transformers.Sam2VideoModel
        processor,                # transformers.Sam2VideoProcessor
        num_classes: int,
        ignore_index: int = 255,
        include_bg_channel: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        max_vision_features_cache_size: int = 2,  # small cache; tune if you want
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.include_bg_channel = bool(include_bg_channel)
        self.torch_dtype = torch_dtype
        self.max_vision_features_cache_size = int(max_vision_features_cache_size)

        # baseline: frozen
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, support_img, support_mask, query_imgs, query_indices=None, support_indices= None, meta=None):
        """
        support_img:  (B,3,H,W) float (usually 0..1 or 0..255)
        support_mask: (B,H,W) long  labels in {0..C-1} plus ignore_index
        query_imgs:   (B,T,3,H,W) float
        return logits: (B,T,C,H,W)
        """

        # import IPython; IPython.embed()
        support_img = support_img.squeeze(1) 
        device = query_imgs.device
        B, T, _, H, W = query_imgs.shape
        C = self.num_classes

        out_prob = torch.zeros((B, T, C, H, W), device=device, dtype=torch.float32)

        # We’ll run SAM2 with autocast on CUDA like HF examples typically do
        use_cuda = (device.type == "cuda")
        autocast_ctx = torch.autocast("cuda", dtype=self.torch_dtype) if use_cuda else torch.autocast("cpu")

        with autocast_ctx:
            for b in range(B):
                # 1) Build video frames: [support] + query
                frames = [self._chw_to_pil_rgb(support_img[b])]
                frames += [self._chw_to_pil_rgb(query_imgs[b, t]) for t in range(T)]

                # 2) Init HF video inference session
                # HF API: processor.init_video_session(video=..., inference_device=..., dtype=...)
                # and then model.propagate_in_video_iterator(session) for propagation.
                session = self.processor.init_video_session(
                    video=frames,
                    inference_device=device,
                    dtype=self.torch_dtype,
                    max_vision_features_cache_size=self.max_vision_features_cache_size,
                )

                # 3) Prepare object prompts on frame 0 from semantic support mask (boxes)
                sup = support_mask[b]
                present = torch.unique(sup)
                present = [int(x) for x in present.tolist()
                           if x != 0 and x != self.ignore_index and 0 <= int(x) < C]

                if len(present) == 0:
                    out_prob[b, :, 0] = 1.0
                    continue

                obj_ids = []
                boxes = []
                cls_for_oid = {}

                oid = 1
                for cls in present:
                    bin_mask = (sup == cls)
                    box = self._mask_to_box_xyxy(bin_mask)
                    if box is None:
                        continue
                    obj_ids.append(oid)
                    boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
                    cls_for_oid[oid] = cls
                    oid += 1

                if len(obj_ids) == 0:
                    out_prob[b, :, 0] = 1.0
                    continue

                # HF expects input_boxes shaped like [ [ [x0,y0,x1,y1], ... ] ] (batch=1, objects=K)
                input_boxes = [boxes]

                # 4) Add prompts to session (frame 0)
                # processor.add_inputs_to_inference_session(..., input_boxes=..., obj_ids=...)
                self.processor.add_inputs_to_inference_session(
                    inference_session=session,
                    frame_idx=0,
                    obj_ids=obj_ids,               # list[int]
                    input_boxes=input_boxes,       # list[list[list[float]]]
                    clear_old_inputs=True,
                )

                # 5) DO NOT call self.model(... frame_idx=0)
                #    It crashes in current transformers (expects object_score_logits).
                #    Instead, start propagation from frame 0 directly.

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

                    if masks.ndim == 4 and masks.shape[1] == 1:
                        masks_nhw = masks[:, 0]
                    elif masks.ndim == 3:
                        masks_nhw = masks
                    else:
                        masks_nhw = masks.squeeze()

                    # Use session.obj_ids order to map each object to its semantic class
                    for i, oid_i in enumerate(session.obj_ids):
                        oid_i = int(oid_i)
                        cls = cls_for_oid.get(oid_i, None)
                        if cls is None:
                            continue
                        out_prob[b, t, cls] = torch.maximum(
                            out_prob[b, t, cls],
                            masks_nhw[i].clamp(0.0, 1.0),
                        )

                # 7) background channel
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
        x = img_chw.detach().float().cpu()
        # tolerate [0,1] or [0,255]
        if x.max().item() <= 1.5:
            x = (x * 255.0).round()
        x = x.clamp(0.0, 255.0).to(torch.uint8)  # (3,H,W)
        x = x.permute(1, 2, 0).numpy()           # (H,W,3)
        return Image.fromarray(x, mode="RGB")

    @staticmethod
    def _mask_to_box_xyxy(bin_mask: torch.Tensor):
        if bin_mask.dtype != torch.bool:
            bin_mask = bin_mask.bool()
        ys, xs = torch.where(bin_mask)
        if ys.numel() == 0:
            return None
        x0 = int(xs.min().item())
        y0 = int(ys.min().item())
        x1 = int(xs.max().item())
        y1 = int(ys.max().item())
        return [x0, y0, x1, y1]



