# simple_svss_model.py
# A minimal SVSS model that uses the support frame+mask to build a class prototype
# and predicts semantic masks for query frames.
#
# Forward:
#   logits = model(support_img, support_mask, query_imgs)
# Shapes:
#   support_img : (B,3,H,W)
#   support_mask: (B,H,W)  labels in {0..C-1} or ignore_index
#   query_imgs  : (B,T,3,H,W)
#   logits      : (B,T,C,H,W)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSVSSProtoModel(nn.Module):
    def __init__(self, num_classes=13, feat_dim=64, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Tiny encoder (fast baseline)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, feat_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, support_img, support_mask, query_imgs):
        B, T, _, H, W = query_imgs.shape

        # Encode support and query
        sup_f = self.encoder(support_img)                         # (B,D,H,W)
        qry_f = self.encoder(query_imgs.view(B*T, 3, H, W))        # (B*T,D,H,W)
        qry_f = qry_f.view(B, T, -1, H, W)                         # (B,T,D,H,W)

        # Build class prototypes from support (mean feature per class)
        # proto: (B,C,D)
        D = sup_f.shape[1]
        proto = sup_f.new_zeros((B, self.num_classes, D))
        counts = sup_f.new_zeros((B, self.num_classes, 1))

        sup_mask = support_mask.clone()
        valid = (sup_mask != self.ignore_index)
        sup_mask = sup_mask.clamp(min=0, max=self.num_classes - 1)

        # one-hot: (B,C,H,W) but masked
        oh = F.one_hot(sup_mask, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        oh = oh * valid.unsqueeze(1).float()

        # sum features per class
        # sup_f: (B,D,H,W) -> (B,1,D,H,W)
        sup_f_ = sup_f.unsqueeze(1)                                # (B,1,D,H,W)
        oh_ = oh.unsqueeze(2)                                      # (B,C,1,H,W)
        summed = (oh_ * sup_f_).sum(dim=(3, 4))                    # (B,C,D)
        cnt = oh.sum(dim=(2, 3), keepdim=False).unsqueeze(-1)      # (B,C,1)

        proto = summed / (cnt + 1e-6)                              # (B,C,D)

        # Score query pixels by dot-product with prototypes:
        # qry_f: (B,T,D,H,W), proto: (B,C,D)
        # logits: (B,T,C,H,W)
        logits = torch.einsum("bt dhw, bcd -> btchw", qry_f, proto)

        return logits
