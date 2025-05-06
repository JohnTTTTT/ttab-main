import timm
import torch.nn as nn
from torch import Tensor
from torch import nn as _nn

class MyAffectNetViT(_nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_large_patch16_224", pretrained=False
        )
        embed_dim = self.backbone.embed_dim
        # authors’ final norm
        self.backbone.fc_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # alias it so forward_features’ self.norm(x) calls your fc_norm
        self.backbone.norm = self.backbone.fc_norm

        in_feats = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_feats, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        # 5) Run through patch‐embed + transformer blocks:
        #    this gives a [B, N+1, C] token tensor
        x = self.backbone.forward_features(x)
        # 6) Grab the CLS token (first token)
        cls = x[:, 0]                       # shape [B, C]
        # 7) Apply the authors’ fc_norm then your head
        return self.backbone.head(self.backbone.fc_norm(cls))
