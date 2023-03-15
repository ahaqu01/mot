import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.resnet_distill import build_resnet_backbone_distill
from .head.embedding_head import EmbeddingHead


class FeatureExtraction(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()

        self.device = device

        self.backbone = build_resnet_backbone_distill(cfg)
        self.backbone.to(device)
        self.backbone.eval()
        self.heads = EmbeddingHead(cfg)
        self.heads.to(device)
        self.heads.eval()

        self.pixel_mean = torch.tensor(cfg["MODEL"]["PIXEL_MEAN"]).view(1, -1, 1, 1).to(device)
        self.pixel_std = torch.tensor(cfg["MODEL"]["PIXEL_STD"]).view(1, -1, 1, 1).to(device)

    def forward(self, batched_inputs):
        batched_inputs = batched_inputs.to(self.device)
        batched_inputs.sub_(self.pixel_mean).div_(self.pixel_std)
        features = self.backbone(batched_inputs)
        predictions = self.heads(features)
        features = F.normalize(predictions)
        features = features.cpu().data
        return features
