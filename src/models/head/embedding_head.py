# encoding: utf-8
import sys
import os
import torch
from torch import nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from ..backbone.layers import GeneralizedMeanPoolingP, GeneralizedMeanPooling, FastGlobalAvgPool2d, \
    AdaptiveAvgMaxPool2d, ClipGlobalAvgPool2d, Flatten, get_norm, ArcSoftmax, CircleSoftmax, CosSoftmax


class EmbeddingHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        feat_dim = cfg["MODEL"]["BACKBONE"]["FEAT_DIM"]
        embedding_dim = cfg["MODEL"]["HEADS"]["EMBEDDING_DIM"]
        num_classes = cfg["MODEL"]["HEADS"]["NUM_CLASSES"]
        neck_feat = cfg["MODEL"]["HEADS"]["NECK_FEAT"]
        pool_type = cfg["MODEL"]["HEADS"]["POOL_LAYER"]
        cls_type = cfg["MODEL"]["HEADS"]["CLS_LAYER"]
        with_bnneck = cfg["MODEL"]["HEADS"]["WITH_BNNECK"]
        norm_type = cfg["MODEL"]["HEADS"]["NORM"]

        if pool_type == 'fastavgpool':
            self.pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':
            self.pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':
            self.pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempoolP':
            self.pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == 'gempool':
            self.pool_layer = GeneralizedMeanPooling()
        elif pool_type == "avgmaxpool":
            self.pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == 'clipavgpool':
            self.pool_layer = ClipGlobalAvgPool2d()
        elif pool_type == "identity":
            self.pool_layer = nn.Identity()
        elif pool_type == "flatten":
            self.pool_layer = Flatten()
        else:
            raise KeyError(f"{pool_type} is not supported!")
        # fmt: on

        self.neck_feat = neck_feat

        bottleneck = []
        if embedding_dim > 0:
            bottleneck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            bottleneck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bottleneck = nn.Sequential(*bottleneck)

        # classification layer
        # fmt: off
        if cls_type == 'linear':
            self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':
            self.classifier = ArcSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'circleSoftmax':
            self.classifier = CircleSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'cosSoftmax':
            self.classifier = CosSoftmax(cfg, feat_dim, num_classes)
        else:
            raise KeyError(f"{cls_type} is not supported!")
        # fmt: on

    def forward(self, features):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        bn_feat = self.bottleneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]
        return bn_feat
