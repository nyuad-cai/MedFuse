
import torch.nn as nn
import torchvision
import torch
import numpy as np

from torch.nn.functional import kl_div, softmax, log_softmax
from .loss import RankingLoss, CosineLoss
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.resnet import ResNet

class FusionDAFT(nn.Module):

    def __init__(self, args, ehr_model, cxr_model):
	
        super(FusionDAFT, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model
        self.layer_after = args.layer_after

        #         self.mmtm0 = MMTM(64, self.ehr_model.feats_dim, self.args.mmtm_ratio)
        # self.mmtm1 = MMTM(64, self.ehr_model.feats_dim, self.args.mmtm_ratio)
        # self.mmtm2 = MMTM(128, self.ehr_model.feats_dim, self.args.mmtm_ratio)
        # self.mmtm3 = MMTM(256, self.ehr_model.feats_dim, self.args.mmtm_ratio)
        # self.mmtm4 = MMTM(512, self.ehr_model.feats_dim, self.args.mmtm_ratio)
        bottleneck_dim_0 = int(((4 * 4) + 64) / 7.0)
        bottleneck_dim_1 = int(((4 * 4) + 64) / 7.0)
        bottleneck_dim_2 = int(((4 * 4) + 128) / 7.0)
        bottleneck_dim_3 = int(((4 * 4) + 256) / 7.0)
        bottleneck_dim_4 = int(((4 * 4) + 512) / 7.0)

        self.daft_layer_0 = DAFTBlock(in_channels=256, ndim_non_img = 64, bottleneck_dim = bottleneck_dim_0, location = 0, activation = args.daft_activation)
        self.daft_layer_1 = DAFTBlock(in_channels=256, ndim_non_img = 64, bottleneck_dim = bottleneck_dim_1, location = 0, activation = args.daft_activation)
        self.daft_layer_2 = DAFTBlock(in_channels=256, ndim_non_img = 128, bottleneck_dim = bottleneck_dim_2, location = 0, activation = args.daft_activation)
        self.daft_layer_3 = DAFTBlock(in_channels=256, ndim_non_img = 256, bottleneck_dim = bottleneck_dim_3, location = 0, activation = args.daft_activation)
        self.daft_layer_4 = DAFTBlock(in_channels=256, ndim_non_img = 512, bottleneck_dim = bottleneck_dim_4, location = 0, activation = args.daft_activation)
        
    def forward(self, ehr, seq_lengths=None, img=None, n_crops=0, bs=16):
        ehr = torch.nn.utils.rnn.pack_padded_sequence(ehr, seq_lengths, batch_first=True, enforce_sorted=False)

        ehr, (ht, _)= self.ehr_model.layer0(ehr)
        ehr_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(ehr, batch_first=True)

        # resnet

        cxr_feats = self.cxr_model.vision_backbone.conv1(img)
        cxr_feats = self.cxr_model.vision_backbone.bn1(cxr_feats)
        cxr_feats = self.cxr_model.vision_backbone.relu(cxr_feats)
        cxr_feats = self.cxr_model.vision_backbone.maxpool(cxr_feats)
        if self.layer_after == 0 or self.layer_after == -1:
            ehr_unpacked = self.daft_layer_0(cxr_feats, ehr_unpacked)


        cxr_feats = self.cxr_model.vision_backbone.layer1(cxr_feats)
        if self.layer_after == 1 or self.layer_after == -1:
            ehr_unpacked = self.daft_layer_1(cxr_feats, ehr_unpacked)
        cxr_feats = self.cxr_model.vision_backbone.layer2(cxr_feats)
        if self.layer_after == 2 or self.layer_after == -1:
            ehr_unpacked = self.daft_layer_2(cxr_feats, ehr_unpacked)
        cxr_feats = self.cxr_model.vision_backbone.layer3(cxr_feats)
        if self.layer_after == 3 or self.layer_after == -1:
            ehr_unpacked = self.daft_layer_3(cxr_feats, ehr_unpacked)
        cxr_feats = self.cxr_model.vision_backbone.layer4(cxr_feats)

        if self.layer_after == 4 or self.layer_after == -1:
            ehr_unpacked = self.daft_layer_4(cxr_feats, ehr_unpacked)

        cxr_feats = self.cxr_model.vision_backbone.avgpool(cxr_feats)
        cxr_feats = torch.flatten(cxr_feats, 1)

        ehr = torch.nn.utils.rnn.pack_padded_sequence(ehr_unpacked, seq_lengths, batch_first=True, enforce_sorted=False)
        ehr, (ht, _)= self.ehr_model.layer1(ehr)
        ehr_feats = ht.squeeze()
        # cxr_preds = self.cxr_model.classifier(cxr_feats)


        out = self.ehr_model.do(ehr_feats)
        out = self.ehr_model.dense_layer(out)
        ehr_preds = torch.sigmoid(out)

        
       

        return {
            'daft_fusion': ehr_preds,
            'daft_fusion_scores': out
            }
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

class DAFTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ndim_non_img: int = 15,
        location: int = 0,
        activation: str = "linear",
        scale: bool = True,
        shift: bool = True,
        bottleneck_dim: int = 7,
    ) -> None:
        super(DAFTBlock, self).__init__()
        self.scale_activation = None
        if activation == "sigmoid":
            self.scale_activation = nn.Sigmoid()
        elif activation == "tanh":
            self.scale_activation = nn.Tanh()
        elif activation == "linear":
            self.scale_activation = None

        self.location = location
        self.film_dims = in_channels
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
        self.bottleneck_dim = bottleneck_dim
        aux_input_dims = self.film_dims
        # shift and scale decoding
        self.split_size = 0
        if scale and shift:
            self.split_size = self.film_dims
            self.scale = None
            self.shift = None
            self.film_dims = 2 * self.film_dims
        elif not scale:
            self.scale = 1
            self.shift = None
        elif not shift:
            self.shift = 0
            self.scale = None

        # create aux net
        layers = [
            ("aux_base", nn.Linear(ndim_non_img + aux_input_dims, self.bottleneck_dim, bias=False)),
            ("aux_relu", nn.ReLU()),
            ("aux_out", nn.Linear(self.bottleneck_dim, self.film_dims, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))
    def forward(self, feature_map, x_aux):
        ehr_avg = torch.mean(x_aux, dim=1)
        
        squeeze = self.global_pool(feature_map)
        squeeze = squeeze.view(squeeze.size(0), -1)
        squeeze = torch.cat((squeeze, ehr_avg), dim=1)

        attention = self.aux(squeeze)
        if self.scale == self.shift:
            v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
            v_scale = v_scale.view(v_scale.size()[0], 1, v_scale.size()[1]).expand_as(x_aux)
            v_shift = v_shift.view(v_shift.size()[0], 1, v_shift.size()[1]).expand_as(x_aux)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale is None:
            v_scale = attention
            v_scale = v_scale.view(v_scale.size()[0], 1, v_scale.size()[1]).expand_as(x_aux)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift is None:
            v_scale = self.scale
            v_shift = attention
            v_shift = v_shift.view(v_shift.size()[0], 1, v_shift.size()[1]).expand_as(x_aux)
        else:
            raise AssertionError(
                f"Sanity checking on scale and shift failed. Must be of type bool or None: {self.scale}, {self.shift}"
            )

        return (v_scale * x_aux) + v_shift