
import torch.nn as nn
import torchvision
import torch
import numpy as np

from torch.nn.functional import kl_div, softmax, log_softmax
from .loss import RankingLoss, CosineLoss, KLDivLoss
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.resnet import ResNet

class MMTM(nn.Module):
    def __init__(self, dim_visual, dim_ehr, ratio):
        super(MMTM, self).__init__()
        dim = dim_visual + dim_ehr
        dim_out = int(2*dim/ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_visual = nn.Linear(dim_out, dim_visual)
        self.fc_skeleton = nn.Linear(dim_out, dim_ehr)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, visual, skeleton):
        squeeze_array = []
        visual_view = visual.view(visual.shape[:2] + (-1,))
        squeeze_array.append(torch.mean(visual_view, dim=-1))
        ehr_avg = torch.mean(skeleton, dim=1)

        squeeze_array.append(ehr_avg)

        squeeze = torch.cat(squeeze_array, 1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        vis_out = self.fc_visual(excitation)
        sk_out = self.fc_skeleton(excitation)

        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)

        dim_diff = len(visual.shape) - len(vis_out.shape)
        vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

        dim_diff = len(skeleton.shape) - len(sk_out.shape)
        sk_out = sk_out.view(sk_out.shape[0], 1 , sk_out.shape[1])

        return visual * vis_out, skeleton * sk_out


class FusionMMTM(nn.Module):

    def __init__(self, args, ehr_model, cxr_model):
	
        super(FusionMMTM, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model

        self.mmtm0 = MMTM(64, self.ehr_model.feats_dim, self.args.mmtm_ratio)
        self.mmtm1 = MMTM(64, self.ehr_model.feats_dim, self.args.mmtm_ratio)
        self.mmtm2 = MMTM(128, self.ehr_model.feats_dim, self.args.mmtm_ratio)
        self.mmtm3 = MMTM(256, self.ehr_model.feats_dim, self.args.mmtm_ratio)
        self.mmtm4 = MMTM(512, self.ehr_model.feats_dim, self.args.mmtm_ratio)

        feats_dim = 2 * self.cxr_model.feats_dim

        self.joint_cls = nn.Sequential(
            nn.Linear(feats_dim, self.args.num_classes),
        )

        

        self.layer_after = args.layer_after
        self.projection = nn.Linear(self.ehr_model.feats_dim, self.cxr_model.feats_dim)

        self.align_loss = CosineLoss()
        self.kl_loss = KLDivLoss()

    def forward(self, ehr, seq_lengths=None, img=None, n_crops=0, bs=16):

        ehr = torch.nn.utils.rnn.pack_padded_sequence(ehr, seq_lengths, batch_first=True, enforce_sorted=False)

        ehr, (ht, _)= self.ehr_model.layer0(ehr)
        ehr_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(ehr, batch_first=True)

        cxr_feats = self.cxr_model.vision_backbone.conv1(img)
        cxr_feats = self.cxr_model.vision_backbone.bn1(cxr_feats)
        cxr_feats = self.cxr_model.vision_backbone.relu(cxr_feats)
        cxr_feats = self.cxr_model.vision_backbone.maxpool(cxr_feats)
        # 64
        if self.layer_after == 0 or self.layer_after == -1:
            cxr_feats, ehr_unpacked = self.mmtm0(cxr_feats, ehr_unpacked)

        cxr_feats = self.cxr_model.vision_backbone.layer1(cxr_feats)
        # 64
        if self.layer_after == 1 or self.layer_after == -1:
            cxr_feats, ehr_unpacked = self.mmtm1(cxr_feats, ehr_unpacked)

        cxr_feats = self.cxr_model.vision_backbone.layer2(cxr_feats)
        # 128
        if self.layer_after == 2 or self.layer_after == -1:
            cxr_feats, ehr_unpacked = self.mmtm2(cxr_feats, ehr_unpacked)
        cxr_feats = self.cxr_model.vision_backbone.layer3(cxr_feats)
        # 256

        if self.layer_after == 3 or self.layer_after == -1:
            cxr_feats, ehr_unpacked = self.mmtm3(cxr_feats, ehr_unpacked)
        cxr_feats = self.cxr_model.vision_backbone.layer4(cxr_feats)
        # 512

        if self.layer_after == 4 or self.layer_after == -1:
            cxr_feats, ehr_unpacked = self.mmtm4(cxr_feats, ehr_unpacked)



        


        cxr_feats = self.cxr_model.vision_backbone.avgpool(cxr_feats)
        cxr_feats = torch.flatten(cxr_feats, 1)


        cxr_preds = self.cxr_model.classifier(cxr_feats)
        cxr_preds_sig = torch.sigmoid(cxr_preds)


        ehr = torch.nn.utils.rnn.pack_padded_sequence(ehr_unpacked, seq_lengths, batch_first=True, enforce_sorted=False)
        ehr, (ht, _)= self.ehr_model.layer1(ehr)
        ehr_feats = ht.squeeze()
        
        ehr_feats = self.ehr_model.do(ehr_feats)
        ehr_preds = self.ehr_model.dense_layer(ehr_feats)
        ehr_preds_sig = torch.sigmoid(ehr_preds)
        
        late_average = (cxr_preds + ehr_preds)/2
        late_average_sig = (cxr_preds_sig + ehr_preds_sig)/2



        projected = self.projection(ehr_feats)
        loss = self.kl_loss(cxr_feats, projected)

        feats = torch.cat([projected, cxr_feats], dim=1)
        joint_preds = self.joint_cls(feats)

        joint_preds_sig = torch.sigmoid(joint_preds)


        
       

        return {
            'cxr_only': cxr_preds_sig,
            'ehr_only': ehr_preds_sig,
            'joint': joint_preds_sig,
            'late_average': late_average_sig,
            'align_loss': loss,

            'cxr_only_scores': cxr_preds,
            'ehr_only_scores': ehr_preds,
            'late_average_scores': late_average,
            'joint_scores': joint_preds,

            }

  