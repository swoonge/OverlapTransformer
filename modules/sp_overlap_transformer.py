#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: OverlapTransformer modules for KITTI sequences


import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
import torch
import torch.nn as nn

from modules.netvlad import NetVLADLoupe
import torch.nn.functional as F
from tools.read_samples import read_one_need_from_seq
import yaml
import math

"""
    Feature extracter of OverlapTransformer.
    Args:
        height: the height of the range image (64 for KITTI sequences). 
                 This is an interface for other types LIDAR.
        width: the width of the range image (900, alone the lines of OverlapNet).
                This is an interface for other types LIDAR.
        channels: 1 for depth only in our work. 
                This is an interface for multiple cues.
        norm_layer: None in our work for better model.
        use_transformer: Whether to use MHSA.
"""
class FeatureExtractor(nn.Module):
    def __init__(self, height=64, width=900, channels=5, norm_layer=None, use_transformer=True):
        super(FeatureExtractor, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.use_transformer = use_transformer

        # CNN Layers
        self.conv1 = nn.Conv2d(channels, 8, kernel_size=(3, 3), stride=(2, 2), bias=False, padding=1)  # [batch size, 8, 32, 450]
        self.bn1 = norm_layer(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), bias=False, padding=1)  # [batch size, 16, 32, 450]
        self.bn2 = norm_layer(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 2), bias=False, padding=1)  # [batch size, 32, 32, 225]
        self.bn3 = norm_layer(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), bias=False, padding=1)  # [batch size, 64, 32, 113]
        self.bn4 = norm_layer(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), bias=False, padding=1)  # [batch size, 128, 16, 57]
        self.bn5 = norm_layer(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), bias=False, padding=1)  # [batch size, 128, 16, 57]
        self.bn6 = norm_layer(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), bias=False, padding=1)  # [batch size, 128, 16, 57]
        self.bn7 = norm_layer(128)
        self.relu = nn.ReLU(inplace=True)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, activation='relu', batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Linear layer to get final descriptor size
        self.final_linear = nn.Linear(128, 128)

    def forward(self, x_l):
        # CNN forward pass
        out_l = self.relu(self.bn1(self.conv1(x_l)))
        out_l = self.relu(self.bn2(self.conv2(out_l)))
        out_l = self.relu(self.bn3(self.conv3(out_l)))
        out_l = self.relu(self.bn4(self.conv4(out_l)))
        out_l = self.relu(self.bn5(self.conv5(out_l)))
        out_l = self.relu(self.bn6(self.conv6(out_l)))
        out_l = self.relu(self.bn7(self.conv7(out_l)))

        # Positional Encoding
        N, C, H, W = out_l.size()
        pe = self._circular_positional_encoding(C, H, W).to(out_l.device).unsqueeze(0).expand(N, -1, -1, -1)
        out_l += pe

        # Transformer Encoder
        if self.use_transformer:
            out_l = out_l.permute(0, 2, 3, 1).contiguous().view(N * H, W, C)  # [N*H, W, C]
            out_l = self.transformer_encoder(out_l)
            out_l = out_l.view(N, H, W, C).permute(0, 3, 1, 2)  # [N, C, H, W]

        # Concatenate Transformer output with CNN output
        out_l = torch.cat((out_l, out_l), dim=1)  # Concatenation

        # Final linear layer to reduce dimensions
        out_l = self.final_linear(out_l.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [N, C, H, W]

        return out_l

    def _circular_positional_encoding(self, channels, height, width):
        """ Generates circular positional encoding for the width dimension. """
        pe = torch.zeros(channels, height, width)
        position = torch.arange(0, width, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2).float() * (-math.log(10000.0) / channels))

        pe[0::2, :, :] = torch.sin(position * div_term).permute(1, 0).unsqueeze(0).expand(height, -1, -1).permute(1, 0, 2)
        pe[1::2, :, :] = torch.cos(position * div_term).permute(1, 0).unsqueeze(0).expand(height, -1, -1).permute(1, 0, 2)

        return pe.to(position.device)


if __name__ == '__main__':
    # load config ================================================================
    config_filename = '../config/config.yml'
    config = yaml.safe_load(open(config_filename))
    seqs_root = config["data_root"]["data_root_folder"]
    # ============================================================================

    combined_tensor = read_one_need_from_seq(seqs_root, "000000","00")
    combined_tensor = torch.cat((combined_tensor,combined_tensor), dim=0)

    feature_extracter=FeatureExtractor(use_transformer=True, channels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extracter.to(device)
    feature_extracter.eval()

    print("model architecture: \n")
    print(feature_extracter)

    gloabal_descriptor = feature_extracter(combined_tensor)
    print("size of gloabal descriptor: \n")
    print(gloabal_descriptor.size())
