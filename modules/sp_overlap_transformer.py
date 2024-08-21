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
import torch.nn.functional as F
from torchvision.models import resnet18
from modules.netvlad import NetVLADLoupe
from tools.read_samples import read_one_need_from_seq
import yaml
import math

# CircularPositionalEncoding 클래스 정의 (열 방향 인코딩에 사용)
class CircularPositionalEncoding(nn.Module):
    def __init__(self, d_model, width, max_len=5000):
        super(CircularPositionalEncoding, self).__init__()
        
        self.width = width
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Modify the encoding to reflect circularity in width dimension
        self.circular_pe = self._create_circular_encoding(pe)

        self.register_buffer('pe', pe)
    
    def _create_circular_encoding(self, pe):
        circular_pe = torch.zeros_like(pe)
        
        for i in range(pe.size(2)):
            circular_pe[:, :, i] = torch.roll(pe[:, :, i], shifts=self.width // 2, dims=1)
        
        return circular_pe

    def forward(self, x):
        seq_len, batch_size, _ = x.size()
        x = x + self.circular_pe[:seq_len, :].squeeze(1)
        return x

# Standard PositionalEncoding 클래스 정의 (행 방향 인코딩에 사용)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
    
class CustomResNet18(nn.Module):
    def __init__(self, channels=1):
        super(CustomResNet18, self).__init__()
        self.resnet = resnet18(pretrained=False)
        
        # 첫 번째 Conv2d 레이어 조정
        self.resnet.conv1 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # MaxPool 레이어를 그대로 사용하여 적절한 다운샘플링
        self.resnet.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # layer4의 채널 수를 256으로 조정 (기존 512 -> 256)
        self.resnet.layer4[0].conv1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.resnet.layer4[0].bn1 = nn.BatchNorm2d(256)
        self.resnet.layer4[0].conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.resnet.layer4[0].bn2 = nn.BatchNorm2d(256)
        self.resnet.layer4[0].downsample = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256)
        )

        self.resnet.layer4[1].conv1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.resnet.layer4[1].bn1 = nn.BatchNorm2d(256)
        self.resnet.layer4[1].conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.resnet.layer4[1].bn2 = nn.BatchNorm2d(256)
        
        # 마지막 fully connected layer 제거 (이미 Identity로 설정)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        x = self.resnet.conv1(x)  # (batch_size, 64, 64, 900)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # (batch_size, 64, 32, 450)

        x = self.resnet.layer1(x)  # (batch_size, 64, 32, 450)
        x = self.resnet.layer2(x)  # (batch_size, 128, 16, 225)
        x = self.resnet.layer3(x)  # (batch_size, 256, 8, 113)
        x = self.resnet.layer4(x)  # (batch_size, 256, 8, 113)

        # 추가적인 업샘플링 레이어를 사용하여 원하는 출력 크기를 얻습니다.
        x = nn.functional.interpolate(x, size=(16, 60), mode='bilinear', align_corners=False)  # (batch_size, 256, 16, 60)

        return x

    def forward(self, x):
        return self.resnet(x)

# OverlapTransformer의 Feature Extractor 정의
class FeatureExtractor(nn.Module):
    def __init__(self, height=64, width=900, channels=1, use_transformer=True):
        super(FeatureExtractor, self).__init__()
        self.use_transformer = use_transformer

        # ResNet18을 사용하여 Feature Extractor 구성
        self.resnet = CustomResNet18()

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, activation='relu', batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Positional Encodings
        self.positional_encoder_row = PositionalEncoding(128)
        self.positional_encoder_col = CircularPositionalEncoding(128, width=60)  # 열 방향 포지셔널 인코딩
        
        # 최종 디스크립터 레이어
        self.final_conv = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.final_bn = nn.BatchNorm2d(256)
        self.final_linear = nn.Linear(256 * 32 * 60, 32 * 60)
    
    def forward(self, x_l):
        # ResNet18으로 피처 추출
        print("x_l size: \n")
        print(x_l.size())
        features = self.resnet(x_l)  # Output shape: (batch_size, 512, h/32, w/32)

        print("features size: \n")
        print(features.size())

        # 채널 수를 줄이고, Transformer에 입력될 크기로 조정
        features_conv = self.final_conv(features)
        features_conv = self.final_bn(features_conv)
        features_conv = features_conv.permute(0, 2, 3, 1)  # Reshape to (batch_size, h, w, channels)
        features_conv = features_conv.flatten(2)  # Flatten the spatial dimensions (batch_size, h, w * channels)

        # Positional Encoding 추가 (행, 열 방향 모두)
        features_pe = self.positional_encoder_row(features_conv)
        features_pe = self.positional_encoder_col(features_pe)

        # Transformer 적용
        if self.use_transformer:
            features_transformed = self.transformer_encoder(features_pe)
        
        # ResNet18 피처와 Transformer 출력 결합
        features_cat = torch.cat((features_conv, features_transformed), dim=-1)

        # 최종 디스크립터 출력
        features_cat = features_cat.view(features_cat.size(0), -1)  # Flatten for the final linear layer
        descriptors = self.final_linear(features_cat)
        descriptors = descriptors.view(features_cat.size(0), 32, 60)  # Reshape to (batch_size, 32, 60)
        
        return descriptors




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
