# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 14:24:34 2022

@author: 10087826
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, pretrained):
        # スーパークラス（Module クラス）の初期化メソッドを実行 
        super().__init__() 
        
        self.pretrained = pretrained
        
        self.pretrained.fc = nn.Linear(2048, 128)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(256, 128)
        self.fc_y1 = nn.Linear(128*2, 64)
        self.fc_y1_1 = nn.Linear(64, 32)
        self.fc_y1_2 = nn.Linear(32, 16)
        self.fc_y1_3 = nn.Linear(16, 32)
        self.fc_y1_4 = nn.Linear(32, 16)
        self.fc_y1_5 = nn.Linear(16, 8)
        self.fc_y1_6 = nn.Linear(8, 2)
        
        self.fc_y2 = nn.Linear(128*2, 64)
        self.fc_y2_1 = nn.Linear(64, 32)
        self.fc_y2_2 = nn.Linear(32, 16)
        self.fc_y2_3 = nn.Linear(16, 32)
        self.fc_y2_4 = nn.Linear(32, 16)
        self.fc_y2_5 = nn.Linear(16, 8)
        self.fc_y2_6 = nn.Linear(8, 2)
        
        self.bn_256 = nn.BatchNorm1d(num_features = 256)
        self.bn_128 = nn.BatchNorm1d(num_features = 128)
        self.bn_64 = nn.BatchNorm1d(num_features = 64)
        self.bn_32 = nn.BatchNorm1d(num_features = 32)
        self.bn_16 = nn.BatchNorm1d(num_features = 16)
        self.bn_8 = nn.BatchNorm1d(num_features = 8)

    def forward(self, x0, x1, x2, x3): # 入力から出力を計算するメソッドを定義
        x0 = self.pretrained(x0)
        x1 = F.relu(self.bn_128(self.fc1(x1)))
        x2 = F.relu(self.bn_128(self.fc2(x2)))
        x3 = F.relu(self.bn_128(self.fc3(x3)))
        
        x123 = x1 + x2 + x3
        x = torch.cat((x0, x123), 1)
        
        y0 = F.relu(self.bn_64(self.fc_y1(x)))
        y0 = F.relu(self.bn_32(self.fc_y1_1(y0)))
        y0 = F.relu(self.bn_16(self.fc_y1_2(y0)))
        y0 = F.relu(self.bn_32(self.fc_y1_3(y0)))
        y0 = F.relu(self.bn_16(self.fc_y1_4(y0)))
        y0 = F.relu(self.bn_8(self.fc_y1_5(y0)))
        y0 = self.fc_y1_6(y0)
        
        y1 = F.relu(self.bn_64(self.fc_y2(x)))
        y1 = F.relu(self.bn_32(self.fc_y2_1(y1)))
        y1 = F.relu(self.bn_16(self.fc_y2_2(y1)))
        y1 = F.relu(self.bn_32(self.fc_y2_3(y1)))
        y1 = F.relu(self.bn_16(self.fc_y2_4(y1)))
        y1 = F.relu(self.bn_8(self.fc_y2_5(y1)))
        y1 = self.fc_y2_6(y1)
        
        return y0, y1