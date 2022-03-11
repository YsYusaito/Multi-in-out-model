
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 11:25:40 2022

@author: 10087826
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 09:05:15 2022

@author: 10087826
"""

import os
from os.path import join
import sys
import numpy as np
import glob
import pathlib
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
import tqdm

import multi_inout_net
import data_management

# os.chdir('C:\\Users\\10087826\\Documents\\multi_in_out_model')
# outputディレクトリの指定
output_dir = f'model\\'

# 一回のパラメータ更新に使うデータ数
size_batch = 128

# 学習データの学習回数
n_epoch = 3

pretrained = models.resnet50(pretrained = True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ' + str(device))

model = multi_inout_net.Net(pretrained).to(device)

# Optimizer(Adamを使用)
pram_to_update = []

# 更新対象の重みパラメータを設定する
# update_param_names = ["fc.weight", "fc.bias",
#                       "fc1.weight", "fc1.bias",
#                       "fc2.weight", "fc2.bias",
#                       "fc3.weight", "fc3.bias",
#                       "fc_y1.weight", "fc_y1.bias",
#                       "fc_y2.weight", "fc_y2.bias",
#                       "fc_y1_1.weight", "fc_y1_1.bias",
#                       "fc_y2_1.weight", "fc_y2_1.bias",
#                       "fc_y1_2.weight", "fc_y1_2.bias",
#                       "fc_y2_2.weight", "fc_y2_2.bias",
#                       "fc_y1_3.weight", "fc_y1_3.bias",
#                       "fc_y2_3.weight", "fc_y2_3.bias",
#                       "fc_y1_4.weight", "fc_y1_4.bias",
#                       "fc_y2_4.weight", "fc_y2_4.bias",
#                       "fc_y1_5.weight", "fc_y1_5.bias",
#                       "fc_y2_5.weight", "fc_y2_5.bias",
#                       "fc_y1_6.weight", "fc_y1_6.bias",
#                       "fc_y2_6.weight", "fc_y2_6.bias"]


# 指定の層の重みのみを更新できるようにする
# for name, param in model.named_parameters():
#     if name in update_param_names:
#         param.requires_grad = True
#         pram_to_update.append(param)
#     else:
#         param.requires_grad = False

# すべての層の重み更新
for name, param in model.named_parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
criterion = nn.CrossEntropyLoss()

# ロスと精度を保存するリスト（訓練用・テスト用）
list_loss_train = []
list_loss_test = []
list_acc_train = []
list_acc_test = []


# 分割されたデータを保存するリスト
data_train = []
data_test = []

train_data_list_name = 'train_data_list.txt'
test_data_list_name = 'test_data_list.txt'

train_data_path_list = []
test_data_path_list = []

with open(train_data_list_name) as f_train_data_list:
    for line in f_train_data_list:
        train_data_path_list.append(line.rstrip("\n"))

with open(test_data_list_name) as f_test_data_list:
    for line in f_test_data_list:
        test_data_path_list.append(line.rstrip("\n"))


print('--- start prepare data set ---')
print('-- train data --')
train_data_set = data_management.DataSet(train_data_path_list)
print('-- test data --')
test_data_set = data_management.DataSet(test_data_path_list)
print('--- finish prepare data set ---')

for epoch in range(n_epoch):
    print("-----------------------------------------")
    print('epoch: {}'.format(epoch))
    print('train')
    model.train()
    
    train_perm = np.random.permutation(len(train_data_path_list))
    test_perm = np.random.permutation(len(test_data_path_list))
    
    sum_loss_train = 0.
    sum_loss_test = 0.
    
    sum_acc_train = 0.
    sum_acc_test = 0.
    
    
    # 訓練
    for i in tqdm.tqdm(range(0, len(train_perm), size_batch)):
        # ミニバッチの用意
        x0_batch, x1_batch, x2_batch, x3_batch, y0_batch, y1_batch = train_data_set.get_batch_tensor(train_perm[i:i+size_batch])
        
        x0_batch = x0_batch.to(device)
        x1_batch = x1_batch.to(device)
        x2_batch = x2_batch.to(device)
        x3_batch = x3_batch.to(device)
        
        y0_batch = y0_batch.to(device)
        y1_batch = y1_batch.to(device)
        

        
        # 順伝搬
        y0_out, y1_out = model(x0_batch, x1_batch, x2_batch, x3_batch)
        
        loss0 = criterion(y0_out, y0_batch)
        loss1 = criterion(y1_out, y1_batch)
        
        acc0 = (y0_out.max(1)[1] == y0_batch).sum().item()
        acc1 = (y1_out.max(1)[1] == y1_batch).sum().item()
        
        
        loss = (loss0 + loss1)/2
        acc = (acc0 + acc1)/2
        
        # 逆伝搬
        loss.backward()
        
        # パラメータ更新
        optimizer.step()
        
        # ロス・精度を蓄積
        sum_loss_train += loss.item()
        sum_acc_train += acc
    
    mean_loss = sum_loss_train / len(train_data_path_list)
    mean_acc = sum_acc_train / len(train_data_path_list)
                                   
    list_loss_train.append(mean_loss)
    list_acc_train.append(mean_acc)
        
    print("- mean loss:", mean_loss)
    print("- mean acc:", mean_acc)
    
    # Evaluate
    print('test')
    model.eval()
    
    
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(test_data_path_list), size_batch)):
            
            # ミニバッチの用意
            x0_batch, x1_batch, x2_batch, x3_batch, y0_batch, y1_batch = test_data_set.get_batch_tensor(test_perm[i:i+size_batch])
                
            x0_batch = x0_batch.to(device)
            x1_batch = x1_batch.to(device)
            x2_batch = x2_batch.to(device)
            x3_batch = x3_batch.to(device)
            
            y0_batch = y0_batch.to(device)
            y1_batch = y1_batch.to(device)
            
            # 順伝搬
            y0_out, y1_out = model(x0_batch, x1_batch, x2_batch, x3_batch)
            
            loss0 = criterion(y0_out, y0_batch)
            loss1 = criterion(y1_out, y1_batch)
            
            acc0 = (y0_out.max(1)[1] == y0_batch).sum().item()
            acc1 = (y1_out.max(1)[1] == y1_batch).sum().item()
            
            loss = (loss0 + loss1)/2
            acc = (acc0 + acc1)/2
            
            # ロスを蓄積
            sum_loss_test += loss.item()
            sum_acc_test += acc
            
        mean_loss = sum_loss_test / len(test_data_path_list)
        mean_acc = sum_acc_test / len(test_data_path_list)
        
        list_loss_test.append(mean_loss)
        list_acc_test.append(mean_acc)
        
        print("- mean loss:", mean_loss)
        print("- mean acc:", mean_acc)
 
        
# Loss
plt.figure(figsize=(8, 5))
plt.grid(True)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(list_loss_train)
plt.plot(list_loss_test)
plt.legend(['train', 'test'])
plt.show()

# acc
plt.figure(figsize=(8, 5))
plt.grid(True)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(list_acc_train)
plt.plot(list_acc_test)
plt.legend(['train', 'test'])
plt.show()

weight = "multi_inout_weight.pth"
model_name = "multi_inout_model.pth"

torch.save(model.state_dict(), output_dir+weight)
torch.save(model, output_dir+model_name)
